#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "pyqt5>=5.15.11",
# ]
# ///

import sys
import os
import signal
import traceback
#traceback.print_exc(file=sys.stderr)
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5 import QtCore, QtWidgets, QtGui

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        safe_text = QtCore.Qt.escape(text) if hasattr(QtCore.Qt, "escape") else text
        self.textWritten.emit(safe_text)

    def flush(self):
        pass

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.setWindowTitle(f"Matplotlib Plot - {os.path.basename(filepath)}")

        # Central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(central_widget)
        self.resize(800, 500)

        # Top bar: toolbar + save button
        self.top_bar_container = QtWidgets.QWidget()
        self.top_bar_layout = QtWidgets.QHBoxLayout(self.top_bar_container)
        self.top_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.top_bar_container.setMaximumSize(300, 40)
        self.main_layout.addWidget(self.top_bar_container)

        # Placeholder for canvas
        self.canvas = None
        self.toolbar = None

        # Log window
        self.log_text_edit = QtWidgets.QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumHeight(500)

        # Increase the font size of the log window
        font = self.log_text_edit.font()
        font.setPointSize(14)  # Set the desired font size here
        self.log_text_edit.setFont(font)

        # Decrease Log line spacing
        #self.log_text_edit.document().setDefaultStyleSheet("p { line-height: 65%; }")

        # Checkbox container
        self.checkbox_container = QtWidgets.QWidget()
        self.checkbox_layout = QtWidgets.QHBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        # Redraw label
        self.reload_log_label = QtWidgets.QLabel("Upon plot reload: ")
        # Checkbox clear log
        self.clear_log_checkbox = QtWidgets.QCheckBox("Clear log")
        self.clear_log_checkbox.setChecked(True)  # default ON
        # Checkbox redraw whole window
        self.redraw_window_checkbox = QtWidgets.QCheckBox("Redraw window")
        self.redraw_window_checkbox.setChecked(True)  # default ON
        # Add checkbox to this container
        self.checkbox_layout.setSpacing(2) # set spacing between items
        self.checkbox_layout.addWidget(self.reload_log_label)
        self.checkbox_layout.addWidget(self.clear_log_checkbox)
        self.checkbox_layout.addWidget(self.redraw_window_checkbox)
        self.checkbox_layout.addStretch() # fill space on right side of boxes -> boxes are left-aligned
        self.main_layout.addWidget(self.checkbox_container)
        
        self.log_container = QtWidgets.QWidget()
        self.log_layout = QtWidgets.QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        # Add checkbox and log window to this container
        self.log_layout.addWidget(self.log_text_edit)
        self.main_layout.addWidget(self.log_container)
        

        # Redirect stdout
        stdout_stream = EmittingStream()
        stdout_stream.textWritten.connect(self.onTextWritten)
        sys.stdout = stdout_stream

        # Watcher
        self.file_watcher = QtCore.QFileSystemWatcher([self.filepath])
        self.file_watcher.fileChanged.connect(self.reload_plot)

        # Load initial plot
        self.reload_plot()

    def onTextWritten(self, text):

        cursor = self.log_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(f'{text}')
        self.log_text_edit.setTextCursor(cursor)
        self.log_text_edit.ensureCursorVisible()
        

    def build_canvas_and_toolbar(self, fig):
        # Remove old canvas and toolbar if they exist
        if self.toolbar is not None:
            self.top_bar_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas is not None:
            self.main_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        # Create new canvas and toolbar
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Rebuild top bar: insert toolbar
        while self.top_bar_layout.count():
            item = self.top_bar_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.top_bar_layout.addWidget(self.toolbar)

        # Add canvas below
        self.main_layout.insertWidget(1, self.canvas)

    def execute_plot_code(self):
        plt.close('all')
        # Use the current globals as the execution environment
        ns = globals().copy()
        ns["plt"] = plt  # explicitly ensure plt is in there
        with open(self.filepath, 'r') as f:
            code = f.read()
        exec(code, ns)
        return plt.gcf()

    def reload_plot(self):
        # Clear log window when we draw a new plot
        if self.clear_log_checkbox.isChecked():
            self.log_text_edit.clear()
        try:
            fig = self.execute_plot_code()
            if self.redraw_window_checkbox.isChecked():
                # Resize based on figure size
                dpi = fig.get_dpi()
                w_in, h_in = fig.get_size_inches()
                self.resize(int(w_in * dpi), int(h_in * dpi + 200))
            self.build_canvas_and_toolbar(fig)
        except Exception as e:
            print(f"Error reloading plot: {e}")

        # Re-add watcher if needed
        if not self.file_watcher.files():
            self.file_watcher.addPath(self.filepath)

def excepthook(exc_type, exc_value, exc_traceback):
    # Don't swallow KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = excepthook
            
def main():
    if len(sys.argv) != 2:
        print(f"Usage: `{os.path.basename(__file__)} <matplotlib_code_file>`")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)

    # hack to enable Ctrl-C closing of the application -- ignores any cleanup and exits ungracefully.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    window = PlotWindow(filepath)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
