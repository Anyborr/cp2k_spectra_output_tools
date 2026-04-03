#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from collections import deque
from collections.abc import Generator
from decimal import Decimal
import math
        
def get_str_nbr_from_deque(deque, line_key_string, index) -> float:
    # look for 'line_key_string' as a key and extract the index item as a float point number
    # we expect there will only be one number to extract from the deque. When found we exit.
    for line in deque:
        if line.startswith(line_key_string):
            parts = line.split()
            if parts:
                try:
                    number = float(parts[index])
                    return number
                except (IndexError, ValueError):
                    continue
    raise KeyError("String not found in deque buffer!")
                
def extract_with_buffer_from_string(filename, buffer_key_string, line_key_string, index, buffer_size) -> Generator[float, None, None]:
    # This creates an iterator
    buffer = deque(maxlen=buffer_size)
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            buffer.append(line)
            if line.startswith(buffer_key_string):
                number = get_str_nbr_from_deque(buffer, line_key_string, index)
                yield number

def extract_from_string(filename, line_key_string, index) -> Generator[float, None, None]:
    # This creates an iterator
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(line_key_string):
                parts = line.split()
                if parts:
                    try:
                        number = float(parts[index])
                        yield number
                    except Exception as e:
                        print(f"Fail: {e}")

def extract_many_from_string(filename, line_key_string, indexes) -> Generator[list[float], None, None]:
    if len(indexes) == 1:
        raise ValueError("This function requires multiple indexes, try instead 'extract_from_string()'")
    # This creates an iterator
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(line_key_string):
                parts = line.split()
                if parts:
                    try:
                        number = []
                        for i in range(len(parts)):
                            if i in indexes:
                                number.append(float(parts[i]))
                        yield number
                    except Exception as e:
                        print(f"Fail: {e}")


# Function used for zero-padding
def findNextPwrOfTwo(n: int, mult: int|None) -> int:
    if mult is None:
        return n

    # finds the next power of two value m larger than n
    # multiplies m by mult, where mult must be a power of two as well    
    assert n != 0
    # check that mult is a power of two
    isPowerOfTwo = True if Decimal(math.log2(mult)) % 1 == 0 else False
    if not isPowerOfTwo:
        raise ValueError('ERROR: supplied padding_mult value is not a nonzero power of two!')
    m = 2 ** int(np.ceil(np.log2(n)))  # Next power of 2
    largePwrOfTwo = m * mult
    return largePwrOfTwo


def generate_window_function(time_in_fs, wtype=None) -> npt.NDArray[float]:
    match wtype:
        case None:
            window = np.ones(len(time_in_fs))
        case 'gauss':
            gaussian_edge_value = 1E-10  # desired value at edge of gaussian (the smallest value)
            alpha = -np.log(gaussian_edge_value) / time_in_fs[-1]**2
            window = np.exp(-alpha * (time_in_fs - time_in_fs[0])**2)
            print(f"gauss alpha: {alpha}")
        case 'hann':
            window = np.hanning(len(time_in_fs))
        case 'lorentzian':
            eta = 0.01  # fs^-1, controls lifetime
            window = np.exp(-eta * (time_in_fs - time_in_fs[0]))
        case 'orca':
            t_max = time_in_fs[-1]
            window = 1 - 3*(time_in_fs/t_max)**2 + 2*(time_in_fs/t_max)**3
        case _:
            raise KeyError("Unknown window type specified!")
    return window

def apply_gaussian_broadening(energy, fdip, width, resolution=100, energy_min=None, energy_max=None) -> list[npt.NDArray[float]]:
    if isinstance(width,float):
        L_width = np.ones(len(energy))*width
    elif isinstance(width,list) or isinstance(width, np.ndarray):
        L_width = np.array(width)
    else:
        raise TypeError('Width variable needs to be supplied as Type float, list, or numpy.ndarray')

    #construct dynamic resolution based on energy span
    if (energy_min is None) or (energy_max is None):
        energy_padding = (max(energy) - min(energy))*0.1
        energy_min = min(energy) - energy_padding
        energy_max = max(energy) + energy_padding

    # print(f'energy_min: {energy_min}')
    # print(f'energy_max: {energy_max}')
    
        
    spec_resolution = int((energy_max-energy_min)*resolution)
    L_omega = np.linspace(energy_min, energy_max, spec_resolution)    
    L_y = np.zeros(len(L_omega))
    for i in range(len(energy)):
        #L_y += fdip[i]/(L_width[i]*np.sqrt(2.*np.pi))*np.exp(-0.5*((L_omega-energy[i])/L_width[i])**2)
        L_y += fdip[i]*np.exp(-0.5*((L_omega-energy[i])/L_width[i])**2)
    return L_omega, L_y



#=================
# RTP
#=================
def rtp_get_energy_from_out(filename):
    buffer_key = 'Time needed for propagation'
    line_key = 'Total energy'
    number_placement = 2
    buffer_size = 4
    RTP_energy = np.fromiter(extract_with_buffer_from_string(filename, buffer_key, line_key, number_placement, buffer_size), float)
    return RTP_energy

# 
def rtp_get_current_from_jint(filename):
    data = np.loadtxt(filename)
    current = data[:, 2:5]
    return current

def rtp_get_time_from_jint(filename):
    data = np.loadtxt(filename)
    time_in_fs = data[:,1]
    return time_in_fs
    
# numpy array of dim 3 with dipoles [x,y,z]
# NOTE: cp2k defaults to print dipole for the groundstate first, which should be removed
#       manually. Also, the time of each step is not included in the output file.
def rtp_get_dipole_from_dipole(filename):
    # returns dipole in x,y,z as a tuple of numpy arrays
    RTP_dipole = np.fromiter(extract_many_from_string(filename, 'X= ', (1,3,5)), dtype=np.dtype((float, 3)))
    return RTP_dipole

def rtp_get_efield_from_field(filename) -> tuple[list, list, list]:
    # returns efield in x,y,z as a tuple of numpy arrays
    field_txt = np.loadtxt(filename)
    field = (field_txt[:,2], field_txt[:,3], field_txt[:,4])
    return field
    
def rtp_get_vecpot_from_field(filename) -> tuple[list, list, list]:
    # returns vector potential in x,y,z as a tuple of numpy arrays
    field_txt = np.loadtxt(filename)
    vecpot = (field_txt[:,5], field_txt[:,6], field_txt[:,7])
    return vecpot

def rtp_correct_for_berryphase_jump(dipole, berry_vol):
    # if the system is periodic we should calculate the dipole within the berry phase
    # therefore, we need to look for any phase shifts and correct for it

    # initial value of traj to use as reference
    traj_start = dipole[0]
    # we calculate if the difference between the signal value and the traj_start is large enough to necessitate a shift by the cell length
    # the traj_shift will be zero or an integer multiplied with the dipole
    traj_shift = berry_vol * ((dipole - traj_start) / berry_vol).round()
    # perform the shift (traj_shift is zero if no shift is needed)
    dipole_out = dipole - traj_shift
    return dipole_out


def get_shortest_len(*arrays) -> int:
    shortest = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) < shortest:
            shortest = len(arr)
    return shortest
    

def do_fft_ND(time_array, signals, padding_mult=None, window_type=None):
    # Wrapper to perform fft of real signals, with optional window function and padding
    # accepts N input time-signals in the List(signals) container variable and truncates to the shortest, so they are all same length
    sig_arrays = signals
    
    # All input arrays must be the same length
    truncated = get_shortest_len(time_array, *sig_arrays)
    
    time_array = time_array[:truncated]
    for idx in range(len(sig_arrays)):
        sig_arrays[idx] = sig_arrays[idx][:truncated]
    
    # Compute time step and time interval
    time_step = time_array[1] - time_array[0]  # Assuming evenly spaced time data

    # Apply windowing function to reduce noise from truncated sig_arrays
    window = generate_window_function(time_array, window_type)
    for idx in range(len(sig_arrays)):
        sig_arrays[idx] = sig_arrays[idx] * window

    # Zero padding to increase FFT resolution
    nr_steps_with_padding = findNextPwrOfTwo(truncated, padding_mult)
    nr_padded_zeros = nr_steps_with_padding - truncated
    time_array = np.pad(time_array, (0, nr_padded_zeros))
    for idx in range(len(sig_arrays)):
        sig_arrays[idx] = np.pad(sig_arrays[idx], (0, nr_padded_zeros))

    # Perform real-signal fast-fourier-transform
    arr_fft = []
    for idx in range(len(sig_arrays)):
        arr_fft.append(np.fft.rfft(sig_arrays[idx]))
    frequency = np.fft.rfftfreq(nr_steps_with_padding, time_step)

    return frequency, arr_fft


def rtp_generate_spectrum(time_array, signals, fields=None, window_type=None, mode=None, padding_mult=None):
    # Make sure we are working in R3
    if len(signals) != 3 or (fields is not None and len(fields) != 3):
        raise ValueError('Dimension of the signal and/or fields arrays must be 3!')
    # Constants
    h_over_e = 4.135667696  # Planck's constant over elementary charge in eV·fs
    speed_of_light = 299792458  # Speed of light in m/s

    # high resolution FFT of our signals
    # # padding multiplier, larger -> artifically increases fft resolution
    freq_fs, fft_signals = do_fft_ND(time_array, signals, padding_mult, window_type)
    
    if fields is not None:
        _, fft_fields = do_fft_ND(time_array, fields, padding_mult)

        # Here we shift the signal based on the applied field so they end up in the
        # correct parts of the FFT output
        for idx in range(3):        
            ## TODO: review if we want to divide by the full complex fft_fields
            ## (see https://pubs.acs.org/doi/10.1021/acs.jctc.6b00511 equation [9] )
            field_mod = fft_fields[idx].real**2 + fft_fields[idx].imag**2
            field_conj = np.conj(fft_fields[idx])
            fft_fields[idx] = field_mod / field_conj
            fft_signals[idx] /= fft_fields[idx]

    # Compute the field-scaled trace of the FFT output
    fft_trace = np.zeros(len(fft_signals[0]), dtype=np.complex128)  # fft_trace = 0
    for idx in range(3):
        fft_trace += fft_signals[idx] # / fft_fields[idx]

        
    # # We assume imaginary component of fft_fields are very small and can be discarded
    # for idx in range(3):        
    #     fft_fields[idx] = fft_fields[idx].real
        
    # # Compute the field-scaled trace of the FFT output
    # fft_trace = 0
    # for idx in range(3):
    #     fft_trace += fft_signals[idx] / fft_fields[idx]



    # We want the spectrum in electron volts [eV]
    freq_eV = freq_fs * h_over_e

    # We only want non-zero frequencies to avoid dividing by zero later
    freq_idx = freq_eV > 0
    freq_eV = freq_eV[freq_idx]
    # [WARNING] multiply by `1j` is a hack !! not sure why this works??
    fft_trace = fft_trace[freq_idx]#*1j
    
    match mode:
        case 'dipole':
            # Scaling
            numer = 4*np.pi
            denom = 3*speed_of_light
            prefactor = 1 #numer/denom
            # Compute absorption
            absorption = prefactor * fft_trace.imag# * freq_eV
            
        case 'current':
            # Currently the correct scaling for the macroscopic current density as printed from CP2K is missing,
            # therefore we obtain higher magnitude scaling for the current than the dipole moment abs spectrum.
            # Scaling
            prefactor = 1 #4*np.pi
            # Compute absorption
            absorption = prefactor * fft_trace.real# / freq_eV

        case _:
            raise TypeError('You need to specify the source of your signal (dipole / current suppported)')
    
    return freq_eV, absorption


def rtp_generate_spectrum_LEGACY(time_in_fs, signal_xx, signal_yy, signal_zz, field_xx, field_yy, field_zz, window_type, mode):
    # Constants
    h_over_e = 4.135667696e-15  # Planck's constant over elementary charge in eV·s
    speed_of_light = 299792458  # Speed of light in m/s
    
    # Ensure same length of all series
    L_data = [time_in_fs, signal_xx, signal_yy, signal_zz]
    data_min = 0
    for i in range(len(L_data)):
        if data_min !=0:
            if len(L_data[i]) < data_min:
                data_min = len(L_data[i])
        else:
            data_min = len(L_data[i])
    time_in_fs = time_in_fs[:data_min]
    signal_xx = signal_xx[:data_min]
    signal_yy = signal_yy[:data_min]
    signal_zz = signal_zz[:data_min]
    field_xx = field_xx[:data_min]
    field_yy = field_yy[:data_min]
    field_zz = field_zz[:data_min]

    
    # Compute time step and time interval
    time_step = time_in_fs[1] - time_in_fs[0]  # Assuming evenly spaced time data
    time_interval = time_step * 1e-15  # Convert time step from femtoseconds to seconds
    
    # Zero padding to increase FFT resolution
    # nr_steps_padded = len(time_in_fs)  # This ignores padding
    nr_steps_padded = findNextPwrOfTwo(len(time_in_fs), 4)  # This enables padding

    # Multiply the current J(t) by a window function
    # (e.g., Gaussian or exponential decay) to suppress end effects and smoothen the FFT.
    window = generate_window_function(time_in_fs, window_type)
#    plt.plot(window)
    
    signal_damped_xx = signal_xx * window
    signal_damped_yy = signal_yy * window
    signal_damped_zz = signal_zz * window
    
    padded_signal_xx = np.pad(signal_damped_xx, (0, nr_steps_padded - len(time_in_fs)))
    padded_signal_yy = np.pad(signal_damped_yy, (0, nr_steps_padded - len(time_in_fs)))
    padded_signal_zz = np.pad(signal_damped_zz, (0, nr_steps_padded - len(time_in_fs)))
    padded_field_xx = np.pad(field_xx, (0, nr_steps_padded - len(time_in_fs)))
    padded_field_yy = np.pad(field_yy, (0, nr_steps_padded - len(time_in_fs)))
    padded_field_zz = np.pad(field_zz, (0, nr_steps_padded - len(time_in_fs)))

    # re-create time array to account for zero-padding
    time_in_fs = np.arange(nr_steps_padded) * time_step

    fft_J_damped_xx = np.fft.fft(padded_signal_xx)
    fft_J_damped_yy = np.fft.fft(padded_signal_yy)
    fft_J_damped_zz = np.fft.fft(padded_signal_zz)

    fft_field_xx = np.fft.fft(padded_field_xx).real
    fft_field_yy = np.fft.fft(padded_field_yy).real
    fft_field_zz = np.fft.fft(padded_field_zz).real
    
    # Compute Fourier transform of each component
    freqs_Hz = np.fft.fftfreq(len(time_in_fs), time_interval)
    freqs_eV = freqs_Hz * h_over_e
        
    # Compute conductivity for x-component
    sigma_xx = fft_J_damped_xx / fft_field_xx
    sigma_yy = fft_J_damped_yy / fft_field_yy
    sigma_zz = fft_J_damped_zz / fft_field_zz
    trace_sigma = sigma_xx+sigma_yy+sigma_zz
    # We only care about the frequencies from the real part of the time dependent signals (could have used np.fft.rfft instead ...
    pos_idx = freqs_eV > 0
    freqs_pos = freqs_eV[pos_idx]

    
    if mode == 'dipole':
        # Scaling
        numer = 4*np.pi
        denom = 3*speed_of_light
        prefactor = numer/denom

        # Compute absorption
        absorption = freqs_pos * trace_sigma[pos_idx].imag * prefactor
    elif mode == 'current':
        # Currently the correct scaling for the macroscopic current density as printed from CP2K is missing,
        # therefore we obtain higher magnitude scaling for the current than the dipole moment abs spectrum.
        # Scaling
        prefactor = 4*np.pi
        # Compute absorption
        absorption =  prefactor * trace_sigma[pos_idx].real / freqs_pos

    return freqs_pos, absorption


#==========================================================================================
# LR-TDDFT
# Author: Adaptation of an initial code by Augustin Bussy and Guillaume Le Breton (2022).
#         Extended for compatibility with trajectories by André Borrfors (2023)
#==========================================================================================


def read_lrtddft_file(filename, atom_kind, donor_type, excitation_type="singlet"):
    '''
    Reads the spectrum file and return the list of energy and excitation
    '''
    index = []
    energy = []
    fdip = []
    skip_blank = True # This makes sure no data is saved until we arrive at desired kind
    do_type = False
    ex_type = False
    at_kind = False

    with open(filename, "r") as myfile:

        for line in myfile:
            # Saves last index of data block when blank line is encountered. Requires that blank
            # lines are present after each data block.
            if not len(line.split()) and skip_blank:
                continue
            if not len(line.split()):
                index.append(int(parts[0]))
                skip_blank = True
                continue

            # If the line starts with XAS TDP, need to look at the excitation_type and donor_type
            if line.startswith("XAS TDP"):
                if donor_type in line:
                    do_type = True
                else:
                    do_type = False

                if excitation_type in line:
                    ex_type = True
                else:
                    ex_type = False

            # If line starts woth "from EXCITED ATOM" then look for atomic kind
            if line.startswith("from EXCITED ATOM"):
                if atom_kind in line:
                    at_kind = True
                else:
                    at_kind = False

            # Only if previous checks are all True will we start storing data from the block
            parts = line.split()
            # If ex_type and do_type and at_kind and parts[0].isdigit():
            if do_type and at_kind and parts[0].isdigit():
                energy.append(float(parts[1]))
                fdip.append(float(parts[2]))
                skip_blank = False

        energy = np.array(energy)
        fdip = np.array(fdip)
        index = np.array(index) # Index is the length of each data block
                                # Used to easily separate data from separate LR calculations
    if not len(energy):
        raise ValueError('ERROR! Atomic kind that was requested cannot be found in the specified file!')
    return(energy, fdip, index)


def lrtddft_generate_spectrum(filename, atom_kind, donor_type, excitation_type, width):
    L_energy, L_oscillator, index = read_lrtddft_file(filename, atom_kind, donor_type, excitation_type)
    L_omega, L_osc_gauss = apply_gaussian_broadening(L_energy, L_oscillator, width)
    return L_omega, L_osc_gauss, index

