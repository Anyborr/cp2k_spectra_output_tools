#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/users/anyber/scripts_and_jn/jn/Andre/modules')
import lr_module_Andre as lr
import os
wd = os.getcwd() #current working dir

# [this_script.py] [gauss width in eV] [x-axis lower bound] [x-axis upper bound]
sFile = str(sys.argv[1]) # XAS file
width = float(sys.argv[2])
omega_min = float(sys.argv[3])
omega_max = float(sys.argv[4])

# read absorption spectrum file
with open(sFile, 'r') as myfile:

    # check that this is a valid XAS file
    if not myfile.readline().strip().startswith("Absorption spectrum for atom"):
        myfile.seek(0)
        print(f"first line reads: {myfile.readline()}")
        raise SystemExit('Error: This does not appear to be a valid XAS file. Please check your input!')

    
    # reset to top of file (needed because we have used .readline() once above)
    myfile.seek(0)

    # as of now this will only save the last spectrum in the file.
    # Should be expanded to store all of them (needed for output from Real Time Propagation)
    for line in myfile:

        
        if line.strip().startswith("Absorption spectrum for atom"): #spectra header line
            line = line.split()
            nLines = int(line[-1])
            L_energy = np.zeros(nLines)
            L_xas = np.zeros(nLines)
            counter = 0
            continue #iterate the foor loop forward

        if counter < nLines:
            line = line.split()
            L_energy[counter] = line[1]
            L_xas[counter] = line[5]
            counter += 1
            
#omega_min = 2460 #min(L_energy)-2
#omega_max = 2490 #max(L_energy)+2


N_omega = 10000
L_omega = np.linspace(omega_min, omega_max, N_omega)

### change to FWHM broadening, with linear energy dependence using lower and upper limits, outside of which the broadening is constant.
# L_width = lr.make_linscale_array(L_energy, emin=L_energy[0], emax=L_energy[0]+20, wmin=1.2, wmax=8)
# gauss_fdip = lr.gaussian_broadening_from_lr_results(L_energy, L_xas, width=L_width, L_omega=L_omega, FWHM=True)

### Change to energy independent gaussian broadening
gauss_fdip = lr.gaussian_broadening_from_lr_results(L_energy, L_xas, width=width, L_omega=L_omega, FWHM=False)



color = 'blue'
#color = 'red'
#color = 'orange'


p1 = plt.figure()
plt.vlines(L_energy, 0, L_xas, color=color, label='dipole moments')
plt.plot(L_omega, gauss_fdip, color=color, label='spectrum')

#plt.ylim([0, 0.018])
plt.grid()
#plt.legend()

#plt.savefig(fname=wd+'/'+sFile+'.png')#, transparent=True)


# with open(wd+'/'+sFile+'-gauss_FWHM.dat', 'w') as fOut:
#     for row in range(len(L_omega)):
#         fOut.write(str(L_omega[row])+' '+str(gauss_fdip[row])+'\n')
for row in range(len(L_omega)):
    sys.stdout.write(str(L_omega[row])+' '+str(gauss_fdip[row])+'\n')
