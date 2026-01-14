#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/users/anyber/scripts_and_jn/jn/Andre/modules')
import lr_module_Andre as lr
import os
wd = os.getcwd() #current working dir


# [.spec file],  [6/S], [1s], [singlet]  
#sFile, aKind, dType, excType = sys.argv[1], str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4])
#energy, fdip, index = lr.read_spectrum_file(sFile, aKind, dType, excType)


# [this_script.py] [gauss width in eV] [x-axis lower bound] [x-axis upper bound]
sFile = str(sys.argv[1]) # XAS file
aKind, dType, excType = str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]) # 6/S 1s singlet
width = float(sys.argv[5]) # [eV]
omega_min = float(sys.argv[6]) # [eV]
omega_max = float(sys.argv[7]) # [eV]

L_energy, L_xas, _ = lr.read_spectrum_file(sFile, aKind, dType, excType)

# hard-coded XASTDP settings, need to fix or change !!
#L_energy, L_xas, _ = lr.read_spectrum_file(sFile, "4/S", "1s", "open-shell spin-conserving (no SOC)")


N_omega = 1000

#width = 0.3 # in eV
#L_width = lr.make_linscale_array(energy, emin=energy[0], emax=energy[0]+20, wmin=1.2, wmax=8)

L_omega = np.linspace(omega_min, omega_max, N_omega)

# change to FWHM broadening, with linear energy dependence using lower and upper limits, outside of which the broadening is constant.
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

outFile = wd+'/'+sFile+'-gauss.dat'
with open(outFile,'w') as fOut:
    print(f"saving file as '{os.path.basename(outFile)}'")
    for row in range(len(L_omega)):
        fOut.write(str(L_omega[row])+' '+str(gauss_fdip[row])+'\n')

