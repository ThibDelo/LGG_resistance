
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Adaptive therapy for a single patient.

# Author: Jesus J Bosque, Thibault Delobel, Salvador Chulian
# last edit: 24/09/2023

import sys
import numpy as np
import pandas as pd
from matplotlib import cycler
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_solve_ODE import solve_ode_adaptive, solve_ode_ID

# Choose a patient out of [1, 2, 3, 4, 5, 6, 7]
p_id = 3
critical_volume = 175
thr = 0.8
n_doses = 50
t0_treatment = 60
space_between_doses = 21
screening_gap = 180


# initial conditions
Vs0 = 100
Vd0 = 0
Vpi0 = 0
Vp0 = 0 
Vr0 = 0
D0 = 0
y0 = [Vs0, Vd0, Vpi0, Vp0, Vr0, D0]


# change default matplotlib style
custom_plt_param = plt.rcParams
custom_plt_param["ytick.minor.visible"] = True
custom_plt_param["xtick.minor.visible"] = True
custom_plt_param["grid.linestyle"] = (0, (15, 10))
custom_plt_param["grid.color"] = "#B6BBBF"
custom_plt_param["grid.linewidth"] = 0.25
custom_plt_param['font.size'] = 24
custom_plt_param["axes.prop_cycle"] = cycler(color=['#1f7bb6', '#7cc499', '#F0C929', 
                                     '#F48B29', '#AC0D0D']) 
custom_plt_param.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
custom_plt_param.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})


# values of params for all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

rho1 = patients_param.loc[int(p_id), 'rho1']
rho2 = patients_param.loc[int(p_id), 'rho2']
tau = patients_param.loc[int(p_id), 'tau']
psi = patients_param.loc[int(p_id), 'psi']
a1 = patients_param.loc[int(p_id), 'alpha1']
a2 = patients_param.loc[int(p_id), 'alpha2']
b = patients_param.loc[int(p_id), 'beta']
l1 = patients_param.loc[int(p_id), 'lambda1']
lamb = - (np.log(1/2)/2)*24

param = [rho1, rho2, tau, psi, a1, a2, b, l1, lamb]

Y_ada, t_ada, screening, dosing, treatment = solve_ode_adaptive(rLGG_model_v3, y0, param, threshold=thr, space_between_screening=screening_gap, space_between_doses=space_between_doses, number_of_doses=n_doses, t0=0, t0_treatment=t0_treatment, tf=10000, tstep=0.1)
volume_ada = Y_ada[:, 0] + Y_ada[:, 1] + Y_ada[:, 2] + Y_ada[:, 3] + Y_ada[:, 4]

t_crit_ada = t_ada[np.where(volume_ada > critical_volume)][0]

Y_ID, t_ID, dosing_ID = solve_ode_ID(rLGG_model_v3, y0, param, space_between_doses=space_between_doses, number_of_doses=n_doses, t0=0, t0_treatment=t0_treatment, tf=10000, tstep=0.1)
arg_doses = np.argwhere(dosing_ID)
t_last_dose = t_ID[arg_doses[-1]]

volume_ID = Y_ID[:, 0] + Y_ID[:, 1] + Y_ID[:, 2] + Y_ID[:, 3] + Y_ID[:, 4]

t_crit_ID = t_ID[np.where(volume_ID > critical_volume)][0]

t_crit = max(t_crit_ada,t_crit_ID)

# Plots
colors = pl.cm.YlGnBu_r(np.linspace(0, 1, 15))

fig = plt.figure(tight_layout=True, figsize=(10, 7))
plt.plot(t_ID/30, volume_ID, c='k', lw=2.2, label='ID',)
plt.plot(t_ada/30, volume_ada, c='#FF69B4', lw=2.2, label='Adaptive',)
plt.plot(t_ada/30, Y_ada, lw=1, label='Adaptive',)
plt.plot(t_ada/30, screening*critical_volume*1.2, color='#E6E6E6', lw=0.6, linestyle = '--', label='screening')
plt.axvline(x = t_last_dose/30, color = '#FF7D61', lw=1, linestyle = ':', label='Last dose')
#plt.plot(t_ada, dosing*200, c='b', lw=0.3, linestyle = ':', label='dosing')
for i in range(treatment.shape[0]):
    plt.axvspan(treatment[i,0]/30, treatment[i,1]/30, 
                color='#eff9b6', alpha=0.5, label='chemotherapy') 
       
plt.xlabel('Time (months)')
plt.ylabel('Tumor volume ($cm^3$)')
plt.axhline(y = critical_volume, color = 'k', lw=0.5, linestyle = '-.', label='Fatal Volume')
plt.axhline(y = thr*Vs0, color = 'k', lw=0.5, linestyle = '--', label='Threshold')
plt.xlim([0, (t_crit/30+1)])
plt.ylim([0, critical_volume*1.2])
plt.savefig('3_Plots/3_6_Adaptative_therapy/adaptative_therapy.png', dpi=300)
plt.show()
