
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Simulation of ID protocols with real patient parameter

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 21/01/2023

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cycler
import json
from matplotlib.colors import Normalize
import matplotlib.pylab as pl
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_tools import convert_date_to_days, convert_treatment_dic, get_last_y_values


# Modification of Matplotlib default params
p = plt.rcParams
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.linestyle"] = (0, (15, 10))
p["grid.color"] = "#B6BBBF"
p["grid.linewidth"] = 0.25
p['font.size'] = 14
p["axes.prop_cycle"] = cycler(color=['#1f7bb6', '#7cc499', '#F0C929', '#F48B29', '#AC0D0D'])
p.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
p.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})


# Function ==========================================================

def solve_ode_simulation(y0, param, space_between_doses, number_of_doses, t0_treatment):
    ''' Simulation of an artificial chemotherapy schedule consisting
    of a certain amont of time between 2 doses, for a given number 
    of doses
    '''
    # pre-treatment -------------------------------------------------
    t_pre_treatment = np.arange(t0, treatment['c'][0], tstep)
    Y_pre_treatment = odeint(rLGG_model_v3, y0, t_pre_treatment, args=(param, ))
    Y_global = np.array(Y_pre_treatment)
    t_global = np.array(t_pre_treatment.reshape(-1, 1))

    # treatment - artificial chemotherapy  --------------------------
    # tj vector
    tj = []
    for t in np.arange(t0_treatment, t0_treatment+(number_of_doses*(space_between_doses)), space_between_doses):
            tj.append(t)
    # simulation
    for i in range(0, len(tj)-1):
        # update initial condition
        y0 = get_last_y_values(Y_global, treatment_dose=1)
        # simulation time
        t = np.arange(tj[i], tj[i+1], tstep)
        # resolution
        res = odeint(rLGG_model_v3, y0, t, args=(param, ))
        # add to global vector
        Y_global = np.vstack((Y_global, res))
        t_global = np.vstack((t_global, t.reshape(-1, 1)))

    # post-traitement -----------------------------------------------
    y0 = get_last_y_values(Y_global, treatment_dose=1)
    t_post_treatment = np.arange(tj[-1], tf, tstep) 
    res_post_treatment = odeint(rLGG_model_v3, y0, t_post_treatment, 
                                args=(param, ))
    Y_global = np.vstack((Y_global, res_post_treatment))
    t_global = np.vstack((t_global, t_post_treatment.reshape(-1, 1)))
    
    return (Y_global, t_global)
    
# values of params for all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# plot for each patient
for p_id in [1, 2, 3, 4, 5, 6, 7]:

    # patient data ------------------------------
    # get vector from real data
    data_fit = np.genfromtxt(f'1_Data/1_2_Patient_Y_Vector/values_fit_p{p_id}.csv', delimiter=',')
    volume_fit = data_fit[:, 1] + data_fit[:, 2] + data_fit[:, 3] + data_fit[:, 4] + data_fit[:, 5]
    t_fit = data_fit[:, 0]
    # get params values
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
    # get tumor data
    data = pd.read_csv(f'1_Data/1_1_Patient_data/data_p{p_id}.csv')
    # get patient data
    with open('1_Data/LGG_selected_patient_data.json', 'r') as f:
        patient_data = json.load(f)
    # volume vector
    data_volume = data.volume
    # time vector
    data_time = convert_date_to_days(data.date)
    # treatment
    treatment = convert_treatment_dic(patient_data[f'{p_id}']['treatments'], data.date[0])
    # define a time vector
    t0 = 0
    tf = data_time[-1] + 10000
    tstep = 0.1
    # initial conditions
    Vs0 = data_volume[0]
    Vd0 = 0
    Vpi0 = 0
    Vp0 = 0 
    Vr0 = 0
    D0 = 0
    y0 = [Vs0, Vd0, Vpi0, Vp0, Vr0, D0]
    critical_volume = 280

    # plot --------------------------------------
    colors = pl.cm.YlGnBu_r(np.linspace(0, 1, 15))

    time_between_doses = []
    final_volume = []
    final_volume_sensitive = []
    time_critical_volume = []

    # plot simulation
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
    # plot real data
    ax.plot(t_fit/30, volume_fit, c='k', zorder=10, lw=2, label='Model',)
    # error bar
    if p_id == '7':
        yerr = data_volume*0
    else:
        yerr=data_volume*0.2
        
    plt.errorbar(data_time/30, data_volume, yerr=yerr, fmt="o", 
                 ecolor='k', c='w', markeredgecolor='k',
                 markeredgewidth=2, zorder=10, label='Data')
    # plot artificial data
    for i, j in zip(range(7, 105, 7), range(len(colors))):
        Y_sim, t_sim = solve_ode_simulation(y0, param, i, 124, treatment['c'][0])
        volume_sim = Y_sim[:, 0] + Y_sim[:, 1] + Y_sim[:, 2] + Y_sim[:, 3] + Y_sim[:, 4]
        
        ax.plot(t_sim/30, volume_sim, color=colors[j], lw=1)
                
        time_between_doses.append(i)
        final_volume.append(volume_sim[-1])
        final_volume_sensitive.append(Y_sim[-1, 0])
        time_critical_volume.append(t_sim[np.where(volume_sim > critical_volume)][0])

    norm = Normalize(vmin=7, vmax=98)
    axcb = fig.colorbar(pl.cm.ScalarMappable(norm=norm, cmap='YlGnBu_r'), ax=ax, ticks=range(7, 105, 7))
    axcb.set_label('Time~between~doses~(days)')
    axcb.ax.set_yticklabels([f'{i}' for i in range(7, 105, 7)])

    ax.hlines(280, 0, 300, colors='k', linestyles='-.', label='Fatal Volume')
    #ax.hlines(42.8, 0, 300, colors='k', linestyles='--', label='Loss of control threshold')

    # decoration
    ax.set_xlabel('Time~(months)')
    ax.set_ylabel('Tumor~volume~($cm^3$)')
    ax.set_ylim([0, 300])
    ax.set_xlim([0, (t_sim[np.where(volume_sim > critical_volume)][0]+20)/30])
    plt.title(f'Patient {p_id}')
    plt.tight_layout()
    #plt.legend(loc=6)
    #plt.show()
    fig.savefig(f'3_Plots/3_3_Real_patients_clinical_study/sim_{p_id}_legend.png', dpi=300)
    
    print(f'Patient {p_id} ok')
