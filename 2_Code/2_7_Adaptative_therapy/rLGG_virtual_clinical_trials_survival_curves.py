
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Adaptive therapy clinical trials. Gives survival plots.

# Author: Jesus J Bosque, Thibault Delobel, Salvador Chulian
# last edit: 24/09/2023

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import cycler
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.append('2_Code/')

from rLGG_package.rLGG_tools import cgc_checker
from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_plots import custom_plt_param
from rLGG_package.rLGG_virtual_patient_generator import VirtualPatient
from rLGG_package.rLGG_solve_ODE import solve_ode_adaptive, solve_ode_ID


# Patients param ==========================================================

# Initial tumor size from real patients 
V0_real = pd.read_csv('1_Data/initial_size_chemo.csv')

# values of params of all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# Simulation ==============================================================

n_patient = 100
space_between_doses = 21

for threshold in np.array([0.3,0.4,0.5,0.6,0.7,0.8]):

    print(f'trial Vs ID{space_between_doses} Vs Adaptive')
    np.random.seed(100)
    
    # initialization vector
    time_critical_volume_Ada = np.zeros(n_patient)
    time_critical_volume_ID = np.zeros(n_patient)
    patient_param = []
    
    # simulation ID (experimental)
    for i in tqdm(range(n_patient)):
        # create a virtual patient
        patient = VirtualPatient(patients_param, V0_real)
        # tumor simulation
        y, t, _ = solve_ode_ID(rLGG_model_v3, patient.y0, patient.param, 
                               space_between_doses=space_between_doses,
                               number_of_doses=patient.n_doses)
        # get critical volume
        volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
        time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
        time_critical_volume_ID[i] = time_critical_volume
        # get patient param
        patient_param.append(np.array(patient.param))
        # get Vr and Vs at the end of treatment
        t_end_treatment = np.where(t >= 476)[0][0]
        # Control growth
        cgc_status = cgc_checker(volume)
        if cgc_status == 1:
            print('Control growth criteria not respected')
            print(f'V0 = {patient.y0}')
            print(f'param = {patient.param}')
            print(f'number of doses = {patient.n_doses}')

    # simulation Adaptive
    for i in tqdm(range(n_patient)):
        # create a virtual patient
        patient = VirtualPatient(patients_param, V0_real)
        # tumor simulation
        y, t, _, _, _ = solve_ode_adaptive(rLGG_model_v3, patient.y0, patient.param,
                                  threshold=threshold, 
                                  space_between_doses=space_between_doses, 
                                  number_of_doses=patient.n_doses)
        # get critical volume
        volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
        time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
        time_critical_volume_Ada[i] = time_critical_volume
        # get patient param
        patient_param.append(np.array(patient.param))
        # get Vr and Vs at the end of treatment
        t_end_treatment = np.where(t >= 476)[0][0]
    
    # Exportation =======================================================
    '''
    # exportation of survival data
    df = pd.DataFrame({'Adapt': time_critical_volume_Ada, 
                       'ID': time_critical_volume_ID}, 
                       columns=['Adapt', 'ID'])
    df.to_excel('Data/survival_data.xlsx', index=False)
    
    # exportation of virtual patients parameters
    np.savetxt('Data/param_clinical_virutal_study.csv',
               np.array(patient_param), delimiter=",")
    '''
    # Plot ================================================================
    
    # log rank statistic test
    res_logrank = logrank_test(time_critical_volume_Ada, 
                               time_critical_volume_ID)

    # Change default Matplotlib params 
    custom_plt_param["axes.prop_cycle"] = cycler(color=['#2195c0', '#9ed9b8', '#F0C929', 
                                                        '#FBB454', '#EB4747']) 
    custom_plt_param["xtick.minor.visible"] = True
    custom_plt_param["axes.grid"] = True
    
    # kaplan meyer curves -------------------------------------------------
    if not os.path.exists(f'3_plots/3_6_Adaptative_therapy/Clinical_trial_ID{space_between_doses}/'):
        os.makedirs(f'3_plots/3_6_Adaptative_therapy/Clinical_trial_ID{space_between_doses}/')
    
    plt.figure(figsize=(10, 6))
    kmf1 = KaplanMeierFitter(label=f"Adaptive thr={threshold} (n={n_patient})",)
    kmf2 = KaplanMeierFitter(label=f"ID{space_between_doses} (n={n_patient})",)
    kmf1.fit(time_critical_volume_Ada/30)
    kmf2.fit(time_critical_volume_ID/30)
    kmf1.plot(color='#081D58', zorder=8, ci_alpha=0.2)
    kmf2.plot(color='#61B579', zorder=8, ci_alpha=0.2)
    plt.xlabel('Time~(months)')
    plt.ylabel('Survival~probability')
    plt.text(5, 0.1, f'p = {float(res_logrank.p_value):g}')
    plt.title(f'ID{space_between_doses} Vs adaptive thr={threshold}')
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(f'3_Plots/3_6_Adaptative_therapy/Clinical_trial_ID{space_between_doses}/km_curve_ID{space_between_doses}_thr{threshold}.png', dpi=300)
    plt.show()
    
 