
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Virtual clinical study
# Surival curves plot

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 22/01/2023


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_tools import cgc_checker
from rLGG_package.rLGG_plots import custom_plt_param
from rLGG_package.rLGG_virtual_patient_generator import VirtualPatient
from rLGG_package.rLGG_solve_ODE import solve_ode_C28, solve_ode_ID


# Patients param ==========================================================

# Initial tumor size from real patients 
V0_real = pd.read_csv('1_Data/initial_size_chemo.csv')

# values of params of all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# Simulation ==============================================================

n_patient = 100

for space_between_doses in range(7, 49, 7):

    print(f'trial C28 Vs ID{space_between_doses}')
    np.random.seed(space_between_doses+100)
    
    # initialization vector
    time_critical_volume_C28 = np.zeros(n_patient)
    time_critical_volume_ID14 = np.zeros(n_patient)
    Vr_list_C28 = np.zeros(n_patient)
    Vs_list_C28 = np.zeros(n_patient)
    Vd_list_C28 = np.zeros(n_patient)
    Vpi_list_C28 = np.zeros(n_patient)
    Vp_list_C28 = np.zeros(n_patient)
    Vr_list_ID = np.zeros(n_patient)
    Vs_list_ID = np.zeros(n_patient)
    Vd_list_ID = np.zeros(n_patient)
    Vpi_list_ID = np.zeros(n_patient)
    Vp_list_ID = np.zeros(n_patient)
    patient_param = []
    
    # simulation ID (experimental)
    for i in tqdm(range(n_patient)):
        # create a virtual patient
        patient = VirtualPatient(patients_param, V0_real)
        # tumor simulation
        y, t = solve_ode_ID(rLGG_model_v3, patient.y0, patient.param, 
                              space_between_doses=space_between_doses,
                              number_of_doses=patient.n_doses)
        # get critical volume
        volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
        time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
        time_critical_volume_ID14[i] = time_critical_volume
        # get patient param
        patient_param.append(np.array(patient.param))
        # get Vr and Vs at the end of treatment
        t_end_treatment = np.where(t >= 476)[0][0]
        Vr_list_ID[i] = y[t_end_treatment, 4] / volume[t_end_treatment] * 100
        Vs_list_ID[i] = y[t_end_treatment, 0] / volume[t_end_treatment] * 100
        Vd_list_ID[i] = y[t_end_treatment, 1] / volume[t_end_treatment] * 100
        Vpi_list_ID[i] = y[t_end_treatment, 2] / volume[t_end_treatment] * 100
        Vp_list_ID[i] = y[t_end_treatment, 3] / volume[t_end_treatment] * 100
        # Control growth
        cgc_status = cgc_checker(volume)
        if cgc_status == 1:
            print('Control growth criteria not respected')
            print(f'V0 = {patient.y0}')
            print(f'param = {patient.param}')
            print(f'number of doses = {patient.n_doses}')

    # simulation C28 (reference)
    for i in tqdm(range(n_patient)):
        # create a virtual patient
        patient = VirtualPatient(patients_param, V0_real)
        # tumor simulation
        y, t = solve_ode_C28(rLGG_model_v3, patient.y0, patient.param, 
                             cycle_mumber=patient.n_cycle)
        # get critical volume
        volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
        time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
        time_critical_volume_C28[i] = time_critical_volume
        # get patient param
        patient_param.append(np.array(patient.param))
        # get Vr and Vs at the end of treatment
        t_end_treatment = np.where(t >= 476)[0][0]
        Vr_list_C28[i] = y[t_end_treatment, 4] / volume[t_end_treatment] * 100
        Vs_list_C28[i] = y[t_end_treatment, 0] / volume[t_end_treatment] * 100
        Vd_list_C28[i] = y[t_end_treatment, 1] / volume[t_end_treatment] * 100
        Vpi_list_C28[i] = y[t_end_treatment, 2] / volume[t_end_treatment] * 100
        Vp_list_C28[i] = y[t_end_treatment, 3] / volume[t_end_treatment] * 100
    
    # Exportation =======================================================
    
    # exportation of survival data
    df = pd.DataFrame({'C28': time_critical_volume_C28, 
                       'ID14': time_critical_volume_ID14}, 
                       columns=['C28', 'ID14'])
    df.to_excel('1_Data/survival_data.xlsx', index=False)
    
    # exportation of virtual patients parameters
    np.savetxt('1_Data/param_clinical_virutal_study.csv',
               np.array(patient_param), delimiter=",")
    
    # Plot ================================================================
    
    # log rank statistic test
    res_logrank = logrank_test(time_critical_volume_C28, 
                               time_critical_volume_ID14)

    # Change default Matplotlib params 
    custom_plt_param["axes.prop_cycle"] = cycler(color=['#2195c0', '#9ed9b8', '#F0C929', 
                                                        '#FBB454', '#EB4747']) 
    custom_plt_param["xtick.minor.visible"] = True
    custom_plt_param["axes.grid"] = True
    
    # kaplan meyer curves -------------------------------------------------
    plt.figure(figsize=(10, 6))
    kmf1 = KaplanMeierFitter(label=f"C28 (n={n_patient})",)
    kmf2 = KaplanMeierFitter(label=f"ID{space_between_doses} (n={n_patient})",)
    kmf1.fit(time_critical_volume_C28/30)
    kmf2.fit(time_critical_volume_ID14/30)
    kmf1.plot(color='#081D58', zorder=8, ci_alpha=0.2)
    kmf2.plot(color='#61B579', zorder=8, ci_alpha=0.2)
    plt.xlabel('Time~(months)')
    plt.ylabel('Survival~probability')
    plt.text(5, 0.1, f'p = {float(res_logrank.p_value):g}')
    plt.title(f'C28 Vs ID{space_between_doses}')
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/km_curve_ID{space_between_doses}.png', dpi=300)
    #plt.show()
    
    # Sensitive volume / resistant volume plot C28 ------------------------
    '''
    # Change default Matplotlib params 
    custom_plt_param["ytick.minor.visible"] = True
    custom_plt_param["xtick.minor.visible"] = False
    custom_plt_param["axes.grid"] = False
    
    width = 1
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(Vs_list_C28)), Vs_list_C28, width, 
            label='Sensitive Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vd_list_C28)), Vd_list_C28, width, bottom=Vs_list_C28,
            label='Domaged Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vpi_list_C28)), Vpi_list_C28, width, 
            bottom=Vs_list_C28+Vd_list_C28,
            label='Persister intermediate Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vp_list_C28)), Vp_list_C28, width, 
            bottom=Vs_list_C28+Vd_list_C28+Vpi_list_C28, 
            label='Persister Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vr_list_C28)), Vr_list_C28, width, 
            bottom=Vs_list_C28+Vd_list_C28+Vpi_list_C28+Vp_list_C28, 
            label='Resistant Volume', edgecolor='k', linewidth=0.5)
    plt.title(f'C28 (n={n_patient})')
    plt.ylabel('Volume~($\%$)')
    plt.xlim([-0.5, n_patient-0.5])
    plt.xticks([])
    plt.ylim([0.5, 100])
    plt.tight_layout()
    plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/Vs_Vr_C28.png', dpi=300)
    #plt.show()
    
    # Sensitive volume / resistant volume plot ID -------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(Vs_list_ID)), Vs_list_ID, width, 
            label='Sensitive Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vd_list_ID)), Vd_list_ID, width, bottom=Vs_list_ID, 
            label='Domaged Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vpi_list_ID)), Vpi_list_ID, width, 
            bottom=Vs_list_ID+Vd_list_ID,
            label='Persister intermediate Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vp_list_ID)), Vp_list_ID, width, 
            bottom=Vs_list_ID+Vd_list_ID+Vpi_list_ID, 
            label='Persister Volume', edgecolor='k', linewidth=0.5)
    plt.bar(range(len(Vr_list_ID)), Vr_list_ID, width, 
            bottom=Vs_list_ID+Vd_list_ID+Vpi_list_ID+Vp_list_ID, 
            label='Resistant Volume', edgecolor='k', linewidth=0.5)
    plt.title(f'ID{space_between_doses} (n={n_patient})')
    plt.ylabel('Volume~($\%$)')
    plt.xlim([-0.5, n_patient-0.5])
    plt.xticks([])
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/Vs_Vr_ID{space_between_doses}.png', dpi=300)
    #plt.show()
    '''