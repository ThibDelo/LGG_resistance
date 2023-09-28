
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Virtual clinical study
# Surival curves plot

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 20/09/2023


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

for n_fixed_doses in [60, 120]:

    for space_between_doses in [14, 21]:
    
        print(f'trial C28 Vs ID{space_between_doses}')
        #np.random.seed(space_between_doses+100)
        
        # initialization vector
        time_critical_volume_C28 = np.zeros(n_patient)
        time_critical_volume_ID14 = np.zeros(n_patient)
        patient_param = []
        
        # simulation ID (experimental)
        for i in tqdm(range(n_patient)):
            # create a virtual patient
            patient = VirtualPatient(patients_param, V0_real)
            patient.n_doses = n_fixed_doses
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
            patient.n_cycle = int(n_fixed_doses / 5)
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

        
        # Exportation =======================================================
        '''
        # exportation of survival data
        df = pd.DataFrame({'C28': time_critical_volume_C28, 
                           'ID14': time_critical_volume_ID14}, 
                           columns=['C28', 'ID14'])
        df.to_excel('1_Data/survival_data.xlsx', index=False)
        
        # exportation of virtual patients parameters
        np.savetxt('1_Data/param_clinical_virutal_study.csv',
                   np.array(patient_param), delimiter=",")
        '''
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
        plt.title(f'C28 Vs ID{space_between_doses} ({n_fixed_doses} doses)')
        plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/km_curve_ID{space_between_doses}_{n_fixed_doses}_doses.png', dpi=300)
        #plt.show()
        
