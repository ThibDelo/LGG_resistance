
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Virtual clinical study
# p_value according to number of patients per arm plot

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 22/01/2023

import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lifelines.statistics import logrank_test

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_plots import custom_plt_param
from rLGG_package.rLGG_solve_ODE import solve_ode_C28, solve_ode_ID
from rLGG_package.rLGG_virtual_patient_generator import VirtualPatient

custom_plt_param['font.size'] = 16
plt.rcParams = custom_plt_param


# Patients param ==========================================================

# Initial tumor size from real patients 
V0_real = pd.read_csv('1_Data/initial_size_chemo.csv')

# values of params of all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# constant
space_between_doses = 21
number_trials = 20

# Simulation ==============================================================

p_value_list_global = []
nb_patients_list = range(10, 110, 10)

for j in range(number_trials):
    print(f'trial {j+1}/{number_trials}')
    p_value_list = []
    for npatient in tqdm(nb_patients_list):
        time_critical_volume_C28 = np.zeros(npatient)
        time_critical_volume_ID14 = np.zeros(npatient)
        # simulation ID
        np.random.RandomState(seed=j+4242)
        for i in range(npatient):
            patient = VirtualPatient(patients_param, V0_real)
            y, t = solve_ode_ID(rLGG_model_v3, patient.y0, patient.param, 
                                  space_between_doses=space_between_doses,
                                  number_of_doses=patient.n_doses)
            volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
            time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
            time_critical_volume_ID14[i] = time_critical_volume
        # simulation C28 
        np.random.RandomState(seed=j+number_trials+42)
        for i in range(npatient):
            patient = VirtualPatient(patients_param, V0_real)
            y, t = solve_ode_C28(rLGG_model_v3, patient.y0, patient.param, cycle_mumber=patient.n_cycle)
            volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
            time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
            time_critical_volume_C28[i] = time_critical_volume

        # log rank statistic test
        res_logrank = logrank_test(time_critical_volume_C28, 
                                   time_critical_volume_ID14)
        p_value_list.append(float(res_logrank.p_value))
    p_value_list_global.append(p_value_list)

p_value_list_global = np.array(p_value_list_global)

# Save/load simulation result =============================================
'''
with open("p_value_list_global_2.pickle", "wb") as f: 
    pickle.dump(p_value_list_global, f)

with open('p_value_list_global_2.pickle', "rb") as f2: 
    p_value_list_global = pickle.load(f2)
'''
# Plot ====================================================================

norm = Normalize(vmin=0, vmax=0.06)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(np.rot90(p_value_list_global), aspect='auto', norm=norm, cmap='YlGnBu_r')
ax.set_title(f"ID{space_between_doses} Vs C28")
cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm, cmap='YlGnBu_r'), ax=ax)
cbar.set_label('p-value')
ax.set_ylabel('Number~of~patients~per~arm')
plt.yticks(range(len(nb_patients_list)),  np.flip([f'{nb_patients_list[i]}' for i in range(len(nb_patients_list))]))
ax.set_xlabel(f'Trials~Id')
plt.xticks(range(0, 20, 1), list(range(1, 21, 1)))
plt.tight_layout()
plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/n_patient_pvalue_ID{space_between_doses}.png', dpi=300)
plt.show()
