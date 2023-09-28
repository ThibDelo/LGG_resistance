
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Virtual clinical study
# box plot of survival gain according to space between individual doses

# Author: Thibault Delobel
# creation: 20/08/2022
# last edit: 22/01/2023

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from tqdm import tqdm
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_plots import custom_plt_param
from rLGG_package.rLGG_virtual_patient_generator import VirtualPatient
from rLGG_package.rLGG_solve_ODE import solve_ode_C28, solve_ode_ID

np.random.seed(42)


# Patients param ==========================================================

# Initial tumor size from real patients 
V0_real = pd.read_csv('1_Data/initial_size_chemo.csv')

# values of params of all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# constant
critical_volume = np.random.normal(280, 20)
n_patient = 100
n_trials = 20
space_between_doses = range(7, 49, 7)

# Simulation ==============================================================

survival_gain_list_global = []

for i in range(n_trials):

    print(f'trials {i}/{n_trials}')

    # initialisation vector
    time_critical_volume_C28 = np.zeros(n_patient)
    time_critical_volume_ID14 = np.zeros(n_patient)
    survival_gain_list = []
    number_cgc_not_respect_list = []

    for i in range(n_patient):
        patient = VirtualPatient(patients_param, V0_real)
        y, t = solve_ode_C28(rLGG_model_v3, patient.y0, patient.param, 
                             cycle_mumber=patient.n_cycle)
        volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
        time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
        time_critical_volume_C28[i] = time_critical_volume

    for sbd in tqdm(space_between_doses):
        # initialisation
        volume_decrease = np.zeros(n_patient)
        number_cgc_not_respect = 0
        # simulation
        for i in range(n_patient):
            patient = VirtualPatient(patients_param, V0_real)
            y, t = solve_ode_ID(rLGG_model_v3, patient.y0, patient.param, 
                                space_between_doses=sbd, number_of_doses=patient.n_doses)
            volume = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3] + y[:, 4]
            time_critical_volume = t[np.where(volume > patient.critical_volume)][0]
            time_critical_volume_ID14[i] = time_critical_volume

        # kaplan meyer curves
        kmf1 = KaplanMeierFitter(label="C28")
        kmf2 = KaplanMeierFitter(label=f"ID{space_between_doses}")
        kmf1.fit(time_critical_volume_C28/30)
        kmf2.fit(time_critical_volume_ID14/30)
        survival_gain = kmf2.median_survival_time_ - kmf1.median_survival_time_
        survival_gain_list.append(survival_gain)
        
    survival_gain_list_global.append(survival_gain_list)

# Save / load of simulation result ========================================
'''
with open("survival_gain_list_global.pickle", "wb") as f: 
    pickle.dump(survival_gain_list_global, f)

with open('survival_gain_list_global.pickle', "rb") as f1: 
    survival_gain_list_global = pickle.load(f1)
'''
# Plot ====================================================================

sglg = np.array(survival_gain_list_global)

custom_plt_param['font.size'] = 14

fig, ax = plt.subplots(figsize=(10, 6))
bplot = plt.boxplot([sglg[:, i] for i in range(6)], widths=0.5, showfliers=False, 
                    showcaps=True, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bplot[element], color='k')
for patch in bplot['boxes']:
    patch.set(facecolor='#d1edb3')

ax.set_xlabel('Protocols')
ax.set_ylabel('Median~survival~gain~(months)')
plt.xticks(range(1, 7), [f'ID{i}' for i in range(7, 49, 7)])
ax.set_ylim([0, 35])
fig.tight_layout()
#fig.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/survival_gain_space_doses_2.pdf', dpi=300)
fig.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/survival_gain_space_doses_2.png', dpi=300)
plt.show()
