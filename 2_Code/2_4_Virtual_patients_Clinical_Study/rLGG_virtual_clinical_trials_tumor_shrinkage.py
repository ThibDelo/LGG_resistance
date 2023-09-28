
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Virtual clinical study
# boxplot of tumoral volume decrease according to space between individual doses

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 22/01/2023


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines import KaplanMeierFitter
import pickle
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_tools import cgc_checker
from rLGG_package.rLGG_plots import custom_plt_param
from rLGG_package.rLGG_virtual_patient_generator import VirtualPatient
from rLGG_package.rLGG_solve_ODE import solve_ode_ID

np.random.seed(42)

plt.rcParams = custom_plt_param


# Patients param ====================================================

# Initial tumor size from real patients 
V0_real = pd.read_csv('1_Data/initial_size_chemo.csv')

# values of params of all patients
patients_param = pd.read_excel('1_Data/param_model_v5.xlsx')
patients_param.index = patients_param['Patient']
patients_param = patients_param.drop('Patient', axis=1)

# constant
n_patient = 1000
space_between_doses = range(7, 49, 7)

# Simulation ========================================================

# initialisation vector
time_critical_volume_C28 = np.zeros(n_patient)
time_critical_volume_ID14 = np.zeros(n_patient)
survival_gain_list = []
number_cgc_not_respect_list = []
volume_decrease_list = []
volume_decrease = np.zeros(n_patient)

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
        # Controled growth check
        cgc_status = cgc_checker(volume)
        if cgc_status == 1:
            number_cgc_not_respect += 1
        # compute max volume decrease
        t0_treatment_volume = volume[int(30*(1/0.1))]
        min_volume_after_t0 = min(volume[int(30*(1/0.1)):])
        volume_decrease[i] = min_volume_after_t0 - t0_treatment_volume

    # Volume decrease
    volume_decrease_list.append(volume_decrease)
    # Controled growth check list        
    number_cgc_not_respect_list.append(number_cgc_not_respect )
    # kaplan meyer curves
    kmf1 = KaplanMeierFitter(label="C28")
    kmf2 = KaplanMeierFitter(label=f"ID{space_between_doses}")
    kmf1.fit(time_critical_volume_C28/30)
    kmf2.fit(time_critical_volume_ID14/30)
    survival_gain = kmf2.median_survival_time_ - kmf1.median_survival_time_
    survival_gain_list.append(survival_gain)

# Save / load of simulation result ========================================
'''
with open("survival_gain_list.pickle", "wb") as f: 
    pickle.dump(survival_gain_list, f)
    
with open("number_cgc_not_respect_list.pickle", "wb") as f2: 
    pickle.dump(number_cgc_not_respect_list, f2)
    
with open("volume_decrease_list.pickle", "wb") as f3: 
    pickle.dump(volume_decrease_list, f3)

with open('survival_gain_list.pickle', "rb") as f4: 
    survival_gain_list = pickle.load(f4)

with open('number_cgc_not_respect_list.pickle', "rb") as f5: 
    number_cgc_not_respect_list = pickle.load(f5)
    
with open('volume_decrease_list.pickle', "rb") as f6: 
    volume_decrease_list = pickle.load(f6)
'''
# Plot ==============================================================

# box plot
fig, ax = plt.subplots(figsize=(10, 6))
bplot = ax.boxplot(volume_decrease_list[:6], showfliers=False, showcaps=True, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bplot[element], color='k')
for patch in bplot['boxes']:
    patch.set(facecolor='#d1edb3')
plt.xticks(range(1, 7), [f'ID{i}' for i in range(7, 49, 7)])
ax.set_ylabel('Maximun~volume~loss~during~treatment~($cm^3$)')
ax.set_xlabel('Protocols')
ax.set_ylim([-120, 10])
plt.tight_layout()
plt.savefig(f'3_Plots/3_4_Virtual_patients_clinical_study/survival_gain_space_doses_boxplot.png', dpi=300)
plt.show()
