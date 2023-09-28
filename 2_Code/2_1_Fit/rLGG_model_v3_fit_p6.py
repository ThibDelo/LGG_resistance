
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Fit of patient 6 data

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 19/01/2023

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

sys.path.append('2_Code/')

from rLGG_package.rLGG_plots import plot_fit
from rLGG_package.rLGG_model import rLGG_model_v3
from rLGG_package.rLGG_treatment_simulation import chemotherapy_simulation
from rLGG_package.rLGG_tools import convert_date_to_days, convert_treatment_dic, get_last_y_values, log_fit


# Function ==========================================================

def solve_ode(y0, param):
    
    # pre-treatment -------------------------------------------------
    t_pre_treatment = np.arange(t0, treatment['c'][0], tstep)
    Y_pre_treatment = odeint(rLGG_model_v3, y0, t_pre_treatment, args=(param, ))
    Y_global = np.array(Y_pre_treatment)
    t_global = np.array(t_pre_treatment.reshape(-1, 1))
    
    # treatment - chemotherapy  -------------------------------------
    Y_global, t_global = chemotherapy_simulation(rLGG_model_v3, Y_global, t_global, 
                                                 treatment['c'], param)
    
    # between treatment ---------------------------------------------    
    y0 = get_last_y_values(Y_global)
    t_between_treatment = np.arange(treatment['c'][1], treatment['c2'][0], tstep)
    res_between_treatment = odeint(rLGG_model_v3, y0, t_between_treatment, 
                                args=(param, ))
    Y_global = np.vstack((Y_global, res_between_treatment))
    t_global = np.vstack((t_global, t_between_treatment.reshape(-1, 1)))
    
    # treatment - chemotherapy  -------------------------------------
    Y_global, t_global = chemotherapy_simulation(rLGG_model_v3, Y_global, 
                                                 t_global, treatment['c2'], param)
    
    # post-traitement -----------------------------------------------
    y0 = get_last_y_values(Y_global)
    t_post_treatment = np.arange(treatment['c2'][1], tf, tstep) 
    res_post_treatment = odeint(rLGG_model_v3, y0, t_post_treatment, 
                                args=(param, ))
    Y_global = np.vstack((Y_global, res_post_treatment))
    t_global = np.vstack((t_global, t_post_treatment.reshape(-1, 1)))
    
    return (Y_global, t_global)

def rmse(param_to_optimize, volume_data):
    # param to optimize
    rho1, rho2, tau, psi, a1, a2 = param_to_optimize
    # solve ODE
    res, _ = solve_ode(y0, [rho1, rho2, tau, psi, a1, a2, b, l1, lamb])
    # obtention of results at the time of volume data
    volume_model = []
    for time in data_time:
        i = time * (1/tstep)
        volume_model.append(res[int(i), 0] + res[int(i), 1] + res[int(i), 2] + \
                            res[int(i), 3] + res[int(i), 4])    
    # RMSE calculus
    rmse = np.sqrt(np.sum((volume_data - volume_model)**2) / len(volume_data))
    
    if rho1 > rho2:
        rmse += 50
    
    return rmse


# Patient Data ======================================================

p_id = '6'

# tumor data
data = pd.read_csv(f'1_Data/1_1_Patient_data/data_p{p_id}.csv')
# patient data
with open('1_Data/LGG_selected_patient_data.json', 'r') as f:
    patient_data = json.load(f)
# volume vector
data_volume = data.volume
# time vector
data_time = convert_date_to_days(data.date)
# y0
Vs0_data = data_volume[0]
# treatment
treatment = convert_treatment_dic(patient_data[f'{p_id}']['treatments'], data.date[0])

# Parameters ========================================================

n_opti = 20
cost_list = np.zeros(n_opti)
p_opt_list = np.zeros(n_opti, dtype=object)

# time
t0 = 0
tf = data_time[-1] + 5000
tstep = 0.1

# initial conditions
Vs0 = Vs0_data
Vd0 = 0
Vpi0 = 0
Vp0 = 0 
Vr0 = 0
D0 = 0
y0 = [Vs0, Vd0, Vpi0, Vp0, Vr0, D0]

# param search space
bounds = ((5e-4, 0.01), (1e-3, 0.02), (1e-3, 0.1), (0.1, 1), (0.15, 0.6), (0.15/10, 0.6/3))

# Fit ===============================================================

if not os.path.exists(f'2_Code/2_1_Fit/fit_log/fit_p{p_id}/'):
    os.makedirs(f'2_Code/2_1_Fit/fit_log/fit_p{p_id}/')

for i in range(n_opti):

    # random initial parameter values
    rho1 = np.random.uniform(5e-4, 0.01)
    rho2 = np.random.uniform(1e-3, 0.02)
    tau = np.random.uniform(1e-3, 0.1)
    psi = np.random.uniform(0.1, 1)
    a1 =  np.random.uniform(0.15, 0.6)
    a2 = np.random.uniform(0.15/10, 0.6/3)
    b = 0.1
    l1 = 15
    lamb = - (np.log(1/2)/2)*24
    param_to_optimize = [rho1, rho2, tau, psi, a1, a2] 

    # minimization of the cost function
    res = minimize(rmse, param_to_optimize, args=(data_volume), method='Nelder-Mead', bounds=bounds)
    p_opt_list[i] = res['x']
    cost_list[i] = res['fun']
    
    # log
    log_fit(n_opti, i, res, p_id)
        
# select best score
id_best_score = np.argmin(cost_list)
p_opt = p_opt_list[id_best_score]

# Solve with new param
new_param = [p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5], b, l1, lamb]
Y_global, t_global = solve_ode(y0, new_param)

# Y vector exportation 
df = np.hstack([t_global, Y_global])
np.savetxt(f'1_Data/1_2_Patient_Y_Vector/values_fit_p{p_id}.csv', df, delimiter=",")

# Final plot 
plot_fit(t_global, Y_global, data_time, data_volume, treatment, p_id, save_img=False, ylim=300)
