
import numpy as np
from datetime import date, datetime


def convert_date_to_days(dates, initial_date:str=None):
    """convert absolute date into a relative number of days
    
    Parameters
    ----------
    dates : ArrayLike
        dates vector with the format YYYY-MM-DD
    initial_date : str 
        date with the format YYYY-MM-DD if we want to set a 
        different initial date than the one given at the
        begining of the dates vector
        
    Returns
    -------
    Array
    """
    if initial_date != None:
        days = []
        # first day
        d0 = date(int(initial_date[0:4]), int(initial_date[5:7]), int(initial_date[8:10]))
        # difference between d0 and each dk in days
        for d in dates[0:]:
            dk = date(int(d[0:4]), int(d[5:7]), int(d[8:10]))
            dif = (dk - d0).days
            days.append(dif)
    else:
        days = [0]
        # first day
        d0 = date(int(dates[0][0:4]), int(dates[0][5:7]), int(dates[0][8:10]))
        # difference between d0 and each dk in days
        for d in dates[1:]:
            dk = date(int(d[0:4]), int(d[5:7]), int(d[8:10]))
            dif = (dk - d0).days
            days.append(dif)
    return np.array(days)

def convert_treatment_dic(treatment_dic:dict, initial_date:str) -> dict:
    """convert date in the treatment dictionary into relative days
    
    Parameters
    ----------
    treatment_dic : dict
        original treatment dictionary
    initial_date : str 
        initial date with the format YYYY-MM-DD 
        
    Returns
    -------
    dict
    """
    for key in treatment_dic.keys():
        if key in ('s', 'nb_tmz_doses'):
            pass
        else:
            treatment_dic[f'{key}'] = convert_date_to_days(treatment_dic[f'{key}'], initial_date)
    return treatment_dic

def get_last_y_values(Y_global, treatment_dose=0):
    Vs0 = Y_global[-1, 0]
    Vd0 = Y_global[-1, 1]
    Vpi0 = Y_global[-1, 2]
    Vp0 = Y_global[-1, 3]
    Vr0 = Y_global[-1, 4]
    D0 = Y_global[-1, 5] + treatment_dose
    y0 = [Vs0, Vd0, Vpi0, Vp0, Vr0, D0]
    return y0

def cgc_checker(volume_vector, t0_treatment=30, t_step=0.1):
    '''check if the control growth criteria is respected in a give
    data vector
    
    Returns
    -------
    bool: 
    - 1 : control growth criteria is not respected
    - 0 : control growth criteria is respected
    '''
    t0_treatment_volume = volume_vector[int(30*(1/t_step))]
    min_volume_after_t0 = min(volume_vector[int(30*(1/t_step)):])
    if min_volume_after_t0 >= t0_treatment_volume:
        return 1
    if min_volume_after_t0 < t0_treatment_volume:
        return 0
    
def log_fit(n_opti, i, res, p_id):
    print(f"minimization {i+1}/{n_opti}: Done, cost: {res['fun']}")
    with open(f'2_Code/2_1_Fit/fit_log/fit_p{p_id}/log_{datetime.now():%m_%d_%Y_%H_%M}.csv', 'a+') as f:
        if i == 0:
            f.write(f"iter, cost, rho1, rho2, tau, psi, a1, a2\n")
        f.write(f"{i+1}, {res['fun']}, {res['x'][0]}, {res['x'][1]}, {res['x'][2]}, {res['x'][3]}, {res['x'][4]}, {res['x'][5]}\n")
