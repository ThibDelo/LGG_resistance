
import numpy as np
from scipy.integrate import odeint
import sys


from rLGG_package.rLGG_tools import get_last_y_values


def solve_ode_ID(model, y0, param, space_between_doses=14, number_of_doses=85, 
                 t0=0, t0_treatment=36, tf=10000, tstep=0.1):
    ''' Simulation of an artificial chemotherapy schedule consisting
    of a certain amont of time between 2 doses, for a given number 
    of doses
    '''
    # pre-treatment -------------------------------------------------------
    t_pre_treatment = np.arange(t0, t0_treatment, tstep)
    Y_pre_treatment = odeint(model, y0, t_pre_treatment, args=(param, ))
    Y_global = np.array(Y_pre_treatment)
    t_global = np.array(t_pre_treatment.reshape(-1, 1))
    dosing = np.zeros(t_pre_treatment.shape)
    # treatment - artificial chemotherapy  --------------------------------
    # tj vector
    tj = np.arange(t0_treatment, 
                   t0_treatment+(number_of_doses*(space_between_doses)), 
                   space_between_doses)

    # simulation
    for i in range(0, len(tj)-1):
        # update initial condition
        y0 = get_last_y_values(Y_global, treatment_dose=1)
        # simulation time
        t = np.arange(tj[i], tj[i+1], tstep)
        # resolution
        res = odeint(model, y0, t, args=(param, ))
        # add to global vector
        Y_global = np.vstack((Y_global, res))
        t_global = np.vstack((t_global, t.reshape(-1, 1)))
        dosing[-1] = 1
        dosing = np.concatenate((dosing, np.zeros(len(t))))

    # post-traitement -----------------------------------------------------
    y0 = get_last_y_values(Y_global, treatment_dose=1)
    t_post_treatment = np.arange(tj[-1], tf, tstep) 
    res_post_treatment = odeint(model, y0, t_post_treatment, args=(param, ))
    Y_global = np.vstack((Y_global, res_post_treatment))
    t_global = np.vstack((t_global, t_post_treatment.reshape(-1, 1)))
    dosing[-1] = 1
    dosing = np.concatenate((dosing, np.zeros(len(t_post_treatment))))

    return (Y_global, t_global, dosing)
    

def solve_ode_C28(model, y0, param, cycle_duration=28, cycle_mumber=17, 
                  doses_spacing=1, doses_number=5, t0=0, 
                  t0_treatment=36, tf=10000, tstep=0.1):
    ''' Simulation of a classical chemotherapy schedule consisting
    of 5 days of treatment followed by 23 days of rest.
    '''
    # pre-treatment -------------------------------------------------------
    t_pre_treatment = np.arange(t0, t0_treatment, tstep)
    Y_pre_treatment = odeint(model, y0, t_pre_treatment, args=(param, ))
    Y_global = np.array(Y_pre_treatment)
    t_global = np.array(t_pre_treatment.reshape(-1, 1))

    # treatment - artificial chemotherapy  --------------------------------
    # tj vector
    tj = []
    for t in range(t0_treatment, (t0_treatment+cycle_duration*cycle_mumber)):
        if (t - t0_treatment) % cycle_duration in range(0, doses_number, 
                                                        doses_spacing):
            tj.append(t)

    for i in range(0, len(tj)-1):
        # update initial condition
        y0 = get_last_y_values(Y_global, treatment_dose=1)
        # simulation time
        t = np.arange(tj[i], tj[i+1], tstep)
        # resolution
        res = odeint(model, y0, t, args=(param, ))
        # add to global vector
        Y_global = np.vstack((Y_global, res))
        t_global = np.vstack((t_global, t.reshape(-1, 1)))
            
    # for the last tk
    y0 = get_last_y_values(Y_global, treatment_dose=1)

    t = np.arange(tj[-1], (t0_treatment+cycle_duration*cycle_mumber), tstep)
    res = odeint(model, y0, t, args=(param, ))

    Y_global = np.vstack((Y_global, res))
    t_global = np.vstack((t_global, t.reshape(-1, 1)))

    # post-traitement -----------------------------------------------------
    y0 = get_last_y_values(Y_global, treatment_dose=1)
    t_post_treatment = np.arange(tj[-1], tf, tstep) 
    res_post_treatment = odeint(model, y0, t_post_treatment, 
                                args=(param, ))
    Y_global = np.vstack((Y_global, res_post_treatment))
    t_global = np.vstack((t_global, t_post_treatment.reshape(-1, 1)))
    
    return (Y_global, t_global)



def solve_ode_adaptive(model, y0, param, threshold=0.5, space_between_screening=90, 
                       space_between_doses=14, number_of_doses=85, 
                       t0=0, t0_treatment=36, tf=10000, tstep=0.1):
    ''' Simulation of an adaptative artificial chemotherapy schedule
    '''
    thr = threshold
    space_between_screening = space_between_screening    
    
    V_ini = y0[0] + y0[1] + y0[2] + y0[3] + y0[4]
    
    # pre-treatment -------------------------------------------------------
    t_pre_treatment = np.arange(t0, t0_treatment, tstep)
    Y_pre_treatment = odeint(model, y0, t_pre_treatment, args=(param, ))
    Y_global = np.array(Y_pre_treatment)
    t_global = np.array(t_pre_treatment.reshape(-1, 1))     # Vertical now
    screening = np.zeros(t_pre_treatment.shape)
    dosing = np.zeros(t_pre_treatment.shape)
         
    
    V_scr = np.copy(V_ini)
    doses_left = number_of_doses
    tj = np.array([-space_between_doses])
    treatment = np.array([[0,0]])
    while doses_left > 0: 

        if V_scr > (thr * V_ini):
            give_dose = 1
            treatment = np.vstack((treatment, t_global[-1]*np.ones([1,2])))
        else:
            give_dose = 0
            
        t_end = min(t_global[-1] + space_between_screening, 
                    t_global[-1] + doses_left*space_between_doses)
    
        # Interval from screening to first dose of the new period   
        t_pre = np.arange(t_global[-1]+tstep, tj[-1] + space_between_doses, tstep)
        y0 = get_last_y_values(Y_global, treatment_dose=0)    
        res_pre = odeint(model, y0, t_pre, args=(param, ))
        Y_global = np.vstack((Y_global, res_pre))
        t_global = np.vstack((t_global, t_pre.reshape(-1, 1)))
        screening = np.concatenate((screening,np.zeros(len(t_pre))))
        dosing = np.concatenate((dosing, np.zeros(len(t_pre))))

        tj = np.arange(t_global[-1]+tstep, t_end, space_between_doses)

        for i in range(0, len(tj)-1):
            y0 = get_last_y_values(Y_global, treatment_dose=give_dose)
            t = np.arange(tj[i], tj[i+1], tstep)
            res = odeint(model, y0, t, args=(param, ))
            Y_global = np.vstack((Y_global, res))
            t_global = np.vstack((t_global, t.reshape(-1, 1)))
            doses_left = doses_left - 1*give_dose
            #print("Quedan", doses_left, "dosis")
            screening = np.concatenate((screening,np.zeros(len(t))))
            dosing[-1] = 1*give_dose
            dosing = np.concatenate((dosing, np.zeros(len(t))))
            
        # Last dose till next screening    
        y0 = get_last_y_values(Y_global, treatment_dose=give_dose)    
        t_last = np.arange(t_global[-1]+tstep, t_end, tstep) 
        res_last = odeint(model, y0, t_last, args=(param, ))
        Y_global = np.vstack((Y_global, res_last))
        t_global = np.vstack((t_global, t_last.reshape(-1, 1)))
        doses_left = doses_left - 1*give_dose
        #print("Quedan", doses_left, "dosis")
        screening = np.concatenate((screening,np.zeros(len(t_last))))
        dosing[-1] = 1*give_dose
        dosing = np.concatenate((dosing, np.zeros(len(t_last))))
            
   
        V_scr = Y_global[-1,0] + Y_global[-1,1] + Y_global[-1,2] + Y_global[-1,3] + Y_global[-1,4]             
        screening[-1] = 1
        if give_dose == 1:
            treatment[-1,1] = t_global[-1]


    # post-traitement -----------------------------------------------------
    y0 = get_last_y_values(Y_global, treatment_dose=0)
    t_post_treatment = np.arange(t_global[-1]+tstep, tf, tstep) 
    res_post_treatment = odeint(model, y0, t_post_treatment, args=(param, ))
    Y_global = np.vstack((Y_global, res_post_treatment))
    t_global = np.vstack((t_global, t_post_treatment.reshape(-1, 1)))
    screening = np.concatenate((screening,np.zeros(len(t_post_treatment))))
    dosing = np.concatenate((dosing, np.zeros(len(t_post_treatment))))   

    return (Y_global, t_global, screening, dosing, treatment)