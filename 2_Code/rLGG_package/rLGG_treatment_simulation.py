
import numpy as np
from scipy.integrate import odeint


def __get_last_y_values(Y_global, treatment_dose=0):
    Vs0 = Y_global[-1, 0]
    Vd0 = Y_global[-1, 1]
    Vpi0 = Y_global[-1, 2]
    Vp0 = Y_global[-1, 3]
    Vr0 = Y_global[-1, 4]
    D0 = Y_global[-1, 5] + treatment_dose
    y0 = [Vs0, Vd0, Vpi0, Vp0, Vr0, D0]
    return y0

def chemotherapy_simulation(model, Y_global, t_global, treatment, param, 
                            tstep=0.1, cycle_duration=28, doses_spacing=1, 
                            doses_number=5):
    ''' Simulate one cycle of chemotherapy
    '''
    # treatment parameters
    t0_treatment = treatment[0]
    tf_treatment = treatment[1]

    # tj vector
    tj = []
    for t in range(t0_treatment, tf_treatment+1):
        if (t - t0_treatment) % cycle_duration in range(0, doses_number, 
                                                        doses_spacing):
            tj.append(t)

    for i in range(0, len(tj)-1):
        # update initial condition
        y0 = __get_last_y_values(Y_global, treatment_dose=1)
        # simulation time
        t = np.arange(tj[i], tj[i+1], tstep)
        # resolution
        res = odeint(model, y0, t, args=(param, ))
        # add to global vector
        Y_global = np.vstack((Y_global, res))
        t_global = np.vstack((t_global, t.reshape(-1, 1)))
            
    # for the last tk
    y0 = __get_last_y_values(Y_global, treatment_dose=1)

    t = np.arange(tj[-1], treatment[1], tstep)
    res = odeint(model, y0, t, args=(param, ))
    
    Y_global = np.vstack((Y_global, res))
    t_global = np.vstack((t_global, t.reshape(-1, 1)))
        
    return (Y_global, t_global)

def chemotherapy_stupp_simulation(model, Y_global, t_global, treatment, param, 
                                  tstep=0.1, cycle_duration=7, doses_spacing=1, 
                                  doses_number=5):
    ''' Simulate one cycle of stupp protocol
    '''
    
    # treatment parameters
    t0_treatment = treatment[0]
    tf_treatment = treatment[1]

    # tj vector
    tj = []
    for t in range(t0_treatment, tf_treatment+1):
        if (t - t0_treatment) % cycle_duration in range(0, doses_number, 
                                                        doses_spacing):
            tj.append(t)

    for i in range(0, len(tj)-1):
        # update initial condition
        y0 = __get_last_y_values(Y_global, treatment_dose=0.5)
        # simulation time
        t = np.arange(tj[i], tj[i+1], tstep)
        # resolution
        res = odeint(model, y0, t, args=(param, ))
        # add to global vector
        Y_global = np.vstack((Y_global, res))
        t_global = np.vstack((t_global, t.reshape(-1, 1)))
            
    # for the last tk
    y0 = __get_last_y_values(Y_global, treatment_dose=0.5)

    t = np.arange(tj[-1], treatment[1], tstep)
    res = odeint(model, y0, t, args=(param, ))

    Y_global = np.vstack((Y_global, res))
    t_global = np.vstack((t_global, t.reshape(-1, 1)))
        
    return (Y_global, t_global)

