
import numpy as np

def rLGG_model_v3(y, t, param):
    '''Macroscopic growth model of Low-grade gliomas with
    resistance.
    '''
    # Unpack param   
    rho1, rho2, tau, psi, a1, a2, b, l1, lamb = param
    Vs, Vd, Vpi, Vp, Vr, D = y
    # Activation function
    f = lambda x: (1/2) * (1 - np.tanh((x - 0.01)/0.01))
    # EDO
    dVsdt = rho1*Vs - psi*Vs*D - a1*Vs*D + b*Vp
    dVddt = -tau*Vd + psi*Vs*D
    dVPidt = a1*Vs*D - l1*Vpi*f(D) 
    dVPdt = l1*Vpi*f(D) - a2*Vp*D - b*Vp
    dVrdt = rho2*Vr + a2*Vp*D
    dDdt = - lamb*D    
    return [dVsdt, dVddt, dVPidt, dVPdt, dVrdt, dDdt]


