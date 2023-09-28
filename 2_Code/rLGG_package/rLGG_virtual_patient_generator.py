
import numpy as np
from sklearn.neighbors import KernelDensity


class VirtualPatient:
    
    def __init__(self, patients_param, V0_real_patients):
        ''' Creation of a virtual patient with parameters and initial
        values from uniform distributions, which the bounds come
        from real patients data. Some parameters are generated with
        a cholenski decomposition, that able to reproduce parameters
        correlation observe in real patients data.
        
        See 'Materials and Methods' from original papers for more 
        information.
        
        Parameters
        ----------
        patients_param: DataFrame
            DataFrame with real patients param
        V0_real_patients: DataFrame
            DataFrame containing real patient volume values that are used 
            to estimate the initial volume distribution
        
        Attributes
        ---------
        param: list
            generated patient parameters 
        y0: list
            initial condition of the system, with a random initial tumor volume
        n_cycle: int
            number of chemotherapy cycle
        n_doses: int
            total number of TMZ doses
        critical_volume: float
            random critical volume
        '''
        self.param = self.random_param(patients_param)
        self.y0 = self.random_v0(V0_real_patients)
        self.n_cycle = self.random_number_cycle()
        self.n_doses = self.n_cycle * 5
        self.critical_volume = self.random_critical_volume()

    def random_param(self, patients_param):
        ''' Compute random parameter by taking account of real correlation
        '''
        # take min and max value of each parameters
        p_range = []
        for p in patients_param:
            p_max = max(patients_param[p])
            p_min = min(patients_param[p])
            p_range.append((p_max, p_min))
        # uniform distribution (initial value)
        rho1 = np.random.uniform(p_range[0][1], p_range[0][0])
        tau = np.random.uniform(p_range[2][1], p_range[2][0])
        a1 = np.random.uniform(p_range[4][1], p_range[4][0])
        # random distribution
        U_rho1 = np.random.uniform(p_range[0][1], p_range[0][0])
        U_tau = np.random.uniform(p_range[2][1], p_range[2][0])
        U_a1 = np.random.uniform(p_range[4][1], p_range[4][0])
        # Cholenski decomposition            
        rho2 = self.cholenski(0.749, 0.0008, 0.785, tau, U_tau) 
        psi = self.cholenski(722, -0.1834, 0.821, rho1, U_rho1)
        a2 = self.cholenski(0.168, 0.0074, 0.954, a1, U_a1)
        
        # fix param
        b = 0.1
        l1 = 15
        lamb = - (np.log(1/2)/2)*24
        
        return [rho1, rho2, tau, psi, a1, a2, b, l1, lamb]
    
    def cholenski(self, a, b, r, x, x_dist):
        '''Cholenski decomposition
        
        Parameters:
        -----------
        a : float
            slope of the regression line
        b : float 
            intercept of the regression line
        r : float
            correlation coefficient 
        x : float
            value of correlated parameter
        x_dist : float
            a random value which follows the same distribution as the
            correlated parameters.
        '''
        return r * (a*x + b) + np.sqrt((1-r**2)) * (a*x_dist + b)  
    
    def random_v0(self, V0_real):
        '''compute random initial tumor size according to a gaussian 
        kernel density estimation distribution
        '''
        X = np.array(V0_real['Volume']).reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=10).fit(X)
        Vs0 = 0
        while Vs0 < 1.2 or Vs0 > 181:
            Vs0 = kde.sample(1)[0][0]
        return [Vs0, 0, 0, 0, 0, 0]
    
    def random_number_cycle(self):
        '''compute random number of cycle of doses
        '''
        number_cycle = 0
        while number_cycle <= 5 or number_cycle >= 33:
            number_cycle = (round(np.random.normal(19, 7)))
        return number_cycle
    
    def random_critical_volume(self):
        '''compute the critical volume
        '''
        return np.random.normal(280, 20)