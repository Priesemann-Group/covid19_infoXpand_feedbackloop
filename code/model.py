from scipy.integrate import solve_ivp
from scipy.special import gamma as gamma_func
import numpy as np
import pickle


class Model:
    def __init__(
        self,

        y0, Cs,
        beta, kappa, sigma, delta, rho,
        gamma, gamma_ICU,
        Theta, Theta_ICU,
        omega_v, omega_n,

        mu, d_0, d_mu,

        a_Rt, b_Rt, a_vac, b_vac, gamma_cutoff,
        tau_vac1, tau_vac2,

        k_ICUcap, epsilon_k,
        k_lowH_NPI, k_highH_NPI,
        k_lowH_noNPI, k_highH_noNPI,

        alpha_u, alpha_w,
        u_base, w_base,
        chi_u, chi_w,
        time_u, time_w,
        epsilon_u, epsilon_w,

        epsilon_free,
        influx,
        t_max, step_size,
        feedback_off,
    ):
        self.y0 = y0
        self.Cs = Cs

        self.beta = beta
        self.kappa = kappa
        self.sigma = sigma
        self.delta = delta
        self.rho = rho
        self.gamma = gamma
        self.gamma_ICU = gamma_ICU
        self.Theta = Theta
        self.Theta_ICU = Theta_ICU
        self.omega_v = omega_v
        self.omega_n = omega_n

        self.mu = mu
        self.d_0 = d_0
        self.d_mu = d_mu

        self.a_Rt = a_Rt
        self.b_Rt = b_Rt
        self.a_vac = a_vac
        self.b_vac = b_vac
        self.gamma_cutoff = gamma_cutoff
        self.tau_vac1 = tau_vac1
        self.tau_vac2 = tau_vac2

        self.k_ICUcap = k_ICUcap
        self.epsilon_k = epsilon_k
        self.k_lowH_NPI = k_lowH_NPI
        self.k_highH_NPI = k_highH_NPI
        self.k_lowH_noNPI = k_lowH_noNPI
        self.k_highH_noNPI = k_highH_noNPI

        self.alpha_u = alpha_u
        self.alpha_w = alpha_w
        self.u_base = u_base
        self.w_base = w_base
        self.chi_u = chi_u
        self.chi_w = chi_w
        self.time_u = time_u
        self.time_w = time_w
        self.epsilon_u = epsilon_u
        self.epsilon_w = epsilon_w

        self.epsilon_free = epsilon_free
        self.influx = influx
        self.t_max = t_max
        self.step_size = step_size
        self.feedback_off = feedback_off

        self.eqs = 18                   # number equations
        self.ags = len(y0)//self.eqs    # number agegroups

        self.M = self.y0.reshape([self.eqs,self.ags])[:-4,:].sum(axis=0)
        self.u_max = 1-chi_u
        self.w_max = 1-chi_w
        self.t_min = self.gamma_cutoff + max(self.tau_vac1,self.tau_vac2)

        self.exp = 1.0


    def time2index(self, t):
        return round((t+self.t_min)/self.step_size)

    def softplus(self, slope, base, threshold, epsilon):
        return lambda x: slope*epsilon*np.log(np.exp(1/epsilon*(threshold-x))+1) + base

    def H(self,t,tau,a,b):
        x = -np.arange(-self.gamma_cutoff,0,self.step_size)
        d = self.data[self.time2index(t-self.gamma_cutoff-tau):self.time2index(t-tau),10:12].sum(axis=(1,2))
        g = b**a * x**(a-1) * np.exp(-b*x) / gamma_func(a)
        return (d*g).sum()*self.step_size

    def H_Rt(self, t):
        return self.H(t, 0, self.a_Rt, self.b_Rt)

    def H_vac1(self, t):
        return self.H(t, self.tau_vac1, self.a_vac, self.b_vac)

    def H_vac2(self, t):
        return self.H(t, self.tau_vac2, self.a_vac, self.b_vac)

    def I_eff(self, I, IBn, IBv):
        return (I + self.sigma*(IBn+IBv)) + self.influx*self.M/self.M.sum()

    def pre_Gamma(self, t):
        return np.cos(2*np.pi*(t+self.d_0-self.d_mu)/360.)

    def Gamma(self, t):
        return 1 + self.mu * np.sign(self.pre_Gamma(t)) * np.abs(self.pre_Gamma(t))**self.exp

    def k_lowH(self, t):
        if t<180-self.epsilon_free:
            return self.k_lowH_NPI
        if t>180+self.epsilon_free:
            return self.k_lowH_noNPI
        else:
            return self.k_lowH_NPI + (self.k_lowH_noNPI-self.k_lowH_NPI)*(t-180+self.epsilon_free)/2/self.epsilon_free
 
    def k_highH(self, t):
        if t<180-self.epsilon_free:
            return self.k_highH_NPI
        if t>180+self.epsilon_free:
            return self.k_highH_noNPI
        else:
            return self.k_highH_NPI + (self.k_highH_noNPI-self.k_highH_NPI)*(t-180+self.epsilon_free)/2/self.epsilon_free

    def k_selfregulation(self,t):
        if self.feedback_off:
            res = self.k_lowH(t)
        else:
            a = (self.k_lowH(t)-self.k_highH(t))/self.k_ICUcap
            res = self.softplus(a, self.k_highH(t), self.k_ICUcap, self.epsilon_k)(self.H_Rt(t))
        # scale down households
        res[0] *= res[1:].sum()/3.
        return res
 
    def Rt(self, t): # not directly used in the code, for plotting only
        #CM = np.zeros([6,6])
        #for i in range(4):
        #    CM += self.Cs[i,:,:] * self.new_sr(t)[i]
        CM = (np.moveaxis(self.Cs,0,2) * self.k_selfregulation(t)).sum(axis=2)
        return self.beta / self.gamma * self.Gamma(t) * max(np.linalg.eigvals(CM))

    def u_w(self, t):
        return self.u_base + (self.u_max-self.u_base)*(1-np.exp(-self.alpha_u*self.H_vac1(t)))

    def Phi(self, t, UC, frac):
        return self.softplus(frac/self.time_u, 0, self.u_w(t), self.epsilon_u)(UC/self.M) * self.M

    def w_w(self, t):
        return self.w_base + (self.w_max-self.w_base)*(1-np.exp(-self.alpha_w*self.H_vac2(t)))

    def phi(self, t, UC, WC, frac):
        return self.softplus(frac/self.time_w, 0, self.w_w(t), self.epsilon_w)(WC/UC) * UC

    def get_phis(self, t, y):
        (S,V,Wn,Wv,E,EBn,EBv,I,IBn,IBv,ICU,ICUv,R,Rv,UC,WC,D,C) = np.split(y, self.eqs)
        Phi = self.Phi(t, UC, (S+Wn)/(self.M-UC))
        phi = self.phi(t, UC, WC, (Wv)/(UC-WC))
        return Phi, phi

    def fun(self, t, y):
        #y.reshape([self.eqs,self.ags])
        (S,V,Wn,Wv,E,EBn,EBv,I,IBn,IBv,ICU,ICUv,R,Rv,UC,WC,D,Dv) = np.split(y, self.eqs)

        # definitions to make DEs more readable
        M = self.M
        rho = self.rho
        kappa = self.kappa
        gamma = self.gamma
        gamma_ICU = self.gamma_ICU
        delta = self.delta
        Theta = self.Theta
        Theta_ICU = self.Theta_ICU
        Rt = self.Rt(t)
        I_eff = self.I_eff(I, IBn, IBv)
        omega_n = self.omega_n
        omega_v = self.omega_v
        Phi, phi = self.get_phis(t,y)
#        Phi = self.Phi(t, UC, (S+Wn)/(M-UC))
#        phi = self.phi(t, UC, WC, (Wv)/(UC-WC))
        
        
        #Effective time dependent contact matrix
        #CM = np.zeros([6,6])
        #for i in range(4):
        #    CM += self.Cs[i,:,:] * self.new_sr(t)[i]

        CM = (np.moveaxis(self.Cs,0,2) * self.k_selfregulation(t)).sum(axis=2)
        # PD: CM is no longer symmetric, transposed should be right
        infect = self.beta*self.Gamma(t) * np.matmul(CM.transpose(), I_eff/M)


        # differential equations
        dS = -S*infect - Phi*(S/(S+Wn))
        dV = Phi + phi - omega_v*V
        dWn = -Wn*infect + omega_n*R - Phi*(Wn/(S+Wn))
        dWv = -Wv*infect + omega_v*V + omega_n*Rv - phi
        dE = S*infect - rho*E
        dEBn = Wn*infect - rho*EBn
        dEBv = Wv*infect - rho*EBv
        dI = rho*E - (gamma+delta+Theta)*I
        dIBn = rho*EBn - (gamma + (Theta+delta)*(1-kappa))*IBn
        dIBv = rho*EBv - (gamma + (Theta+delta)*(1-kappa))*IBv
        dICU = delta*(I + (1-kappa)*IBn) - (Theta_ICU+gamma_ICU)*ICU
        dICUv = delta*(1-kappa)*IBv - (Theta_ICU+gamma_ICU)*ICUv
        dR = gamma*(I+IBn) - omega_n*R + gamma_ICU*ICU
        dRv = gamma*IBv - omega_n*Rv + gamma_ICU*ICUv
        dUC = Phi
        dWC = phi
        dD = Theta*I + (1-kappa)*Theta*IBn + Theta_ICU*ICU
        dDv = (1-kappa)*Theta*IBv + Theta_ICU*ICUv

        return np.concatenate([dS,dV,dWn,dWv,dE,dEBn,dEBv,dI,dIBn,dIBv,dICU,dICUv,dR,dRv,dUC,dWC,dD,dDv]).flatten()


    def build_data(self):
        self.times = np.arange(0,self.t_max,self.step_size)
        self.data = np.zeros((self.time2index(self.t_max)+100,self.eqs,self.ags))    # solve_ivp tends to look into the future, H needs values
        self.data[:self.time2index(0)+1,:,:] = [self.y0.reshape([self.eqs,self.ags])]*(self.time2index(0)+1)

    def run(self):
        self.build_data()
        for i in range(len(self.times)-1):
            res = solve_ivp(self.fun, (self.times[i],self.times[i+1]), self.data[self.time2index(i*self.step_size)].flatten())
            self.data[self.time2index(i*self.step_size)+1,:,:] = res["y"][:,-1:].reshape(self.eqs,self.ags)
        return self.times, self.chopped_data()

    def chopped_data(self):
        return self.data[self.time2index(0):-100,:,:]
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

