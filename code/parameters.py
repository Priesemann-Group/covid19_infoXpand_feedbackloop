import numpy as np
import pandas as pd

DEUparams = pd.read_csv('../parameters/DEU_params.csv', sep=';',header=0)

params_base = {
    #y0
    #Cs

    #Age-stratified parameters are commented out and taken from a parameter file
    
    'beta':0.5,
    'kappa': 0.8,
    'sigma': 1,
    #'delta': 0.0019,
    'rho': 0.25,
    #'gamma': 0.1,
    #'gamma_ICU': 0.13,
    #'Theta': 0.0005366,
    #'Theta_ICU': 0.09755,
    'omega_v': 1./(7.5*30),
    'omega_n': 1./(7.5*30),

    'mu': 0.267,
    'd_0': 8*30.,
    'd_mu': 0.,

    'a_Rt': 4.,
    'b_Rt': 0.7,
    'a_vac': 6.,
    'b_vac': 0.4,
    'gamma_cutoff': 45.,
    'tau_vac1': 6*7.,
    'tau_vac2': 2*7.,

    'k_ICUcap': 35.,
    'epsilon_k': 10.,
    #k_lowH_NPI
    #k_highH_NPI
    #k_lowH_noNPI
    #k_highH_noNPI

    #'alpha_u': 0.02,
    #'alpha_w': 0.03,
    #'u_base': 0.5,
    #'w_base': 0.0,
    #'chi_u': 0.1,
    #'chi_w': 0.2,
    'time_u': 7.,
    'time_w': 7.,
    'epsilon_u': 0.01,
    'epsilon_w': 0.01,

    'epsilon_free':14.,
    'influx': 1,
    't_max': 360.,
    'step_size': 0.1,
    'feedback_off': False,
}


def get_params(scen=3, inspiration='60'):

    params = params_base.copy()
    for i in ['delta', 'gamma', 'gamma_ICU', 'Theta', 'Theta_ICU', 'alpha_u', 'alpha_w', 'u_base', 'w_base', 'chi_u', 'chi_w']:
        params[i] = DEUparams[i].values

    params['y0'] = calc_y0(params, inspiration)
    params['Cs'] = calc_Cs()
    params.update( calc_ks(scen) )

    return params 


def calc_y0(params, vacfrac='60'):

    darkfigure = 1
    
    
    vacfrac = DEUparams[f'vacfrac_{vacfrac}'].values    
    frac = DEUparams['Anteil'].values
    
    R_raw = DEUparams['R_raw'].values
    cases = DEUparams['cases'].values
    ICU_raw = DEUparams['ICU'].values
    
    wvfrac = DEUparams['wanedfrac_V'].values
    wrfrac = DEUparams['wanedfrac_R'].values
    

    M = frac * 1e6
    
    V_raw = M * vacfrac
    R_raw = R_raw*darkfigure
    
    Rv_all = V_raw*R_raw/M #Overlap
    
    print('effective immune rate:', np.array((V_raw+R_raw - Rv_all))/M)
    print('total effective immune rate:', np.sum(np.array((V_raw+R_raw - Rv_all))/M*frac))
    
    V_all = V_raw - Rv_all
    V = V_all*(1-wvfrac)
    
    R_all = R_raw-Rv_all
    
    R = R_all*(1-wrfrac)
    Rv = Rv_all*(1-wvfrac) #to discuss (if wrfrac or wvfrac)


    
    Wn = wrfrac*R_all 
    Wv = wvfrac*(V_all+Rv_all) #depending on definition of Rv

    Etot = 1./params['rho']*cases*darkfigure
    Itot = 1./(params['gamma']+params['delta']+params['Theta'])* cases*darkfigure

    
    #S = M*(1-V_raw/M)*(1-R_raw/M) - Etot - Itot - ICU_raw  (the same as below)
    
    S = M - V - R - Rv - Wn - Wv -Etot - Itot - ICU_raw

    for i, s in enumerate(S):
        if s <= 0:
            print('Error')

    En = (Wn) / (S+Wn+Wv) * Etot
    Ev = (Wv) / (S+Wn+Wv) * Etot
    E  =  (S) / (S+Wn+Wv) * Etot

    In = (Wn) / (S+Wn+Wv) * Itot
    Iv = (Wv) / (S+Wn+Wv) * Itot
    I  =  (S) / (S+Wn+Wv) * Itot


    ICUimmune = (1-params['kappa'])*(In+Iv)

    ICUv =   (Iv*(1-params['kappa'])) / (I+ICUimmune) * ICU_raw
    ICU  = (I+In*(1-params['kappa'])) / (I+ICUimmune) * ICU_raw


    

    

    y0= {
            'S':   S,
            'V':   V,
            'Wn':  Wn,
            'Wv':  Wv,
            'E':   E,
            'EBn': En,
            'EBv': Ev,
            'I':   I,
            'IBn': In,
            'IBv': Iv,
            'ICU': ICU,
            'ICUv':ICUv,
            'R':   R,
            'Rv':  Rv,
            'UC':  V_raw,
            'WC':  [0.]*6,
            'D':   [0.]*6,
            'C':   [0.]*6,
        }

        

    
    y0_array = [y0['S'],y0['V'],y0['Wn'],y0['Wv'],y0['E'],y0['EBn'],y0['EBv'],y0['I'],y0['IBn'],y0['IBv'],
                y0['ICU'],y0['ICUv'],y0['R'],y0['Rv'],y0['UC'],y0['WC'],y0['D'],y0['C']]
    
    return np.array(y0_array).flatten()


def calc_Cs():
    C_household = np.loadtxt('../parameters/Germany_country_level_F_household_setting_85.csv', delimiter=',')
    C_school = np.loadtxt('../parameters/Germany_country_level_F_school_setting_85.csv', delimiter=',')
    C_workplace = np.loadtxt('../parameters/Germany_country_level_F_work_setting_85.csv', delimiter=',')
    C_community = np.loadtxt('../parameters/Germany_country_level_F_community_setting_85.csv', delimiter=',')
    
    germany = np.loadtxt('../parameters/germany.csv', delimiter=',')
    germany[84,1]= germany[84:,1].sum()
    germany = germany[:85,1]
    
    w_h = 4.1100
    w_s = 11.4100
    w_w = 8.0700
    w_c = 2.7900
    
    ind = [0, 20, 40, 60, 70, 80, -1];

    for i in range(85):
        for j in range(85):
            C_household[i,j] *= germany[i]*germany[j]
            C_school[i,j] *= germany[i]*germany[j]
            C_workplace[i,j] *= germany[i]*germany[j]
            C_community[i,j] *= germany[i]*germany[j]
    
    C_H = np.zeros([6,6])
    C_S = np.zeros([6,6])
    C_W = np.zeros([6,6])
    C_C = np.zeros([6,6])
    
    def M(i):
        return germany[ind[i]:ind[i+1]].sum()

    for i in range(6):
        for j in range(6):
            C_H[i,j] = C_household[ind[i]:ind[i+1],ind[j]:ind[j+1]].sum()/M(i)/M(j)
            C_S[i,j] = C_school[ind[i]:ind[i+1],ind[j]:ind[j+1]].sum()/M(i)/M(j)
            C_W[i,j] = C_workplace[ind[i]:ind[i+1],ind[j]:ind[j+1]].sum()/M(i)/M(j)
            C_C[i,j] = C_community[ind[i]:ind[i+1],ind[j]:ind[j+1]].sum()/M(i)/M(j)
    
    C = w_h*C_H + w_s*C_S + w_w*C_W + w_c*C_C
    rho = max(np.linalg.eigvals(C))
    
    C_H = w_h * C_H / rho
    C_S = w_s * C_S / rho
    C_W = w_w * C_W / rho
    C_C = w_c * C_C / rho
    
    Cs = np.array([C_H, C_S, C_W, C_C])

    return Cs


def calc_ks(scen):
    ks = pd.read_csv('../parameters/scenariosDefinition.csv', sep=';')
    ks_NPI = ks.loc[ks['Scenarios']==scen]
    ks_noNPI = ks.loc[ks['Scenarios']==1]

    d = {}
    d['k_lowH_NPI']    = np.array([ks_NPI['householdhigh'],  ks_NPI['schoolhigh'],  ks_NPI['workplacehigh'],  ks_NPI['communityhigh']]).flatten()
    d['k_highH_NPI']   = np.array([ks_NPI['householdlow'],   ks_NPI['schoollow'],   ks_NPI['workplacelow'],   ks_NPI['communitylow']]).flatten()
    d['k_lowH_noNPI']  = np.array([ks_noNPI['householdhigh'],ks_noNPI['schoolhigh'],ks_noNPI['workplacehigh'],ks_noNPI['communityhigh']]).flatten()
    d['k_highH_noNPI'] = np.array([ks_noNPI['householdlow'], ks_noNPI['schoollow'], ks_noNPI['workplacelow'], ks_noNPI['communitylow']]).flatten()

    return d

