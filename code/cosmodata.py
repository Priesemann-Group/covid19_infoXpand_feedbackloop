import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma as gammafunc
from scipy.stats import pearsonr


#germany_owid = pd.read_csv('../parameters/germany_ourworldindata.csv', sep=',', header=None, usecols=[3,18,45])
owid_raw = pd.read_csv('../parameters/owid-covid-data.csv', sep=',', header=0)
owid = owid_raw.dropna(axis=0, how='any', subset=['icu_patients_per_million', 'date', 'stringency_index'])

germany_owid=owid[owid['iso_code'] == 'DEU']


owid_ICU = list(germany_owid['icu_patients_per_million'])
owid_NPI = list(germany_owid['stringency_index'])
owid_dates = list(germany_owid['date'])


ICUtime=[]
NPItime =[]
datesdict = {}
for times, dates in zip(range(len(owid_dates)), owid_dates):
    ICUtime.append(float(owid_ICU[times]))
    NPItime.append(float(owid_NPI[times])/100)
    datesdict[dates] = times
    
t=np.linspace(0,len(NPItime),len(NPItime))


feiernnoages = pd.read_csv('../parameters/PrivateFeiern_no_ages.csv', sep=',', header=None, usecols=[0,3])

datesCosmo = feiernnoages[0][:-1]

cosmotimeline=[]
cosmotimelineICU=[]
for i in datesCosmo[1:]:
    cosmotimeline.append(datesdict[i])
    cosmotimelineICU.append(ICUtime[datesdict[i]])

avggroup = []
cosmoraw = []
for i in range(len(datesCosmo)-1):
    avggroup.append((float(feiernnoages[3][i+1])))
    cosmoraw.append(float(feiernnoages[3][i+1]))

    



# ----------------------------------------- ROMANIA -----------------------------------------------


romania_owid=owid[owid['iso_code'] == 'ROU']

ROU_owid_ICU = list(romania_owid['icu_patients_per_million'])
ROU_owid_vaccines = list(romania_owid['new_vaccinations_smoothed_per_million'])
ROU_owid_dates = list(romania_owid['date'])


ROU_t=np.arange(0,len(ROU_owid_dates)-1,1)

ROU_ICUtime=[]
ROU_vaccinetime=[]
ROU_datesdict = {}
for times, dates in zip(range(len(ROU_owid_dates)-1), ROU_owid_dates[1:]):
    ROU_ICUtime.append(float(ROU_owid_ICU[1+times]))
    ROU_vaccinetime.append(float(ROU_owid_vaccines[1+times]))
    ROU_datesdict[dates] = times
    
#ax.plot(ROU_t, ROU_ICUtime)
#ax.plot(ROU_t, ROU_vaccinetime)
