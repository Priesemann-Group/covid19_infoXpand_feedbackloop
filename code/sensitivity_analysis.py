import numpy as np
import time
from copy import deepcopy


def sweep(model, param_name, param_values, vector=False):

    ms = [None]*len(param_values)
    for i in range(len(param_values)):
        ms[i] = deepcopy(model)
        if vector:
            setattr(ms[i], param_name, param_values[i]*getattr(ms[i], param_name))
        else:
            setattr(ms[i], param_name, param_values[i])

        start_time = time.time()
        times, data = ms[i].run()
        print(str(i+1)+'/'+str(len(param_values)), time.time() - start_time, end='\r')

    icu_winter = [0]*len(ms)
    icu_spring = [0]*len(ms)
    for i,m in enumerate(ms):
        tmp = m.chopped_data()[:,10:12,:].sum(axis=(1,2))
        icu_winter[i] = tmp[:1800].max()
        icu_spring[i] = tmp[1800:].max()

    d = {'models':ms, 'icu_winter':icu_winter, 'icu_spring':icu_spring, 'param_name':param_name, 'param_values':param_values}

    return d



