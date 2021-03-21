# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)
# %%
folder_path = "C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15_cropped"
results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
results_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15_cropped'

folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
results_path = results_path.replace('C:/Users/yj/','/Users/yeonsu/')

pi = piv.ParticleImage(folder_path,results_folder_path)
# %%
step = 25
x,y,u_avg,v_avg,u_std,v_std = pi.get_entire_avg_velocity_map(step,'003_90')

# %%
start_i = 0
stop_i = -1
step_i = 50

v_array = -v_avg[start_i:stop_i:step_i,:]
y_array = y[start_i:stop_i:step_i,0]
N = len(v_array)

s = []
for y in (y_array - 55):
    s.append('%.2f'%y)

C = 0.01
fig,ax = plt.subplots(figsize=(40,10))
for i,vv in enumerate(v_array):
    # ax.plot(vv + C*y_array[i],x[0,:],'k--')
    ax.plot(C*vv + y_array[i],x[0,:],'k--')
    ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
    ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
    for j, vvv in enumerate(vv):         
        ax.arrow(y_array[i], x[0,j], C*(vvv-50), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

ax.set_xticks(y_array)
ax.set_xticklabels(s)
# %%
