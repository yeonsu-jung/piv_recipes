# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)

# %%
def get_drag_by_integration(folder_path,results_folder_path):
    pi = piv.ParticleImage(folder_path,results_folder_path)
    stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]
    pi.set_param_string_list(stitch_list)
    pi.piv_dict_list = pi.param_dict_list
    pi.check_piv_dict_list()

    x,y,u_l,v_l,u_r,v_r = pi.get_left_right_velocity_map('003_90')

    step = 25
    x_t,y_t,u_t,v_t,ut_std,vt_std,x_d,y_d,u_d,v_d,ud_std,vd_std = pi.get_entire_ul_velocity_map(step,'003_90')

    # Correcting direction and signs
    y_lr = x[3,:]

    u_left = -v_l[3,:]
    v_left = u_l[3,:]

    u_right = -v_r[-3,:]
    v_right = u_r[-3,:]

    x_top = y_d[:,-1] - 50
    u_top = -v_d[:,-1]
    v_top = -u_d[:,-1]

    x_bottom = y_t[:,-1]
    u_bottom = -v_t[:,-1]
    v_bottom = -u_t[:,-1]
    
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].plot(u_left,y_lr,label='left')
    ax[0].plot(u_right,y_lr,label = 'right')
    ax[1].plot(x_top,v_top,label = 'top')
    ax[1].plot(x_top,v_bottom,label = 'bottom')

    ax[0].set_xlabel('u (mm/s)')
    ax[0].set_ylabel('y (mm)')
    ax[1].set_xlabel('x (mm)')
    ax[1].set_ylabel('v (mm/s)')

    rho = 1e3
    span = 0.048

    flowrate_left = rho*np.trapz(u_left,y_lr)*1e-6*span
    flowrate_right = rho*np.trapz(u_right,y_lr)*1e-6*span
    flowrate_top = rho*np.trapz(v_top,x_top)*1e-6*span
    flowrate_bottom = rho*np.trapz(-v_bottom,x_top)*1e-6*span

    momentum_left = rho*np.trapz(u_left**2,y_lr)*1e-9*span # in
    momentum_right = rho*np.trapz(u_right**2,y_lr)*1e-9*span # out
    momentum_top = rho*np.trapz(v_top*u_top,x_top)*1e-9*span # out
    momentum_bottom = rho*np.trapz(-v_bottom*u_bottom,x_top)*1e-9*span # out

    (flowrate_left,flowrate_right,flowrate_top,flowrate_bottom)
    print(momentum_left,momentum_right,momentum_top,momentum_bottom)

    momentum_deficit = momentum_left - momentum_right - momentum_top - momentum_bottom
    return momentum_deficit
# %%
folder_path = "/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/2_1_1_10 (black)_motor15"
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

drag = get_drag_by_integration(folder_path,results_folder_path)
print(drag)
# %%
ss = folder_path
ss2 = ss.replace('/Users/yeonsu/','C:/Users/yj/')
print(ss,ss2)

# %%
folder_path = "/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15"
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'
drag_flat = get_drag_by_integration(folder_path,results_folder_path)
print(drag_flat)

# %%

# %%

# %%
