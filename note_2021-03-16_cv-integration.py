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
# folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
# results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-11\\Flat_10 (black)_motor15"
results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)
# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,0,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0.25, "save_result": True,"show_result": False
}
pi.set_piv_param(piv_cond)
# %%
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]

pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()

# %%
os.listdir('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]')
# %%
x_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/x.txt'
y_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/y.txt'

ul_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/u_tavg_003_90.txt'
vl_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/v_tavg_003_90.txt'

ur_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/u_tavg_003_90.txt'
vr_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/v_tavg_003_90.txt'
# %%
x = np.loadtxt(x_path)
y = np.loadtxt(y_path)

u_left = np.loadtxt(ul_path)
v_left = np.loadtxt(vl_path)

u_right = np.loadtxt(ur_path)
v_right = np.loadtxt(vr_path)
# %%
plt.plot(-v_left[10,:],x[0,:])
plt.plot(-v_right[10,:],x[0,:])
# %%
rho = 1e3
D = np.sum(0.5*rho*(v_left[10,:-1]**2 - v_right[10,:-1]**2)*1e-6 * (x[0,1:] - x[0,0:-1]) )*1e-3 * 0.048
print(D)

# %%
pi.set_piv_param({'raw_or_cropped': True})
pi.piv_dict_list[0]
# %%
# pos 0, VOFFSET 0
sd1 = {'pos': 1, 'VOFFSET': 0}
# pos 6, VOFFSET 840
sd2 = {'pos': 6, 'VOFFSET': 840}


# %%
pi.piv_over_time2(sd1,start_index=,N=0)
pi.piv_over_time2(sd2,start_index=,N=0)
# %%
x_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/x_full.txt'
y_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/y_full.txt'

ul_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/u_full_tavg_003_90.txt'
vl_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/v_full_tavg_003_90.txt'

ur_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/u_full_tavg_003_90.txt'
vr_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/v_full_tavg_003_90.txt'
# %%
x = np.loadtxt(x_path)
y = np.loadtxt(y_path)

u_left = np.loadtxt(ul_path)
v_left = np.loadtxt(vl_path)

u_right = np.loadtxt(ur_path)
v_right = np.loadtxt(vr_path)
# %%
plt.plot(-v_left[0,:],x[0,:],'o-')
plt.plot(-v_right[15,:],x[0,:],'o-')
# %%
rho = 1e3
D = np.sum(0.5*rho*(v_left[10,:-1]**2 - v_right[10,:-1]**2)*1e-6 * (x[0,1:] - x[0,0:-1]) )*1e-3 * 0.048
print(D)
# %%
