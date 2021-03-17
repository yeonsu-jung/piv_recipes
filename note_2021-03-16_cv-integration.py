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

# folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-11\\Flat_10 (black)_motor15"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

folder_path = "/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15"
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

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
x_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/x_full.txt'
y_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/y_full.txt'

ul_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/u_full_tavg_003_90.txt'
vl_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/v_full_tavg_003_90.txt'

ur_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/u_full_tavg_003_90.txt'
vr_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos6_VOFFSET840_timing100_ag1_dg1_laser5_[03-11]/v_full_tavg_003_90.txt'

entire_u_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/entire_u_tavg.txt'
# %%
x = np.loadtxt(x_path)
y = np.loadtxt(y_path)

u_left = np.loadtxt(ul_path)
v_left = np.loadtxt(vl_path)

u_right = np.loadtxt(ur_path)
v_right = np.loadtxt(vr_path)

entire_u = np.loadtxt(entire_u_path)
# %%
plt.plot(-v_left[0,:],x[0,:],'o-',label='left')
plt.plot(-v_right[15,:],x[0,:],'o-',label='right')
plt.xlabel('u (mm/s)')
plt.ylabel('y (mm)')
plt.legend()
# %%
x[0,:]
# %%
np.mean(v_right)
np.mean(v_left)
print(np.mean(v_right), np.mean(v_left))
print((np.mean(v_right) - np.mean(v_left))*0.015)
# %%
1e3*np.sum((-v_left[0,:-1] + v_right[0,:-1])* (x[0,1:] - x[0,0:-1]))*1e-6 * 0.048
# %%
rho = 1e3
mdot = rho*np.sum((-v_left[0,:-1] + v_right[0,:-1])* (x[0,1:] - x[0,0:-1])) * 1e-6 * 0.048

mdot
# %%
np.sum(rho*(v_left[0,:-1]**2 - v_right[16,:-1]**2)*1e-6 * (x[0,1:] - x[0,0:-1]) )*1e-3 * 0.048
# %%
rho = 1e3
D = np.sum(rho*(v_left[0,:-1]**2 - v_right[16,:-1]**2)*1e-6 * (x[0,1:] - x[0,0:-1]) )*1e-3 * 0.048 - mdot*np.mean(-v_left[0,:-1])*1e-3
print(D)
# %%
u_right[10,-4]
# %%
0.01 * 0.048 * 0.1 * 0.5 * 1e3
# %%
entire_u.shape
# %%
plt.plot(entire_u[:900,-1])
# %%
np.sum(entire_u[:900,-1])

# %%

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,580,0],    
    "sn_threshold": 1.0001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": False,
    "check_angle": False
}
pi.set_piv_param(piv_cond)
# pi.quick_piv({'pos':3, 'VOFFSET': 420})
# %%
for pd in pi.piv_dict_list:
    pi.piv_over_time3(pd,start_index=3,N=90,tag='upper')

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,0,580],    
    "sn_threshold": 1.0001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": False,
    "check_angle": False
}
pi.set_piv_param(piv_cond)
pi.quick_piv({'pos':3, 'VOFFSET': 420})
# %%
for pd in pi.piv_dict_list:
    pi.piv_over_time3(pd,start_index=3,N=90,tag='lower')
# %%

u_upper = np.loadtxt('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/u_upper_tavg_003_90.txt')

u_upper = np.loadtxt('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos1_VOFFSET0_ag1_dg1_laser5_[03-11]/u_lower_tavg_003_90.txt')

# %%

step = 25
xu,yu,uu_avg,vu_avg,uu_std,vu_std,xl,yl,ul_avg,vl_avg,ul_std,vl_std = pi.get_entire_ul_velocity_map(step,'003_90')

# %%
(np.mean(uu_avg[:,0]) - np.mean(ul_avg[:,-1])) * 0.001 * 0.048 * (0.1766 - 0.02535) * 0.5 * 1e3

# %%
plt.plot(yu[:,0]-55,uu_avg[:,0])
# plt.plot(vu_avg[:,0])
plt.xlabel('x (mm)')
plt.ylabel('v_lower (mm/s)')
# %%
plt.plot(yu[:,0]-55,ul_avg[:,0])
# plt.plot(vl_avg[:,0])
plt.xlabel('x (mm)')
plt.ylabel('v_upper (mm/s)')
# %%
uu = uu_avg[:,-1]
vu = vu_avg[:,-1]

yy = yu[:,-1]

np.sum(uu[:-1] * vu[:-1] * (yy[1:] - yy[:-1])) * 0.048 * 1e-9 * 1e3
# %%
ul = ul_avg[:,0]
vl = vl_avg[:,0]

yy = yl[:,0]

np.sum(ul[:-1] * vl[:-1] * (yy[1:] - yy[:-1])) * 0.048 * 1e-9 * 1e3
# %%
mdot_upper = np.sum(uu[:-1] * (yy[1:] - yy[:-1])) * 1e-6 * 0.048 * 1e3
mdot_lower = np.sum(ul[:-1] * (yy[1:] - yy[:-1])) * 1e-6 * 0.048 * 1e3
mdot = mdot_upper - mdot_lower
(mdot_upper,mdot_lower,mdot)
# %%
U = -(np.mean(vu) + np.mean(vl))/2 * 1e-3
# %%
mdot_lower * U
# %%
np.mean(uu)
# %%
np.mean(ul_avg[:,3])
# %%
