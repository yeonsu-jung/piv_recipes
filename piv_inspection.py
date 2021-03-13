# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time

from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)

folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

# folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-09"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)

piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 23,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,450,0],    
    "sn_threshold": 1.1,'dt': 0.0001,
    "rotate": 0.25, "save_result": True,"show_result": False,
    "check_angle": False
}
pi.set_piv_param(piv_cond)
# %%
sd = {'path': 'Flat_10 (black)_motor15.00_pos5_VOFFSET210_timing100_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)
# %%
pi.set_piv_param({"dt": 0.0001,"check_angle": True})
sd = {'path': 'Flat_10 (black)_motor15.00_pos5_VOFFSET210_timing100_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)
# %%
xyuv[3].shape
np.mean(xyuv[3][20:30,20:30])
# %%
pi.set_piv_param({"dt": 0.0005,"check_angle": True})
sd = {'path': 'Flat_10 (black)_motor15.00_pos4_VOFFSET210_timing500_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)
# %%
pi.set_piv_param({"dt": 0.0005,"check_angle": False})
sd = {'path': 'Flat_10 (black)_motor15.00_pos4_VOFFSET210_timing500_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)

# %%
pi.set_piv_param({"dt": 0.0001,"check_angle": False})
sd = {'path': 'Flat_10 (black)_motor15.00_pos1_VOFFSET210_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)

np.mean(xyuv[3])
# %%
pi.set_piv_param({"dt": 0.0001,"check_angle": False,"crop":[0,0,0,0],"show_result":True,'u_bound': [-100,100]})
sd = {'path': 'Flat_10 (black)_motor15.00_pos2_VOFFSET630_timing100_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)
# %%

piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 23,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,450,0],    
    "sn_threshold": 1.01,'dt': 0.0001,
    "rotate": 0.25, "save_result": True,"show_result": False,
    "check_angle": False
}
pi.set_piv_param(piv_cond)

pi.set_piv_param({"dt": 0.0001,"check_angle": False,"crop":[0,0,0,370],"show_result":True,'u_bound': [-100,100]})
sd = {'path': 'Flat_10 (black)_motor15.00_pos3_VOFFSET0_timing100_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)

# %%
pi.set_piv_param({"dt": 0.0001,"check_angle": False,"crop":[0,0,450,0],"show_result":True,'u_bound': [-100,100]})
sd = {'path': 'Flat_10 (black)_motor15.00_pos4_VOFFSET0_timing100_ag1_dg1_laser5_[03-11]'}
ind = pi.check_proper_index(sd,index_a = 10)
xyuv = pi.quick_piv(sd,index_a = ind, index_b = ind + 1)
# %%
