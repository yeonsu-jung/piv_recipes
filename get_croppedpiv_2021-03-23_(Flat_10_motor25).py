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
folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/Flat_10 (black)_motor25_cropped'
results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'

# folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
# results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')

pi = piv.ParticleImage(folder_path,results_folder_path)
# %% check point
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,10,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
d = pi.quick_piv({'pos': 3,'VOFFSET': 630})
# %%
pi.set_piv_param({"show_result": False})
for sd in pi.piv_dict_list:
    pi.piv_over_time(sd,start_index=3,N=90)   
# %%
