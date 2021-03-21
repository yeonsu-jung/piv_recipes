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
def get_boundary_velocity(pi, start_index = 3, N = 2):    
    # left
    sd1 = {'pos': 1, 'VOFFSET': 0}
    pi.piv_over_time2(sd1,start_index= start_index,N= N)

    # right
    sd2 = {'pos': 6, 'VOFFSET': 840}
    pi.piv_over_time2(sd2,start_index= start_index ,N= N)
        
    piv_cond = {
        "winsize": 28, "searchsize": 34, "overlap": 22,
        "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
        "u_bound": [-50,50],"v_bound": [-300,0],
        "transpose": False, "crop": [0,0,580,0],
        "sn_threshold": 1.0001,'dt': 0.0001,
        "rotate": 0, "save_result": True,"show_result": False,
        "check_angle": False
    }
    pi.set_piv_param(piv_cond)
    for pd in pi.piv_dict_list:
        pi.piv_over_time3(pd,start_index= start_index,N=N,tag='upper')

    piv_cond = {
        "winsize": 28, "searchsize": 34, "overlap": 22,
        "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
        "u_bound": [-50,50],"v_bound": [-300,0],
        "transpose": False, "crop": [0,0,0,580],
        "sn_threshold": 1.0001,'dt': 0.0001,
        "rotate": 0, "save_result": True,"show_result": False,
        "check_angle": False
    }

    pi.set_piv_param(piv_cond)    
    for pd in pi.piv_dict_list:
        pi.piv_over_time3(pd,start_index= start_index,N=N,tag='lower')
# %%
try:
    folder_path = "C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/Flat_10 (black)_motor25"
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-100,100],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,0,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": False, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]
pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()
# %% check piv setting
# pi.set_piv_param({'show_result': True, 'crop': [0,0,400,0]})
# pi.quick_piv({'pos':3,'VOFFSET':420})
# pi.set_piv_param({'show_result': False})
# %%
get_boundary_velocity(pi, start_index=3,N=90)
# %%
