# %%
import imageio as io
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import logging

import threading

import openpiv
import tunnel_control as tc
import camera_control as cc
import datetime
import time
import logging
import os
import sys
# %%
import openpiv_recipes as piv
from importlib import reload
reload(piv)

folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-18'
pi = piv.ParticleImage(folder_path)

exp_cond_dict = {'sample': 'Flat_10', 'motor': 5}
pi.set_piv_list(exp_cond_dict)
for x in pi.piv_dict_list:
    print(x)

piv_param = {
    "winsize": 48,
    "searchsize": 52,
    "overlap": 24,
    "dt": 0.0001,
    "image_check": False,    
    "show_vertical_profiles": False,            
    "figure_export_name": '_quick_piv.tiff',
    "text_export_name": '_quick_piv.txt',
    "scale_factor": 1,            
    "pixel_density": 36.74,
    "arrow_width": 0.02,
    "show_result": True,        
    }

larger = {"winsize": 48, "searchsize": 52, "overlap": 24, "scale_factor": 1}
smaller = {"winsize": 16, "searchsize": 20, "overlap": 8, "scale_factor": 1e3}

pi.set_piv_param(piv_param)
# pi.set_piv_param(larger)
pi.set_piv_param(smaller)

search_dict = {'pos': 2, 'VOFFSET': 480}
pi.quick_piv_by_key(search_dict,index_a=101,index_b=102)

# %%
import numpy as np
from matplotlib import pyplot as plt

from matplotlib import interactive
import imageio as io

interactive(True)

%matplotlib qt

bgd, bgd2 = pi.read_two_images(search_dict,index_a=101,index_b=102)
bgd = -np.array(bgd).T
piv_result = np.loadtxt('_quick_piv.txt')

elongate = 1

window_size = 16
scaling_factor = 1
arrow_width = 0.002

xx = piv_result[:,0].T * bgd.shape[1] / np.max(piv_result[:,0])
yy = piv_result[:,1].T * bgd.shape[0] / np.max(piv_result[:,1])
uu = piv_result[:,2].T
vv = piv_result[:,3].T*0

fig, ax = plt.subplots(figsize=(10,10))

xmax = np.amax(piv_result[:, 0]) + window_size / (2 * scaling_factor)
ymax = np.amax(piv_result[:, 1]) + window_size / (2 * scaling_factor)

ax.quiver(xx,yy,uu,vv,angles='xy',width=arrow_width,color='r')
ax.imshow(bgd,origin="lower", cmap="Greys_r",interpolation='nearest',aspect='auto')
# ax.invert_yaxis()


plt.show()

# %%
ax.get_

# %%
ax.imshow(bgd,)

plt.show()
# ax.imshow(im, origin="lower", cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

# %%
# locals().update(piv_param)
from argparse import Namespace

def a_fun(dict):
    ns = Namespace(**dict)
    return ns.winsize

a_fun(piv_param)

# %%
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.close()
        pass    

sys.stdout = Logger()


# %%
logger = logging.getLogger('test')


hdlr = logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
# %%
try:
    file_a_path = 'abc.txt'           
    with open(file_a_path) as f:
        f.write('1')
except FileNotFoundError:
    print('abc')
except Exception as e:
        logger.exception('Failed: ' + str(e))
# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
# %%
logging.
# %%
