# %%
import os
import numpy as np
from matplotlib import pyplot as plt

import piv_class as pi
from importlib import reload
from matplotlib import pyplot as plt

reload(pi)


# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
def foo5(path):

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    os.chdir(owd)

    return x,y,u,v
# %%
d = foo5('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')

# %%
