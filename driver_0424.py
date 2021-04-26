# %%
import piv_class as pi
from importlib import reload
from matplotlib import pyplot as plt
import numpy as np

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-08/Flat_10 (black)_motor10_stitching'
parent_path = parent_path.replace('C:/Users/yj/','/Users/yeonsu/')

ins = pi.piv_class(parent_path)
os.listdir(parent_path)
# %%
for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue
    if 'pos1_' in x:
        continue
    if 'pos2_' in x:
        continue
    if 'pos12_' in x:
        continue
    # print(x)
    ins.get_wall_position(path=x)


# %%
ins.get_wall_position()

# %%
_ = ins.quick_piv(index = 2)

# %%
ins.piv_over_sample(3,2)

# %%
parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-08/Flat_10 (black)_motor10_stitching'
