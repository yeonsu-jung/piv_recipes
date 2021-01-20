# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpiv_recipes
import imageio
import cv2
import time


# from scipy import ndimage
# import math
# import matplotlib.cm as cm

# import os


# %%
path = 'test/12-09_004_'
img_a = imageio.imread(path + 'frame_000055.tiff')
img_b = imageio.imread(path + 'frame_000056.tiff')

openpiv_recipes.run_piv(img_a,img_b,
    show_vertical_profiles=True,
    image_check=True)
# %%

# # %%
# path = '023/'
# img_a = imageio.imread(path + 'frame_000301.tiff')
# img_b = imageio.imread(path + 'frame_000302.tiff')        

# openpiv_recipes.show_vertical_profiles(img_a,img_b,animation_flag=False,scale_factor = 1)
# # %%
# path = 'test/12-09_004_'
# img_a = imageio.imread(path + 'frame_000055.tiff')
# img_b = imageio.imread(path + 'frame_000056.tiff')        

# openpiv_recipes.show_vertical_profiles(img_a,img_b,animation_flag=False)

