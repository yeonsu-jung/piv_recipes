# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpiv_recipes
import imageio
import cv2
import time

# %%
path = '../../piv(raw)/2020-12-11/12-11_012/'
# path = '../../piv(raw)/2020-12-11/Flat/12-11_007/'

img_a = imageio.imread(path + 'frame_000353.tiff')
img_b = imageio.imread(path + 'frame_000354.tiff')

openpiv_recipes.run_piv(img_a,img_b,
    winsize=48,
    searchsize=50,
    overlap=24,
    show_vertical_profiles=True,
    image_check=True,
    figure_export_name='results.png')
# %%
