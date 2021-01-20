# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import cv2
import time
from scipy import ndimage
import math
import openpiv_recipes

# image frames to piv tables (x,y,u,v,sig2noise,mask)

vidcap = cv2.VideoCapture('022.mp4')
# %%
success,img_a = vidcap.read(301)
success,img_b = vidcap.read(302)
img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
openpiv_recipes.run_piv(img_a,img_b,winsize=32,searchsize=35,overlap=16,
                        figure_export_name='from_video.png')

# %%
path = '022/'
img_a = tools.imread(path + 'frame_000301.tiff')
img_b = tools.imread(path + 'frame_000302.tiff')        
openpiv_recipes.run_piv(img_a,img_b,winsize=32,searchsize=35,overlap=16,
                        figure_export_name='from_tiff.png')



# %%
