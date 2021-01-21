# %%
import re
import imageio as io
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

from matplotlib import pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling
from scipy.ndimage.morphology import binary_hit_or_miss 
import openpiv_recipes

import skimage
import skimage.feature
import skimage.viewer

import importlib

# %%
def param_string_to_dictionary(pstr):      
    running_parameter = re.findall("_[a-z]+[0-9]+[.]*[0-9]*", pstr, re.IGNORECASE)
    sample_parameter = pstr.replace("img_","")

    for k in running_parameter:
        sample_parameter = sample_parameter.replace(k,"")

    param_dict = {'sample': sample_parameter}
    for k in running_parameter:
        kk = re.findall('[a-x]+', k,re.IGNORECASE)
        vv = re.findall('[0-9]+[.]*[0-9]*', k,re.IGNORECASE)
        param_dict[kk[0]] = vv[0]

    return param_dict

def split_image(img):
    vertically_averaged = np.mean(img,axis=0)
    filtered = gaussian_filter1d(vertically_averaged,10)
    center_by_minimum = np.argmin(vertically_averaged)

    left_max = np.argmax(filtered[0:center_by_minimum]) - 20
    right_max = np.argmax(filtered[center_by_minimum:-1]) +center_by_minimum + 20
    
    img_left = img_a[:,0:left_max]
    img_right = img_a[:,right_max:-1]

    return img_left,img_right

def split_image_arg(img):
    vertically_averaged = np.mean(img,axis=0)
    filtered = gaussian_filter1d(vertically_averaged,10)
    center_by_minimum = np.argmin(vertically_averaged)

    left_max = np.argmax(filtered[0:center_by_minimum]) - 20
    right_max = np.argmax(filtered[center_by_minimum:-1]) +center_by_minimum + 20
    
    return left_max, right_max

def match_images(img_a,img_b):
    span_a = img_a.shape[1]
    span_b = img_b.shape[1]

    if span_a > span_b:
        return img_a[:,span_a-span_b:-1], img_b
    else:
        return img_a, img_b[:,span_b-span_a:-1]
# %%
folder_path = 'D:/Rowland/piv-data/2021-01-20'

param_string_list = os.listdir(folder_path)
param_string_to_dictionary(param_string_list[5])
# %%
importlib.reload(openpiv_recipes)
param_string_list = [param_string_list[5]]

for param_string in param_string_list:
    img_a_name = 'frame_000102.tiff'
    img_b_name = 'frame_000103.tiff'

    param_dict = param_string_to_dictionary(param_string)
    # print(param_dict)  
    try:
        file_path_a = os.path.join(folder_path,param_string,img_a_name)
        file_path_b = os.path.join(folder_path,param_string,img_b_name)        
        img_a = io.imread(file_path_a)
        img_b = io.imread(file_path_b)
    except:
        file_path_a = os.path.join(folder_path,'img_' + param_string,img_a_name)
        file_path_b = os.path.join(folder_path,'img_' + param_string,img_b_name)

        img_a = io.imread(file_path_a)
        img_b = io.imread(file_path_b)

    
    
    a_left, a_right = split_image_arg(img_a)
    b_left, b_right = split_image_arg(img_b)

    right_edge_index = max(a_right,b_right)
    
    img_a_right = img_a[:,right_edge_index:-1]
    img_b_right = img_b[:,right_edge_index:-1]
    
    openpiv_recipes.run_piv(img_a_right,img_b_right,
        winsize=48,
        searchsize=50,
        overlap=24,
        show_vertical_profiles=False,
        image_check=True,
        figure_export_name='results.png')

    
    

# %%
img_a_left,img_a_right = split_image(img_a)
img_b_left,img_b_right = split_image(img_b)

# %%
trimmed_a,trimmed_b = match_images(img_a_right,img_b_right)

plt.imshow(trimmed_b)

# %%
importlib.reload(openpiv_recipes)
openpiv_recipes.run_piv(trimmed_a,trimmed_b,
    winsize=48,
    searchsize=50,
    overlap=24,
    show_vertical_profiles=False,
    image_check=True,
    figure_export_name='results.png')
        
# %%
importlib.reload(openpiv_recipes)
openpiv_recipes.run_piv(img_a,img_b,
    winsize=48,
    searchsize=50,
    overlap=24,
    show_vertical_profiles=False,
    image_check=True,
    figure_export_name='results.png')
# %%
importlib.reload(openpiv_recipes)
openpiv_recipes.run_piv(img_a[:,700:-1],img_b[:,700:-1],
    winsize=48,
    searchsize=50,
    overlap=24,
    show_vertical_profiles=False,
    image_check=True,
    figure_export_name='results.png')
# %%
