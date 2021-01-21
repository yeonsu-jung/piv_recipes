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


# %%
folder_path = 'D:/Rowland/piv-data/2021-01-20'

param_string_list = os.listdir(folder_path)
param_string_to_dictionary(param_string_list[5])
# %%

param_string_list = [param_string_list[0]]

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


    plt.imshow(img_a)
    plt.imshow(img_b)
# %%

# %%
openpiv_recipes.run_piv(img_a,img_b,
    winsize=40,
    searchsize=42,
    overlap=20,
    show_vertical_profiles=False,
    image_check=True,
    figure_export_name='results.png')
        
# %%
sigma = 2
low_threshold = 0.1
high_threshold = 0.2

edges = skimage.feature.canny(
    image=img_a,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

viewer = skimage.viewer.ImageViewer(edges)
viewer.show()


# %%
vertically_averaged = np.mean(img_a,axis=0)
horizontally_averaged = np.mean(img_a,axis=1)

# %%


plt.plot(vertically_averaged[350:500],'.-')
plt.plot(vertically_averaged)

center_by_minimum = np.argmin(vertically_averaged)
span = 200



right_max = np.argmax(vertically_averaged[center_by_minimum:center_by_minimum+span]- vertically_averaged[center_by_minimum+1:center_by_minimum+span+1]) + center_by_minimum + 10
left_max = np.argmax(-vertically_averaged[0:center_by_minimum-2] + vertically_averaged[2:center_by_minimum]) + 10


# %%
img_a_right = img_a[:,right_max:-1]

img_a_left = img_a[:,0:350+140]
img_a_left = img_a[:,0:left_max]
plt.imshow(img_a_left)


# %%
plt.imshow(img_a_right)
# %%
img_filtered = gaussian_filter1d(vertically_averaged,10)
plt.plot(img_filtered,'o')

right_max = np.argmax(vertically_averaged[center_by_minimum:-1]) +center_by_minimum + 20
left_max = np.argmax(vertically_averaged[0:center_by_minimum]) - 20

img_a_right = img_a[:,right_max:-1]
img_a_left = img_a[:,0:left_max]
plt.imshow(img_a_left)
# %%
plt.imshow(img_a_right)
# %%
