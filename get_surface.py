# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy import ndimage
import imageio as io
import sys
import json

from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)

folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'

# folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-15\\Flat_10 (black)_motor25"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]

pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()
# %%
depth = 1200
height = 150

# %%
crop_info = {
    1: (0.42, 422),
    2: (0.42, 422),
    3: (0.42, 412),
    4: (0.5, 410),
    5: (0.5, 410),
    6: (0.5, 395)
    }

whole_stack = np.zeros((depth*len(crop_info),height))
# %%
pos = 1
offset = 430
angle = 0.42

crop_info[1] = (angle,offset)
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%
pos = 2
offset = 434
angle = 0.42

crop_info[1] = (angle,offset)
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%
pos = 3
offset = 424
angle = 0.41
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%

pos = 4
offset = 425
angle = 0.5
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r--')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%
pos = 5
offset = 422
angle = 0.5
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r--')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%
pos = 6
offset = 410
angle = 0.5
crop_info[pos] = (angle,offset)

sd_list = []
for vo in [0,210,420,630,840]:    
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11,raw=True)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, angle)
stack_a = stack_a[:depth,offset: offset+ height]
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(stack_a)
ax.plot([30,30],[0,depth],'r--')
whole_stack[(pos-1) * depth:pos * depth,:] = stack_a
# %%
fig,ax = plt.subplots(figsize=(20,100))
ax.imshow(whole_stack)
# ax.plot([30,30],[0,depth*pos],'r')# %%

# %%
start_index = 1
stop_index = 201 # not included
# %%
parent_path, folder_name = os.path.split(folder_path)
try:
    path_for_cropped = os.path.join(parent_path, folder_name +'_cropped')
    os.makedirs(path_for_cropped)
except:
    y_n = input('Overwrite can occur, go ahead? [y/n]')    
    if y_n is y:
        pass
    else:
        sys.exit(1)
# %%
for ind in range(start_index,stop_index,2):       
    for k in range(0,30,5):
        path_list = []
        for pd in pi.piv_dict_list[k:k+5]:    
            angle = crop_info[pd['pos']][0]
            offset = crop_info[pd['pos']][1]

            img_a,img_b = pi.read_two_images(pd,index_a = ind,index_b = ind+1,raw=True)
            path_list.append(pd['path'])
            try:
                stack_a = np.vstack((stack_a,img_a))
                stack_b = np.vstack((stack_b,img_b))
            except:
                stack_a = img_a
                stack_b = img_b
            
        stack_a = ndimage.rotate(stack_a, angle)
        stack_a = stack_a[0:1200,offset:offset+height]

        stack_b = ndimage.rotate(stack_b, angle)
        stack_b = stack_b[0:1200,offset:offset+height]

        stack_a_reshaped = stack_a.reshape(num_offset,1200//num_offset,height)
        stack_b_reshaped = stack_b.reshape(num_offset,1200//num_offset,height)

        i = 0
        for im_a,im_b in zip(stack_a_reshaped,stack_b_reshaped):
            try:
                os.makedirs(os.path.join(path_for_cropped, path_list[i]))
            except:
                pass

            im_a_path = os.path.join(path_for_cropped, path_list[i],'frame_%06d.tiff'%ind)
            im_b_path = os.path.join(path_for_cropped, path_list[i],'frame_%06d.tiff'%(ind+1))

            io.imwrite(im_a_path,im_a)
            io.imwrite(im_b_path,im_b)
            i = i + 1       

with open(os.path.join(path_for_cropped,'_crop_info.txt'),'w') as f:
    for k, v in crop_info.items():
        f.write('%d: (%.2f,%d)\n' %(k,v[0],v[1]))

# %%

# For 03-15 motor 5
# crop_info = {
#     1: (0.42, 422),
#     2: (0.42, 422),
#     3: (0.42, 412),
#     4: (0.5, 410),
#     5: (0.5, 410),
#     6: (0.5, 395)
#     }