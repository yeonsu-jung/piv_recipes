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

# %%
path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-07/Flat_10 (black)_stitching process'
os.listdir(path)

parent_path, folder_name = os.path.split(path)

# %%
try:
    path_for_cropped = os.path.join(parent_path, folder_name +'_cropped')
    os.makedirs(path_for_cropped)
except:
    y_n = input('Overwrite in %s can occur, go ahead? [y/n]'%path_for_cropped)    
    if y_n is 'y':
        pass
    else:
        sys.exit(1)


# %%
import path_class as pc
import piv_class as piv
ins = pc.path_class(path)
pos_list, voffset_list, path_list = ins.get_stitching_lists()

# %%
pos_list
voffset_list
path_list
# %%

pv_list = []
i = 0
for pos in pos_list:
    for voffset in voffset_list:
        pv_list.append((pos,voffset,path_list[i]))
        i = i + 1

# %%
piv_ins = piv.piv_class(path)
img_a = piv_ins.read_image_from_path(pv_list[15][2],index = 10)
# %%
img_a.shape
plt.imshow(img_a)

# %%
depth = 960
height = 150
whole_stack = np.zeros((depth*len(pos_list),height))
whole_stack.shape
# %%
for pos in pos_list:
# for pos in range(12,13):    
    crop_info = {
        1: (0.42, 812),
        2: (0.42, 812),
        3: (0.42, 781),
        4: (0.42, 770),
        5: (0.42, 762),
        6: (0.42, 770),
        7: (0.42, 774),
        8: (0.42, 775),
        9: (0.42, 776),
        10: (0.42, 774),
        11: (0.42, 766),
        12: (0.42, 766)
        }
    # pos = 11
    
    angle,offset = crop_info[pos]
    paths_for_fixed_pos = [x[2] for x in pv_list if x[0] == pos]

    stack_a = []
    for pth in paths_for_fixed_pos:
        img_a = piv_ins.read_image_from_path(pth,index = 10)
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
    plt.show()    

# %%
fig,ax = plt.subplots(figsize=(20,100))
ax.imshow(whole_stack)
for pos in pos_list:
    ax.text(50,300+(pos-1)*depth,'%d'%pos,fontsize=100)
# ax.plot([30,30],[0,depth*pos],'r')# %%
crop_info
fig.savefig(os.path.join(path_for_cropped,'_cropped.png'))

# %%
start_index = 1
stop_index = 202
# stop_index = 202 # not included
num_offset = len(voffset_list)

# %%
for ind in range(start_index,stop_index):
    # ith_path = os.path.join(path_for_cropped, paths_for_fixed_pos[0],'frame_%06d.tiff'%ind)
    # assert not os.path.isfile(first_path), "Must not overwrite files in %s"%ith_path

    for pos in pos_list:        
        paths_for_fixed_pos = [x[2] for x in pv_list if x[0] == pos]
        for pth in paths_for_fixed_pos:
            img_a = piv_ins.read_image_from_path(pth,index = ind)
            try:
                stack_a = np.vstack((stack_a,img_a))
            except:
                stack_a = img_a

        angle,offset = crop_info[pos]

        stack_a = ndimage.rotate(stack_a, angle)
        stack_a = stack_a[0:depth,offset:offset+height]    
        stack_a_reshaped = stack_a.reshape(num_offset,depth//num_offset,height)

        i = 0
        for im_a in stack_a_reshaped:
            try:
                os.makedirs(os.path.join(path_for_cropped, paths_for_fixed_pos[i]))
            except:                
                pass

            im_a_path = os.path.join(path_for_cropped, paths_for_fixed_pos[i],'frame_%06d.tiff'%ind)                        
            io.imwrite(im_a_path,im_a)
            i = i + 1
    
with open(os.path.join(path_for_cropped,'_crop_info.txt'),'w') as f:
    for k, v in crop_info.items():
        f.write('%d: (%.2f,%d)\n' %(k,v[0],v[1]))
