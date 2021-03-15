# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy import ndimage
import imageio as io

from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)

folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

# folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-09"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]

pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()
# %%

# %%
height = 120
num_offset = 5

for ind in range(1,101,2):       
    for k in range(0,30,5):
        path_list = []
        for pd in pi.piv_dict_list[k:k+5]:    
            angle = pi.crop_info[pd['pos']][0]
            offset = pi.crop_info[pd['pos']][1]

            img_a,img_b = pi.read_two_images(pd,index_a = ind,index_b = ind+1)
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
            im_a_path = os.path.join(pi.path, path_list[i],'cropped_%06d.tiff'%ind)
            im_b_path = os.path.join(pi.path, path_list[i],'cropped_%06d.tiff'%(ind+1))

            io.imwrite(im_a_path,im_a)
            io.imwrite(im_b_path,im_b)
            i = i + 1       

# %%
for k in range(0,30,5):
    print(k,k+5)
# %%
for ii in range(0,100,2):
    print(ii)
# %%
pos = 2
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a        

stack_a = ndimage.rotate(stack_a, 0.5)
stack_a = stack_a[0:1200,460:460+120]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
stack_reshaped = stack_a.reshape(len(sd_list),1200//len(sd_list),height)




# %%
sd = {'pos': 3, 'VOFFSET': 210}
img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)
# %%
pos = 2
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.5)
stack_a = stack_a[:,460:]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
pos = 3
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.5)
stack_a = stack_a[:,450:]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
pos = 4
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.58)
stack_a = stack_a[:,453:]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
pos = 5
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.45)
stack_a = stack_a[:,446:]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
pos = 6
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.45)
stack_a = stack_a[:,435:]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
def crop_images(img,angle,offset):
    img = ndimage.rotate(img,angle)
    img = img[:,int(offset):]
    return img
# %%


im = crop_images(img_a,0.45,446)
plt.imshow(im)
# %%
pos = 5
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.45)
stack_a = stack_a[:,446:446+100]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)
# %%
print(stack_a.shape)
# %%
sliced = stack_a[0:1200,:].reshape((5,1200//5,100))
# %%
fig,ax = plt.subplots(5,figsize=(5,20))
for i, s in enumerate(sliced):
    io.imwrite('')
# %%
s.shape
# %%
sd = {'pos': 3, 'VOFFSET': 0}
img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
# %%
pos = 5
sd_list = []
for vo in [0,210,420,630,840]:
    sd_list.append({'pos': pos,'VOFFSET': vo})

stack_a = []
for sd in sd_list:
    img_a,img_b = pi.read_two_images(sd,index_a = 10,index_b = 11)        
    try:
        stack_a = np.vstack((stack_a,img_a))
    except:
        stack_a = img_a

stack_a = ndimage.rotate(stack_a, 0.45)
stack_a = stack_a[:,446:446+100]
fig, ax = plt.subplots(figsize=(5,20))
ax.imshow(stack_a)