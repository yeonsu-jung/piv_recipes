# %%
import sys
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import yaml
import matplotlib.patches as patches

from openpiv import tools, process, validation, filters, scaling, pyprocess
# %%
sys.path.append(os.path.dirname('../'))

import openpiv_recipes as piv
importlib.reload(piv)

# %%
def read_folder(parent_folder):
    try:
        folder_path = os.path.join('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/',parent_folder)
        results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
        pi = piv.ParticleImage(folder_path,results_folder_path)
    except:
        folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
        results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
        pi = piv.ParticleImage(folder_path,results_folder_path)    
    return pi

def get_windows(img,winsize,overlap):    
    m = img.shape[0]
    n = img.shape[1]
    m2 = (m-overlap)//(winsize-overlap)
    n2 = (n-overlap)//(winsize-overlap)
    
    windows = []
    for i in range(m2):
        win2 = []
        left = i * (winsize-overlap)
        for j in range(n2):            
            top = j * (winsize - overlap)
            win2.append(img[left:left+winsize,top:top+winsize])
        windows.append(win2)    

    return windows

def see_windows(i,j,im1,win1,win2,_winsize,_overlap):       
    
    left = i * (_winsize - _overlap)
    top = j * (_winsize - _overlap)

    fig, ax = plt.subplots()
    ax.imshow(im1)
    rect = patches.Rectangle((top,left), _winsize, _winsize, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    fig,ax = plt.subplots(3,3,figsize=(20,15))
    ax[0,0].imshow(win1[i][j])
    ax[0,1].imshow(win1[i][j+1])
    ax[0,2].imshow(win1[i][j+2])

    ax[1,0].imshow(win2[i][j])
    ax[1,1].imshow(win2[i][j+1])
    ax[1,2].imshow(win2[i][j+2])

    ax[2,0].imshow(win2[i][j] - win1[i][j])
    ax[2,1].imshow(win2[i][j+1] - win1[i][j+1])
    ax[2,2].imshow(win2[i][j+2] - win1[i][j+2])

# %%
pi = read_folder('2021-04-05')
with open('piv_setting.yml') as f:
    piv_cond = yaml.safe_load(f)
pi.set_piv_param(piv_cond)

# %%
path_to_study = '2_1_1_10 (black)_motor10'
d = pi.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,67,img_a,windows1,windows2,28,22)

# %%
path_to_study = '2_1_1_10 (black)_motor25'
d = pi.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,67,img_a,windows1,windows2,28,22)
# %%
path_to_study = '1_1_1_10 (clear 3dp)_motor10'
d = pi.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,67,img_a,windows1,windows2,28,22)

# %%
plt.plot(d[3][:,1],'o-')
# %%
path_to_study = '1_1_1_10 (clear 3dp)_motor25'
d = pi.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,67,img_a,windows1,windows2,28,22)
# %%
path_to_study = 'Flat_10 (acrylic, polished)_motor10'
d = pi.quick_piv({'path': path_to_study},index_a=21,index_b=22)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,60,img_a,windows1,windows2,28,22)
# %%
path_to_study = 'Flat_10 (acrylic, polished)_motor25'
d = pi.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi.read_two_images({'path': path_to_study},index_a=21,index_b=22)

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
see_windows(15,60,img_a,windows1,windows2,28,22)
# %%
pi2 = read_folder('2021-04-06')
with open('piv_setting.yml') as f:
    piv_cond = yaml.safe_load(f)
pi2.set_piv_param(piv_cond)
# %%
path_to_study = 'Flat_10 (black)_motor10_particle1'
d = pi2.quick_piv({'path': path_to_study},index_a=20,index_b=21)
# %%
from argparse import Namespace

img_a,img_b = pi2.read_two_images({'path': path_to_study},index_a=20,index_b=21)

ns = Namespace(**piv_cond)

img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
# %%
i = 25
j = 1

see_windows(i,j,img_a,windows1,windows2,28,22)
print(d[3][i,j])
# %%
plt.plot(d[3][:,j],'o-')
# %%
i = 25
j = 15

see_windows(i,j,img_a,windows1,windows2,28,22)
print(d[3][i,j])
# %%
plt.plot(d[3][:,j],'o-')
# %%
path_to_study = 'Flat_10 (black)_motor10_particle2'
d = pi2.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi2.read_two_images({'path': path_to_study},index_a=20,index_b=21)


windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
# %%

# %%
path_to_study = 'Flat_10 (black)_motor10_particle3'
d = pi2.quick_piv({'path': path_to_study},index_a=20,index_b=21)
img_a,img_b = pi2.read_two_images({'path': path_to_study},index_a=20,index_b=21)

img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]

windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)
# %%
i = 25
j = 1

see_windows(i,j,img_a,windows1,windows2,28,22)
print(d[3][i,j])
# %%
path_to_study = 'Flat_10 (black)_motor10_particle4'
d = pi2.quick_piv({'path': path_to_study},index_a=21,index_b=22)
# %%
with open('piv_setting.yml') as f:
    piv_cond = yaml.safe_load(f)
pi2.set_piv_param(piv_cond)

path_to_study = 'Flat_10 (black)_motor10_particle4_hori1920'
d = pi2.quick_piv({'path': path_to_study},index_a=101,index_b=102)
# %%
