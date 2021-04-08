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
import matplotlib.patches as patches
import cv2
# %%
sys.path.append(os.path.dirname('../'))

import openpiv_recipes as piv
importlib.reload(piv)

# %%
def plot_velocity_profile(x, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
    v_array = -v_avg[start_i:stop_i:step_i,:]
    v_std_array = v_std[start_i:stop_i:step_i,:]
    y_array = y[start_i:stop_i:step_i,0]
    N = len(v_array)
    s = []
    for yy in (y_array - 55): # -55 for rearrange x axis with LE being zero.
        s.append('%.2f'%yy)

    C = 0.005
    
    for i,vv in enumerate(v_array):
        # ax.plot(vv + C*y_array[i],x[0,:],'k--')
        ax.plot(C*vv + y_array[i],x[0,:],'--',**kwargs)
        ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
        ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
        for j, vvv in enumerate(vv):
            ax.arrow(y_array[i], x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

    ax.set_xticks(y_array)
    ax.set_xticklabels(s)

    return ax

def plot_dimensionless_velocity_profile(x, y, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
    v_array = -v_avg[start_i:stop_i:step_i,:]
    v_std_array = v_std[start_i:stop_i:step_i,:]
    y_array = y[start_i:stop_i:step_i,0]
    N = len(v_array)
    s = []
    
    C = 4
    
    rho = 1e3
    mu = 1e-3

    U = np.mean(-v_avg[400:800,-1])*1e-3       

    for yy in (y_array): # -55 for rearrange x axis with LE being zero.
        Re = rho*U*(yy-55)/mu*1e-3
        s.append('%d'%Re)

    for i,vv in enumerate(v_array):
        # ax.plot(vv + C*y_array[i],x[0,:],'k--')
        # U = np.max(vv)*1e-3
        # U = 0.5       

        Re = rho*U*(y_array[i]-55)/mu*1e-3       
        # x_c = (U/(mu/rho)/((y_array[i]-55)*1e-3))**0.5*1e-3
        x_c = 1

        ax.plot(C*vv + Re,x[0,:]*x_c,'--',**kwargs)
        ax.plot([Re,Re],[0,np.max(x)*x_c],'b-')
        ax.plot([Re+C*U*1e3,Re+C*U*1e3],[0,np.max(x)*x_c],'b--')
        for j, vvv in enumerate(vv):
            ax.arrow(Re, x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
    
    xx = np.linspace(0,0.12,100)
    Re_x = np.linspace(0,rho*U*0.12/mu,100)
    delta = 5*((mu/rho)*xx/U)**0.5*1e3
    ax.plot(Re_x,delta,'g-')

    ax.set_xticks(rho*U*(y_array-55)/mu*1e-3)
    ax.set_xticklabels(s)
    ax.set_xlabel('Re_x')
    ax.set_ylabel('y (mm)')
    ax.axis([0,np.max(Re_x),0,2.5])

    return ax

def plot_single_profile(i,x,y,v_array,v_std_array,ax,x_cut = 0,**kw):
    C = 1        
    # y_array[i]

    rho = 1e3
    mu = 1e-3
    U = np.mean(-v_avg[400:800,-1])*1e-3       

    vv = v_array[i,:]
    ax.plot(C*vv[x_cut:],y[x_cut:],'--',**kw)
    ax.plot([0,0],[0,np.max(y)],'b-')
    ax.plot([0+C*U*1e3,C*U*1e3],[0,np.max(y)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(0, y[j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
        ax.arrow(0 + C*(vvv), y[j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')
        ax.arrow(0 + C*(vvv), y[j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')

    ax.set_title('x = %.2f mm; Re_x = %d'%(x[i]-55,rho*U*(x[i]-55)/mu*1e-3))
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    ax.axis([0,U*1.1*1e3,0,2.2])
    # plt.savefig('individual_profile.png')
    # fig.savefig('individual_profile.png',dpi=900)
    return ax

def velocity_array(x,y,ul,vl,ur,vr):
    y_lr = x[0,:]
    u_left = -vl[:,3,:]
    v_left = ul[:,3,:]

    u_right = -vr[:,-1,:]
    v_right = ur[:,-1,:]

    x_top = y2[:,0]
    u_top = -v2[:,:,0]
    v_top = -u2[:,:,0]

    x_bottom = y1[:,-3]
    u_bottom = -v1[:,:,-3]
    v_bottom = -u1[:,:,-3]
    return y_lr,u_left,v_left,u_right,v_right,x_top,u_top,v_top,x_bottom,u_bottom,v_bottom

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
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-07/'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path,version='2')
# %%
piv_cond = {
    "winsize": 32, "searchsize": 45, "overlap": 26,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-25,25],"v_bound": [-1500,100],
    "transpose": False, "crop": [0,0,750,0],    
    "sn_threshold": 1.000001,'dt': 0.0005,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)


# %%
path_to_study = '2_1_1_10 (black)_timing250_ag1_dg1_laser1_shutter30_motor10_pos8_[04-07]'
d = pi.quick_piv({'path': path_to_study},index_a=22,index_b=23)


# %%
img_a,img_b = pi.read_two_images({'path': '2_1_1_10 (black)_timing250_ag1_dg1_laser1_shutter30_motor10_pos8_[04-07]'},index_a=22,index_b=23)

img_a = img_a[piv_cond["crop"][0]:-piv_cond["crop"][1]-1,piv_cond["crop"][2]:-piv_cond["crop"][3]-1]
img_b = img_b[piv_cond["crop"][0]:-piv_cond["crop"][1]-1,piv_cond["crop"][2]:-piv_cond["crop"][3]-1]

# %%
windows1 = get_windows(np.array(img_a),32,28)
windows2 = get_windows(np.array(img_b),32,28)
see_windows(15,1,img_a,windows1,windows2,32,28)


# %%
def get_edge(img, th = 110):
    bw = img > th
    n, labels = cv2.connectedComponents(bw.astype(np.uint8))
    edge = labels==1

    return edge

def get_edge_removed_img(img,edge):
    img_wo_edge = img * (np.invert(edge)).astype(np.uint8)
    return img_wo_edge

def get_rightmost(row,col):
    rightmost = np.zeros(np.max(row))
    idx = 0
    for i in range(np.max(row)):
        idx = idx + len([x for x in row if x == i])-1   
        rightmost[i] = col[idx]
    return rightmost

def get_img_wo_sample(img,rightmost):
    img_wo_sample = img
    for i in range(len(rightmost)):        
        # print(i,int(rightmost[i]))
        img_wo_sample[i,:int(rightmost[i])] = 0
    return img_wo_sample
    
    
# %%
img_a_new = get_edge_removed_img(img_a,get_edge(img_a,th = 150))
img_b_new = get_edge_removed_img(img_b,get_edge(img_b,th = 150))

row, col = np.where(img_a_new == 0)
rightmost_a = get_rightmost(row,col)
row, col = np.where(img_b_new == 0)
rightmost_b = get_rightmost(row,col)

iws_a = get_img_wo_sample(img_a,rightmost_a)
iws_b = get_img_wo_sample(img_b,rightmost_a)

fig,ax = plt.subplots(2)
ax[0].imshow(iws_a)
ax[1].imshow(iws_b)
# %%
fig,ax = plt.subplots(2)
ax[0].imshow(img_a)
ax[1].imshow(img_b)
# %%
import imageio as io

io.imwrite('iws_a.tiff',iws_a)
io.imwrite('iws_b.tiff',iws_b)

# %%
path_to_study = 'vid_2021-04-07_11-39-27'
d = pi.quick_piv({'path': path_to_study},index_a=503,index_b=504)
# %%
