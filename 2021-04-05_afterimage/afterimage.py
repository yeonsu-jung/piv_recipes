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

from openpiv import tools, process, validation, filters, scaling, pyprocess
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

def make_PI_instance(s):
    try:
        folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/' + s
        results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
        pi = piv.ParticleImage(folder_path,results_folder_path)
    except:
        folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
        results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
        pi = piv.ParticleImage(folder_path,results_folder_path)    
    return pi

# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-02/Clear 3dp'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)    
# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,380,0],    
    "sn_threshold": 1.000001,'dt': 0.00025,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
d = pi.quick_piv({'path': 'Flat_10 (clear3dp)_timing250_ag1_dg1_laser1_motor25.00_pos2_[04-02]_RE_VOFFSET420'},index_a=20,index_b=21)
# %%
img_a,img_b = pi.read_two_images({'path': 'Flat_10 (clear3dp)_timing250_ag1_dg1_laser1_motor25.00_pos2_[04-02]_RE_VOFFSET420'},index_a=20,index_b=21)

# %%
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
# %%
windows = get_windows(np.array(img_a),28,22)

np.array(windows).shape
# %%
import matplotlib.patches as patches

img = img_a
winsize = 28
overlap = 22

m = img.shape[0]
n = img.shape[1]
m2 = (m-overlap)//(winsize-overlap)
n2 = (n-overlap)//(winsize-overlap)

m2 = (m-overlap)//(winsize-overlap)
n2 = (n-overlap)//(winsize-overlap)

print(m, n, m2, n2)

i = 15
j = 67

left = i * (winsize-overlap)
top = j * (winsize - overlap)

fig, ax = plt.subplots()
ax.imshow(img_a)
rect = patches.Rectangle((top,left), winsize, winsize, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[i][j])
ax[0,1].imshow(windows[i][j+1])
ax[0,2].imshow(windows[i][j+2])
ax[1,0].imshow(windows[i+1][j])
ax[1,1].imshow(windows[i+1][j+1])
ax[1,2].imshow(windows[i+1][j+2])
ax[2,0].imshow(windows[i+2][j])
ax[2,1].imshow(windows[i+2][j+1])
ax[2,2].imshow(windows[i+2][j+2])

# %%
i = 15
j = 85

left = i * (winsize-overlap)
top = j * (winsize - overlap)

fig, ax = plt.subplots()
ax.imshow(img_a)
rect = patches.Rectangle((top,left), winsize, winsize, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[i][j])
ax[0,1].imshow(windows[i][j+1])
ax[0,2].imshow(windows[i][j+2])
ax[1,0].imshow(windows[i+1][j])
ax[1,1].imshow(windows[i+1][j+1])
ax[1,2].imshow(windows[i+1][j+2])
ax[2,0].imshow(windows[i+2][j])
ax[2,1].imshow(windows[i+2][j+1])
ax[2,2].imshow(windows[i+2][j+2])

# %%

################################

try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-01/Clear acrylic'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)    
# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,380,0],    
    "sn_threshold": 1.000001,'dt': 0.00025,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
d = pi.quick_piv(pi.piv_dict_list[10],index_a=20,index_b=21)
# %%
img_a,img_b = pi.read_two_images(pi.piv_dict_list[10],index_a=20,index_b=21)

# %%
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
windows1 = get_windows(np.array(img_a),28,22)
windows2 = get_windows(np.array(img_b),28,22)

# %%
import matplotlib.patches as patches

img = img_a
winsize = 28
overlap = 22

m = img.shape[0]
n = img.shape[1]
m2 = (m-overlap)//(winsize-overlap)
n2 = (n-overlap)//(winsize-overlap)

m2 = (m-overlap)//(winsize-overlap)
n2 = (n-overlap)//(winsize-overlap)

print(m, n, m2, n2)

i = 15
j = 70

left = i * (winsize-overlap)
top = j * (winsize - overlap)

fig, ax = plt.subplots()
ax.imshow(img_a)
rect = patches.Rectangle((top,left), winsize, winsize, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[i][j])
ax[0,1].imshow(windows[i][j+1])
ax[0,2].imshow(windows[i][j+2])
ax[1,0].imshow(windows[i+1][j])
ax[1,1].imshow(windows[i+1][j+1])
ax[1,2].imshow(windows[i+1][j+2])
ax[2,0].imshow(windows[i+2][j])
ax[2,1].imshow(windows[i+2][j+1])
ax[2,2].imshow(windows[i+2][j+2])

# %%

    
see_windows(15,80,img_a,windows1,windows2,28,22)
# %%
see_windows(15,75,img_a,windows1,windows2,28,22)

# %%
i = 15
j = 85

left = i * (winsize-overlap)
top = j * (winsize - overlap)

fig, ax = plt.subplots()
ax.imshow(img_a)
rect = patches.Rectangle((top,left), winsize, winsize, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[i][j])
ax[0,1].imshow(windows[i][j+1])
ax[0,2].imshow(windows[i][j+2])
ax[1,0].imshow(windows[i+1][j])
ax[1,1].imshow(windows[i+1][j+1])
ax[1,2].imshow(windows[i+1][j+2])
ax[2,0].imshow(windows[i+2][j])
ax[2,1].imshow(windows[i+2][j+1])
ax[2,2].imshow(windows[i+2][j+2])

################################


# %%
fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[15][32])
ax[0,1].imshow(windows[15][31])
ax[0,2].imshow(windows[15][30])
ax[1,0].imshow(windows[16][32])
ax[1,1].imshow(windows[16][31])
ax[1,2].imshow(windows[16][30])
ax[2,0].imshow(windows[17][32])
ax[2,1].imshow(windows[17][31])
ax[2,2].imshow(windows[17][30])

# %%
windows = get_windows(np.array(img_a),100,0)
plt.imshow(windows[0][3])
# %%

# %%
#############################

try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-05'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)    
# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,380,0],    
    "sn_threshold": 1.000001,'dt': 0.00025,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
d = pi.quick_piv({'path': '2_1_1_10 (black)_motor10'},index_a=20,index_b=21)
# %%
piv_cond = {
    "winsize": 32, "searchsize": 38, "overlap": 28,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,380,0],    
    "sn_threshold": 1.000001,'dt': 0.00025,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
d = pi.quick_piv({'path': '2_1_1_10 (black)_motor10'},index_a=20,index_b=21)
# %%
img_a,img_b = pi.read_two_images({'path': '2_1_1_10 (black)_motor10'},index_a=20,index_b=21)

windows1 = get_windows(np.array(img_a),32,28)
windows2 = get_windows(np.array(img_b),32,28)
# %%
see_windows(15,100,img_a,windows1,windows2,32,28)



#############################

# %%
process.get_field_shape(img_a, 28, 22)
# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-01/'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)    
# %%
piv_cond = {
    "winsize": 28, "searchsize": 40, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,0,390],    
    "sn_threshold": 1.000001,'dt': 0.00025,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
d = pi.quick_piv({'path': 'nofm_test3'},index_a=100,index_b=101)
# %%
img_a,img_b = pi.read_two_images({'path': 'nofm_test3'},index_a=20,index_b=21)
# %%
d[0].shape
np.array(img_a).shape
# %%
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
# %%
windows = get_windows(np.array(img_a),28,22)
# %%
fig,ax = plt.subplots(3,3,figsize=(20,15))
ax[0,0].imshow(windows[15][75])
ax[0,1].imshow(windows[15][76])
ax[0,2].imshow(windows[15][77])
ax[1,0].imshow(windows[16][75])
ax[1,1].imshow(windows[16][76])
ax[1,2].imshow(windows[16][77])
ax[2,0].imshow(windows[17][75])
ax[2,1].imshow(windows[17][76])
ax[2,2].imshow(windows[17][77])





# %%
from inspect import getmembers, isfunction

print(getmembers(process, isfunction))
# %%
getmembers(process)





# %%

# %%
pi_500 = make_PI_instance('2021-03-30/Laser1_Timing500')
pi_400 = make_PI_instance('2021-03-30/Laser1_Timing400')
pi_300 = make_PI_instance('2021-03-30/Laser1_Timing300')
pi_200 = make_PI_instance('2021-03-30/Laser1_Timing200')

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-200,200],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,420,0],    
    "sn_threshold": 1.000001,'dt': 0.0002,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}

pi_200.set_piv_param(piv_cond)
pi_300.set_piv_param(piv_cond)
pi_400.set_piv_param(piv_cond)
pi_500.set_piv_param(piv_cond)

pi_200.set_piv_param({"dt": 0.0002})
pi_300.set_piv_param({"dt": 0.0003})
pi_400.set_piv_param({"dt": 0.0004})
pi_500.set_piv_param({"dt": 0.0005})
# %%
d = pi_500.quick_piv(pi_500.piv_dict_list[5],index_a=100,index_b=101)
# %%
d = pi_400.quick_piv(pi_400.piv_dict_list[5],index_a=100,index_b=101)
# %%
d = pi_300.quick_piv(pi_300.piv_dict_list[5],index_a=100,index_b=101)
# %%
d = pi_200.quick_piv(pi_200.piv_dict_list[5],index_a=100,index_b=101)
# %%
