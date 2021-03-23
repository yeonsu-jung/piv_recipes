# %%
import sys
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

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

# %%
def plot_dimensionless_velocity_profile(x, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
    v_array = -v_avg[start_i:stop_i:step_i,:]
    v_std_array = v_std[start_i:stop_i:step_i,:]
    y_array = y[start_i:stop_i:step_i,0]
    N = len(v_array)
    s = []
    
    C = 2
    
    rho = 1e3
    mu = 1e-3

    for yy in (y_array): # -55 for rearrange x axis with LE being zero.
        Re = rho*0.5*(yy-55)/mu*1e-3       
        s.append('%.2f'%Re)

    for i,vv in enumerate(v_array):
        # ax.plot(vv + C*y_array[i],x[0,:],'k--')
        # U = np.max(vv)*1e-3
        U = 0.5

        Re = rho*U*(y_array[i]-55)/mu*1e-3       
        # x_c = (U/(mu/rho)/((y_array[i]-55)*1e-3))**0.5*1e-3
        x_c = 1

        ax.plot(C*vv + Re,x[0,:]*x_c,'--',**kwargs)
        ax.plot([Re,Re],[0,np.max(x)*x_c],'b-')
        ax.plot([Re+C*540,Re+C*540],[0,np.max(x)*x_c],'b--')
        for j, vvv in enumerate(vv):
            ax.arrow(Re, x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

    xx = np.linspace(0,0.12,100)
    Re_x = np.linspace(0,60000,100)
    delta = 5*((mu/rho)*xx/U)**0.5*1e3
    ax.plot(Re_x,delta,'g-')

    ax.set_xticks(rho*U*(y_array-55)/mu*1e-3   )
    ax.set_xticklabels(s)
    ax.set_xlabel('Re_x')
    ax.set_ylabel('$\eta$')
    ax.axis([0,60000,0,2.5])

    return ax

# %%
def plot_single_profile(i,y_array,v_array,v_std_array,ax,x_cut = 0,**kw):
    C = 1        
    # y_array[i]
    vv = v_array[i,:]
    ax.plot(C*vv[x_cut:],x[0,x_cut:],'--',**kw)
    ax.plot([0,0],[0,np.max(x)],'b-')
    ax.plot([0+C*540,C*540],[0,np.max(x)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(0, x[0,j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
        ax.arrow(0 + C*(vvv), x[0,j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')
        ax.arrow(0 + C*(vvv), x[0,j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')

    ax.set_title('x = %.2f mm'%(y_array[i]-55))
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    plt.savefig('individual_profile.png')
    # fig.savefig('individual_profile.png',dpi=900)
    return ax

# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15_timing500_cropped'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)

# %%
step = 25
x,y,u_avg,v_avg,u_std,v_std = pi.get_entire_avg_velocity_map(step,'003_90')
x = x - 0.4 # 0.4 
# %%
fig,ax = plt.subplots(figsize=(40,10))
ax = plot_dimensionless_velocity_profile(x,v_avg, v_std, 0,-1,50, ax,color='k')
# ax = plot_dimensionless_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')
# %%
start_i = 0
stop_i = -1
step_i = 50

v_array = -v_avg[start_i:stop_i:step_i,:]
v_std_array = v_std[start_i:stop_i:step_i,:]
y_array = y[start_i:stop_i:step_i,0]
N = len(v_array)
# %%
s = []
for yy in (y_array - 55): # -55 for rearrange x axis with LE being zero.
    s.append('%.2f'%yy)

C = 0.005
fig,ax = plt.subplots(figsize=(40,10))
for i,vv in enumerate(v_array):
    # ax.plot(vv + C*y_array[i],x[0,:],'k--')
    ax.plot(C*vv + y_array[i],x[0,:],'k--')
    ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
    ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
    for j, vvv in enumerate(vv):         
        ax.arrow(y_array[i], x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

ax.set_xticks(y_array)
ax.set_xticklabels(s)
# %%
x_cut = 0

C = 0.01
fig,ax = plt.subplots(figsize=(40,10))
for i,vv in enumerate(v_array):
    # ax.plot(vv + C*y_array[i],x[0,:],'k--')
    ax.plot(C*vv[x_cut:] + y_array[i],x[0,x_cut:],'k--')
    ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
    ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(y_array[i], x[0,j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
   
ax.set_xticks(y_array)
ax.set_xticklabels(s)
# %%
v_std_array.shape
# %%
x_cut = 0

C = 0.01
fig,ax = plt.subplots(figsize=(40,10))
for i,vv in enumerate(v_array):
    # ax.plot(vv + C*y_array[i],x[0,:],'k--')
    ax.plot(C*vv[x_cut:] + y_array[i],x[0,x_cut:],'k--')
    ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
    ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(y_array[i], x[0,j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
        ax.arrow(y_array[i] + C*(vvv), x[0,j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.02, head_length=0.2, fc='r', ec='r')
        ax.arrow(y_array[i] + C*(vvv), x[0,j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.02, head_length=0.2, fc='r', ec='r')
   
ax.set_xticks(y_array)
ax.set_xticklabels(s)
# %%
v_array.shape
# %%
C = 1
fig,ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
i = 10
y_array[i]
vv = v_array[i,:]
ax.plot(C*vv[x_cut:],x[0,x_cut:],'k--')
ax.plot([0,0],[0,np.max(x)],'b-')
ax.plot([0+C*540,C*540],[0,np.max(x)],'b--')
for j, vvv in enumerate(vv[x_cut:]):         
    ax.arrow(0, x[0,j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
    ax.arrow(0 + C*(vvv), x[0,j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')
    ax.arrow(0 + C*(vvv), x[0,j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')

ax.set_title('x = %.2f mm'%(y_array[i]-55))
ax.set_xlabel('u (mm/s)')
ax.set_ylabel('y (mm)')
plt.savefig('individual_profile.png')
# fig.savefig('individual_profile.png',dpi=900)


# %%
plt.plot(v_std_array.T)

# %%
fig,ax = plt.subplots(figsize=(20,5))
c1 = ax.contourf(y,x,(u_avg**2+v_avg**2)**0.5,cmap=cm.coolwarm)
fig.colorbar(c1, ax=ax)

for pos in [1,2,3,4,5,6]:
    ax.plot([step*pos,step*pos],[0.5,np.max(x)],'k-')

xx = np.linspace(56,156,100)
yy = 5*np.sqrt(1e-6/0.54*(xx-56)/1000)*1000+0.5
ax.plot(xx,yy,'b-')

# %%
fig,ax = plt.subplots(figsize=(20,5))
c1 = ax.contourf(y,x,(u_std**2+v_std**2)**0.5,cmap=cm.coolwarm)
fig.colorbar(c1, ax=ax)

for pos in [1,2,3,4,5,6]:
    ax.plot([step*pos,step*pos],[0.5,np.max(x)],'k-')

xx = np.linspace(56,156,100)
yy = 5*np.sqrt(1e-6/0.54*(xx-56)/1000)*1000+0.5
ax.plot(xx,yy,'b-')
# %%
