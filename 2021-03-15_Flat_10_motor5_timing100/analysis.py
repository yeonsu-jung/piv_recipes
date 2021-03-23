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

def plot_dimensionless_velocity_profile(x, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
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
fig,ax = plt.subplots(figsize=(20,5),dpi=600)
ax = plot_dimensionless_velocity_profile(x,v_avg, v_std, 0,-1,50, ax,color='k')
# ax = plot_dimensionless_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')
fig.savefig('dless_bl.png',bbox_inches='tight', pad_inches=0)
# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/Flat_10 (black)_motor5_cropped'
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
plt.plot(-v_avg[:,-1])
plt.plot(-u_avg[:,-1])
# %%
fig,ax = plt.subplots(figsize=(20,5),dpi=600)
ax = plot_dimensionless_velocity_profile(x,v_avg, v_std, 0,-1,50, ax,color='k')
# ax = plot_dimensionless_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')
fig.savefig('dless_bl.png',bbox_inches='tight', pad_inches=0)
# %%
with open('all_data.txt','a') as f:
    names = ('x','y','u_avg','v_avg','u_std','v_std')
    i = 0
    for k in (x,y,u_avg,v_avg,u_std,v_std):
        f.write('# %s \n' %names[i])
        np.savetxt(f,k)
        # print(names[i])
        i = i + 1
# %%

# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/Flat_10 (black)_motor5_cropped'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)

