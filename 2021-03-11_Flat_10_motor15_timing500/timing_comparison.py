# %%
import sys
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


import matplotlib.animation as animation
# %%
sys.path.append(os.path.dirname('../'))
import openpiv_recipes as piv
importlib.reload(piv)

# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15_timing500_cropped'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi_500 = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi_500 = piv.ParticleImage(folder_path,results_folder_path)
# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15_cropped'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi_100 = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi_100 = piv.ParticleImage(folder_path,results_folder_path)
# %%
def plot_single_profile(i,x_array,y_array,v_array,v_std_array,ax,x_cut = 0,**kw):
    C = 1        
    # y_array[i]
    vv = v_array[i,:]
    ax.plot(C*vv[x_cut:],x_array[0,x_cut:],'--',**kw)
    ax.plot([0,0],[0,np.max(x_array)],'b-')
    ax.plot([0+C*540,C*540],[0,np.max(x_array)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(0, x_array[0,j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
        ax.arrow(0 + C*(vvv), x_array[0,j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')
        ax.arrow(0 + C*(vvv), x_array[0,j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')

    ax.set_title('x = %.2f mm'%(y_array[i]-55))
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    plt.savefig('individual_profile.png')
    # fig.savefig('individual_profile.png',dpi=900)
    return ax

# %%
step = 25
x5,y5,u_avg5,v_avg5,u_std5,v_std5 = pi_500.get_entire_avg_velocity_map(step,'003_90')
x5 = x5 - 0.4 # 0.4 

# %%
step = 25
x,y,u_avg,v_avg,u_std,v_std = pi_100.get_entire_avg_velocity_map(step,'003_90')
x = x - 0.4 # 0.4 

# %%
ix = 200

fig,ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
ax = plot_single_profile(ix,x,y[:,0],-v_avg,-v_std,ax,x_cut = 0,color='k')
ax = plot_single_profile(ix,x,y[:,0],-v_avg5,-v_std5,ax,x_cut = 0,color='r')

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-25,25],"v_bound": [-800,100],
    "transpose": False, "crop": [0,0,12,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi_100.set_piv_param(piv_cond)
pi_500.set_piv_param(piv_cond)
# %%
pi_100.piv_param
# %%
ix = 10
d = pi_100.quick_piv(pi_100.piv_dict_list[ix])

# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-25,25],"v_bound": [-800,100],
    "transpose": False, "crop": [0,0,12,0],    
    "sn_threshold": 1.000001,'dt': 0.0005,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi_500.set_piv_param(piv_cond)

d = pi_500.quick_piv(pi_500.piv_dict_list[ix])

# %%
step = 25
x5,y5,u_avg5,v_avg5,u_std5,v_std5 = pi_500.get_entire_avg_velocity_map(step,'003_90')
x5 = x5 - 0.4 # 0.4 
# %%

# %%
step = 25
x,y,u_avg,v_avg,u_std,v_std = pi_100.get_entire_avg_velocity_map(step,'003_90')
x = x - 0.4 # 0.4 
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

        ax.plot(C* + Re,x[0,:]*x_c,'--')
        for j, vvv in enumerate(vv):
            ax.arrow(Re, x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

    x = np.linspace(0,0.12,100)
    Re_x = np.linspace(0,60000,100)
    delta = 5*((mu/rho)*x/U)**0.5*1e3
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
fig,ax = plt.subplots(figsize=(40,10))
ax = plot_dimensionless_velocity_profile(x,v_avg, v_std, 0,-1,50, ax,color='k')
ax = plot_dimensionless_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')
# %%
fig,ax = plt.subplots(figsize=(40,10))
ax = plot_velocity_profile(x,v_avg,v_std,0,-1,50,ax,color='k')
ax = plot_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')

# %%
fig,ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
ax = plot_single_profile(500,y[:,0],-v_avg,-v_std,ax,x_cut = 0,color='k')
ax = plot_single_profile(500,y[:,0],-v_avg5,-v_std5,ax,x_cut = 0,color='r')


# %%


C = 1        
fig,ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
def update(i):
    ax.clear() 
    plot_single_profile(5*i,y[:,0],-v_avg,-v_std,ax,x_cut = 0,color='k')
    plot_single_profile(5*i,y[:,0],-v_avg5,-v_std5,ax,x_cut = 0,color='g')

ani = animation.FuncAnimation(fig,update,200,interval=30)
writer = animation.writers['ffmpeg'](fps=10)

ani.save('boundary_layer_movie.mp4',writer=writer)
# %%


piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 1e4, "arrow_width": 0.001,
    "u_bound": [-25,25],"v_bound": [-800,100],
    "transpose": False, "crop": [0,0,12,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0, "save_result": True,"show_result": True, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
d = pi.quick_piv(pi.piv_dict_list[15])