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

# %%
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/2_1_1_10 (black)_motor15_cropped'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)

# %% For velocity field over sample
step = 25
x_entire,y_entire,u_avg,v_avg,u_std,v_std = pi.get_entire_avg_velocity_map(step,'003_90')
x_entire = x_entire - 0.4 # 0.4

fig,ax = plt.subplots(figsize=(20,5),dpi=600)
ax = plot_dimensionless_velocity_profile(x_entire,y_entire,v_avg, v_std, 0,-1,50, ax,color='k')
# ax = plot_dimensionless_velocity_profile(x5,v_avg5, v_std5, 0,-1,50, ax,color='r')
fig.savefig('dless_bl.png',bbox_inches='tight', pad_inches=0)
# %%
with open('all_data.txt','a') as f:
    names = ('x','y','u_avg','v_avg','u_std','v_std')
    i = 0
    for k in (x_entire,y_entire,u_avg,v_avg,u_std,v_std):
        f.write('# %s \n' %names[i])
        np.savetxt(f,k)
        # print(names[i])
        i = i + 1
# %% For drag analysis
try:
    folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/2_1_1_10 (black)_motor15'
    results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'    
    pi = piv.ParticleImage(folder_path,results_folder_path)
except:
    folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
    pi = piv.ParticleImage(folder_path,results_folder_path)
# %%
x,y,ul,vl,ur,vr = pi.get_left_right_velocity_map_series('003_90')

step = 25
x1,y1,u1,v1,x2,y2,u2,v2 = pi.get_top_bottom_velocity_series(25,'003_90')
# %%
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

t = np.linspace(0,90/15,90)
# %%
rho = 1e3
span = 0.048
flowrate_left = rho*np.trapz(u_left,y_lr)*1e-6*span
flowrate_right = -rho*np.trapz(u_right,y_lr)*1e-6*span
flowrate_top = -rho*np.trapz(v_top,x_top)*1e-6*span
flowrate_bottom = rho*np.trapz(v_bottom,x_top)*1e-6*span

momentum_left = rho*np.trapz(u_left**2,y_lr)*1e-9*span # in
momentum_right = -rho*np.trapz(u_right**2,y_lr)*1e-9*span # out
momentum_top = -rho*np.trapz(v_top*u_top,x_top)*1e-9*span # out
momentum_bottom = rho*np.trapz(v_bottom*u_bottom,x_top)*1e-9*span # out

momentum_deficit = momentum_left + momentum_right + momentum_top + momentum_bottom
net_flowrate = flowrate_left + flowrate_right + flowrate_top + flowrate_bottom

flowrate_left2 = rho*np.trapz(np.mean(u_left,axis=0),y_lr)*1e-6*span
flowrate_right2 = -rho*np.trapz(np.mean(u_right,axis=0),y_lr)*1e-6*span
flowrate_top2 = -rho*np.trapz(np.mean(v_top,axis=0),x_top)*1e-6*span
flowrate_bottom2 = rho*np.trapz(np.mean(v_bottom,axis=0),x_top)*1e-6*span

momentum_left2 = rho*np.trapz(np.mean(u_left,axis=0)**2,y_lr)*1e-9*span # in
momentum_right2 = -rho*np.trapz(np.mean(u_right,axis=0)**2,y_lr)*1e-9*span # out
momentum_top2 = -rho*np.trapz(np.mean(v_top,axis=0)*np.mean(u_top,axis=0),x_top)*1e-9*span # out
momentum_bottom2 = rho*np.trapz(np.mean(v_bottom,axis=0)*np.mean(u_bottom,axis=0),x_top)*1e-9*span # out
# %%
print(np.mean(momentum_deficit))
# %%
momentum_deficit2 = momentum_left2 + momentum_right2 + momentum_top2 + momentum_bottom2
print(momentum_deficit2)
# %%
plt.plot(np.mean(u_bottom,axis=0)[400:800])
# %%
momentum_deficit3 = momentum_left2 + momentum_right2 - (flowrate_left2+flowrate_right2)*np.mean((u_top+u_bottom)/2*0.001)
print(momentum_deficit3)

# %% For pressure calculation
u_bottom_avg = np.mean(u_bottom,axis=0)
v_bottom_avg = np.mean(v_bottom,axis=0)
U_bottom_avg = (u_bottom_avg**2 + v_bottom_avg**2)**0.5

# %% check U_bottom
# plt.plot(U_bottom_avg)
plt.plot(U_bottom_avg[1:10])
plt.plot(U_bottom_avg[-10:-1])
# %%
U_inf_first = np.mean(U_bottom_avg[1:5])*1e-3
U_inf_last = np.mean(U_bottom_avg[-10:-5])*1e-3
pressure_difference = 0.5 * rho * (U_inf_last**2 - U_inf_first**2)* span * (np.max(x) - np.min(x)) * 0.001
print(pressure_difference)
# %%
Drag = momentum_deficit3 + pressure_difference
print(Drag)
# %%
T = 80
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(u_left[T,:],y_lr,'o-',label='left')
ax[0].plot(u_right[T,:],y_lr,'o-',label = 'right')
ax[1].plot(x_top,v_top[T,:],'ko-',label = 'top')
ax[1].plot(x_top,v_bottom[T,:],'ro-',label = 'bottom')

ax[0].set_xlabel('u (mm/s)')
ax[0].set_ylabel('y (mm)')
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('v (mm/s)')
fig.legend()
fig.savefig('boundary_velocities.png',bbox_inches='tight', pad_inches=0)
# %%
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(np.mean(u_left,axis=0),y_lr,'o-',label='left')
ax[0].plot(np.mean(u_right,axis=0),y_lr,'o-',label = 'right')
ax[1].plot(x_top,np.mean(v_top,axis=0),'ko-',label = 'top')
ax[1].plot(x_top,np.mean(v_bottom,axis=0),'ro-',label = 'bottom')

ax[0].set_xlabel('u (mm/s)')
ax[0].set_ylabel('y (mm)')
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('v (mm/s)')
fig.legend()
fig.savefig('averaged_boundary_velocities.png',bbox_inches='tight', pad_inches=0)

# %%
plt.figure(dpi=300)
plt.plot(t,momentum_left,label='left')
plt.plot(t,momentum_right,label='right')
plt.plot(t,momentum_top,label='top')
plt.plot(t,momentum_bottom,label='bottom')
plt.xlabel('time (s)')
plt.ylabel('Momentum in (N s)')
plt.legend()
plt.savefig('momentum_entry.png')

# %% Animation
# C = 1        
# fig,ax = plt.subplots(figsize=(3.5,3.5),dpi=600)
# def update(i):
#     ax.clear() 
#     plot_single_profile(5*i,y_entire[:,0],x_entire[0,:],-v_avg,-v_std,ax,x_cut = 0,color='k')    

# ani = animation.FuncAnimation(fig,update,200,interval=30)
# writer = animation.writers['ffmpeg'](fps=10)
# plt.tight_layout()

# ani.save('boundary_layer_movie.mp4',writer=writer)
# %%

# %%



