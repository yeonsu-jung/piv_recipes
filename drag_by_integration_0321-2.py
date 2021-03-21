# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)
# %%
folder_path = "C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/Flat_10 (black)_motor5"
results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
folder_path = folder_path.replace('C:/Users/yj/','/Users/yeonsu/')
results_folder_path = results_folder_path.replace('C:/Users/yj/','/Users/yeonsu/')

pi = piv.ParticleImage(folder_path,results_folder_path)
# %%
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]
pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()
# %%
x,y,ul,vl,ur,vr = pi.get_left_right_velocity_map_series('003_90')
# %%
step = 25
x1,y1,u1,v1,x2,y2,u2,v2 = pi.get_top_bottom_velocity_series(25,'003_90')
# %%

for i in range(90):
    try:
        u1_new = np.vstack((u1_new,u1[:,i,:,0].reshape((1,1080))))
    except:
        u1_new = u1[:,0,:,0].reshape((1,1080))
# %%
u1_new2 = np.moveaxis(u1,0,1)
u1_new2 = u1_new2.reshape(90,1080,4)
# u1_new = u1_new.reshape(1080,4)
# %%

# %%
t = np.linspace(0,90/15,90)
# plt.plot(t,np.mean(u1_new,axis=1))
plt.plot(t,np.mean(u1,axis=1))
# %%
y_lr = x[0,:]
u_left = -vl[:,0,:]
v_left = ul[:,0,:]

u_right = -vr[:,-1,:]
v_right = ur[:,-1,:]

x_top = y2[:,0]
u_top = -v2[:,:,0]
v_top = u2[:,:,0]

x_bottom = y1[:,-1]
u_bottom = -v1[:,:,-1]
v_bottom = u1[:,:,-1]
# %%
t = np.linspace(0,90/15,90)
# %%
T = 80
fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].plot(u_left[T,:],y_lr,'o-',label='left')
ax[0].plot(u_right[T,:],y_lr,'o-',label = 'right')
ax[1].plot(x_top,v_top[T,:],'o-',label = 'top')
ax[1].plot(x_top,v_bottom[T,:],'o-',label = 'bottom')

ax[0].set_xlabel('u (mm/s)')
ax[0].set_ylabel('y (mm)')
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('v (mm/s)')
# %%

# %%
rho = 1e3
span = 0.048
flowrate_left = rho*np.trapz(u_left,y_lr)*1e-6*span
flowrate_right = rho*np.trapz(u_right,y_lr)*1e-6*span
flowrate_top = rho*np.trapz(v_top,x_top)*1e-6*span
flowrate_bottom = rho*np.trapz(-v_bottom,x_top)*1e-6*span

momentum_left = rho*np.trapz(u_left**2,y_lr)*1e-9*span # in
momentum_right = rho*np.trapz(u_right**2,y_lr)*1e-9*span # out
momentum_top = rho*np.trapz(v_top*u_top,x_top)*1e-9*span # out
momentum_bottom = rho*np.trapz(-v_bottom*u_bottom,x_top)*1e-9*span # out

(flowrate_left,flowrate_right,flowrate_top,flowrate_bottom)
# print(momentum_left,momentum_right,momentum_top,momentum_bottom)

momentum_deficit = momentum_left - momentum_right - momentum_top - momentum_bottom
# %%
np.mean(momentum_deficit)
# %%
plt.plot(t,momentum_left)
plt.plot(t,-momentum_right)
plt.plot(t,-momentum_top)
plt.plot(t,-momentum_bottom)
plt.xlabel('time (s)')
plt.ylabel('Momentum in (N s)')
# %%
v_top.shape
# %%
plt.plot(t,np.mean(v_top,axis=1))
plt.plot(t,np.mean(v_bottom,axis=1))
# %%

# %%
plt.plot(t,momentum_deficit)
# %%
plt.plot(flowrate_top,'o-')
plt.plot(flowrate_bottom,'o-')
# %%
plt.plot(flowrate_left)
plt.plot(flowrate_right)
plt.plot(flowrate_top)
plt.plot(flowrate_bottom)
# %%
net_flowrate = flowrate_left - flowrate_right - flowrate_top - flowrate_bottom
# %%
plt.plot(net_flowrate)
# %%
plt.plot(flowrate_left-flowrate_right)
# %%
mdot = (flowrate_left-flowrate_right)
print(np.mean(mdot))
# %%
D = momentum_left - momentum_right - mdot * np.mean(u_top)/1000
print(np.mean(D))
# %%
mom_lr = momentum_left - momentum_right
plt.plot(mom_lr)
np.mean(mom_lr)
# %%
frate_lr = flowrate_left -flowrate_right
plt.plot(frate_lr)
# %%
plt.plot(D)
# %%
np.mean(momentum_deficit)
# %%
import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 300

fig = plt.figure()
ax = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
fig.set_size_inches([10,10])

# ax.set_aspect('equal')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
line, = ax.plot(x_top,v_top[0,:])
ax.set_xlabel('x (mm)')
ax.set_ylabel('v_top (mm/s)')

line2, = ax2.plot(t[0], np.mean(v_top,axis=1)[0] )
ax2.axis([0,6,-40,40])
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Mean of v_top (mm/s)')

line3, = ax3.plot(x_bottom,v_bottom[0,:])
ax3.set_xlabel('x (mm)')
ax3.set_ylabel('v_bottom (mm/s)')

line4, = ax4.plot(t[0], np.mean(v_bottom,axis=1)[0] )
ax4.axis([0,6,-40,40])
ax4.set_xlabel('time (s)')
ax4.set_ylabel('Mean of v_bottom (mm/s)')

line5, = ax5.plot(u_left[0,:],y_lr,'o-')
ax5.axis([0,400,0,18])
ax5.set_xlabel('u_left (mm/s)')
ax5.set_ylabel('y (mm)')

line6, = ax6.plot(u_right[0,:],y_lr,'o-')
ax6.axis([0,400,0,18])
ax6.set_xlabel('u_right (mm/s)')
ax6.set_ylabel('y (mm)')
# tight_layout()

def update(n):    
    line.set_data(x_top,v_top[n,:])
    line2.set_data(t[:n],np.mean(v_top,axis=1)[:n] )
    line3.set_data(x_bottom,v_bottom[n,:])
    line4.set_data(t[:n],np.mean(v_bottom,axis=1)[:n] )
    line5.set_data(u_left[n,:],y_lr)
    line6.set_data(u_right[n,:],y_lr)

    return line, line2, line3, line4, line5, line6

#legend(loc=0)
ani = animation.FuncAnimation(fig,update,90,interval=30)
writer = animation.writers['ffmpeg'](fps=10)

ani.save('Flat_10_motor5.mp4',writer=writer,dpi=dpi)
# %%
