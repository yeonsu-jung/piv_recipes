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
def get_left_right_velocity_map_series(self,s):
    sd_left = {'pos': 1, 'VOFFSET': 0}
    left_path = [x['path'] for x in self.piv_dict_list if sd_left.items() <= x.items()][0]

    sd_right = {'pos': 6, 'VOFFSET': 840}
    right_path = [x['path'] for x in self.piv_dict_list if sd_right.items() <= x.items()][0]               
    
    x_path = os.path.join(self.results_path,left_path,'x_full.txt')
    y_path = os.path.join(self.results_path,left_path,'y_full.txt')

    ul_path = os.path.join(self.results_path,left_path,'u_full_series_%s.txt'%s)
    vl_path = os.path.join(self.results_path,left_path,'v_full_series_%s.txt'%s)

    ur_path = os.path.join(self.results_path,right_path,'u_full_series_%s.txt'%s)
    vr_path = os.path.join(self.results_path,right_path,'v_full_series_%s.txt'%s)

    x = np.loadtxt(x_path)
    y = np.loadtxt(y_path)

    # u_left = np.loadtxt(ul_path)    
    # v_left = np.loadtxt(vl_path)
    # u_right = np.loadtxt(ur_path)
    # v_right = np.loadtxt(vr_path)

    u_left = piv.load_nd_array(ul_path)
    v_left = piv.load_nd_array(vl_path)
    u_right = piv.load_nd_array(ur_path)
    v_right = piv.load_nd_array(vr_path)

    return x,y, u_left, v_left, u_right, v_right
# %%
def get_top_bottom_velocity_series(self,camera_step,s):
    lis = self.piv_dict_list    
    for pd in lis:        
        print(pd['path'])
        xu = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_upper.txt'))
        yu = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_upper.txt'))
        yu = yu + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']

        xl = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_lower.txt'))
        yl = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_lower.txt'))
        yl = yl + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']
        
        uu_path = os.path.join(self.results_path, pd['path'], 'u_upper_series_%s.txt'%s)
        vu_path = os.path.join(self.results_path, pd['path'], 'v_upper_series_%s.txt'%s)

        ul_path = os.path.join(self.results_path, pd['path'], 'u_lower_series_%s.txt'%s)
        vl_path = os.path.join(self.results_path, pd['path'], 'v_lower_series_%s.txt'%s)
        
        uu_series = piv.load_nd_array(uu_path)
        vu_series = piv.load_nd_array(vu_path)        

        ul_series = piv.load_nd_array(ul_path)
        vl_series = piv.load_nd_array(vl_path)

        try:
            entire_xu = np.vstack((entire_xu,xu))
            entire_yu = np.vstack((entire_yu,yu))

            entire_xl = np.vstack((entire_xl,xl))
            entire_yl = np.vstack((entire_yl,yl))

            entire_uu_series = np.vstack((entire_uu_series,uu_series.reshape(1,*uu_series.shape)))
            entire_vu_series = np.vstack((entire_vu_series,vu_series.reshape(1,*uu_series.shape)))
            
            entire_ul_series = np.vstack((entire_ul_series,ul_series.reshape(1,*uu_series.shape)))
            entire_vl_series = np.vstack((entire_vl_series,vl_series.reshape(1,*uu_series.shape)))
        except:
            entire_xu = xu
            entire_yu = yu

            entire_xl = xl
            entire_yl = yl

            entire_uu_series = uu_series.reshape(1,*uu_series.shape)
            entire_vu_series = vu_series.reshape(1,*vu_series.shape)
        
            entire_ul_series = ul_series.reshape(1,*ul_series.shape)
            entire_vl_series = vl_series.reshape(1,*vl_series.shape)

    a,b,c,d = entire_uu_series.shape   
    entire_uu_series = entire_uu_series.reshape(b,a*c,d)
    entire_vu_series = entire_vu_series.reshape(b,a*c,d)
    entire_ul_series = entire_ul_series.reshape(b,a*c,d)
    entire_vl_series = entire_vl_series.reshape(b,a*c,d)

    return entire_xu, entire_yu, entire_uu_series,entire_vu_series, entire_xl, entire_yl, entire_ul_series, entire_vl_series
# %%
folder_path = "C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/2_1_1_10 (black)_motor15"
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
x,y,ul,vl,ur,vr = get_left_right_velocity_map_series(pi,'003_90')
# %%
plt.plot(vl[0,:,:])
# %%
plt.plot(vr[0,24,:])
# %%
step = 25
x1,y1,u1,v1,x2,y2,u2,v2 = get_top_bottom_velocity_series(pi,25,'003_90')
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
print(momentum_left,momentum_right,momentum_top,momentum_bottom)

momentum_deficit = momentum_left - momentum_right - momentum_top - momentum_bottom
# %%

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
plt.plot()
# %%
T = 1
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
ax5.axis([250,500,0,18])
ax5.set_xlabel('u_left (mm/s)')
ax5.set_ylabel('y (mm)')

line6, = ax6.plot(u_right[0,:],y_lr,'o-')
ax6.axis([200,500,0,18])
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

ani.save('2_1_1.mp4',writer=writer,dpi=dpi)
# %%
plt.plot(t,np.mean(v_top,axis=1))
# %%
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
line, = ax.plot(x_top,v_top[20,:])
# %%
