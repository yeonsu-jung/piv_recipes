# %%
import openpiv_recipes as piv
import importlib
import numpy as np
import os
import time
import matplotlib.cm as cm
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

t = time.time()
importlib.reload(piv)
# %%
folder_path = "C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-15/2_1_1_10 (black)_motor15_cropped"
results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)
# %%
piv_cond = {
    "winsize": 28, "searchsize": 34, "overlap": 22,
    "pixel_density": 40,"scale_factor": 3e4, "arrow_width": 0.001,
    "u_bound": [-50,50],"v_bound": [-1000,0],
    "transpose": False, "crop": [0,0,10,0],    
    "sn_threshold": 1.000001,'dt': 0.0001,
    "rotate": 0.25, "save_result": True,"show_result": False, 'raw_or_cropped':True
}
pi.set_piv_param(piv_cond)
# %%
stitch_list = [x for x in os.listdir(folder_path) if not x.startswith(('.','_')) and not 'timing500' in x]

pi.set_param_string_list(stitch_list)
pi.piv_dict_list = pi.param_dict_list
pi.check_piv_dict_list()
# %%
pi.set_piv_param({"show_result": True})
xyuv = pi.quick_piv({'pos': 4, 'VOFFSET':630})
# %%
pi.set_piv_param({"show_result": False})

for sd in pi.search_dict_list:
    pi.piv_over_time(sd,start_index=3,N=90)
# %%
step = 25
x,y,u_avg,v_avg,u_std,v_std = pi.get_entire_avg_velocity_map(step,'003_90')
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

results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'
results_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15_cropped'

results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results'
results_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15_cropped'

x_path = os.path.join(results_path,'entire_x.txt')
y_path = os.path.join(results_path,'entire_y.txt')

u_path = os.path.join(results_path,'entire_u_tavg.txt')
v_path = os.path.join(results_path,'entire_v_tavg.txt')
us_path = os.path.join(results_path,'entire_u_tstd.txt')
vs_path = os.path.join(results_path,'entire_v_tstd.txt')
# %%
x = np.loadtxt(x_path) - 0.25
y = np.loadtxt(y_path)


u = np.loadtxt(u_path)
v = np.loadtxt(v_path)
us = np.loadtxt(us_path)
vs = np.loadtxt(vs_path)
# %%
start_i = 0
stop_i = -1
step_i = 50

v_array = -v[start_i:stop_i:step_i,:]
y_array = y[start_i:stop_i:step_i,0]
N = len(v_array)

fig,ax = plt.subplots(1,N,figsize=(20,5))
for i,vv in enumerate(v_array):
    ax[i].plot(vv,x[0,:])
    ax[i].axis([0,520,0,2.5])
# %%
s = []
for y in (y_array - 55):
    s.append('%.2f'%y)

C = 0.01
fig,ax = plt.subplots(figsize=(40,10))
for i,vv in enumerate(v_array):
    # ax.plot(vv + C*y_array[i],x[0,:],'k--')
    ax.plot(C*vv + y_array[i],x[0,:],'k--')
    ax.plot([y_array[i],y_array[i]],[0,2.5],'b-')
    ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,2.5],'b--')
    for j, vvv in enumerate(vv):         
        ax.arrow(y_array[i], x[0,j], C*(vvv-50), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

ax.set_xticks(y_array)
ax.set_xticklabels(s)

# %%
eta = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,100]
fprime = [0.0,0.03321,0.06441,0.09960,0.13276,0.16589,0.19894,0.26471,0.32978,0.39378,0.45626,0.51676,0.57476,0.62977,0.68131,0.72898,0.77245,0.81151,0.84604,0.91304,0.95552,0.97951,0.99154,0.99688,0.99897,0.99970,0.99992,1.0,1.0,1.0,1.0]

from scipy.interpolate import interp1d
f2 = interp1d(eta,fprime,kind='cubic')          

def blasius(x,U,p,q):                  
    # q = 0
    return U*f2(p*(x+q))

# %%
xx = x[0,:]
v_i = v_array[10]

fig,ax = plt.subplots()
ax.plot(v_i,xx,'k--')
ax.axis([0,540,0,2.5])
for j, vvv in enumerate(v_i):         
        ax.arrow(0, x[0,j], (vvv-20), 0, head_width=0.02, head_length=10, fc='k', ec='k')

# %%
popt,pcov = curve_fit(blasius, xx,v_i,bounds=((0,0.5,-1),(500,5,0.3)))
# %%
plt.plot(v_i,xx,'o',label='piv data')
yy = np.linspace(0,np.max(xx),100)
v_fit = blasius(xx,*popt)
plt.plot(v_fit,xx,'-',label='Blasius')
# plt.axis([0,4,0,300])
plt.xlabel('y (mm)')
plt.ylabel('u (mm/s)')
plt.legend()
print(popt)
# %%
x_fit = np.linspace(0.3,2.5,100)
v_fit = blasius(x_fit,*popt)
# %%
fig,ax = plt.subplots()
ax.plot(v_i,xx,'k--',label='PIV data')
ax.axis([0,540,0,2.5])
for j, vvv in enumerate(v_i):         
    ax.arrow(0, x[0,j], (vvv-20), 0, head_width=0.02, head_length=10, fc='k', ec='k')
ax.plot(v_fit,x_fit,'b-',label='Blasius')
ax.set_xlabel('u (mm/s)')
ax.set_ylabel('y (mm)')
ax.legend(loc='lower right')
# %%
def fit_with_blasius(xx,v_i):  
    popt,pcov = curve_fit(blasius, xx,v_i,bounds=((0,0.5,-0.3),(500,10,0)))            
    x_fit = np.linspace(-popt[2],2.5,100)
    print(x_fit)
    v_fit = blasius(x_fit,*popt)    
    fig,ax = plt.subplots(figsize=(1.5,3))
    # ax.plot(v_i,xx,'k--',label='PIV data')
    ax.axis([0,540,0,2.5])
    for j, vvv in enumerate(v_i):         
        ax.arrow(0, x[0,j], (vvv-20), 0, head_width=0.02, head_length=10, fc='k', ec='k')
    ax.plot(v_fit,x_fit,'b-',label='Blasius')
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    ax.legend(loc='lower right')    
    print(*popt)
    return popt
# %%
U_array = []
p_array = []
q_array = []

fig,ax = plt.subplots(12,figsize=(20,5))
for i in range(6,18):
    u,p,q = fit_with_blasius(xx,v_array[i])
    U_array.append(u)
    p_array.append(p)
    q_array.append(q)
    # plt.savefig('bl_%d.png'%i)

# %%
U_array_conversion = np.array(U_array) * 0.001
nux_array = U_array_conversion/(np.array(p_array)*1000)**2
wall_shear_stress = U_array_conversion*0.33206*(U_array_conversion/nux_array)**0.5*0.001
# %%
fig,ax = plt.subplots(3)
ax[0].plot(U_array_conversion)
ax[1].plot(nux_array,'o-')
ax[2].plot(wall_shear_stress,'o-')
# %%

x = y_array[6:18] - 55
y = wall_shear_stress

plt.plot(x,y,'o-')
plt.xlabel('x (mm)')
plt.ylabel('$\\tau$ (Pa)')

x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10,label='From PIV fitting')
plt.bar(x_left,y_left,width=x[1:]-x[:-1],alpha=0.2,align='edge',edgecolor='b')
plt.axis([0,100,0,2])

xt = np.linspace(0,100,100)
yt = 0.332*1e-3*0.5**(3/2)/(1e-6*xt/1000)**0.495

plt.plot(xt,yt,label='Blasius solution')
plt.legend()
# %%
np.sum(y[:-1]*(x[1:]-x[:-1]))
# %%
np.sum(y[:-1]*(x[1:]-x[:-1]))/1000*0.048

# %%

