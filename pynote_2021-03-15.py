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
folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15'
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'
results_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-03-11/Flat_10 (black)_motor15'

# folder_path = "C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-data\\2021-03-11\\Flat_10 (black)_motor15"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

x_path = os.path.join(results_path,'entire_x.txt')
y_path = os.path.join(results_path,'entire_y.txt')

u_path = os.path.join(results_path,'entire_u_tavg.txt')
v_path = os.path.join(results_path,'entire_v_tavg.txt')
us_path = os.path.join(results_path,'entire_u_tstd.txt')
vs_path = os.path.join(results_path,'entire_v_tstd.txt')
# %%
x = np.loadtxt(x_path)
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


# %%
y_array = y[start_i:stop_i:step_i,0]
y_array
# %%
eta = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,100]
fprime = [0.0,0.03321,0.06441,0.09960,0.13276,0.16589,0.19894,0.26471,0.32978,0.39378,0.45626,0.51676,0.57476,0.62977,0.68131,0.72898,0.77245,0.81151,0.84604,0.91304,0.95552,0.97951,0.99154,0.99688,0.99897,0.99970,0.99992,1.0,1.0,1.0,1.0]

from scipy.interpolate import interp1d
f2 = interp1d(eta,fprime,kind='cubic')          

def blasius(x,U,p):                  
    q = -0.1
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
popt,pcov = curve_fit(blasius, xx,v_i,bounds=((0,0.5),(500,5)))
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
x_fit = np.linspace(0.1,2.5,100)
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
ax.legend()
# %%
def fit_with_blasius(xx,v_i):  
    popt,pcov = curve_fit(blasius, xx,v_i,bounds=((0,0.5),(500,10)))            
    x_fit = np.linspace(0.1,2.5,100)
    v_fit = blasius(x_fit,*popt)    
    fig,ax = plt.subplots(figsize=(1.5,3))
    # ax.plot(v_i,xx,'k--',label='PIV data')
    ax.axis([0,540,0,2.5])
    for j, vvv in enumerate(v_i):         
        ax.arrow(0, x[0,j], (vvv-20), 0, head_width=0.02, head_length=10, fc='k', ec='k')
    ax.plot(v_fit,x_fit,'b-',label='Blasius')
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    ax.legend()    
    print(*popt)
    return popt
# %%
U_array = []
p_array = []

fig,ax = plt.subplots(12,figsize=(20,5))
for i in range(6,18):
    u,p = fit_with_blasius(xx,v_array[i])
    U_array.append(u)
    p_array.append(p)
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
plt.axis([0,100,0,1.3])

xt = np.linspace(0,100,100)
yt = 0.332*1e-3*0.5**(3/2)/(1e-6*xt/1000)**0.495

plt.plot(xt,yt,label='Blasius solution')
plt.legend()
# %%
np.sum(y[:-1]*(x[1:]-x[:-1]))
# %%
np.sum(y[:-1]*(x[1:]-x[:-1]))/1000*0.048

# %%
