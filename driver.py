# %%
import piv_class as pi
from importlib import reload
from matplotlib import pyplot as plt
import numpy as np

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-05'
parent_path = parent_path.replace('C:/Users/yj/','/Users/yeonsu/')

ins = pi.piv_class(parent_path)
os.listdir(parent_path)
# %%
_ = ins.quick_piv(index = 2)
# %%
for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue
    if x.startswith('Experiment'):
        continue

    print(x)
# %%
yn = input('Go ahead?')
if yn == 'yes':
    pass
else:
    raise AssertionError

for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue
    if x.startswith('Experiment'):
        continue
    if x.startswith('Flat_10 (clear 3dp)_motor10'):
        continue
    
    pth = os.path.join(parent_path,x)    

    t = time.time()
    ins.piv_over_time(path = x,start_index=3,N=95)
    elapsed = time.time() - t
    print('Elapsed time: %.3f s'%elapsed)


    
# %%
d = ins.quick_piv(index = 101)

# %%
import time

t = time.time()
ins.piv_over_time(start_index=3,N=95)
elapsed = t - time.time()
print('Elapsed time: %.3f s'%elapsed)

# %%
ins.piv_over_sample(start_index=3,N=95)


# %%
def foo(path):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    for i in range(u.shape[0]):
        plt.plot(-v[i,0,:],x[0,:],'b.')
    os.chdir(owd)




# %%
def foo2(path):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    
    u_avg = np.mean(u,axis=0)
    v_avg = np.mean(v,axis=0)

    plt.plot(-v_avg[0,:],x[0,:],'o-')
    os.chdir(owd)

# %%
def foo3(path,k):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    for i in range(u.shape[0]):
        plt.plot(-v[i,k,:],x[0,:],'.')
    os.chdir(owd)
# %%
def foo4(path,k):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    
    u_avg = np.mean(u,axis=0)
    v_avg = np.mean(v,axis=0)

    plt.plot(-v_avg[k,:],x[0,:],'o-')
    os.chdir(owd)
# %%
def foo5(path):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    os.chdir(owd)

    return x,y,u,v

# %%
path_flat1 = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/Flat_10 (black)_timing400/Flat_10 (black) (9)_motor10.53/0,0,700,0_32,26,40/series_003_95'
foo(path_flat1)
foo2(path_flat1)
plt.axis([0,500,0,6])

# %%
path_flat2 = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/Flat_10 (black)_timing400/Flat_10 (black) (9)_motor10.53/0,0,700,0_32,26,50/series_003_95'
foo(path_flat2)
foo2(path_flat2)
plt.axis([0,500,0,6])

# %%
foo2(path_flat1)
foo2(path_flat2)

# %%

path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/1_1_1_10 (black)_timing400/1_1_1_10 (black)_motor10.53/0,0,870,0_32,26,40/series_003_95'
foo(path)
foo2(path)
plt.axis([0,500,0,6])

# %%
path2 = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/1_1_1_10 (black)_timing400/1_1_1_10 (black)_motor10.53/0,0,870,0_32,26,50/series_003_95'
foo(path)
foo2(path2)
plt.axis([0,500,0,6])

# %%
foo2(path)
foo2(path2)

plt.axis([0,500,0,6])
# %%
path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/2_1_1_10 (black)_timing400/2_1_1_10 (black)_motor10.53/0,0,860,0_32,26,40/series_003_95'

foo(path)
foo2(path)
plt.axis([0,500,0,6])
# %%
path_211_1 = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/2_1_1_10 (black)_timing400/2_1_1_10 (black)_motor10.53/0,0,860,0_32,26,40/series_003_95'
path_211_2 = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-12/2_1_1_10 (black)_timing400/2_1_1_10 (black)_motor10.53/0,0,860,0_32,26,50/series_003_95'

foo2(path_211_1)
foo2(path_211_2)
plt.axis([0,500,0,6])


# %%
foo(path_211_1)
plt.axis([0,500,0,6])
# %%

foo(path_211_2)
plt.axis([0,500,0,6])

# %%

foo('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')
# %%
foo('/Users/yeonsu/Documents/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1600_laser1-4_nd0p7_RE/0,0,0,0_32,26,50/series_003_95')
# %%
foo3('/Users/yeonsu/Documents/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1600_laser1-4_nd0p7_RE/0,0,0,0_32,26,50/series_003_95',5)

# %%
foo3('/Users/yeonsu/Documents/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1600_laser1-4_nd0p7_RE/0,0,0,0_32,26,50/series_003_95',4)
# %%

from matplotlib import pyplot as plt
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',4)
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',5)
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',6)

plt.plot([0,500],[822.32/40, 822.32/40],'k-')
plt.ylim([822.32/40,32])

# %%
# (822.3208333333333, 617.0041666666667)
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',4)
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',5)
foo4('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',6)

plt.plot([0,500],[617.00/40, 617.0/40],'k-')
plt.ylim([0,617.0/40])
# %%
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',4)
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',5)
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',6)

plt.plot([0,500],[617.00/40, 617.0/40],'k-')
# plt.ylim([0,617.0/40])
plt.ylim([0,650.0/40])
# %%
# %%
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',4)
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',5)
foo3('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95',6)

plt.plot([0,500],[822.32/40, 822.32/40],'k-')
plt.ylim([812.32/40,32])
# %%
x,y,u,v = foo5('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')

# %%
x.shape,u.shape
# %%
w1 = 617.0/40
w2 = 822.32/40

plt.plot(x[0,:],'o')
# %%

xtmp = x[0,:]
vtmp = v[:,0,:]

xtmp2 = xtmp[xtmp < w1]
xtmp3 = xtmp[xtmp > w2]

# %%
xfront = xtmp[xtmp > w2]
vfront = -v[:,0,xtmp>w2]

# %%

k = 15

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
for i in range(u.shape[0]):
    plt.plot(-v[i,k,xtmp<w1],x[0,xtmp<w1],'.')
    
plt.plot(-v[:,k,xtmp<w1].mean(axis=0),x[0,xtmp<w1],'k-',linewidth=2)
plt.plot([0,400],[w1,w1],'k--')
plt.xlim([0,400])

plt.subplot(1,2,2)
for i in range(u.shape[0]):
    plt.plot(-v[i,k,xtmp>w2],x[0,xtmp>w2],'.')

plt.plot(-v[:,k,xtmp>w2].mean(axis=0),x[0,xtmp>w2],'k-',linewidth=2)
plt.plot([0,400],[w2,w2],'k--')
plt.xlim([0,400])
# %%
eta = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,100]
fprime = [0.0,0.03321,0.06441,0.09960,0.13276,0.16589,0.19894,0.26471,0.32978,0.39378,0.45626,0.51676,0.57476,0.62977,0.68131,0.72898,0.77245,0.81151,0.84604,0.91304,0.95552,0.97951,0.99154,0.99688,0.99897,0.99970,0.99992,1.0,1.0,1.0,1.0]
# https://www.chegg.com/homework-help/questions-and-answers/table-10-3-solution-blasius-laminar-flat-plate-boundary-layer-similarity-variables-5-6-5-5-q28526523

from scipy.interpolate import interp1d
f2 = interp1d(eta,fprime,kind='cubic')


def blasius(x,U,p,q):
    # q = 0
    return U*f2(p*(x+q))
# %%
from scipy.optimize import curve_fit

# xtmp = x[0,:]
# %%
k = 22
xtmp = np.flip(w1 - x[0,x[0,:]<w1])
ytmp = np.flip(-v[:,k,x[0,:]<w1].mean(axis=0))

# plt.plot(xtmp,ytmp,'o')
# plt.ylim([0,350])

st = 3
popt,pcov = curve_fit(blasius,xtmp[st:],ytmp[st:],bounds=([300,1,-50],[350,10,50]))

import numpy as np
plt.plot(xtmp[st:],ytmp[st:],'o')
xtmp2 = np.linspace(-popt[2],14,100)
plt.plot(xtmp2,blasius(xtmp2,*popt))
plt.ylim([0,350])
plt.text(0.5,20,'delta: %.2f mm'%popt[2])
plt.text(10,300,'U_inf: %.2f mm/s'%popt[0])
plt.text(8,150,'x = %.2f mm'%(popt[0]/popt[1]**2))

# ==> make this routine as function or class
# %%
path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-08/Flat_10 (black)_motor10_stitching/Flat_10 (black)_timing400_ag1_dg1_laser1_shutter100_motor10.00_pos3_[04-08]_VOFFSET0/0,0,0,0_32,26,50/series_003_95'
x,y,u,v = foo5(path)

# %%
from pathlib import Path
# `path.parents[1]` is the same as `path.parent.parent`
# d = Path(__file__).resolve().parents[1]

w1,w2 = np.loadtxt(Path(path).resolve().parents[1].joinpath('wall_a_position.txt'))

# %%
w1, w2
# %%
import yaml

with open(Path(path).joinpath('piv_setting.yaml')) as f:
    piv_param = yaml.safe_load(f)

pix_den = piv_param["pixel_density"]
# %%
w1 = w1/pix_den
w2 = w2/pix_den

# %%

xtmp = x[0,:]
vtmp = v[:,0,:]

xtmp2 = xtmp[xtmp < w1]
xtmp3 = xtmp[xtmp > w2]

# %%
xfront = xtmp[xtmp > w2]
vfront = -v[:,0,xtmp>w2]

# %%
w1,w2
# %%
foo4(path,6)
plt.plot([0,600],[w1,w1],'k-')
plt.plot([0,600],[w2,w2],'k-')

# %%
k = 6

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
for i in range(u.shape[0]):
    plt.plot(-v[i,k,xtmp<w1],x[0,xtmp<w1],'.')
    
plt.plot(-v[:,k,xtmp<w1].mean(axis=0),x[0,xtmp<w1],'k-',linewidth=2)
plt.plot([0,600],[w1,w1],'k--')
plt.xlim([0,600])

plt.subplot(1,2,2)
for i in range(u.shape[0]):
    plt.plot(-v[i,k,xtmp>w2],x[0,xtmp>w2],'.')

plt.plot(-v[:,k,xtmp>w2].mean(axis=0),x[0,xtmp>w2],'k-',linewidth=2)
plt.plot([0,600],[w2,w2],'k--')
plt.xlim([0,600])
# %%
eta = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,100]
fprime = [0.0,0.03321,0.06441,0.09960,0.13276,0.16589,0.19894,0.26471,0.32978,0.39378,0.45626,0.51676,0.57476,0.62977,0.68131,0.72898,0.77245,0.81151,0.84604,0.91304,0.95552,0.97951,0.99154,0.99688,0.99897,0.99970,0.99992,1.0,1.0,1.0,1.0]
# https://www.chegg.com/homework-help/questions-and-answers/table-10-3-solution-blasius-laminar-flat-plate-boundary-layer-similarity-variables-5-6-5-5-q28526523

from scipy.interpolate import interp1d
f2 = interp1d(eta,fprime,kind='cubic')


def blasius(x,U,p,q):
    # q = 0
    return U*f2(p*(x+q))
# %%
from scipy.optimize import curve_fit

# xtmp = x[0,:]
# %%
k = 6
xtmp = np.flip(w1 - x[0,x[0,:]<w1])
ytmp = np.flip(-v[:,k,x[0,:]<w1].mean(axis=0))

# plt.plot(xtmp,ytmp,'o')
# plt.ylim([0,350])

st = 1
popt,pcov = curve_fit(blasius,xtmp[st:],ytmp[st:],bounds=([450,1,-50],[650,10,50]))

import numpy as np
plt.plot(xtmp[st:],ytmp[st:],'o')
xtmp2 = np.linspace(-popt[2],14,100)
plt.plot(xtmp2,blasius(xtmp2,*popt))
plt.ylim([0,popt[0]*1.2])
plt.text(0.5,20,'delta: %.2f mm'%popt[2])
plt.text(10,popt[0],'U_inf: %.2f mm/s'%popt[0])
plt.text(8,150,'x = %.2f mm'%(popt[0]/popt[1]**2))
