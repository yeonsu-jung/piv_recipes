# %%
import numpy as np
import os
import re
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from importlib import reload
# %%
# import matplotlib
# reload(matplotlib)
# matplotlib.get_backend()
# %%
# import matplotlib
# matplotlib.use('agg')
# matplotlib.get_backend()

# %%
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')
out_path.is_dir()

# %%
matches = out_path.glob("**/u.txt")
matches_list = sorted(matches)

# %%
for pth in matches_list:
    if '2021-04-06' in str(pth):
        print(pth)

# %%
matches_partial_list = [x for x in matches_list if '2021-04-06' in str(x)]
len(matches_partial_list)

# %%
def load_nd_array(path):
    with open(path, 'r') as file:
        temp = file.readline()            
        a = re.findall("\d+",temp)            

        ar = np.loadtxt(path)

        no_slices = ar.shape[0]//int(a[0])
        field_shape = (no_slices,int(a[0]),int(a[1]))
        
        ar = ar.reshape(field_shape)        
    return ar

# %%
# matplotlib.use('Agg')
for pth in matches_partial_list:
    setting_path = os.path.join(pth.parent,'piv_setting.yaml')
    with open(setting_path) as f:
        piv_setting = yaml.safe_load(f)
    
    xpath = os.path.join(pth.parent,'x.txt')
    ypath = os.path.join(pth.parent,'x.txt')
    upath = os.path.join(pth.parent,'u.txt')
    vpath = os.path.join(pth.parent,'v.txt')

    x_data = np.loadtxt(xpath)
    y_data = np.loadtxt(ypath)
    u_data = load_nd_array(upath)
    v_data = load_nd_array(vpath)

    w1,w2 = np.loadtxt(pth.parents[2].joinpath('wall_a_position.txt'))

    pix_den = piv_setting['pixel_density']

    w1 = w1/pix_den
    w2 = w2/pix_den
   
    xtmp = x_data[0,:]

    k = 15
    fig = plt.figure(figsize=(15,5))    
    plt.subplot(1,2,1)
    for i in range(u_data.shape[0]):
        plt.plot(-v_data[i,k,xtmp<w1],x_data[0,xtmp<w1],'.')
        
    plt.plot(-v_data[:,k,xtmp<w1].mean(axis=0),x_data[0,xtmp<w1],'k-',linewidth=2)
    # plt.title(str(x.parent).replace(out_path,''))
    left, right = plt.xlim()

    plt.plot([0, right],[w1,w1],'k--')    
    plt.xlim([0, right])

    plt.subplot(1,2,2)
    for i in range(u_data.shape[0]):
        plt.plot(-v_data[i,k,xtmp>w2],x_data[0,xtmp>w2],'.')

    plt.plot(-v_data[:,k,xtmp>w2].mean(axis=0),x_data[0,xtmp>w2],'k-',linewidth=2)
    plt.plot([0,right],[w2,w2],'k--')
    plt.xlim([0,right])       
    
    plt.savefig(pth.parent.joinpath('raw+mean.png'))
    plt.close()


# %%
matches = out_path.glob("**/u.txt")
matches_list = sorted(matches)

# %%
for pth in matches_list:
    if '2021-04-06' in str(pth) and 'Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95' in str(pth):
        print(pth)

# %%
matches_partial_list = [x for x in matches_list if '2021-04-06' in str(x)]
len(matches_partial_list)

# %%
# matches_partial_list = [Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95/u.txt')]

for pth in matches_list:
    setting_path = os.path.join(pth.parent,'piv_setting.yaml')
    with open(setting_path) as f:
        piv_setting = yaml.safe_load(f)
    
    xpath = os.path.join(pth.parent,'x.txt')
    ypath = os.path.join(pth.parent,'y.txt')
    upath = os.path.join(pth.parent,'u.txt')
    vpath = os.path.join(pth.parent,'v.txt')

    x_data = np.loadtxt(xpath)
    y_data = np.loadtxt(ypath)
    u_data = load_nd_array(upath)
    v_data = load_nd_array(vpath)

    u_mean = np.mean(u_data,axis=0)
    v_mean = np.mean(v_data,axis=0)

    u_sd = np.std(u_data,axis = 0, ddof=1)
    v_sd = np.std(v_data,axis = 0, ddof=1)

    u_se = u_sd / np.sqrt(u_sd.shape[0])
    v_se = v_sd / np.sqrt(v_sd.shape[0])

    np.savetxt(pth.parent.joinpath('u_mean.txt'),u_mean)
    np.savetxt(pth.parent.joinpath('v_mean.txt'),v_mean)
    np.savetxt(pth.parent.joinpath('u_sd.txt'),u_sd)
    np.savetxt(pth.parent.joinpath('v_sd.txt'),v_sd)
    np.savetxt(pth.parent.joinpath('u_se.txt'),u_se)
    np.savetxt(pth.parent.joinpath('v_se.txt'),v_se)
    
# %%
import falkner_skan
reload(falkner_skan)
from falkner_skan import falkner_skan

# %%
falkner_skan(-0.02)
# %%
def fs_query(eta_q,delta_eta,m,U0,p):
    eta,f0,f1,f2 = falkner_skan(m)    
    return np.interp(eta_q,p*(eta-delta_eta),f1)*U0
# %%
def fs_wall_shear(m):
    eta,f0,f1,f2 = falkner_skan(m)
    return f2[0]

m = 0
f20 = fs_wall_shear(m)
f20*1e-3*((m+1)/2)**0.5*(346**3/1e-6/50)**0.5*1e-3
# %%
def falkner_scan_fitting(xx,yy,ee,img_path):
    _p0 = (0,0,600,10)
    _bounds = ((-1,-0.08,200,0.1),(1,3,1200,100))
    
    popt,pcov = curve_fit(fs_query,xx,yy,p0=_p0,bounds=_bounds)
    perr = np.sqrt(np.diag(pcov))
    rel_err = perr/popt
        
    nu = 1e-6
    L = 100
    U0 = popt[2]
    m = popt[1]

    x = (2*nu*L/U0/(m+1)*L**(m-1)/popt[3]**2*1e6)**(1/(m-1))
    x_err = x*(rel_err[0] + rel_err[2] + 2*rel_err[3])

    delta_y = popt[0]*popt[3]
    delta_y_err = perr[0]*popt[3]

    f20 = fs_wall_shear(m)
    tau = f20*1e-3*((m+1)/2)**0.5*(346**3/1e-6/50)**0.5*1e-3
    tau_err = tau*(rel_err[1]*0.5 + rel_err[2]*1.5 + x_err/x*0.5)

    xtmp2 = np.linspace(-delta_y,np.max(xtmp),100)

    plt.errorbar(xx,yy,ee,fmt='o',capsize=4)
    plt.plot(xtmp2,fs_query(xtmp2,*popt))

    plt.ylim((0,popt[2]*1.2))

    left, right = plt.xlim()
    bottom, top = plt.ylim()

    plt.title('%d,%d,%d,%d_%d,%d,%d,%d-%d,%d,%d,%d'%(*_p0,*_bounds[0],*_bounds[1]))
    plt.text(0.5*(right-left),0.5*top,'m = %.2f $\pm$ %.3f'%(popt[0],perr[0]))
    plt.text(0.5*(right-left),0.4*top,'U0 = %.2f $\pm$ %.3f mm/s'%(popt[2],perr[2]))
    plt.text(0.5*(right-left),0.3*top,'x = %.2f $\pm$ %.3f mm'%(x,x_err))
    plt.text(0.5*(right-left),0.2*top,'$\Delta$ y = %.2f $\pm$ %.3f mm'%(delta_y,delta_y_err))
    plt.text(0.5*(right-left),0.1*top,'$\\tau$ = %.2f $\pm$ %.3f mPa'%(1e3*tau,1e3*tau_err))

    plt.xlabel('y (mm)')
    plt.ylabel('u (mm)')

    plt.savefig(img_path)
    plt.close()
# %%
lst = [Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor25_particle4_hori1408_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95/u.txt')]

matches_partial_list = []

for pth in matches_list:
    if '2021-04-06' in str(pth) and '003_95' in str(pth):

        numbers = re.findall('(\d+),(\d+),(\d+),(\d+)',str(pth))

        assert len(numbers) == 1
        pos = (int(numbers[0][0]),int(numbers[0][1]),int(numbers[0][2]),int(numbers[0][3]))

        print(pth)

        if pos == (0,0,0,0):
            matches_partial_list.append(pth)
            print(pth)      
        
# %%


# %%
print(len(matches_partial_list))

for pth in matches_partial_list:
    print(pth.parent.parent.parent.name)

# %%
for pth in matches_partial_list[9:10]:
    xpath = os.path.join(pth.parent,'x.txt')
    ypath = os.path.join(pth.parent,'y.txt')
    upath = os.path.join(pth.parent,'u.txt')
    vpath = os.path.join(pth.parent,'v.txt')

    w1,w2 = np.loadtxt(pth.parents[2].joinpath('wall_a_position.txt'))

    x_data = np.loadtxt(xpath)
    y_data = np.loadtxt(ypath)
    u_data = load_nd_array(upath)
    v_data = load_nd_array(vpath)

    u_mean = np.mean(u_data,axis=0)
    v_mean = np.mean(v_data,axis=0)

    u_sd = np.std(u_data,axis = 0, ddof=1)
    v_sd = np.std(v_data,axis = 0, ddof=1)

    u_se = u_sd / np.sqrt(u_sd.shape[0])
    v_se = v_sd / np.sqrt(v_sd.shape[0])

    w1_index = int((w1-26)//(32-26))
    w2_index = int((w2-26)//(32-26))

    delta = 0
    starting_index = w1_index - delta

    x1 = -x_data[0,:starting_index]+x_data[0,starting_index]
    y1 = -v_mean[0,:starting_index]
    e1 = v_se[0,:starting_index]

    falkner_scan_fitting(x1,y1,e1,pth.parent.joinpath('fitting_back_%d.png'%starting_index))

    delta = 4
    starting_index = w2_index + delta

    x2 = x_data[0,starting_index:]-x_data[0,starting_index]
    y2 = -v_mean[0,starting_index:]
    e2 = v_se[0,starting_index:]

    falkner_scan_fitting(x2,y2,e2,pth.parent.joinpath('fitting_front_%d.png'%starting_index))
# %%
pth

# w2_index
# %%
y_data.shape



# %%
x2 = x_data[0,starting_index:]-x_data[0,starting_index]
y2 = -v_mean[0,starting_index:]
e2 = v_se[0,starting_index:]

plt.plot(x2,y2,'o')

x2.shape


# %%
falkner_scan_fitting(x1,y1,e1,pth.parent.joinpath('fitting_back.png'))
# %%
pth = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor25_particle4_hori1408_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95/u.txt')

xpath = os.path.join(pth.parent,'x.txt')
ypath = os.path.join(pth.parent,'y.txt')
upath = os.path.join(pth.parent,'u.txt')
vpath = os.path.join(pth.parent,'v.txt')

w1,w2 = np.loadtxt(pth.parents[2].joinpath('wall_a_position.txt'))

x_data = np.loadtxt(xpath)
y_data = np.loadtxt(ypath)
u_data = load_nd_array(upath)
v_data = load_nd_array(vpath)

u_mean = np.mean(u_data,axis=0)
v_mean = np.mean(v_data,axis=0)

u_sd = np.std(u_data,axis = 0, ddof=1)
v_sd = np.std(v_data,axis = 0, ddof=1)

u_se = u_sd / np.sqrt(u_sd.shape[0])
v_se = v_sd / np.sqrt(v_sd.shape[0])

w1_index = int((w1-26)//(32-26))
w2_index = int((w2-26)//(32-26))

w1_index,w2_index
# %%
starting_index = w1_index

xtmp = -x_data[0,:starting_index]+x_data[0,starting_index]
ytmp = -v_mean[0,:starting_index]
etmp = v_se[0,:starting_index]

# plt.errorbar(ytmp,xtmp,xerr=etmp,fmt='o',capsize=3)
# %%




# %%

popt,pcov = curve_fit(fs_query,xtmp,ytmp,p0=(0,0,800,10),bounds=((-1,-3,600,0.1),(1,3,950,100)))
perr = np.sqrt(np.diag(pcov))
rel_err = perr/popt

nu = 1e-6
L = 100
U0 = popt[2]
m = popt[1]
pth = (2*nu*L/U0/(m+1)*L**(m-1)/popt[3]**2)*1e6
x_err = pth*(rel_err[0] + rel_err[2] + 2*rel_err[3])
delta_y = popt[0]*popt[3]
delta_y_err = perr[0]*popt[3]

# %%
xtmp2 = np.linspace(-delta_y,np.max(xtmp),100)

plt.errorbar(xtmp,ytmp,etmp,fmt='o',capsize=4)
plt.plot(xtmp2,fs_query(xtmp2,*popt))

plt.ylim((0,popt[2]*1.2))

left, right = plt.xlim()
bottom, top = plt.ylim()

plt.text(0.5*right,0.5*top,'m = %.2f $\pm$ %.2f'%(popt[0],perr[0]))
plt.text(0.5*right,0.5*top-100,'U0 = %.2f $\pm$ %.2f mm/s'%(popt[2],perr[2]))
plt.text(0.5*right,0.5*top-200,'x = %.2f $\pm$ %.2f mm'%(pth,x_err))
plt.text(0.5*right,0.5*top-300,'delta y = %.2f $\pm$ %.2f mm'%(delta_y,delta_y_err))

plt.xlabel('y (mm)')
plt.ylabel('u (mm)')

# %%


# %%


# %%
# plt.plot(v_mean[0,w2_index:],'o-')



# %%
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.errorbar(x_data[0,:],-v_mean[0,:],tst[0,:],capsize=5,fmt='o-')
plt.subplot(2,1,2)
plt.errorbar(x_data[0,:],-v_mean[0,:],-v_sd[0,:]/np.sqrt(95),capsize=5,fmt='o-')

# %%
rel_path = '2021-04-06/Flat_10 (black)_motor25_particle4_hori1408_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95'
pth = out_path.joinpath(rel_path)
for pth in pth.iterdir():
    print(pth)

# %%
matches = out_path.glob('**/u.txt')
matches = sorted(matches)

for pth in matches:
    if '2021-04-06' in str(pth):
        print(pth)

# %%
pth.parents[2]

# %%
u_data.shape

# %%