# %%
import numpy as np
import os
import re
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
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
plt.plot([1,1],[2,2])

# %%
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')
out_path.is_dir()

# %%
matches = out_path.glob("**/u.txt")
matches_list = sorted(matches)

# %%
for x in matches_list:
    if '2021-04-06' in str(x):
        print(x)

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
for x in matches_partial_list:
    setting_path = os.path.join(x.parent,'piv_setting.yaml')
    with open(setting_path) as f:
        piv_setting = yaml.safe_load(f)
    
    xpath = os.path.join(x.parent,'x.txt')
    ypath = os.path.join(x.parent,'x.txt')
    upath = os.path.join(x.parent,'u.txt')
    vpath = os.path.join(x.parent,'v.txt')

    x_data = np.loadtxt(xpath)
    y_data = np.loadtxt(ypath)
    u_data = load_nd_array(upath)
    v_data = load_nd_array(vpath)

    w1,w2 = np.loadtxt(x.parents[2].joinpath('wall_a_position.txt'))

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
    
    plt.savefig(x.parent.joinpath('raw+mean.png'))
    plt.close()

# %%


# %%
matches = out_path.glob("**/u.txt")
matches_list = sorted(matches)

# %%
for x in matches_list:
    if '2021-04-06' in str(x) and 'Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95' in str(x):
        print(x)

# %%
matches_partial_list = [x for x in matches_list if '2021-04-06' in str(x)]
len(matches_partial_list)

# %%
# matches_partial_list = [Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95/u.txt')]

for x in matches_list:
    setting_path = os.path.join(x.parent,'piv_setting.yaml')
    with open(setting_path) as f:
        piv_setting = yaml.safe_load(f)
    
    xpath = os.path.join(x.parent,'x.txt')
    ypath = os.path.join(x.parent,'y.txt')
    upath = os.path.join(x.parent,'u.txt')
    vpath = os.path.join(x.parent,'v.txt')

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

    np.savetxt(x.parent.joinpath('u_mean.txt'),u_mean)
    np.savetxt(x.parent.joinpath('v_mean.txt'),v_mean)
    np.savetxt(x.parent.joinpath('u_sd.txt'),u_sd)
    np.savetxt(x.parent.joinpath('v_sd.txt'),v_sd)
    np.savetxt(x.parent.joinpath('u_se.txt'),u_se)
    np.savetxt(x.parent.joinpath('v_se.txt'),v_se)
    

# %%


# %%
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.errorbar(x_data[0,:],-v_mean[0,:],tst[0,:],capsize=5,fmt='o-')
plt.subplot(2,1,2)
plt.errorbar(x_data[0,:],-v_mean[0,:],-v_sd[0,:]/np.sqrt(95),capsize=5,fmt='o-')

# %%
rel_path = '2021-04-06/Flat_10 (black)_motor25_particle4_hori1408_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95'
pth = out_path.joinpath(rel_path)
for x in pth.iterdir():
    print(x)

# %%
matches = out_path.glob('**/u.txt')
matches = sorted(matches)

for x in matches:
    if '2021-04-06' in str(x):
        print(x)

# %%
x.parents[2]

# %%
u_data.shape

# %%