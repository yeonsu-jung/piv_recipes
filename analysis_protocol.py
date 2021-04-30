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
import matplotlib
matplotlib.use('agg')
matplotlib.get_backend()
# %%
plt.plot([1,1],[2,2])

# %%
out_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/'

# %%
matches = Path(out_path).glob("**/u.txt")
matches_list = sorted(matches)

# %%
for x in matches_list:
    if '2021-04-07' in str(x):
        print(x)

# %%
matches_partial_list = [x for x in matches_list if '2021-04-07' in str(x)]
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
    plt.title(str(x.parent).replace(out_path,''))
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
    plt.title(str(x.parent).replace(out_path,''))
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
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')
# %%
matches = out_path.glob('**/raw+mean.png')
matches = sorted(matches)

# %%
for x in matches:
    print(x)



# %%
x.parents[2]

# %%
u_data.shape

# %%