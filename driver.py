# %%
from re import T
import piv_class as pi
from importlib import reload

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/'
ins = pi.piv_class(parent_path)

# %%
os.listdir(parent_path)

# %%
d = ins.quick_piv(index = 100)
# %%
ins.piv_over_time(start_index=3,N=1)

# %%
os.listdir(parent_path)

# %%
lst = [
 'Flat_10 (black)_motor10_particle4_hori1920_laser1-4_nd0p7',
 'Flat_10 (black)_motor10_particle4_hori1920_laser1-4_nd0p7_RE',
 'Flat_10 (black)_motor10_particle4_hori640_laser1-4_nd0p7',
 'Flat_10 (black)_motor10_particle4_hori800_laser1-4_nd0p7',
 'Flat_10 (black)_motor25_particle4_hori1024_laser1-4_nd0p7',
 'Flat_10 (black)_motor25_particle4_hori1280_laser1-4_nd0p7',
 'Flat_10 (black)_motor25_particle4_hori1408_laser1-4_nd0p7',
 'Flat_10 (black)_motor25_particle4_hori1600_laser1-4_nd0p7',
 'Flat_10 (black)_motor25_particle4_hori1920_laser1-4_nd0p7']
# %%
for pth in lst:  
    ins.piv_over_time(path=pth,start_index=3,N=95)

# %%
ins.piv_over_time(start_index=3,N=95)

# %%

os.path.join('a','b') + 'c'
# %%
import time
t = time.time()
ins.piv_over_sample(3,95)
el = time.time() - t
print('Elapsed time: %.2f sec'%el)
# %%

ins.piv_over_time(start_index=3,N=95)
# %%
for x in os.listdir(parent_path):
    if x.startswith('vid'):
        continue
    if x.startswith('Flat_10 (black)_stitching process'):
        continue
    if x.startswith('Flat_10 (black)_stitching process_cropped'):
        continue
    print(x)

# %%
for x in os.listdir(parent_path):
    if x.startswith('vid'):
        continue
    if x.startswith('Flat_10 (black)_stitching process'):
        continue
    if x.startswith('Flat_10 (black)_stitching process_cropped'):
        continue
    print(x)

    ins.piv_over_time(path = x,start_index=3,N=95)

# %%



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

path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor25_particle4_hori1280_laser1-4_nd0p7/[0,0,810,0]_[32,26,40]/series_003_95'
foo(path)



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