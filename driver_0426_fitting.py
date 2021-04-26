# %%
import piv_class as pi
from importlib import reload
from matplotlib import pyplot as plt
import numpy as np

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06'
parent_path = parent_path.replace('C:/Users/yj/','/Users/yeonsu/')

ins = pi.piv_class(parent_path)
# %%
os.listdir(parent_path)

# %%
for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue    
    # print(x)
    ins.get_wall_position(path=x)

# %%
os.listdir(parent_path)
ins.path.results_path

# %%
from pathlib import Path
import re
import yaml

def foo(data_path):    
    with open('path_setting.yml') as f:
        path_setting = yaml.safe_load(f)   

    rel_rpath = re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",data_path)[0] # + version        
    par_rpath = path_setting['result_path'][0]
    result_path = os.path.join(par_rpath, rel_rpath)
    
    return result_path
    
result_path = foo(parent_path)


# %%
os.listdir(result_path)

# %%
# '0,0,0,0_32,26,50'
# 'series_003_95'


# %%
for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue
    if 'pos1_' in x:
        continue
    if 'pos2_' in x:
        continue
    if 'pos12_' in x:
        continue
    print(x)

# %%
for x in os.listdir(parent_path):
    if x.startswith('.'):
        continue
    if 'pos1_' in x:
        continue
    if 'pos2_' in x:
        continue
    if 'pos12_' in x:
        continue
    # print(x)
    
    ins.get_wall_position(path=x)


# %%
from pathlib import Path
import yaml

ipth = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-08/Flat_10 (black)_motor10_stitching/Flat_10 (black)_timing400_ag1_dg1_laser1_shutter100_motor10.00_pos3_[04-08]_VOFFSET0/0,0,0,0_32,26,50/series_003_95'
w1,w2 = np.loadtxt(Path(ipth).resolve().parents[1].joinpath('wall_a_position.txt'))
w1, w2

with open(Path(ipth).joinpath('piv_setting.yaml')) as f:
    piv_param = yaml.safe_load(f)

pix_den = piv_param["pixel_density"]

w1 = w1/pix_den
w2 = w2/pix_den
w1,w2
# %%



# %%
from pathlib import Path

data_parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/'
os.listdir(data_parent_path)

result_parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/'
date = '2021-04-08'

date_abspath = os.path.join(data_parent_path,date)
p = Path(date_abspath)

[x.name for x in p.iterdir() if x.is_dir()]


for x in p.iterdir():
    if x.is_dir() and x.joinpath('frame_000001.tiff').is_file():
        # print(x.name)
        print(x)        
        print(len([y for y in x.iterdir() if 'frame' in y.name]))
    
    elif x.is_dir():
        # print(x)
        xx = Path(x)

        for k in xx.iterdir():
            print(foo(str(k)))
# %%
date_result_abspath = foo(date_abspath)
date_result_abspath

pr = Path(date_result_abspath)

for x in pr.iterdir():
    if x.is_dir():
        print(x) # image dir or collection
    
# %%
for x in p.iterdir():

    if x.is_dir() and x.joinpath('frame_000001.tiff').is_file(): # image dir
        # print(x.name)
        # print(foo(str(x)))
        
        res_p = Path(foo(str(x)))
        print(res_p.resolve())

# %%
for x in p.iterdir():    
    if x.is_dir(): # collection        
        num = 0

        for k in x.iterdir():              
            if k.is_dir():
                # print(foo(str(k)))
                res_p = Path(foo(str(k)))
                print(res_p.resolve())
                num = num + 1



        print(num)   
# %%
result_parent_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/'
rp = Path(result_parent_path)
# %%
for x in rp.iterdir():
    print(x)

# %%

matches = Path(result_parent_path).glob("**/u.txt")
matches_list = sorted(matches)

# %%
matches_list

# %%
for x in matches_list:
    if '2021-04-06' in str(x):
        print(x)




# %%
# back up
for x in matches_list:
    if x.parents[2].joinpath('wall_a_position.txt').is_file():
        print(x.parents[2])

# %%
# possible collection
exp = 'Flat_10 (black)_motor10_particle3'

data_path_tuple = (data_parent_path,date,exp)
result_path_tuple = (result_parent_path,date,exp)

data_full_path = os.path.join(*data_path_tuple)
result_full_path = os.path.join(*result_path_tuple)

dp = Path(data_full_path)
rp = Path(result_full_path)

crop_win = [str(f.name) for f in p.iterdir() if f.is_dir()]
crop_win

# %%

piv_instance = pi.piv_class()
# %%
[x for x in dp.iterdir()]

# %%
[x for x in p.iterdir()]




# %%
result_path_tuple + ('test',)


# %%
os.scandir(result_full_path)


# %%





# %%
dirs = filter(os.path.isdir, os.listdir(result_full_path))

for x in dirs:
    print(x)

# %%

def analyze_a_folder(pth1,pth2):
    # pth1 should include x,y,u,v txt files and piv_setting.yaml
    # .../{exp cond}/{crop-win info}/series%03d_%d(possible numbering)/
    # 