# %%
import numpy as np
from pathlib import Path
import re
import datetime
import yaml
import os
import shutil
import matplotlib.image as mpimg
from argparse import Namespace
from openpiv import tools, process, validation, filters, scaling, pyprocess
from time import time

from piv_class import run_piv
from importlib import reload


# %%
base_path = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/')

# %%
base_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')

# %%
matches = out_path.glob('**/series_003_95/')
matches = sorted(matches)
# %%
for subd in out_path.iterdir():
    # print(subd.name)
    #     
    if not subd.is_dir():
        continue

    y,m,d = re.findall('\d+',subd.name)
    d0 = datetime.datetime(2021,3,29)
    d = datetime.datetime(int(y),int(m),int(d))

    if d < d0:
        continue

    
# %%
date = '2021-04-06'


# %%
def base_to_out(path_in):    
    assert isinstance(path_in,Path)        
    return out_path.joinpath(path_in.relative_to(base_path))

def out_to_base(path_in):    
    assert isinstance(path_in,Path)        
    return base_path.joinpath(path_in.relative_to(out_path))

# %%
active_paths = []

for subd in base_path.iterdir():
    if not subd.is_dir():
        continue

    y,m,d = re.findall('\d+',subd.name)
    d0 = datetime.datetime(2021,3,29)
    d = datetime.datetime(int(y),int(m),int(d))

    if d < d0:
        continue

    print(subd.name)
    active_paths.append(subd)

# %%
all_paths = []
for apth in active_paths:
    # print(sorted(apth.iterdir()))

    first_image_path_list = apth.glob('**/frame_000001.tiff')
    for first_image in first_image_path_list:
        containing_dir = first_image.parent
        # print(containing_dir)
        all_paths.append(containing_dir)


# %%
already_analyzed_paths = []
for containing_dir in all_paths:
    u_path_list = base_to_out(containing_dir).glob('**/u.txt')

    for u in u_path_list:
        u_containing_dir = u.parent            

        num_frames = int(re.findall('\d+_(\d+)',str(u_containing_dir.name))[0])
        print(num_frames)
        if num_frames > 60:
            already_analyzed_paths.append(containing_dir)

       
# %%
not_analyzed_paths = [x for x in all_paths if not x in already_analyzed_paths]
# %%
len(not_analyzed_paths)


# %%
not_analyzed_paths_set = set(not_analyzed_paths)
contains_duplicates = len(not_analyzed_paths) != len(not_analyzed_paths_set)

contains_duplicates
# %%
len(all_paths), len(already_analyzed_paths), len(not_analyzed_paths)

# %%
with open('not_yet_analyzed.txt','w') as f:
    for x in not_analyzed_paths:
        f.write(str(x)+'\n')

# %%
[x for x in already_analyzed_paths if not x in all_paths]
[x for x in not_analyzed_paths if not x in all_paths]
# %%
[x for x in not_analyzed_paths if x in already_analyzed_paths]
# %%
tmpth = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-30/Laser1_Timing300/Flat_10 (black)_timing300_ag1_dg1_laser1_motor25.00_pos1_[03-30]_VOFFSET210')
parent_name = str(tmpth.parent.parent.name)

matched = re.match('\d{4}-\d{2}-\d{2}', parent_name)
bool(matched)
# %%
def check_proper_index(image_dir_path,index,piv_setting_path):   

    with open(piv_setting_path) as f:
        piv_param = yaml.safe_load(f)                

    ns = Namespace(**piv_param)

    path_a = image_dir_path.joinpath('frame_%06d.tiff'%index)
    path_b = image_dir_path.joinpath('frame_%06d.tiff'%index)
    path_c = image_dir_path.joinpath('frame_%06d.tiff'%index)

    img_a = mpimg.imread(path_a)
    img_b = mpimg.imread(path_b)
    img_c = mpimg.imread(path_c)

    img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
    img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        
    img_c = img_c[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
    
    corr1 = pyprocess.correlate_windows(img_a,img_b)
    corr2 = pyprocess.correlate_windows(img_b,img_c)

    if np.max(corr1) > np.max(corr2):
        out = index
    else:
        out = index + 1
    return out
# %%
def check_piv_setting_file(image_dir_path = None):
    assert image_dir_path != None
    assert isinstance(image_dir_path,Path)    
    
    current_path = image_dir_path

    num_img = len([x for x in image_dir_path.glob('frame_*.tiff')])           
    assert(num_img > 10)

    while True:
        try:
            piv_setting_path = current_path.joinpath('_piv_setting.yaml')
            # print(piv_setting_path)
            with open(piv_setting_path) as f:
                piv_param = yaml.safe_load(f)                       
                return True             
            
        except:
            if bool(re.match('\d{4}-\d{2}-\d{2}',str(current_path.name))):
                break          

            current_path = current_path.parent            
            # print(current_path.name)

    return False

tmpth = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-30/Laser1_Timing300/Flat_10 (black)_timing300_ag1_dg1_laser1_motor25.00_pos1_[03-30]_VOFFSET210')
tmpth = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-30/Laser1_Timing300/Flat_10 (black)_timing300_ag1_dg1_laser1_motor25.00_pos1_[03-30]_VOFFSET210')

tmpth = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-30/Laser1_Timing300/Flat_10 (black)_timing300_ag1_dg1_laser1_motor25.00_pos1_[03-30]_VOFFSET210')
check_piv_setting_file(tmpth)
# %%
not_analyzed_paths = [x for x in all_paths if not x in already_analyzed_paths]

for pth in not_analyzed_paths:    
    if not check_piv_setting_file(pth):
        print(pth)


# %%

def piv_over_time(image_dir_path = None,start_index=3,N=2):
    assert image_dir_path != None
    assert isinstance(image_dir_path,Path)
    
    t = time()    
    imdir_out_path = base_to_out(image_dir_path)
    current_path = image_dir_path

    num_img = len([x for x in image_dir_path.glob('frame_*.tiff')])
    assert(num_img>10) # raise what error, to continue in a loop

    start_index = 3
    N = num_img//2-5
           
    while True:
        try:
            piv_setting_path = current_path.joinpath('_piv_setting.yaml')
            # print(piv_setting_path)
            with open(piv_setting_path) as f:
                piv_param = yaml.safe_load(f)            
                break
            
        except:
            if bool(re.match('\d{4}-\d{2}-\d{2}',str(current_path.name))):
                break          

            current_path = current_path.parent            
            # print(current_path.name)

        # if bool(re.match('\d{4}-\d{2}-\d{2}',str(current_path.name))):
        #     break          

    # piv_param = update_piv_param(setting_file=self.piv_setting_path)
    ns = Namespace(**piv_param)
    relative_path = '%d,%d,%d,%d_%d,%d,%d'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
        
    # relative_path = '[%d,%d,%d,%d]_[%d,%d,%d]'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
    # relative_path = get_rel_path(piv_setting_path=self.piv_setting_path)

    # full_path = os.path.join(results_path,relative_path,'series_%03d_%d'%(start_index,N))
    cropwin_path = imdir_out_path.joinpath(relative_path)
    series_path = cropwin_path.joinpath('series_%03d_%d'%(start_index,N))

    try:
        os.makedirs(series_path)
    except:
        lst = [x for x in os.listdir(cropwin_path) if 'series_%03d_%d'%(start_index,N) in x]
        
        numbering = []
        for pth in lst:
            num = re.findall('\((\d)\)',pth)
            if num == []:
                numbering.append(1)
            else:
                numbering.append(int( num[0] ))

        new_no = max(numbering) + 1

        #full_path = os.path.join(results_path,relative_path,'series_%03d_%d(%d)'%(start_index,N,new_no))
        series_path = cropwin_path.joinpath('series_%03d_%d(%d)'%(start_index,N,new_no))
        os.makedirs(series_path)
        
    shutil.copy(piv_setting_path,series_path.joinpath('piv_setting.yaml')) # to be replaced with class member
    # time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    x_path = os.path.join(series_path, 'x.txt')
    y_path = os.path.join(series_path, 'y.txt')
    u_path = os.path.join(series_path, 'u.txt')
    v_path = os.path.join(series_path, 'v.txt')

    # x,y,U,V = quick_piv(path=image_dir_path,index = start_index,mute=True)

    path_a = image_dir_path.joinpath('frame_%06d.tiff'%start_index)
    path_b = image_dir_path.joinpath('frame_%06d.tiff'%(start_index+1))
    
    x,y,U,V = run_piv(path_a,path_b, export_parent_path = imdir_out_path, piv_setting_path = piv_setting_path, mute = True)

    np.savetxt(x_path,x)
    np.savetxt(y_path,y)

    with open(u_path, 'w') as uf, open(v_path, 'w') as vf:          
        uf.write('# Array shape: {0}\n'.format(U.shape))
        vf.write('# Array shape: {0}\n'.format(U.shape))                       
    
    ind = check_proper_index(image_dir_path,start_index,piv_setting_path)

    for i in range(N):

        path_a = image_dir_path.joinpath('frame_%06d.tiff'%ind)
        path_b = image_dir_path.joinpath('frame_%06d.tiff'%(ind+1))
    
        x,y,U,V = run_piv(path_a,path_b, export_parent_path = imdir_out_path, piv_setting_path = piv_setting_path, mute = True)
        # x,y,U,V = quick_piv(path=image_dir_path,index = ind,mute=True)

        with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
            np.savetxt(uf,U,fmt='%-7.5f')
            np.savetxt(vf,V,fmt='%-7.5f')

        ind = ind + 2

    elapsed = time() - t
    print('PIV over time has been done. Elapsed time is %.2f'%elapsed)
# %%
tmpth = Path(r'C:\Users\yj\Dropbox (Harvard University)\Riblet\data\piv-data\2021-03-31\PBS160_ND0p3_motor25')
piv_over_time(image_dir_path=tmpth,start_index=3,N=1)


# %%
t = time()
not_analyzed_paths = [x for x in all_paths if not x in already_analyzed_paths]

for pth in not_analyzed_paths:
    print(pth.name)
    piv_over_time(pth,start_index=3,N=95)

el = time()-t
print('Total elapsed time: %.4f sec'% el)
# %%

     

# %%
len(matches)


# %%
base_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')

# %%
tmpth = base_path.joinpath('2021-04-06/Flat_10 (black)_motor10_particle4_hori1024_laser1-4_nd0p7')
# %%
tmpth.is_dir()
# %%
piv_over_time(image_dir_path=tmpth,start_index=3,N=1)

