# %%
import numpy as np
from pathlib import Path
import re
import datetime


# %%
base_path = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/')

# %%
matches = out_path.glob('**/series_003_95/')
matches = sorted(matches)
# %%
for subd in out_path.iterdir():
    # print(subd.name)

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
        
# %%
not_analyzed_paths = [x for x in all_paths if not x in already_analyzed_paths]
# %%
with open('not_yet_analyzed.txt','w') as f:
    for x in not_analyzed_paths:
        f.write(str(x)+'\n')

# %%
len(all_paths), len(already_analyzed_paths), len(not_analyzed_paths)

# %%
495-174
# %%
[x for x in already_analyzed_paths if not x in all_paths]
[x for x in not_analyzed_paths if not x in all_paths]
# %%
[x for x in not_analyzed_paths if x in already_analyzed_paths]

# %%
from time import time

def piv_over_time(path = None,start_index=1,N=2):
    assert path != None
    assert isinstance(path,Path)
    
    t = time()    

    results_path = os.path.join(self.path.results_path,path)

    results_path = base_to_out(path)

    piv_setting_path = 

    piv_param = update_piv_param(setting_file=self.piv_setting_path)
    ns = Namespace(**piv_param)

    

    # relative_path = '[%d,%d,%d,%d]_[%d,%d,%d]'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
    relative_path = get_rel_path(piv_setting_path=self.piv_setting_path)
    full_path = os.path.join(results_path,relative_path,'series_%03d_%d'%(start_index,N))
    try:
        os.makedirs(full_path)
    except:
        lst = [x for x in os.listdir(os.path.join(results_path,relative_path)) if 'series_%03d_%d'%(start_index,N) in x]
        numbering = []
        for pth in lst:
            num = re.findall('\((\d)\)',pth)
            if num == []:
                numbering.append(1)
            else:
                numbering.append(int( num[0] ))

        new_no = max(numbering) + 1

        full_path = os.path.join(results_path,relative_path,'series_%03d_%d(%d)'%(start_index,N,new_no))
        os.makedirs(full_path)

        pass

    shutil.copy('piv_setting.yaml',os.path.join(full_path,'piv_setting.yaml')) # to be replaced with class member

    # time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    x_path = os.path.join(full_path, 'x.txt')
    y_path = os.path.join(full_path, 'y.txt')
    u_path = os.path.join(full_path, 'u.txt')
    v_path = os.path.join(full_path, 'v.txt')

    x,y,U,V = self.quick_piv(path=path,index = start_index,mute=True)

    np.savetxt(x_path,x)
    np.savetxt(y_path,y)

    with open(u_path, 'w') as uf, open(v_path, 'w') as vf:          
        uf.write('# Array shape: {0}\n'.format(U.shape))
        vf.write('# Array shape: {0}\n'.format(U.shape))                       
    
    ind = self.check_proper_index(path,index = start_index)

    for i in range(N):
        x,y,U,V = self.quick_piv(path=path,index = ind,mute=True)

        with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
            np.savetxt(uf,U,fmt='%-7.5f')
            np.savetxt(vf,V,fmt='%-7.5f')

        ind = ind + 2

    elapsed = time() - t
    print('PIV over time has been done. Elapsed time is %.2f'%elapsed)




# %%
    

    


# %%

     

# %%
len(matches)


# %%
