# %%
import numpy as np
import os
import re
import yaml
import matplotlib.image as mpimg

from matplotlib import pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import argrelmax

# from medpy.filter.smoothing import anisotropic_diffusion

from importlib import reload

from falkner_skan import falkner_skan


# %%
base_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')
base_path.is_dir(),out_path.is_dir()

# %%
def base_to_out(path_in):
    assert isinstance(path_in,Path)        
    return out_path.joinpath(path_in.relative_to(base_path))

def out_to_base(path_in):
    assert isinstance(path_in,Path)
    return base_path.joinpath(path_in.relative_to(out_path))

def load_nd_array(path):
    with open(path, 'r') as file:
        temp = file.readline()            
        a = re.findall("\d+",temp)            

        ar = np.loadtxt(path)

        no_slices = ar.shape[0]//int(a[0])
        field_shape = (no_slices,int(a[0]),int(a[1]))
        
        ar = ar.reshape(field_shape)        
    return ar

def get_wall_pos2(img, check_img = False, check_plots = False): # img or img path
    num_col = img.shape[1]
    row_averaged = np.zeros(num_col)
    for i in range(num_col):
        row_averaged[i] = np.sum(img[:,i])    

    peaks, _ = find_peaks(row_averaged,distance=100)
    peaks = sorted(peaks,key = lambda x: -row_averaged[x])
    
    idx1 = min(peaks[0],peaks[1])
    idx2 = max(peaks[0],peaks[1])

    w1_new = np.zeros(img.shape,dtype=bool)
    w2_new = np.zeros(img.shape,dtype=bool)

    w1_new[:,idx1] = 1    
    w2_new[:,idx2] = 1

    if check_img:
        row_dim = img.shape[0]
        plt.imshow(img)
        plt.plot([idx1,idx1],[0,row_dim],'r-',linewidth=0.5)
        plt.plot([idx2,idx2],[0,row_dim],'r-',linewidth=0.5)
        plt.show()

    if check_plots:        
        plt.plot(row_averaged,'o-')        
        # plt.plot(row_averaged[peaks],'o-')
        plt.show()

    return idx1, idx2, w1_new, w2_new

def fs_query(eta_q,delta_eta,m,U0,p):
    eta,f0,f1,f2 = falkner_skan(m)    
    return np.interp(eta_q,p*(eta-delta_eta),f1)*U0

# %%

def get_wall_pos3(img, check_img = False, check_plots = False):    
    def foo(arr):
        idx = np.argsort(2*arr[1:-1] - arr[:-2] - arr[2:])        
        return idx[-1]+1, idx[-2]+1

    num_col = img.shape[1]
    row_averaged = np.zeros(num_col)
    for i in range(num_col):
        row_averaged[i] = np.sum(img[:,i])  

    center = np.argmin(row_averaged)

    # peaks = argrelmax(row_averaged,order = 10)

    max_loc = argrelmax(row_averaged,order = 10)        

    # plt.plot(row_averaged,'o-')
    # plt.plot(*max_loc,row_averaged[max_loc[0]],'o-')        
    # plt.plot(row_averaged)
    # plt.show()
    
    a,b = foo(row_averaged[max_loc[0]])

    # plt.plot(max_loc[0][a], row_averaged[max_loc[0][a]],'o')
    # plt.plot(max_loc[0][b], row_averaged[max_loc[0][b]],'o')
    # plt.show()

    peaks = [max_loc[0][a] , max_loc[0][b]]

    # print(peaks)
    
    idx1 = min(peaks[0],peaks[1])
    idx2 = max(peaks[0],peaks[1])

    w1_new = np.zeros(img.shape,dtype=bool)
    w2_new = np.zeros(img.shape,dtype=bool)

    w1_new[:,idx1] = 1    
    w2_new[:,idx2] = 1

    if check_img:
        row_dim = img.shape[0]
        plt.imshow(img)
        plt.plot([idx1,idx1],[0,row_dim],'r-',linewidth=0.5)
        plt.plot([idx2,idx2],[0,row_dim],'r-',linewidth=0.5)
        plt.show()

    if check_plots:        
        plt.subplot(2,1,1)
        plt.plot(row_averaged,'o-')        
        bottom, top = plt.ylim()
        plt.plot([center,center],[bottom,top])        
        plt.plot(peaks,row_averaged[peaks],'o')        
        plt.show()

    return idx1, idx2, w1_new, w2_new
# %%
matches = out_path.glob("**/u.txt")
matches_list = sorted(matches)

# %% TOTAL LIST
containing_dir_list = []
for pth in matches_list:   
    info_crop = re.findall('(\d+),(\d+),(\d+),(\d+)',pth.parents[1].name)

    assert len(info_crop) == 1
    pos = (int(info_crop[0][0]),int(info_crop[0][1]),int(info_crop[0][2]),int(info_crop[0][3]))

    if pos == (0,0,0,0):
        info_img_pairs = re.findall('(\d+)_(\d+)',pth.parents[0].name)
        assert len(info_img_pairs) == 1

        start_index = int(info_img_pairs[0][0])
        num_img_pairs = int(info_img_pairs[0][1])

        if num_img_pairs > 60:
            containing_dir_list.append(pth.parent)

containing_dir_list

# %% SHORTLISTING
containing_dir_shortlist = []

keyword1 = '2021-04-07'
keyword2 = 'stitching'
exclude1 = 'qq'
exclude2 = 'cropped'
for pth in containing_dir_list:
    if keyword1 in str(pth) and keyword2 in str(pth) and not exclude1 in str(pth) and not exclude2 in str(pth):
        containing_dir_shortlist.append(pth)

len(containing_dir_shortlist)

# %%

img_dir_list = []
for pth in containing_dir_shortlist:
    img_dir_list.append(out_to_base(pth.parents[1]))

img_dir_list

# %%
ii = 0
for img_dir in img_dir_list:
    
    img_a = mpimg.imread(img_dir.joinpath('frame_000010.tiff'))
    img_b = mpimg.imread(img_dir.joinpath('frame_000011.tiff'))

    wa1_pos, wa2_pos, wa1, wa2 = get_wall_pos3(img_a, check_img=True, check_plots = False)
    wb1_pos, wb2_pos, wb1, wb2 = get_wall_pos3(img_b, check_img=True, check_plots = False)

    wpos_a_path = base_to_out(img_dir).joinpath('wall_a_position.txt')
    wpos_b_path = base_to_out(img_dir).joinpath('wall_b_position.txt')

    np.savetxt(wpos_a_path,[wa1_pos,wa2_pos],fmt='%d')
    np.savetxt(wpos_b_path,[wb1_pos,wb2_pos],fmt='%d')

    # if abs(wa1_pos - wb1_pos) > 3 or abs(wa2_pos - wb2_pos) > 3:
    #     print(img_dir.name)    

# %%
for pth in containing_dir_shortlist:
    setting_path = os.path.join(pth,'piv_setting.yaml')
    with open(setting_path) as f:
        piv_setting = yaml.safe_load(f)
    
    xpath = os.path.join(pth,'x.txt')
    ypath = os.path.join(pth,'y.txt')
    upath = os.path.join(pth,'u.txt')
    vpath = os.path.join(pth,'v.txt')

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

    np.savetxt(pth.joinpath('u_mean.txt'),u_mean)
    np.savetxt(pth.joinpath('v_mean.txt'),v_mean)
    np.savetxt(pth.joinpath('u_sd.txt'),u_sd)
    np.savetxt(pth.joinpath('v_sd.txt'),v_sd)
    np.savetxt(pth.joinpath('u_se.txt'),u_se)
    np.savetxt(pth.joinpath('v_se.txt'),v_se)

# %%
import logging
logger = logging.getLogger('piv')
hdlr = logging.FileHandler('piv.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def fs_wall_shear(m):
    eta,f0,f1,f2 = falkner_skan(m)
    return f2[0]

def falkner_scan_fitting(xx,yy,ee,img_path,_p0 = (0,0,300,10),_bounds = ((-10,-0.08,200,0.1),(10,3,1000,100))):    
    
    try:
        popt,pcov = curve_fit(fs_query,xx,yy,p0=_p0,bounds=_bounds)
    except RuntimeError as e:
        logger.error('RuntimeError in %s: '%str(img_path)+ str(e))
        return

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

    xtmp2 = np.linspace(-delta_y,np.max(xx),100)

    plt.errorbar(xx,yy,ee,fmt='o',capsize=4)

    left, right = plt.xlim()
    bottom, top = plt.ylim()

    plt.plot(xtmp2,fs_query(xtmp2,*popt))

    plt.ylim((0,popt[2]*1.2))    

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
containing_dir_shortlist = sorted(containing_dir_shortlist,key=lambda x: (int(re.findall('_pos(\d+)_',x.parents[1].name)[0]),int(re.findall('_VOFFSET(\d+)',x.parents[1].name)[0])))
containing_dir_shortlist
# %%
for ii,pth in enumerate(containing_dir_shortlist):
    print(ii, pth.parents[1].name)
# %%
for pth in containing_dir_shortlist[10:]:
    xpath = os.path.join(pth,'x.txt')
    ypath = os.path.join(pth,'y.txt')
    upath = os.path.join(pth,'u.txt')
    vpath = os.path.join(pth,'v.txt')

    w1,w2 = np.loadtxt(pth.parents[1].joinpath('wall_a_position.txt'))

    x_data = np.loadtxt(xpath)
    y_data = np.loadtxt(ypath)
    u_data = load_nd_array(upath)
    v_data = load_nd_array(vpath)
    
    with open(pth.joinpath('piv_setting.yaml')) as f:
        piv_param = yaml.safe_load(f)

    u_mean = np.mean(u_data,axis=0)
    v_mean = np.mean(v_data,axis=0)

    u_sd = np.std(u_data,axis = 0, ddof=1)
    v_sd = np.std(v_data,axis = 0, ddof=1)

    u_se = u_sd / np.sqrt(u_sd.shape[0])
    v_se = v_sd / np.sqrt(v_sd.shape[0])

    winsize = piv_param['winsize']
    overlap = piv_param['overlap']

    w1_index = int((w1-overlap)//(winsize-overlap))
    w2_index = int((w2-overlap)//(winsize-overlap))

    delta = 0
    starting_index1 = w1_index - delta

    delta = 4
    starting_index2 = w2_index + delta

    si_pth = pth.joinpath(f'starting_index_{starting_index1}_{starting_index2}')

    if not si_pth.is_dir():
        os.makedirs(si_pth)
    elif si_pth.is_dir():        
        m = si_pth.parent.glob(str(si_pth.name)+'*')
        m = sorted(m)
        for tmp in m:
            lst = []
            num = re.findall('\((\d)\)',tmp.name)    
            if num == []:
                lst.append(1)
            else:
                lst.append(int( num[0] ))
            new_no = sorted([x for x in range(lst[0], lst[-1]+2) if x not in lst])[0]
        si_pth = pth.joinpath(f'starting_index_{starting_index1}_{starting_index2} ({new_no})')        
        os.makedirs(si_pth)

    for k in range(x_data.shape[0]//5,4*x_data.shape[0]//5,10):
        x1 = -x_data[k,:starting_index1]+x_data[k,starting_index1]
        y1 = -v_mean[k,:starting_index1]
        e1 = v_se[k,:starting_index1]
        
        p0 = (0,0,400,10)
        bounds = ((-1,-0.08,200,0.1),(3,2,600,10))

        falkner_scan_fitting(x1,y1,e1,si_pth.joinpath('fitting_back_%03d.png'%k),_p0=p0,_bounds=bounds)

        x2 = x_data[k,starting_index2:]-x_data[k,starting_index2]
        y2 = -v_mean[k,starting_index2:]
        e2 = v_se[k,starting_index2:]

        falkner_scan_fitting(x2,y2,e2,si_pth.joinpath('fitting_front_%03d.png'%k),_p0=p0,_bounds=bounds)
# %%
out_to_base(pth)
# %%
print(si_pth)


# %%
for tst in range(x_data.shape[0]//5,4*x_data.shape[0]//5,10):
    print(tst)
# %%
starting_index1 = 1
'starting_index_{si}'.format(si=starting_index1)
# %%
f'starting_index_{starting_index1}'

# %%
'Hey %(name)s, there is a 0x%(errno)x error!' % {
...     "name": 'ddd', "errno": 2 }

# %%
 'Hey {name}, there is a 0x{errno:x} error!'.format(
...     name=name, errno=errno)
# %%
a=1
b=2

f'Five plus ten is {a + b} and not {2 * (a + b)}.'
# %%
print('''\
    A
    B
    C
    ''')