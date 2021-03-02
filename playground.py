# %%
import openpiv_recipes as piv
import importlib
importlib.reload(piv)

import imageio as io
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser5_motor0_frame_000011.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser5_motor0_frame_000013.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    scale_factor=1e4),


# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser5_motor0_frame_000011.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser5_motor0_frame_000012.tiff')


piv.run_piv(img_a,img_b,
    winsize = 32, # pixels, interrogation window size in frame A
    searchsize = 40,  # pixels, search in image B
    overlap = 16,
    scale_factor=1e4),

# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_frame_000014.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_frame_000016.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    scale_factor=1e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_frame_000014.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_frame_000015.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    scale_factor=1e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_particleAdded_2_frame_000032.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_particleAdded_2_frame_000034.tiff')


piv.run_piv(img_a,img_b,
    winsize = 32, # pixels, interrogation window size in frame A
    searchsize = 40,  # pixels, search in image B
    overlap = 16,
    scale_factor=1e3),
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded_frame_000008.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded_frame_000009.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    scale_factor=1e4),
# %%


img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded2_2_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded2_2_frame_000022.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    scale_factor=1e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded2_2_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor10_particleAdded2_2_frame_000022.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,200],
    scale_factor=3e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_particleAdded2_frame_000023.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\_test_ag1_dg1_laser10_motor0_particleAdded2_frame_000024.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [10,1000],
    v_bounds = [-20000,200],
    scale_factor=3e4),

# %%
from openpiv import pyprocess

folder_path = '/Volumes/Backup Plus /ROWLAND/piv-data/2021-02-23/'
results_folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results'

# folder_path = "C:\\Users\\yj\\Documents\\Chronos\\2021-02-22\\1_1_-1 (3dp)"
# results_folder_path = 'C:\\Users\\yj\\Dropbox (Harvard University)\\Riblet\\data\\piv-results'

pi = piv.ParticleImage(folder_path,results_folder_path)

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# search_dict =  {'pos': 7, 'VOFFSET': 0}
search_dict = {'path': '1_1_-1_10 (3dp)__motor10.00_pos7_VOFFSET0_ag1_dg1_laser10_added_more_particle_[02-23]'}

small = {"winsize": 16, "searchsize": 20, "overlap": 8,}
middle = {"winsize": 32, "searchsize": 36, "overlap": 16}

scales = {"pixel_density": 45.75,"scale_factor": 9e3, "arrow_width": 0.001}
bounds = {"u_upper_bound": 200,"u_lower_bound": -200,"v_upper_bound": 0,"v_lower_bound": -500}
tr = {"transpose": False}
ic = {"show_result": True}
crop = {"crop": [0,0,0,398]}
th = {"sn_threshold": 1.3}

piv_cond = merge_dicts(small,
                       scales,bounds,tr,ic,crop,th)
pi.set_piv_param(piv_cond)

xyuv = pi.quick_piv(search_dict,index_a = 100, index_b = 101)
# %%
img_a,img_b = pi.read_two_images(search_dict)

window_a = pyprocess.moving_window_array(img_a,20,8)
window_b = pyprocess.moving_window_array(img_b,20,8)

win_a = window_a[200,:,:]
win_b = window_b[200,:,:]

corr = pyprocess.fft_correlate_strided_images(win_a,win_b,correlation_method="circular",normalized_correlation=False)
print(corr.shape)

# a,b = pyprocess.find_first_peak(corr)
s2n =  pyprocess.sig2noise_ratio(corr,sig2noise_method="peak2peak",width=2)
