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
