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
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\2021_03_01_Exp2_frame_000020.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\2021_03_01_Exp2_frame_000021.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [10,1000],
    v_bounds = [-20000,200],
    scale_factor=3e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\2021-03-01_Exp3_frame_000015.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\2021-03-01_Exp3_frame_000016.tiff')

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-20000,0],
    scale_factor=3e3),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\2021-03-01_Exp8_frame_000005.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\2021-03-01_Exp8_frame_000007.tiff')

img_a = img_a[:,:250]
img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-20000,0],
    scale_factor=3e3),
# %%

img_a = io.imread('D:\\2021-03-01_Exp12\\frame_000029.tiff')
img_b = io.imread('D:\\2021-03-01_Exp12\\frame_000030.tiff')

img_a = img_a[:,:250]
img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-20000,0],
    scale_factor=3e3),
# %%
img_a = io.imread('D:\\2021-03-01_Exp12\\frame_000029.tiff')
img_b = io.imread('D:\\2021-03-01_Exp12\\frame_000030.tiff')

img_a = img_a[:,:250]
img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-20000,0],
    scale_factor=3e3)
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\2021_03_02_Exp1_frame_000018.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\2021_03_02_Exp1_frame_000019.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-20000,0],
    scale_factor=3e3),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\2021-03-02_Exp2_frame_000034.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\2021-03-02_Exp2_frame_000035.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-20000,0],
    scale_factor=3e3),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp6_frame_000015.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp6_frame_000016.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=5e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp7_frame_000023.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp7_frame_000024.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=5e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exo8_frame_000023.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exo8_frame_000024.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a.T,img_b.T,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [0,2000],
    v_bounds = [-100,100],
    scale_factor=1e4),
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp11_frame_000028.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp11_frame_000029.tiff')

img_a = img_a[:,:250]
img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%


img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp13_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp13_frame_000022.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_exp12_frame_000013.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_exp12_frame_000014.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp14_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp14_frame_000022.tiff')

# img_a = img_a[:,:250]
# img_b = img_b[:,:250]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp15_frame_000023.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp15_frame_000024.tiff')

img_a = img_a[:,400:]
img_b = img_b[:,400:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp16_frame_000017.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp16_frame_000018.tiff')

img_a = img_a[:,400:]
img_b = img_b[:,400:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,10],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%


img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp17_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp17_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4),
# %%
from openpiv import pyprocess
from matplotlib import pyplot as plt


win_a = pyprocess.moving_window_array(img_a,20,8)
win_b = pyprocess.moving_window_array(img_b,20,8)

fig,ax = plt.subplots(2,figsize=(10,10))
ax[0].imshow(win_a[50,:,:])
ax[1].imshow(win_b[50,:,:])
# %%
def check_windows(img_a,img_b,no):
    win_a = pyprocess.moving_window_array(img_a,20,8)
    win_b = pyprocess.moving_window_array(img_b,20,8)

    fig,ax = plt.subplots(2,figsize=(10,10))
    ax[0].imshow(win_a[no,:,:])
    ax[1].imshow(win_b[no,:,:])


# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp19_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp19_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4),

check_windows(img_a,img_b,15)
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp20_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp20_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-10,50],
    v_bounds = [-2000,0],
    scale_factor=1e4),

check_windows(img_a,img_b,15)
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp21_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp21_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4),

check_windows(img_a,img_b,15)
# %%


img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp23_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp23_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4),

check_windows(img_a,img_b,15)
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp25_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp25_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4),

check_windows(img_a,img_b,15)
# %%
img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp26_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp26_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4,
    dt = 0.0004),

check_windows(img_a,img_b,15)
# %%

img_a = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp27_frame_000021.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\0302_Exp27_frame_000022.tiff')

img_a = img_a[:,415:]
img_b = img_b[:,415:]

piv.run_piv(img_a,img_b,
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8,
    u_bounds = [-100,100],
    v_bounds = [-2000,0],
    scale_factor=1e4,
    dt = 0.0004),

check_windows(img_a,img_b,15)
# %%
