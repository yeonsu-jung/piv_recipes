# %%
import openpiv_recipes as piv
import importlib
importlib.reload(piv)

import imageio as io
# %%


img_a = io.imread('C:\\Users\\yj\\Downloads\\dg_4_laser_10_frame_000013.tiff')
img_b = io.imread('C:\\Users\\yj\\Downloads\\dg_4_laser_10_frame_000014.tiff')

piv.run_piv(img_a,img_b)



# %%
