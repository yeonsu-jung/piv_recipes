# %%
import openpiv_recipes
from openpiv_recipes import ParticleImage
import importlib
# %%
importlib.reload(openpiv_recipes)
# %%
folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-20'
pi = ParticleImage(folder_path)
# %%
pi.open_two_images(1,1)
# %%

pi.quick_piv(1, 1)
# %%

pi.quick_piv(5, 1)
# %%
