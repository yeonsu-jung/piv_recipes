# %%
import numpy as np
import os

# %%
class save_results():
    def __init__(self,data_path,results_path):
        os.listdir(data_path)


# %%
common_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data'
folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-02-06/Blockage_lower_level'

results_folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-results/'

# norm_path = os.path.normpath(path)
# %%
norm_common_path = os.path.normpath(common_path)
norm_folder_path = os.path.normpath(folder_path)
norm_results_folder_path = os.path.normpath(results_folder_path)
# %%


# %%
os.path.join(results_folder_path,os.path.relpath(folder_path,common_path))
# %%
os.path.split(folder_path)

# %%
import re

a = re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",folder_path)
print(*a)
# %%
os.path.join(results_folder_path,*a)


# %%
