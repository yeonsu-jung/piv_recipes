# %%
import numpy as np
import os
import re

# %%
class path_class:
    def __init__(self, folder_path,results_folder_path):               
    
        self.path = folder_path
        rel_rpath = re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",self.path)[0] # + version
        self.results_path = os.path.join(os.path.normpath(results_folder_path), rel_rpath)
        print(self.results_path)
        
        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            pass

        self.get_image_dirs()

    def listdir(self):
        for s in os.listdir(self.path):
            print(s)

    def get_image_dirs(self):
        self.image_dirs = [x for x in os.listdir(self.path) if not x.startswith('.') and not x.startswith('_')]

    def choose_by_path(self,path):

        chosen_path = [x for x in self.image_dirs if path == x]         
        assert len(chosen_path) == 1, "multiple possibilities"

        return chosen_path        


# %%
folder_path = os.path.join('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/')
results_folder_path = '/Users/yeonsu/Documents/piv-results'

ins = path_class(folder_path,results_folder_path)
# %%
ins.choose_by_path('Flat_10 (black)_motor25_particle4_hori1280_laser1-4_nd0p7')

# %%
