# %%
import numpy as np
import os
import re
import yaml

# %%
class path_class:
    def __init__(self, parent_path):               
    
        self.path = parent_path
        rel_rpath = re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",self.path)[0] # + version
        self.path_list_for_stitching = []

        with open('path_setting.yml') as f:
            path_setting = yaml.safe_load(f)           
        
        try:
            self.results_path = os.path.join(os.path.normpath(path_setting['result_path'][0]), rel_rpath)
        except:
            self.results_path = os.path.join(os.path.normpath(path_setting['result_path'][1]), rel_rpath)

        print(self.results_path)
        
        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            pass

        self.get_image_dirs()
        self.parse_folder_name()
        self.check_stitching_possibility()        

    def listdir(self):
        for s in os.listdir(self.path):
            print(s)

    def get_image_dirs(self):
        self.image_dirs = [x for x in os.listdir(self.path) if not x.startswith('.') and not x.startswith('_')]

    def show_image_dirs(self):
        i = 0
        for x in self.image_dirs:
            print(i,':',x)
            i = i + 1

    def choose_by_path(self,path):

        chosen_path = [x for x in self.image_dirs if path == x]         
        assert len(chosen_path) == 1, "multiple possibilities"

        return chosen_path       

    def set_param_string_list(self,new_param_string_list):
        self.param_string_list = new_param_string_list        
        self.param_dict_list = []

        for x in self.param_string_list:
            self.param_dict_list.append(self.param_string_to_dictionary(x))       

    def parse_folder_name(self):
        self.parameter_list = []
        for pstr in self.image_dirs:

            param_dict = {'path': pstr}
            pstr = pstr.replace('.tiff',"")

            running_parameter = re.findall("_[a-z]+[0-9]+[.]*[0-9]*", pstr, re.IGNORECASE)
            date_parameter = re.findall("_\[.*?\]",pstr)
            sample_parameter = pstr.replace("img_","")

            if date_parameter:
                sample_parameter = pstr.replace(date_parameter[0],"")

            for k in running_parameter:
                sample_parameter = sample_parameter.replace(k,"")

            sample_parameter = sample_parameter.replace('_01-18',"")

            param_dict.update({'sample': sample_parameter})
            for k in running_parameter:
                kk = re.findall('[a-x]+', k,re.IGNORECASE)
                vv = re.findall('[0-9]+[.]*[0-9]*', k,re.IGNORECASE)
                param_dict[kk[0]] = float(vv[0])

            self.parameter_list.append(param_dict)               

    def check_stitching_possibility(self):
        # check if every folder has appropriate pos and VOFFSET
        self.path_list_for_stitching = []
        try:
            self.parameter_list = sorted(self.parameter_list,key=lambda d: (d['pos'],d['VOFFSET']))

            # to do: raise warning if pos voffset are wrong
            # for elm in self.parameter_list:
            #     pos = elm['pos']
            #     voffset = elm['VOFFSET']

            def check_missing_numbers(lst):
                for x in range(lst[0], lst[-1]+1):
                    if x not in lst:                    
                        msg = "\033[1m" + 'The parent folder cannot generate stitched total image.' + "\033[0m"
                        print(msg)
                        return False           
            
            pos_list = []
            voffset_list = []

            [pos_list.append(int(x['pos'])) for x in self.parameter_list if x['pos'] not in pos_list]
            [voffset_list.append(int(x['VOFFSET'])) for x in self.parameter_list if x['VOFFSET'] not in voffset_list]            

            check_missing_numbers(pos_list)
            check_missing_numbers(np.array(voffset_list)//voffset_list[1])
            
        except KeyError:

            msg = "\033[1m" + 'The parent folder cannot generate stitched total image.' + "\033[0m"
            print(msg)
            return False

        msg = "\033[1m" + 'Parameter list is sorted for stitching!' + "\033[0m"
        print(msg)
        return True
        
    # def get_stitching_lists(self):
    #     assert self.check_stitching_possibility(), "Can't stitch images."

    #     pos_list = []
    #     voffset_list = []
    #     path_list = []

    #     [pos_list.append(int(x['pos'])) for x in self.parameter_list if x['pos'] not in pos_list]
    #     [voffset_list.append(int(x['VOFFSET'])) for x in self.parameter_list if x['VOFFSET'] not in voffset_list]            

    def set_piv_list(self,exp_cond_dict):        
        self.piv_dict_list = [x for x in self.param_dict_list if exp_cond_dict.items() <= x.items()]
        self.search_dict_list = self.check_piv_dict_list()       
        
# %%

# parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-08/Flat_10 (black)_motor10_stitching'

# ins = path_class(parent_path)
# ins.parameter_list
# # %%
# parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-08'
# ins = path_class(parent_path)

# ins.parameter_list

# # %%
# parent_path = os.path.join('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/')
# results_folder_path = '/Users/yeonsu/Documents/piv-results'

# ins = path_class(parent_path,results_folder_path)
# # %%
# ins.choose_by_path('Flat_10 (black)_motor25_particle4_hori1280_laser1-4_nd0p7')

# # %%
