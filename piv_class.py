# %%
from openpiv import tools, process, validation, filters, scaling, pyprocess
import yaml
import os
import imageio as io
import numpy as np
import path_class as path
from matplotlib import pyplot as plt
from argparse import Namespace
from scipy import ndimage
from importlib import reload
import datetime
import shutil
from PIL import Image

reload(path)
# %%
class piv_class():
    def __init__(self,parent_path):        
        self.update_piv_param()
        self.parent_path = parent_path       
        self.path_handler = path.path_class(self.parent_path)

    def update_piv_param(self):
        with open('piv_setting.yaml') as f:
            self.piv_param = yaml.safe_load(f)           

        print("- PIV parameters -")
        for x, y in self.piv_param.items():
            print(x +":", y)
        
    def read_image(self,path,index=1):
        assert isinstance(index,int), "Frame index should be of int type."
        
        file_a_path = os.path.join(self.parent_path,path,'frame_%06d.tiff' %index)        
        img_a = io.imread(file_a_path)  
        
        print('Read image from:', file_a_path)

        return img_a

    def run_piv(self, path, index = 100):
        self.update_piv_param()
        ns = Namespace(**self.piv_param)
        
        figure_export_folder = '%d_%d_%d'%(ns.winsize,ns.overlap,ns.searchsize)
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        setting_path = os.path.join(self.path_handler.results_path,figure_export_folder,'piv_setting_%s.yaml'%time_stamp)
        shutil.copy('piv_setting.yaml',setting_path)

        try:
                os.makedirs(os.path.join(self.path_handler.results_path,figure_export_folder))
        except FileExistsError:
            pass

        img_a = self.read_image(path,index)
        img_b = self.read_image(path,index+1)
                
        img_a = ndimage.rotate(img_a, ns.rotate)
        img_b = ndimage.rotate(img_b, ns.rotate)

        img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        
            
        u0, v0, sig2noise = process.extended_search_area_piv(img_a.astype(np.int32),
                                                            img_b.astype(np.int32),
                                                            window_size=ns.winsize,
                                                            overlap=ns.overlap, 
                                                            dt=ns.dt, 
                                                            search_area_size=ns.searchsize, 
                                                            sig2noise_method='peak2peak')

        x, y = process.get_coordinates(image_size=img_a.shape, 
                                        window_size=ns.winsize,                                    
                                        overlap=ns.overlap)

        x, y, u0, v0 = scaling.uniform(x, y, u0, v0, scaling_factor = ns.pixel_density) # no. pixel per distance

        u0, v0, mask = validation.global_val(u0,v0,ns.u_bound,ns.v_bound)        

        # x,y,u0,v0,mask,sig2noise = peel_off_edges((x,y,u0,v0,mask,sig2noise))

        print('Number of invalid vectors:',np.sum(np.isnan(mask)))

        u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                                sig2noise, 
                                                threshold = ns.sn_threshold)

        u3, v3 = filters.replace_outliers( u1, v1,
                                        method='localmean',
                                        max_iter=50,
                                        kernel_size=1)
        
        # if ns.check_angle:
        #     # u3, v3 = angle_mean_check(u3,v3)           
        #     u3, v3 = correct_by_angle(u3,v3)

        #save in the simple ASCII table format        
        txt_path = os.path.join(self.path_handler.results_path,figure_export_folder,'Stream_%05d_%s.txt'%(index,time_stamp))
        img_path = os.path.join(self.path_handler.results_path,figure_export_folder,'Stream_%05d_%s.png'%(index,time_stamp))
        if ns.save_result is True:            
            tools.save(x, y, u3, v3, mask, txt_path)
            io.imwrite(img_path,img_a)

            # quiver_and_contour(x,y,u3,v3,index,self.path_handler.results_path,show_result = ns.show_result)       
               

        if ns.show_result == True:
            fig, ax = plt.subplots(figsize=(24,12))
            tools.display_vector_field( txt_path, 
                                        ax=ax,
                                        scaling_factor= ns.pixel_density, 
                                        scale=ns.scale_factor, # scale defines here the arrow length
                                        width=ns.arrow_width, # width is the thickness of the arrow
                                        on_img=True, # overlay on the image
                                        image_name= img_path)
            fig.savefig(os.path.join(self.path_handler.results_path,figure_export_folder))                    
        
        print('Mean of u: %.3f' %np.mean(u3))
        print('Std of u: %.3f' %np.std(u3))        
        print('Mean of v: %.3f' %np.mean(v3))
        print('Std of v: %.3f' %np.std(v3))

        # output = np.array([np.mean(u3),np.std(u3),np.mean(v3),np.std(v3)])
        # if np.absolute(np.mean(v3)) < 50:
        #     output = self.quick_piv(search_dict,index_a = index_a + 1, index_b = index_b + 1)

        return x,y,u3,v3

    def quick_piv(self,index = 1):
        self.run_piv(self.path_handler.image_dirs[0],index = index)

    def stitch_images(self, step, update = False, index = 1):
        entire_image_path = os.path.join(self.path_handler.results_path,'_entire_image.png')        

        assert self.path_handler.check_stitching_possibility(), "Can't stitch images."            

        try:
            if update == True:
                raise FileNotFoundError
            else:
                im = Image.open(entire_image_path)
        except FileNotFoundError:            

            sorted_parameters = self.path_handler.parameter_list

            pos_list = [int(x['pos']) for x in sorted_parameters]
            voffset_list = [int(x['VOFFSET']) for x in sorted_parameters]
            path_list = [x['path'] for x in sorted_parameters]

            print(pos_list)
            print(voffset_list)
            # print(path_list)

            img_a = self.read_image(path_list[0], index = 10)            
            num_row, num_col = img_a.T.shape            

            num_pos = len(pos_list)
            num_voffset = len(voffset_list)
            voffset_unit = voffset_list[1]

            used_sensor_size = num_voffset * num_col - (num_voffset - 1) * (num_col - voffset_unit)
            overlap = int(used_sensor_size - step * self.piv_param["pixel_density"])                                  

            num_entire_col = voffset_unit*num_voffset*num_pos + (num_col-voffset_unit) - overlap * (num_pos - 1)
            entire_image = np.zeros((num_row,num_entire_col))

            print("entire image shape:",entire_image.shape)

            i = 0
            for path in path_list:
                img_a = self.read_image(path,index=index)
                pos = pos_list[i]
                voffset = voffset_list[i]

                xl = int(pos-pos_list[0]) * num_voffset * voffset_unit + int(voffset) - overlap * (int(pos) - 1)
                xr = xl + num_col

                print(xl,xr)
                print(img_a.T.shape)
                print(entire_image[:,xl:xr].shape)
                entire_image[:,xl:xr] = img_a.T
                i = i + 1

            # for pos in pos_list:
            #     for voffset in voffset_list:
            #         sd = {'pos': pos, 'VOFFSET': voffset}
            #         img_a, img_b = self.read_two_images(sd)

            #         img_a = self.read_image(path_list)

            #         xl = int(pos-pos_list[0]) * num_voffset * voffset_unit + int(voffset) - overlap * (int(pos) - 1)
            #         xr = xl + num_col

            #         print(xl,xr)
            #         print(img_a.T.shape)
            #         print(entire_image[:,xl:xr].shape)
            #         entire_image[:,xl:xr] = img_a.T

            io.imwrite(entire_image_path,entire_image)
            im = Image.open(entire_image_path)
            # im.show(entire_image_path)

            im.show(entire_image_path)           

        return im
        


# # %%
# folder_path = os.path.join('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/')
# pi = piv_class(folder_path)
# # %%
# pi.path_handler.show_image_dirs()
# # %%
# img_a = pi.read_image(pi.path_handler.image_dirs[2],100)
# plt.imshow(img_a)
# # %%
# pi.quick_piv()
