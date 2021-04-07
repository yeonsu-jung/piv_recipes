# to do
# - merge dict
# - sig2noise, image quality, filtering , ...

# %%
from openpiv import tools, process, validation, filters, scaling, pyprocess
from PIL import Image
from argparse import Namespace
import matplotlib.cm as cm
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import imageio as io
import re
import os
import sys

# %%
class ParticleImage:    
    def __init__(self, folder_path, results_folder_path, version = None):
        self.path = os.path.normpath(folder_path)
        if version is not None:
            version = '_' + version
        elif version is None:
            version = ''
            
        rel_rpath = re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",folder_path)[0] + version
        print('Result path:', rel_rpath)
        self.results_path = os.path.join(os.path.normpath(results_folder_path), rel_rpath)
        
        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            pass

        # self.param_string_list = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(folder_path,x)) and not x.startswith('_')]
        self.param_string_list = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(folder_path,x))]
        # temporary code here:        

        self.set_param_string_list(self.param_string_list)        
        
        self.piv_param = {
            "winsize": 48,
            "searchsize": 52,
            "overlap": 24,
            "dt": 0.0001,
            "image_check": False,    
            "show_vertical_profiles": False,            
            "figure_export_name": '_quick_piv.tiff',
            "text_export_name": '_quick_piv.txt',
            "scale_factor": 1,            
            "pixel_density": 40,
            "arrow_width": 0.02,
            "show_result": True,        
            "u_bound": [-2000,2000], # (mm/s)            
            "v_bound": [-2000,2000], # (mm/s)            
            "transpose": False,
            "crop": [0,0,0,0],
            "sn_threshold": 1.3,
            "rotate": 0,
            "save_result": True,
            "check_angle": False,
            "raw_or_cropped": True,
        }

        self.crop_info = {
            1: (0.5,460),
            2: (0.5,460),
            3: (0.5,450),
            4: (0.58,453),
            5: (0.45,446),
            6: (0.45,435)
        }

        self.piv_dict_list = self.param_dict_list
        try:
            self.search_dict_list = self.check_piv_dict_list()
        except:
            pass

    def set_param_string_list(self,new_param_string_list):
        self.param_string_list = new_param_string_list        
        self.param_dict_list = []

        for x in self.param_string_list:
            self.param_dict_list.append(self.param_string_to_dictionary(x))       

    def set_piv_list(self,exp_cond_dict):        
        self.piv_dict_list = [x for x in self.param_dict_list if exp_cond_dict.items() <= x.items()]
        self.search_dict_list = self.check_piv_dict_list()

    def param_string_to_dictionary(self,pstr):
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

        return param_dict

    def read_two_images(self,search_dict,index_a = 100,index_b = 101, open = False, raw=True):
        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        print('Read image from:', location_path[0])

        # file_a_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_a)
        # file_b_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_b)

        if raw is False:
            file_a_path = os.path.join(self.path,*location_path,'cropped_%06d.tiff' %index_a)
            file_b_path = os.path.join(self.path,*location_path,'cropped_%06d.tiff' %index_b)
        elif raw is True:
            file_a_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_a)
            file_b_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_b)

        try:
            img_a = Image.open(file_a_path)
            img_b = Image.open(file_b_path)
        except FileNotFoundError:
            print('No such file: ' + file_a_path)
            print('No such file: ' + file_b_path)
            return None

        if open == True:
            img_a.show()
            img_b.show()

        if self.piv_param['transpose'] == True:
            img_a_array = np.array(img_a).T
            img_b_array = np.array(img_b).T
        else:
            img_a_array = np.array(img_a)
            img_b_array = np.array(img_b)        

        return img_a_array, img_b_array

    def check_piv_dict_list(self):      
        self.piv_dict_list = sorted(self.piv_dict_list,key=lambda d: (d['pos'],d['VOFFSET']))
        lis = self.piv_dict_list
        search_dict_list = []
        for x in lis:
            search_dict = {'pos': x['pos'], 'VOFFSET': x['VOFFSET']}
            search_dict_list.append(search_dict)        

        self.search_dict_list = sorted(search_dict_list, key=lambda e: (e['pos'],e['VOFFSET']))        

    def show_piv_param(self):
        print("- PIV parameters -")
        for x, y in self.piv_param.items():
            print(x +":", y)

    def check_proper_index(self,search_dict,index_a):
        img_a, img_b = self.read_two_images(search_dict,index_a=index_a,index_b=index_a+1)
        img_b, img_c = self.read_two_images(search_dict,index_a=index_a+1,index_b=index_a+2)        

        ns = Namespace(**self.piv_param)

        img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        
        img_c = img_c[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        
        corr1 = pyprocess.correlate_windows(img_a,img_b)
        corr2 = pyprocess.correlate_windows(img_b,img_c)

        if np.max(corr1) > np.max(corr2):
            out = index_a
        else:
            out = index_a + 1
        return out

    def quick_piv(self, search_dict, index_a = 100, index_b = 101, folder = None):
        self.show_piv_param()
        ns = Namespace(**self.piv_param)

        if folder == None:
            img_a, img_b = self.read_two_images(search_dict,index_a=index_a,index_b=index_b,raw=ns.raw_or_cropped)

            location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
            results_path = os.path.join(self.results_path,*location_path)
            try:
                os.makedirs(results_path)
            except FileExistsError:
                pass
        else:
            try:
                file_a_path = os.path.join(self.path,folder,'frame_%06d.tiff' %index_a)
                file_b_path = os.path.join(self.path,folder,'frame_%06d.tiff' %index_b)

                img_a = np.array(Image.open(file_a_path))
                img_b = np.array(Image.open(file_b_path))                
            except:
                return None
        
        # crop
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

        x,y,u0,v0,mask,sig2noise = peel_off_edges((x,y,u0,v0,mask,sig2noise))       

        print('Number of invalid vectors:',np.sum(np.isnan(mask)))

        u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                                sig2noise, 
                                                threshold = ns.sn_threshold)

        u3, v3 = filters.replace_outliers( u1, v1,
                                        method='localmean',
                                        max_iter=50,
                                        kernel_size=1)
        
        if ns.check_angle:
            # u3, v3 = angle_mean_check(u3,v3)           
            u3, v3 = correct_by_angle(u3,v3)

        #save in the simple ASCII table format        
        if ns.save_result is True:
            tools.save(x, y, u3, v3, mask, os.path.join(results_path,'Stream_%05d.txt'%index_a))            
            io.imwrite(os.path.join(results_path,ns.figure_export_name),img_a)
            quiver_and_contour(x,y,u3,v3,index_a,results_path,show_result = ns.show_result)
        
        if ns.image_check == True:
            fig,ax = plt.subplots(2,1,figsize=(24,12))
            ax[0].imshow(img_a)
            ax[1].imshow(img_b)       

        if ns.show_result == True:
            fig, ax = plt.subplots(figsize=(24,12))
            tools.display_vector_field( os.path.join(results_path,'Stream_%05d.txt'%index_a), 
                                        ax=ax,
                                        scaling_factor= ns.pixel_density, 
                                        scale=ns.scale_factor, # scale defines here the arrow length
                                        width=ns.arrow_width, # width is the thickness of the arrow
                                        on_img=True, # overlay on the image
                                        image_name= os.path.join(results_path,ns.figure_export_name))
            fig.savefig(os.path.join(results_path,ns.figure_export_name))                    
        
        print('Mean of u: %.3f' %np.mean(u3))
        print('Std of u: %.3f' %np.std(u3))        
        print('Mean of v: %.3f' %np.mean(v3))
        print('Std of v: %.3f' %np.std(v3))

        output = np.array([np.mean(u3),np.std(u3),np.mean(v3),np.std(v3)])
        # if np.absolute(np.mean(v3)) < 50:
        #     output = self.quick_piv(search_dict,index_a = index_a + 1, index_b = index_b + 1)

        return x,y,u3,v3

        # return np.std(u3)        

    def get_entire_velocity_map(self,camera_step,start_index = 2):       
        x_path = os.path.join(self.results_path, 'entire_x_%03d.txt'%start_index)
        y_path = os.path.join(self.results_path, 'entire_y_%03d.txt'%start_index)
        u_path = os.path.join(self.results_path, 'entire_u_%03d.txt'%start_index)
        v_path = os.path.join(self.results_path, 'entire_v_%03d.txt'%start_index)

        for sd in self.search_dict_list:            
            self.set_piv_param({'save_result': True, 'show_result': False})            
            # angle,offset =  self.crop_info[sd['pos']]
            
            ind = self.check_proper_index(sd,index_a = start_index)
            x,y,u,v = self.quick_piv(sd,index_a = ind, index_b = ind + 1)
            y = y + camera_step * float(sd['pos']) + float(sd['VOFFSET'])/self.piv_param['pixel_density']
            self.set_piv_param({'show_result': True})

            try:
                entire_x = np.vstack((entire_x,x))
                entire_y = np.vstack((entire_y,y))
                entire_u = np.vstack((entire_u,u))
                entire_v = np.vstack((entire_v,v))
            except:
                entire_x = x
                entire_y = y
                entire_u = u
                entire_v = v
        
        np.savetxt(x_path,entire_x)
        np.savetxt(y_path,entire_y)
        np.savetxt(u_path,entire_u)
        np.savetxt(v_path,entire_v)        
        

    def get_entire_velocity_map_series(self,camera_step,start_index = 2,N = 10):
        for i in range(N):
            self.get_entire_velocity_map(camera_step,start_index = start_index+i)

    def average_velocity_series(self,start_index=10,N = 10):

        for i in range(N):            
            u_path = os.path.join(self.results_path, 'entire_u_%03d.txt'%(start_index+i))
            v_path = os.path.join(self.results_path, 'entire_v_%03d.txt'%(start_index+i))
        
            try:                
                entire_u_series = np.vstack(entire_u_series,np.loadtxt(u_path))
                entire_v_series = np.vstack(entire_v_series,np.loadtxt(v_path))
            except:                
                entire_u_series = np.loadtxt(u_path)
                entire_v_series = np.loadtxt(v_path)

        return (entire_u_series,entire_v_series)    

    def get_entire_avg_velocity_map(self,camera_step,s):
        lis = self.piv_dict_list    
        
        for pd in lis:
            u_path = os.path.join(self.results_path, pd['path'], 'u_tavg_%s.txt'%s)
            v_path = os.path.join(self.results_path, pd['path'], 'v_tavg_%s.txt'%s)

            us_path = os.path.join(self.results_path, pd['path'], 'u_tstd_%s.txt'%s)
            vs_path = os.path.join(self.results_path, pd['path'], 'v_tstd_%s.txt'%s)

            u_tavg = np.loadtxt(u_path)
            v_tavg = np.loadtxt(v_path)

            u_tstd = np.loadtxt(us_path)
            v_tstd = np.loadtxt(vs_path)
            
            x = np.loadtxt(os.path.join(self.results_path, pd['path'],'x.txt'))
            y = np.loadtxt(os.path.join(self.results_path, pd['path'],'y.txt'))
            y = y + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']

            try:
                entire_x = np.vstack((entire_x,x))
                entire_y = np.vstack((entire_y,y))
                entire_u_tavg = np.vstack((entire_u_tavg,u_tavg))
                entire_v_tavg = np.vstack((entire_v_tavg,v_tavg))
                entire_u_tstd = np.vstack((entire_u_tstd,u_tstd))
                entire_v_tstd = np.vstack((entire_v_tstd,v_tstd))
            except:
                entire_x = x
                entire_y = y
                entire_u_tavg = u_tavg
                entire_v_tavg = v_tavg
                entire_u_tstd = u_tstd
                entire_v_tstd = v_tstd

        np.savetxt(os.path.join(self.results_path,'entire_x.txt'),entire_x)
        np.savetxt(os.path.join(self.results_path,'entire_y.txt'),entire_y)
        np.savetxt(os.path.join(self.results_path,'entire_u_tavg.txt'),entire_u_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_v_tavg.txt'),entire_v_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_u_tstd.txt'),entire_u_tstd)
        np.savetxt(os.path.join(self.results_path,'entire_v_tstd.txt'),entire_v_tstd)

        return (entire_x,entire_y,entire_u_tavg,entire_v_tavg,entire_u_tstd,entire_v_tstd)

    def get_top_bottom_average_velocity_map(self,camera_step,s):
        lis = self.piv_dict_list    
        
        for pd in lis:
            uu_path = os.path.join(self.results_path, pd['path'], 'u_upper_tavg_%s.txt'%s)
            vu_path = os.path.join(self.results_path, pd['path'], 'v_upper_tavg_%s.txt'%s)

            uus_path = os.path.join(self.results_path, pd['path'], 'u_upper_tstd_%s.txt'%s)
            vus_path = os.path.join(self.results_path, pd['path'], 'v_upper_tstd_%s.txt'%s)

            ul_path = os.path.join(self.results_path, pd['path'], 'u_lower_tavg_%s.txt'%s)
            vl_path = os.path.join(self.results_path, pd['path'], 'v_lower_tavg_%s.txt'%s)

            uls_path = os.path.join(self.results_path, pd['path'], 'u_lower_tstd_%s.txt'%s)
            vls_path = os.path.join(self.results_path, pd['path'], 'v_lower_tstd_%s.txt'%s)

            uu_tavg = np.loadtxt(uu_path)
            vu_tavg = np.loadtxt(vu_path)

            uu_tstd = np.loadtxt(uus_path)
            vu_tstd = np.loadtxt(vus_path)

            ul_tavg = np.loadtxt(ul_path)
            vl_tavg = np.loadtxt(vl_path)

            ul_tstd = np.loadtxt(uls_path)
            vl_tstd = np.loadtxt(vls_path)
            
            xu = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_upper.txt'))
            yu = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_upper.txt'))
            yu = yu + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']

            xl = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_lower.txt'))
            yl = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_lower.txt'))
            yl = yl + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']

            try:
                entire_xu = np.vstack((entire_xu,xu))
                entire_yu = np.vstack((entire_yu,yu))

                entire_xl = np.vstack((entire_xl,xl))
                entire_yl = np.vstack((entire_yl,yl))

                entire_uu_tavg = np.vstack((entire_uu_tavg,uu_tavg))
                entire_vu_tavg = np.vstack((entire_vu_tavg,vu_tavg))
                entire_uu_tstd = np.vstack((entire_uu_tstd,uu_tstd))
                entire_vu_tstd = np.vstack((entire_vu_tstd,vu_tstd))

                entire_ul_tavg = np.vstack((entire_ul_tavg,ul_tavg))
                entire_vl_tavg = np.vstack((entire_vl_tavg,vl_tavg))
                entire_ul_tstd = np.vstack((entire_ul_tstd,ul_tstd))
                entire_vl_tstd = np.vstack((entire_vl_tstd,vl_tstd))
            except:
                entire_xu = xu
                entire_yu = yu

                entire_xl = xl
                entire_yl = yl

                entire_uu_tavg = uu_tavg
                entire_vu_tavg = vu_tavg
                entire_uu_tstd = uu_tstd
                entire_vu_tstd = vu_tstd

                entire_ul_tavg = ul_tavg
                entire_vl_tavg = vl_tavg
                entire_ul_tstd = ul_tstd
                entire_vl_tstd = vl_tstd

        np.savetxt(os.path.join(self.results_path,'entire_xu.txt'),entire_xu)
        np.savetxt(os.path.join(self.results_path,'entire_yu.txt'),entire_yu)
        np.savetxt(os.path.join(self.results_path,'entire_uu_tavg.txt'),entire_uu_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_vu_tavg.txt'),entire_vu_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_uu_tstd.txt'),entire_uu_tstd)
        np.savetxt(os.path.join(self.results_path,'entire_vu_tstd.txt'),entire_vu_tstd)

        np.savetxt(os.path.join(self.results_path,'entire_xl.txt'),entire_xl)
        np.savetxt(os.path.join(self.results_path,'entire_yl.txt'),entire_yl)
        np.savetxt(os.path.join(self.results_path,'entire_ul_tavg.txt'),entire_ul_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_vl_tavg.txt'),entire_vl_tavg)
        np.savetxt(os.path.join(self.results_path,'entire_ul_tstd.txt'),entire_ul_tstd)
        np.savetxt(os.path.join(self.results_path,'entire_vl_tstd.txt'),entire_vl_tstd)

        return (entire_xu,entire_yu,entire_uu_tavg,entire_vu_tavg,entire_uu_tstd,entire_vu_tstd,entire_xl,entire_yl,entire_ul_tavg,entire_vl_tavg,entire_ul_tstd,entire_vl_tstd)

    def get_left_right_velocity_map(self,s):
        sd_left = {'pos': 1, 'VOFFSET': 0}
        left_path = [x['path'] for x in self.piv_dict_list if sd_left.items() <= x.items()][0]

        sd_right = {'pos': 6, 'VOFFSET': 840}
        right_path = [x['path'] for x in self.piv_dict_list if sd_right.items() <= x.items()][0]               
        
        x_path = os.path.join(self.results_path,left_path,'x_full.txt')
        y_path = os.path.join(self.results_path,left_path,'y_full.txt')

        ul_path = os.path.join(self.results_path,left_path,'u_full_tavg_%s.txt'%s)
        vl_path = os.path.join(self.results_path,left_path,'v_full_tavg_%s.txt'%s)

        ur_path = os.path.join(self.results_path,right_path,'u_full_tavg_%s.txt'%s)
        vr_path = os.path.join(self.results_path,right_path,'v_full_tavg_%s.txt'%s)

        x = np.loadtxt(x_path)
        y = np.loadtxt(y_path)

        u_left = np.loadtxt(ul_path)
        v_left = np.loadtxt(vl_path)

        u_right = np.loadtxt(ur_path)
        v_right = np.loadtxt(vr_path)

        return x,y, u_left, v_left, u_right, v_right

    def get_left_right_velocity_map_series(self,s):
        sd_left = {'pos': 1, 'VOFFSET': 0}
        left_path = [x['path'] for x in self.piv_dict_list if sd_left.items() <= x.items()][0]

        sd_right = {'pos': 6, 'VOFFSET': 840}
        right_path = [x['path'] for x in self.piv_dict_list if sd_right.items() <= x.items()][0]               
        
        x_path = os.path.join(self.results_path,left_path,'x_full.txt')
        y_path = os.path.join(self.results_path,left_path,'y_full.txt')

        ul_path = os.path.join(self.results_path,left_path,'u_full_series_%s.txt'%s)
        vl_path = os.path.join(self.results_path,left_path,'v_full_series_%s.txt'%s)

        ur_path = os.path.join(self.results_path,right_path,'u_full_series_%s.txt'%s)
        vr_path = os.path.join(self.results_path,right_path,'v_full_series_%s.txt'%s)

        print(x_path)

        x = np.loadtxt(x_path)
        y = np.loadtxt(y_path)

        # u_left = np.loadtxt(ul_path)    
        # v_left = np.loadtxt(vl_path)
        # u_right = np.loadtxt(ur_path)
        # v_right = np.loadtxt(vr_path)

        u_left = load_nd_array(ul_path)
        v_left = load_nd_array(vl_path)
        u_right = load_nd_array(ur_path)
        v_right = load_nd_array(vr_path)

        return x,y, u_left, v_left, u_right, v_right
# %%
    def get_top_bottom_velocity_series(self,camera_step,s):
        lis = self.piv_dict_list    
        for pd in lis:        
            print(pd['path'])
            xu = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_upper.txt'))
            yu = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_upper.txt'))
            yu = yu + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']

            xl = np.loadtxt(os.path.join(self.results_path, pd['path'],'x_lower.txt'))
            yl = np.loadtxt(os.path.join(self.results_path, pd['path'],'y_lower.txt'))
            yl = yl + camera_step * float(pd['pos']) + float(pd['VOFFSET'])/self.piv_param['pixel_density']
            
            uu_path = os.path.join(self.results_path, pd['path'], 'u_upper_series_%s.txt'%s)
            vu_path = os.path.join(self.results_path, pd['path'], 'v_upper_series_%s.txt'%s)

            ul_path = os.path.join(self.results_path, pd['path'], 'u_lower_series_%s.txt'%s)
            vl_path = os.path.join(self.results_path, pd['path'], 'v_lower_series_%s.txt'%s)
            
            uu_series = load_nd_array(uu_path)
            vu_series = load_nd_array(vu_path)        
            ul_series = load_nd_array(ul_path)
            vl_series = load_nd_array(vl_path)

            try:
                entire_xu = np.vstack((entire_xu,xu))
                entire_yu = np.vstack((entire_yu,yu))

                entire_xl = np.vstack((entire_xl,xl))
                entire_yl = np.vstack((entire_yl,yl))

                entire_uu_series = np.vstack((entire_uu_series,uu_series.reshape(1,*uu_series.shape)))
                entire_vu_series = np.vstack((entire_vu_series,vu_series.reshape(1,*uu_series.shape)))
                
                entire_ul_series = np.vstack((entire_ul_series,ul_series.reshape(1,*uu_series.shape)))
                entire_vl_series = np.vstack((entire_vl_series,vl_series.reshape(1,*uu_series.shape)))
            except:
                entire_xu = xu
                entire_yu = yu

                entire_xl = xl
                entire_yl = yl

                entire_uu_series = uu_series.reshape(1,*uu_series.shape)
                entire_vu_series = vu_series.reshape(1,*vu_series.shape)
            
                entire_ul_series = ul_series.reshape(1,*ul_series.shape)
                entire_vl_series = vl_series.reshape(1,*vl_series.shape)

        a,b,c,d = entire_uu_series.shape   
        entire_uu_series = np.moveaxis(entire_uu_series,0,1).reshape(b,a*c,d)
        entire_vu_series = np.moveaxis(entire_vu_series,0,1).reshape(b,a*c,d)
        entire_ul_series = np.moveaxis(entire_ul_series,0,1).reshape(b,a*c,d)
        entire_vl_series = np.moveaxis(entire_vl_series,0,1).reshape(b,a*c,d)        

        return entire_xu, entire_yu, entire_uu_series,entire_vu_series, entire_xl, entire_yl, entire_ul_series, entire_vl_series
        

    def piv_over_time(self,search_dict,start_index=1,N=90):

        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        results_path = os.path.join(self.results_path,*location_path)

        u_path = os.path.join(results_path, 'u_series_%03d_%d.txt'%(start_index,N))
        v_path = os.path.join(results_path, 'v_series_%03d_%d.txt'%(start_index,N))

        x,y,U,V = self.quick_piv(search_dict,index_a = start_index, index_b = start_index + 1)

        with open(u_path, 'w') as uf, open(v_path, 'w') as vf:          
            uf.write('# Array shape: {0}\n'.format(U.shape))        
            vf.write('# Array shape: {0}\n'.format(U.shape))        
                        
        ind = self.check_proper_index(search_dict,index_a = start_index)

        for i in range(N):
            self.set_piv_param({'save_result': False, 'show_result': False})
            x,y,U,V = self.quick_piv(search_dict,index_a = ind,index_b = ind + 1)

            with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
                np.savetxt(uf,U,fmt='%-7.5f')
                np.savetxt(vf,V,fmt='%-7.5f')

            ind = ind + 2
        
        u_series = load_nd_array(u_path)
        v_series = load_nd_array(v_path)

        u_tavg = np.mean(u_series,axis=0)
        v_tavg = np.mean(v_series,axis=0)

        u_tstd = np.std(u_series,axis=0)
        v_tstd = np.std(u_series,axis=0)
        
        x_path = os.path.join(results_path, 'x.txt')
        y_path = os.path.join(results_path, 'y.txt')
        u_tavg_path = os.path.join(results_path, 'u_tavg_%03d_%d.txt' %(start_index,N))
        v_tavg_path = os.path.join(results_path, 'v_tavg_%03d_%d.txt' %(start_index,N))
        u_tstd_path = os.path.join(results_path, 'u_tstd_%03d_%d.txt' %(start_index,N))
        v_tstd_path = os.path.join(results_path, 'v_tstd_%03d_%d.txt' %(start_index,N))

        np.savetxt(x_path,x)
        np.savetxt(y_path,y)
        np.savetxt(u_tavg_path,u_tavg)
        np.savetxt(v_tavg_path,v_tavg)
        np.savetxt(u_tstd_path,u_tstd)
        np.savetxt(v_tstd_path,v_tstd)    

        with open(os.path.join(results_path,'piv_over_time_log.txt'),'a') as f:
            f.write('%s \n' %datetime.datetime.now())
            for k, v in self.piv_param.items():              
                f.write('%s: %s\n' %(k,str(v)))

    def piv_over_time2(self,search_dict,start_index=1,N=90):
        self.set_piv_param({'raw_or_cropped': True})
        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        results_path = os.path.join(self.results_path,*location_path)

        u_path = os.path.join(results_path, 'u_full_series_%03d_%d.txt'%(start_index,N))
        v_path = os.path.join(results_path, 'v_full_series_%03d_%d.txt'%(start_index,N))

        x,y,U,V = self.quick_piv(search_dict,index_a = start_index, index_b = start_index + 1)

        with open(u_path, 'w') as uf, open(v_path, 'w') as vf:
            uf.write('# Array shape: {0}\n'.format(U.shape))
            vf.write('# Array shape: {0}\n'.format(U.shape))
                        
        ind = self.check_proper_index(search_dict,index_a = start_index)

        for i in range(N):
            self.set_piv_param({'save_result': False, 'show_result': False})
            x,y,U,V = self.quick_piv(search_dict,index_a = ind,index_b = ind + 1)

            with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
                np.savetxt(uf,U,fmt='%-7.5f')
                np.savetxt(vf,V,fmt='%-7.5f')

            ind = ind + 2
        
        u_series = load_nd_array(u_path)
        v_series = load_nd_array(v_path)

        u_tavg = np.mean(u_series,axis=0)
        v_tavg = np.mean(v_series,axis=0)

        u_tstd = np.std(u_series,axis=0)
        v_tstd = np.std(u_series,axis=0)
        
        x_path = os.path.join(results_path, 'x_full.txt')
        y_path = os.path.join(results_path, 'y_full.txt')
        u_tavg_path = os.path.join(results_path, 'u_full_tavg_%03d_%d.txt' %(start_index,N))
        v_tavg_path = os.path.join(results_path, 'v_full_tavg_%03d_%d.txt' %(start_index,N))
        u_tstd_path = os.path.join(results_path, 'u_full_tstd_%03d_%d.txt' %(start_index,N))
        v_tstd_path = os.path.join(results_path, 'v_full_tstd_%03d_%d.txt' %(start_index,N))

        np.savetxt(x_path,x)
        np.savetxt(y_path,y)
        np.savetxt(u_tavg_path,u_tavg)
        np.savetxt(v_tavg_path,v_tavg)
        np.savetxt(u_tstd_path,u_tstd)
        np.savetxt(v_tstd_path,v_tstd)          
        self.set_piv_param({'raw_or_cropped': False}) 

        with open(os.path.join(results_path,'piv_over_time2_log.txt'),'a') as f:
            f.write('%s \n' %datetime.datetime.now())
            for k, v in self.piv_param.items():              
                f.write('%s: %s\n' %(k,str(v)))

    def piv_over_time3(self,search_dict,start_index=1,N=90,tag = 'test'):
        self.set_piv_param({'raw_or_cropped': True})
        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        results_path = os.path.join(self.results_path,*location_path)

        u_path = os.path.join(results_path, 'u_%s_series_%03d_%d.txt'%(tag,start_index,N))
        v_path = os.path.join(results_path, 'v_%s_series_%03d_%d.txt'%(tag,start_index,N))

        x,y,U,V = self.quick_piv(search_dict,index_a = start_index, index_b = start_index + 1)

        with open(u_path, 'w') as uf, open(v_path, 'w') as vf:          
            uf.write('# Array shape: {0}\n'.format(U.shape))        
            vf.write('# Array shape: {0}\n'.format(U.shape))        
                        
        ind = self.check_proper_index(search_dict,index_a = start_index)

        for i in range(N):
            self.set_piv_param({'save_result': False, 'show_result': False})
            x,y,U,V = self.quick_piv(search_dict,index_a = ind,index_b = ind + 1)

            with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
                np.savetxt(uf,U,fmt='%-7.5f')
                np.savetxt(vf,V,fmt='%-7.5f')

            ind = ind + 2
        
        u_series = load_nd_array(u_path)
        v_series = load_nd_array(v_path)

        u_tavg = np.mean(u_series,axis=0)
        v_tavg = np.mean(v_series,axis=0)

        u_tstd = np.std(u_series,axis=0)
        v_tstd = np.std(u_series,axis=0)
        
        x_path = os.path.join(results_path, 'x_%s.txt'%tag)
        y_path = os.path.join(results_path, 'y_%s.txt'%tag)
        u_tavg_path = os.path.join(results_path, 'u_%s_tavg_%03d_%d.txt' %(tag,start_index,N))
        v_tavg_path = os.path.join(results_path, 'v_%s_tavg_%03d_%d.txt' %(tag,start_index,N))
        u_tstd_path = os.path.join(results_path, 'u_%s_tstd_%03d_%d.txt' %(tag,start_index,N))
        v_tstd_path = os.path.join(results_path, 'v_%s_tstd_%03d_%d.txt' %(tag,start_index,N))

        np.savetxt(x_path,x)
        np.savetxt(y_path,y)
        np.savetxt(u_tavg_path,u_tavg)
        np.savetxt(v_tavg_path,v_tavg)
        np.savetxt(u_tstd_path,u_tstd)
        np.savetxt(v_tstd_path,v_tstd)          
        self.set_piv_param({'raw_or_cropped': False}) 
               
        with open(os.path.join(results_path,'piv_over_time3_log.txt'),'a') as f:
            f.write('%s \n' %datetime.datetime.now())
            for k, v in self.piv_param.items():              
                f.write('%s: %s\n' %(k,str(v)))

    def point_statistics(self,search_dict,ind_x,ind_y,dt):
        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        results_path = os.path.join(self.results_path,*location_path)

        u_path = os.path.join(results_path,'entire_U.txt')
        v_path = os.path.join(results_path,'entire_V.txt')
        print(u_path)

        with open(u_path, 'r') as ufile:
            temp = ufile.readline()
            print(temp)
            a = re.findall("\d+",temp)            

        entire_U = np.loadtxt(u_path)
        entire_V = np.loadtxt(v_path)       

        no_slices = entire_U.shape[0]//int(a[0])
        field_shape = (no_slices,int(a[0]),int(a[1]))
        
        entire_U = entire_U.reshape(field_shape)
        entire_V = entire_V.reshape(field_shape)

        uu = []
        vv = []
        for x,y in zip(entire_U,entire_V):
            uu.append(x[ind_x,ind_y])
            vv.append(y[ind_x,ind_y])        

        delta_t = dt*len(uu)
        time = np.linspace(0,delta_t,len(uu))

        mean_u = np.mean(uu)
        mean_v = np.mean(vv)

        std_u = np.std(uu,ddof=1)
        std_v = np.std(vv,ddof=1)

        se_u = std_u/len(uu)
        se_v = std_v/len(uu)    

        fig, ax = plt.subplots(2,2,figsize=(10,10))
        ax[0,0].plot(time,uu)
        ax[0,1].plot(time,vv)

        ax[0,0].set_ylabel('$u$ (mm/s)')
        ax[0,1].set_ylabel('$v$ (mm/s)')
        ax[0,0].set_xlabel('time (ms)')
        ax[0,1].set_xlabel('time (ms)')

        ax[0,0].set_title('$u = %.2f \pm %.2f$, $v = %.2f \pm %.2f$' %(mean_u,se_u,mean_v,se_v))
        ax[0,1].set_title('std of u = %.2f , std of v = %.2f' %(std_u,std_v))
        
        ax[1,0].hist(uu,bins = 20)
        ax[1,1].hist(vv,bins = 20)
        ax[1,0].set_xlabel('u (mm/s)')
        ax[1,1].set_xlabel('v (mm/s)')
        ax[1,0].set_ylabel('Frequency (no. samples in a bin)')    

        fig.savefig(os.path.join(results_path,'point_statistics_%d_%d.png')%(ind_x,ind_y))

    def stitch_images(self,step,update = False):
        entire_image_path = os.path.join(self.results_path,'_entire_image.png')        
        try:
            if update == True:
                raise FileNotFoundError
            else:
                im = Image.open(entire_image_path)
        except FileNotFoundError:

            sd_list = self.search_dict_list
            img_a,img_b = self.read_two_images(sd_list[0])
            num_row, num_col = img_a.T.shape

            pos_list = []
            voffset_list = []
            [pos_list.append(x['pos']) for x in self.search_dict_list if x['pos'] not in pos_list]
            [voffset_list.append(x['VOFFSET']) for x in self.search_dict_list if x['VOFFSET'] not in voffset_list]

            pos_list.sort()
            voffset_list.sort()

            num_pos = len(pos_list)
            num_voffset = len(voffset_list)

            voffset_unit = int(voffset_list[1])            

            used_sensor_size = num_voffset * num_col - (num_voffset - 1) * (num_col - voffset_unit)
            overlap = used_sensor_size - step * self.piv_param["pixel_density"]           

            num_entire_col = voffset_unit*num_voffset*num_pos + (num_col-voffset_unit) - overlap * (num_pos - 1)

            entire_image = np.zeros((num_row,num_entire_col))
            print("entire image shape:",entire_image.shape)

            for pos in pos_list:
                for voffset in voffset_list:
                    sd = {'pos': pos, 'VOFFSET': voffset}
                    img_a, img_b = self.read_two_images(sd)

                    xl = int(pos-pos_list[0]) * num_voffset * voffset_unit + int(voffset) - overlap * (int(pos) - 1)
                    xr = xl + num_col

                    print(xl,xr)
                    print(img_a.T.shape)
                    print(entire_image[:,xl:xr].shape)
                    entire_image[:,xl:xr] = img_a.T

            io.imwrite(entire_image_path,entire_image)
            im = Image.open(entire_image_path)
            # im.show(entire_image_path)

            im.show(entire_image_path)           

        return im

    def crop_images(img,a,b,c,d):
        # to be implemented to crop images        
        return img[a:-b,c:-d]    

    def get_entire_vector_field(self,first_position=1, last_position = 11,index_a=100,index_b=101):
        self.set_piv_param({"show_result":False})       

        entire_x = np.empty((49,3))
        entire_y = np.empty((49,3))
        entire_u = np.empty((49,3))
        entire_v = np.empty((49,3))

        for pos in range(first_position,last_position+1,1):
            for voffset in range(1,13,1):                
                search_dict = {'pos': pos, 'VOFFSET': (voffset-1)*80}

                self.quick_piv(search_dict)
                xx,yy,uu,vv = convert_xyuv()

                entire_x = np.hstack((entire_x,xx))
                entire_y = np.hstack((entire_y,yy))
                entire_u = np.hstack((entire_u,uu))
                entire_v = np.hstack((entire_v,vv))

        np.savetxt('_entire_x.txt',entire_x[:,3:-1].T)
        np.savetxt('_entire_y.txt',entire_y[:,3:-1].T)
        np.savetxt('_entire_u.txt',entire_u[:,3:-1].T)
        np.savetxt('_entire_v.txt',entire_v[:,3:-1].T)

        self.set_piv_param({"show_result":True})

    def set_piv_param(self,param):
        for k in param:
            self.piv_param[k] = param[k]        
        
    def my_argsort(lis):
        return sorted(range(len(lis)),key=lis.__getitem__)    

    def calculate_drag(self,start = 1, end = 1):
        entire_x = np.loadtxt('_entire_x.txt')
        entire_y = np.loadtxt('_entire_y.txt')
        entire_u = np.loadtxt('_entire_u.txt')
        entire_v = np.loadtxt('_entire_v.txt')

        left_u = entire_u[start,:]
        right_u = entire_u[-end,:]

        left_v = entire_v[start,:]
        right_v = entire_v[-end,:]

        # top_u = entire_u[3:-1,1]
        # bottom_u = entire_u[3:-1,-1]

        # top_v = entire_v[3:-1,1]
        # bottom_v = entire_v[3:-1,-1]
        
        # X = (52-24)*215*1.5*25.4/1400
        Y = (52-24)*49*1.5*25.4/1400

        # x_coord = np.linspace(0,X,215)
        y_coord = np.linspace(0,Y,49)

        fig,ax = plt.subplots(2,2)
        ax[0,0].plot(y_coord,left_u)
        ax[1,0].plot(y_coord,right_u)

        ax[0,1].plot(y_coord,left_v)
        ax[1,1].plot(y_coord,right_v)

        ax[1,0].set_xlabel('y coordinate (mm)')
        ax[0,0].set_ylabel('u (mm/s)')
        ax[1,0].set_ylabel('u (mm/s)')

        ax[0,1].set_ylabel('v (mm/s)')
        ax[1,1].set_ylabel('v (mm/s)')

        # fig,ax = plt.subplots(2,2)
        # ax[0,0].plot(x_coord,top_u)
        # ax[1,0].plot(x_coord,bottom_u)

        # ax[0,1].plot(x_coord,top_v)
        # ax[1,1].plot(x_coord,bottom_v)

        # ax[1,0].set_xlabel('y coordinate')
        # ax[0,0].set_ylabel('u (mm/s)')
        # ax[1,0].set_ylabel('u (mm/s)')

        # ax[0,1].set_ylabel('v (mm/s)')
        # ax[1,1].set_ylabel('v (mm/s)')
        rho = 1000
        delta_y = (52-24)*1.5*25.4/1400/1000
        W = 0.048
        # delta_x = (52-24)*1.5*25.4/1400/1000

        A = rho*(np.sum((left_u/1000)**2) - np.sum((right_u/1000)**2)) * delta_y * W

        mdot_1 = np.sum((left_u - right_u))/1000 * delta_y * W
        # mdot_2 = np.sum((top_v - bottom_v))/1000*delta_y

        B1 = mdot_1*np.mean(left_u)/1000 * W

        F1 = A - B1
        print('Dynamic pressure difference: %.4f (N)' %A)
        print('m dot: %.8f (kg/s)' %B1)
        print('Force (N): %.4f' %F1)  
# %%

# plt.rcParams['animation.ffmpeg_path'] = '/Users/yeonsu/opt/anaconda3/envs/piv/share/ffmpeg'

def dummy(a,b):
    print('I have done nothing.')

def run_piv(
    frame_a,
    frame_b,
    winsize = 28, # pixels, interrogation window size in frame A
    searchsize = 34,  # pixels, search in image B
    overlap = 24, # pixels, 50% overlap
    dt = 0.0001, # sec, time interval between pulses
    image_check = False,    
    figure_export_name = '_results.png',
    text_export_name =  "_results.txt",
    scale_factor = 1,
    pixel_density = 40,
    arrow_width = 0.001,
    show_result = True,
    u_bounds = (-100,100),
    v_bounds = (-1000,0)
    ):
           
    u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), 
                                                        frame_b.astype(np.int32), 
                                                        window_size=winsize, 
                                                        overlap=overlap, 
                                                        dt=dt, 
                                                        search_area_size=searchsize, 
                                                        sig2noise_method='peak2peak')

    x, y = process.get_coordinates(image_size=frame_a.shape, 
                                    window_size=winsize,
                                    overlap=overlap)

    x, y, u0, v0 = scaling.uniform(x, y, u0, v0,
        scaling_factor = pixel_density) # no. pixel per distance        

    u0, v0, mask = validation.global_val(u0,v0,u_bounds,v_bounds)

    u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                            sig2noise, 
                                            threshold = 1.05 )

    u3, v3 = filters.replace_outliers( u1, v1,
                                    method='localmean',
                                    max_iter=10,
                                    kernel_size=3)

    print('Number of nan elements',np.sum(np.isnan(mask)))

    #save in the simple ASCII table format    
    tools.save(x, y, u3, v3, mask, text_export_name)
    
    if image_check == True:
        fig,ax = plt.subplots(2,1,figsize=(24,12))
        ax[0].imshow(frame_a)
        ax[1].imshow(frame_b)

    io.imwrite(figure_export_name,frame_a)

    if show_result == True:
        fig, ax = plt.subplots(figsize=(24,12))
        tools.display_vector_field(text_export_name, 
                                    ax=ax,
                                    scaling_factor= pixel_density, 
                                    scale=scale_factor, # scale defines here the arrow length
                                    width=arrow_width, # width is the thickness of the arrow
                                    on_img=True, # overlay on the image
                                    image_name= figure_export_name)
        fig.savefig(figure_export_name)
    
    print('Mean of u: %.3f' %np.mean(u3))
    print('Std of u: %.3f' %np.std(u3))
    print('Mean of v: %.3f' %np.mean(v3))
    print('Std of v: %.3f' %np.std(v3))

    return x,y,u3,v3


def vertical_profiles(text_export_name,field_shape):
    df = pd.read_csv(text_export_name,
        delimiter='\t',
        names=['x','y','u','v','s2n','mask'],
        skiprows=1)

    x_array = df.iloc[:,0].to_numpy()
    y_array = df.iloc[:,1].to_numpy()
    uv_array = df.iloc[:,2:4].to_numpy()

    num_rows = field_shape[0] # 11
    num_cols = field_shape[1] # 100

    U_array = np.sqrt(uv_array[:,0]**2+uv_array[:,0]**2).reshape(num_rows,num_cols).T        
    x_coord = x_array.reshape(num_rows,num_cols)[0,:]
    y_coord = np.flipud(y_array.reshape(num_rows,num_cols)[:,0])
    
    fig,axes = plt.subplots(4,3,figsize=(16,16))

    axes_flatten = np.ravel(axes)
    ii = 0  
    for ax in axes_flatten:            
        for i in range(ii*10,(ii+1)*10):
            if i >= num_cols:
                break
            ax.plot(U_array[i,:],y_coord,'o-',label='%.2f'%x_coord[i])
            ax.legend()
            ax.axis([0,np.max(U_array),0,np.max(y_coord)])        
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_ylabel('y-coordinate (mm)')
        ii = ii + 1        
    fig.savefig('vertical_profile.png')

    # if animation_flag is True:
    #     #writer_instance = animation.writers['ffmpeg']
    #     # writer_instance = animation.FFMpegWriter(fps=60) 
    #     # Writer = animation.FFMpegWriter
    #     #Writer = animation.writers['ffmpeg']
    #     Writer = animation.FFMpegWriter(fps=15)
    #     writer = Writer(fps=15,bitrate=1800)


    #     ani_frames = []
    #     fig_ani = plt.figure()
    #     plt.xlabel('Velocity (mm/s)')
    #     plt.ylabel('y-coordinate (mm)')
    #     plt.axis([0,280,0,6])
    #     for i in range(num_cols):
    #         ani_frames.append(plt.plot(U_array[i,:],y_coord,'o-'))
            
    #     ani = animation.ArtistAnimation(fig_ani, ani_frames,
    #                 interval=10,
    #                 repeat_delay = 10,
    #                 blit=True)                    
    #     ani.save('animation.mp4',writer=writer)   

def convert_xyuv():
    saved_txt = np.loadtxt('_quick_piv.txt')
    
    xx = saved_txt[:,0]
    yy = saved_txt[:,1]
    uu = saved_txt[:,2]
    vv = saved_txt[:,3]

    xx2 = xx.reshape(49,3)
    yy2 = yy.reshape(49,3)
    uu2 = uu.reshape(49,3)
    vv2 = vv.reshape(49,3)

    return xx2,yy2,uu2,vv2

def negative(image):
    """ Return the negative of an image
    
    Parameter
    ----------
    image : 2d np.ndarray of grey levels

    Returns
    -------
    (255-image) : 2d np.ndarray of grey levels

    """
    return 255 - image

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def angle_mean_check(Ux,Vy):
    angle = np.arctan(Vy/Ux)*(180/np.pi)
    Mean_val = np.zeros([angle.shape[0],angle.shape[1]])
    Mean_vel = np.zeros([angle.shape[0],angle.shape[1]])
    for i in range(0,angle.shape[0]):
        for j in range(0,angle.shape[1]):
            if (j == 0 and i != 0 and i != angle.shape[0]-1):
                Mean_val[i,j] = (angle[i-1,j]+angle[i+1,j]+angle[i,j+1])/4
                if abs(Mean_val[i,j]/angle[i,j]-1) >0.25:
                    Ux[i,j] = (Ux[i,j+1]+Ux[i-1,j]+Ux[i+1,j])/4
                    Vy[i,j] = (Vy[i,j+1]+Vy[i-1,j]+Vy[i+1,j])/4
            elif (i == 0 and j != 0 and j != angle.shape[1]-1):
                Mean_val[i,j] = (angle[i,j-1]+angle[i+1,j]+angle[i,j+1])/4
                if abs(Mean_val[i,j]/angle[i,j]-1) >0.25:
                    Ux[i,j] = (Ux[i,j+1]+Ux[i,j-1]+Ux[i+1,j])/4
                    Vy[i,j] =  (Vy[i,j+1]+Vy[i,j-1]+Vy[i+1,j])/4
            elif j == angle.shape[1]-1 and i != angle.shape[0]-1 and i != 0 :
                Mean_val[i,j] = (angle[i-1,j]+angle[i+1,j]+angle[i,j-1])/4
                if abs(Mean_val[i,j]/angle[i,j]-1) >0.25:
                    Ux[i,j] = (Ux[i,j-1]+Ux[i-1,j]+Ux[i+1,j])/4
                    Vy[i,j] = (Vy[i,j-1]+Vy[i-1,j]+Vy[i+1,j])/4
            elif i == angle.shape[0]-1 and j != angle.shape[1]-1 and j != 0:
                Mean_val[i,j] = (angle[i-1,j]+angle[i,j+1]+angle[i,j-1])/4
                if abs(Mean_val[i,j]/angle[i,j]-1) >0.25:
                    Ux[i,j] = (Ux[i,j-1]+Ux[i-1,j]+Ux[i,j+1])/4
                    Vy[i,j] = (Vy[i,j-1]+Vy[i-1,j]+Vy[i,j+1])/4
            elif (i>0 and i<angle.shape[0]-1 and j>0 and j<angle.shape[1]-1):
                Mean_val[i,j] = (angle[i-1,j]+angle[i,j-1]+angle[i+1,j]+angle[i,j+1])/4
                if abs(Mean_val[i,j]/angle[i,j]-1) >0.25:
                    Ux[i,j] = (Ux[i,j-1]+Ux[i,j+1]+Ux[i-1,j]+Ux[i+1,j])/4
                    Vy[i,j] = (Vy[i,j-1]+Vy[i,j+1]+Vy[i-1,j]+Vy[i+1,j])/4
    angle_new = np.arctan(Vy/Ux)*(180/np.pi)
    vel = (Ux**2+Vy**2)**0.5
    for i in range(1,vel.shape[0]-1):
        for j in range(1,vel.shape[1]-1):
            Mean_vel[i,j] = (vel[i-1,j]+vel[i+1,j]+vel[i,j-1]+vel[i,j+1])/4
            if abs(Mean_vel[i,j]/vel[i,j]-1)>0.5:
                Ux[i,j] = (Ux[i,j-1]+Ux[i,j+1]+Ux[i-1,j]+Ux[i+1,j])/4
                Vy[i,j] = (Vy[i,j-1]+Vy[i,j+1]+Vy[i-1,j]+Vy[i+1,j])/4
    return Ux, Vy

def get_adjacent_indices(i, j, m, n):
    adjacent_indices = []
    if i > 0:
        adjacent_indices.append((i-1,j))
    if i+1 < m:
        adjacent_indices.append((i+1,j))
    if j > 0:
        adjacent_indices.append((i,j-1))
    if j+1 < n:
        adjacent_indices.append((i,j+1))
    return adjacent_indices

def get_adjacent_mag_angle(i,j,u,v):
    if u.shape != v.shape:
        raise Exception('u and v should be in the same shape.')
    m,n = u.shape
    indices = get_adjacent_indices(i,j,m,n)
    mag = []
    ang = []
    for k in indices:
        u_k = u[k]
        v_k = v[k]
        mag.append(np.sqrt(u_k**2+v**2))
        ang.append(np.arctan(u_k/v_k)*180/np.pi)
    return mag,ang

def get_adjacent_uv(i,j,u,v):
    if u.shape != v.shape:
        raise Exception('u and v should be in the same shape.')
    m,n = u.shape
    indices = get_adjacent_indices(i,j,m,n)
    u2 = []
    v2 = []
    for k in indices:
        u2.append(u[k])
        v2.append(v[k])
        
    return u2,v2
       
def correct_by_angle(u,v):
    if u.shape != v.shape:
        raise Exception('u and v should be in the same shape.')
    
    N_i,N_j = u.shape

    for i in range(N_i):
        for j in range(N_j):
            u_ij = u[i,j]
            v_ij = v[i,j]

            mag_ij = np.sqrt(u_ij**2+v_ij**2)
            ang_ij = np.arctan(u_ij/v_ij)

            mag,ang = get_adjacent_mag_angle(i,j,u,v)
            u_adj,v_adj = get_adjacent_uv(i,j,u,v)

            if np.abs(np.mean(ang)/ang_ij-1) > 0.1:
                u[i,j] = np.mean(u_adj)
                v[i,j] = np.mean(v_adj)            

            # mag,ang = get_adjacent_mag_angle(i,j,u,v)
            # u_adj,v_adj = get_adjacent_uv(i,j,u,v)

            # if np.abs(np.mean(mag)/mag_ij - 1) > 0.1:
            #     u[i,j] = np.mean(u_adj)
            #     v[i,j] = np.mean(v_adj)
            
    return u,v

def quiver_and_contour(x,y,Ux,Vy,img_a_count,results_path, show_result=False):
    fig = plt.figure(figsize=(20, 5), dpi= 400, constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    CS = ax.contourf(y,x,(Ux**2+Vy**2)**0.5, 50, vmin = 0.00, vmax=np.max(np.absolute(Vy)), cmap = cm.coolwarm)
    # CS = ax.contourf(y,x,(Ux**2+Vy**2)**0.5, 50, vmin = np.max(np.absolute(Vy))*0.8, vmax=np.max(np.absolute(Vy)), cmap = cm.coolwarm)

    m = plt.cm.ScalarMappable(cmap = cm.coolwarm)
    m.set_array((Ux**2+Vy**2)**0.5)
    m.set_clim(np.max(np.absolute(Vy))*0,np.max(np.absolute(Vy)))
    # m.set_clim(np.max(np.absolute(Vy))*0.8,np.max(np.absolute(Vy)))
    ax.set_aspect('auto')

    plt.colorbar(m, orientation = 'vertical')
    ax.quiver(y,x,-Vy,-Ux, color = 'black',
            angles='xy', scale_units='xy', #scale=5000, width = 0.003,
            headlength = 2, headwidth = 2, headaxislength = 2, pivot = 'tail')

    ax.set_title('Frame = %0.5f s' %img_a_count)

    pic = 'Stream_%05d.png' %img_a_count

    if show_result is True:
        plt.savefig(os.path.join(results_path,pic), dpi=400, facecolor='w', edgecolor='w')
        plt.show()        
    elif show_result is False:
        plt.savefig(os.path.join(results_path,pic), dpi=400, facecolor='w', edgecolor='w')
        plt.close()

def peel_off_edges(xyuv):
    field_shape = xyuv[0].shape    
    out = []
    for x in xyuv:        
        out.append(x[:,1:-1])
    out = tuple(out)
    return out

def point_statistics(entire_U,entire_V,ind_x,ind_y,dt = 0.1):
    uu = []
    vv = []
    for x,y in zip(entire_U,entire_V):
        uu.append(x[ind_x,ind_y])
        vv.append(y[ind_x,ind_y])        

    

    delta_t = dt*len(uu)
    time = np.linspace(0,delta_t,len(uu))

    

    fig, ax = plt.subplots(2,figsize=(8,3))
    ax[0].plot(time,uu)
    ax[1].plot(time,vv)

    ax[0].set_ylabel('$u$ (mm/s)')
    ax[1].set_ylabel('$v$ (mm/s)')
    ax[1].set_xlabel('time (ms)')
    ax[0].set_title('$u = %.2f \pm %.2f$, $v = %.2f \pm %.2f$' %(mean_u,mean_v,se_u,se_v))
    
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].hist(uu,bins = 20)
    ax[1].hist(vv,bins = 20)
    ax[0].set_xlabel('u (mm/s)')
    ax[1].set_xlabel('v (mm/s)')
    ax[0].set_ylabel('Frequency (no. samples in a bin)')




    
def ensemble_statistics(uu,vv):
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].hist(uu.flatten(),bins = 20)
    ax[1].hist(vv.flatten(),bins = 20)
    ax[0].set_xlabel('u (mm/s)')
    ax[1].set_xlabel('v (mm/s)')
    ax[0].set_ylabel('Frequency (no. samples in a bin)')    

def save_nd_array(path,ndarray):
    with open(path, 'w') as file:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        file.write('# Array shape: {0}\n'.format(ndarray[0].shape))        

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for slice in ndarray:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(file, slice, fmt='%-7.5f')
            # Writing out a break to indicate different slices...
            file.write('# New slice\n')

def load_nd_array(path):        
    with open(path) as f:
        shape_info = f.readline()
        shape = re.findall("\d+",shape_info)
        shape = (int(shape[0]), int(shape[1]))

    out = np.loadtxt(path)        
    out = out.reshape( (out.shape[0]//shape[0],shape[0],shape[1]) )        

    return out

def load_nd_array(path):
    with open(path, 'r') as file:
        temp = file.readline()            
        a = re.findall("\d+",temp)            

        ar = np.loadtxt(path)

        no_slices = ar.shape[0]//int(a[0])
        field_shape = (no_slices,int(a[0]),int(a[1]))
        
        ar = ar.reshape(field_shape)        
    return ar