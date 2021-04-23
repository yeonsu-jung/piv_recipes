# %%
from time import time
from openpiv_recipes import get_adjacent_indices
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
import pathlib as Path
import re

reload(path)
# %%
class piv_class():
    def __init__(self,parent_path):                        
        owd = os.getcwd()

        self.path = path.path_class(parent_path)
        self.piv_setting_path = os.path.join(owd,'piv_setting.yaml')

    def get_image_path(self,path,index):
        # assuming image file names start with frame_
        img_path = os.path.join(self.path.path,path,'frame_%06d.tiff' %index)        
        return img_path

    def choose_path(self):
        
        try:           
            _,_,path_list = self.path.get_stitching_lists()            
        except:
            path_list = os.listdir(self.path.path)


        for i, pth in enumerate(path_list):
            print(i,'\t', pth)
        path_no = int(input('Choose number corresponding to the path you want.') )       
        assert path_no in range(i), "Choose number between %d and %d"%(0,i)

        print('Chose: %d'%path_no, path_list[path_no])

        return path_list[path_no]

    def read_image_from_path(self,path,index=1):
        assert isinstance(index,int), "Frame index should be of int type."
        
        file_a_path = os.path.join(self.path.path,path,'frame_%06d.tiff' %index)        
        img_a = io.imread(file_a_path)  
        
        # print('Read image from:', file_a_path)

        return img_a

    def check_proper_index(self,path,index):
        piv_param = update_piv_param(setting_file=self.piv_setting_path,mute = True)
        ns = Namespace(**piv_param)

        img_a = self.read_image_from_path(path,index)
        img_b = self.read_image_from_path(path,index+1)
        img_c = self.read_image_from_path(path,index+2)

        img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        
        img_c = img_c[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        
        corr1 = pyprocess.correlate_windows(img_a,img_b)
        corr2 = pyprocess.correlate_windows(img_b,img_c)

        if np.max(corr1) > np.max(corr2):
            out = index
        else:
            out = index + 1
        return out
    
    def piv_over_time(self,path = None,start_index=1,N=2):
        t = time()
        if path == None:            
            path = self.choose_path()

        results_path = os.path.join(self.path.results_path,path)
        print('Result_path: %s'%results_path)

        piv_param = update_piv_param(setting_file=self.piv_setting_path)
        ns = Namespace(**piv_param)

        # relative_path = '[%d,%d,%d,%d]_[%d,%d,%d]'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
        relative_path = get_rel_path(piv_setting_path=self.piv_setting_path)
        full_path = os.path.join(results_path,relative_path,'series_%03d_%d'%(start_index,N))
        try:
            os.makedirs(full_path)
        except:
            pass

        shutil.copy('piv_setting.yaml',os.path.join(full_path,'piv_setting.yaml')) # to be replaced with class member

        # time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        x_path = os.path.join(full_path, 'x.txt')
        y_path = os.path.join(full_path, 'y.txt')
        u_path = os.path.join(full_path, 'u.txt')
        v_path = os.path.join(full_path, 'v.txt')

        x,y,U,V = self.quick_piv(path=path,index = start_index,mute=True)

        np.savetxt(x_path,x)
        np.savetxt(y_path,y)

        with open(u_path, 'w') as uf, open(v_path, 'w') as vf:          
            uf.write('# Array shape: {0}\n'.format(U.shape))
            vf.write('# Array shape: {0}\n'.format(U.shape))                       
        
        ind = self.check_proper_index(path,index = start_index)

        for i in range(N):
            x,y,U,V = self.quick_piv(path=path,index = ind,mute=True)

            with open(u_path, 'a') as uf, open(v_path, 'a') as vf:
                np.savetxt(uf,U,fmt='%-7.5f')
                np.savetxt(vf,V,fmt='%-7.5f')

            ind = ind + 2

        elapsed = time() - t
        print('PIV over time has been done. Elapsed time is %.2f'%elapsed)

    def piv_over_sample(self,start_index,N):
        pos_list,voffset_list,path_list = self.path.get_stitching_lists()

        for pth in path_list:
            self.piv_over_time(pth,start_index=start_index,N=N)           
        
    def temporal_average(self,series_path):

        # relative_path = '[%d,%d,%d,%d]_[%d,%d,%d]'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
        
        x_path = os.path.join(series_path, 'x.txt')
        y_path = os.path.join(series_path, 'y.txt')
        u_path = os.path.join(series_path, 'u.txt')
        v_path = os.path.join(series_path, 'v.txt')

        x = np.loadtxt(x_path)
        y = np.loadtxt(y_path)
        u = load_nd_array(u_path)
        v = load_nd_array(v_path)

        u_tavg = np.mean(u,axis=0)
        v_tavg = np.mean(v,axis=0)

        u_tstd = np.std(u,axis=0)
        v_tstd = np.std(u,axis=0)
        
        
        u_tavg_path = os.path.join(results_path, 'u_%s_tavg_%03d_%d.txt' %(note,start_index,N))
        v_tavg_path = os.path.join(results_path, 'v_%s_tavg_%03d_%d.txt' %(note,start_index,N))
        u_tstd_path = os.path.join(results_path, 'u_%s_tstd_%03d_%d.txt' %(note,start_index,N))
        v_tstd_path = os.path.join(results_path, 'v_%s_tstd_%03d_%d.txt' %(note,start_index,N))

        np.savetxt(u_tavg_path,u_tavg)
        np.savetxt(v_tavg_path,v_tavg)
        np.savetxt(u_tstd_path,u_tstd)
        np.savetxt(v_tstd_path,v_tstd)          
        self.set_piv_param({'raw_or_cropped': False}) 
               
        with open(os.path.join(results_path,'piv_over_time3_log.txt'),'a') as f:
            f.write('%s \n' %datetime.datetime.now())
            for k, v in self.piv_param.items():              
                f.write('%s: %s\n' %(k,str(v)))

    def quick_piv(self,path = None,index = 1,mute = False):
        # choose path
        if path == None:            
            path = self.choose_path()

        path_a = self.get_image_path(path,index)
        path_b = self.get_image_path(path,index+1)

        return run_piv(path_a,path_b, export_parent_path = os.path.join(self.path.results_path,path), piv_setting_path = self.piv_setting_path,mute=mute)
        # return x,y,U,V
    

    def stitch_images(self, step, update = False, index = 1):
        entire_image_path = os.path.join(self.path.results_path,'_entire_image.png')        

        assert self.path.check_stitching_possibility(), "Can't stitch images."            

        try:
            if update == True:
                raise FileNotFoundError
            else:
                im = Image.open(entire_image_path)
        except FileNotFoundError:                        

            pos_list,voffset_list,path_list = self.path.get_stitching_lists()

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
            for pos in pos_list:
                for voffset in voffset_list:                   

                    img_a = self.read_image(path_list[i],index=index)

                    xl = int(pos-pos_list[0]) * num_voffset * voffset_unit + int(voffset) - overlap * (int(pos) - 1)
                    xr = xl + num_col

                    # print(xl,xr)
                    # print(img_a.T.shape)
                    # print(entire_image[:,xl:xr].shape)
                    entire_image[:,xl:xr] = img_a.T
                    i = i + 1

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

# %%
def update_piv_param(setting_file = 'piv_setting.yaml', mute = False):
        with open(setting_file) as f:
            piv_param = yaml.safe_load(f)

        if not mute:
            print("- PIV parameters -")
            for x, y in piv_param.items():
                print(x +":", y)
        
        return piv_param  

def run_piv(img_a_path, img_b_path, export_parent_path = None, piv_setting_path = 'piv_setting.yaml', mute = False):
    if export_parent_path is None:
        export_parent_path = '_test'

    owd = os.getcwd()
    try:
        img_a = io.imread(img_a_path)
        img_b = io.imread(img_b_path)
    except:
        raise AssertionError("Can't open images")

    name_a = os.path.splitext(os.path.basename(img_a_path))[0]    
    name_b = os.path.splitext(os.path.basename(img_b_path))[0]

    piv_param = update_piv_param(setting_file=piv_setting_path,mute = mute)
    ns = Namespace(**piv_param)

    if mute:
        ns.show_result = False
        ns.save_result = False

    # export_param = ()    

    # relative_path = '%d_%d_%d'%(ns.winsize,ns.overlap,ns.searchsize)
    relative_path = get_rel_path(piv_setting_path=piv_setting_path)
    # full_path = os.path.join(export_parent_path,relative_path)
    full_path = relative_path

    if mute:
        time_stamp = ''
    else:        
        time_stamp = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')

    try:
        os.makedirs(os.path.join(export_parent_path,relative_path))
    except FileExistsError:
        pass        
    
    os.chdir(export_parent_path)
                
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
    txt_path = os.path.expanduser( os.path.join(full_path,'%s%s.txt'%(name_a,time_stamp)) )
    img_path = os.path.join(full_path,'%s%s.png'%(name_a,time_stamp))
    # field_path = os.path.normcase(os.path.join(full_path,'Field_%s%s.png'%(name_a,time_stamp)))
    field_path = os.path.normcase(os.path.join(full_path,'Field_%s.png'%(time_stamp)))
    # field_path = os.path.normcase(os.path.join(full_path,'test.png'))

    if ns.save_result is True:            
        tools.save(x, y, u3, v3, mask, txt_path)
        io.imwrite(img_path,img_a)
        setting_path = os.path.join(full_path,'piv_setting%s.yaml'%time_stamp)
        shutil.copy(os.path.join(owd,'piv_setting.yaml'),setting_path)

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
        
        fig.savefig(field_path)
        # fig.savefig('test.png')
    
    if not mute:
        print('Mean of u: %.3f' %np.mean(u3))
        print('Std of u: %.3f' %np.std(u3))        
        print('Mean of v: %.3f' %np.mean(v3))
        print('Std of v: %.3f' %np.std(v3))

    # output = np.array([np.mean(u3),np.std(u3),np.mean(v3),np.std(v3)])
    # if np.absolute(np.mean(v3)) < 50:
    #     output = self.quick_piv(search_dict,index_a = index_a + 1, index_b = index_b + 1)
    os.chdir(owd)

    return x,y,u3,v3



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
    print('foo')
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

def get_rel_path(piv_setting_path = 'piv_setting.yaml'):
    piv_param = update_piv_param(setting_file=piv_setting_path,mute = True)
    ns = Namespace(**piv_param)

    relative_path = '%d,%d,%d,%d_%d,%d,%d'%(ns.crop[0],ns.crop[1],ns.crop[2],ns.crop[3],ns.winsize,ns.overlap,ns.searchsize)
    return relative_path

