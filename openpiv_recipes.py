# to do
# - merge dict
# - sig2noise, image quality, filtering , ...

# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
from PIL import Image
from argparse import Namespace

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
    def __init__(self, folder_path, results_folder_path, exception_list = None):
        self.path = os.path.normpath(folder_path)
        self.results_path = os.path.join(os.path.normpath(results_folder_path),*re.findall("[\d]{4}-[\d]{2}-[\d]{2}.*",folder_path))
        
        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            pass

        # self.param_string_list = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(folder_path,x)) and not x.startswith('_')]
        self.param_string_list = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(folder_path,x))]
        # temporary code here:
        try:
            for x in exception_list:
                self.param_string_list.remove(x)            
        except:
            pass

        self.param_dict_list = []

        for x in self.param_string_list:
            self.param_dict_list.append(self.param_string_to_dictionary(x))
        
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
            "pixel_density": 36.74,
            "arrow_width": 0.02,
            "show_result": True,        
            "u_upper_bound": 2000, # (mm/s)
            "u_lower_bound": -2000, # (mm/s)
            "v_upper_bound": 2000, # (mm/s)
            "v_lower_bound": -2000, # (mm/s)
            "transpose": False,
            "crop": [0,0,0,0],
            "sn_threshold": 1.3,
        }
        self.piv_dict_list = self.param_dict_list
        try:
            self.search_dict_list = self.check_piv_dict_list()
        except:
            pass

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

    def read_two_images(self,search_dict,index_a = 100,index_b = 101, open = False):
        location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
        print('Read image from:', location_path[0])

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

    def check_image_pair():
        # to be implemented to check if PIV result is successful or not
        return True

    def check_piv_dict_list(self):
        lis = self.piv_dict_list
        search_dict_list = []
        for x in lis:
            search_dict = {'pos': x['pos'], 'VOFFSET': x['VOFFSET']}
            search_dict_list.append(search_dict)

        return search_dict_list

    def show_piv_param(self):
        print("- PIV parameters -")
        for x, y in self.piv_param.items():
            print(x +":", y)

    def quick_piv(self, search_dict, index_a = 100, index_b = 101, folder = None):
        self.show_piv_param()
        ns = Namespace(**self.piv_param)

        if folder == None:
            img_a, img_b = self.read_two_images(search_dict,index_a=index_a,index_b=index_b)

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
        img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        
        img_a = ndimage.rotate(img_a, ns.rotate)
        img_b = ndimage.rotate(img_b, ns.rotate)
            
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(img_a.astype(np.int32),
                                                            img_b.astype(np.int32),
                                                            window_size=ns.winsize,
                                                            overlap=ns.overlap, 
                                                            dt=ns.dt, 
                                                            search_area_size=ns.searchsize, 
                                                            sig2noise_method='peak2peak')

        x, y = pyprocess.get_coordinates(image_size=img_a.shape, 
                                        search_area_size=ns.searchsize,                                    
                                        overlap=ns.overlap)

        x, y, u0, v0 = scaling.uniform(x, y, u0, v0, scaling_factor = ns.pixel_density) # no. pixel per distance

        u0, v0, mask = validation.global_val(u0,v0,ns.u_bound,ns.v_bound)

        u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                                sig2noise, 
                                                threshold = ns.sn_threshold)

        u3, v3 = filters.replace_outliers( u1, v1,
                                        method='localmean',
                                        max_iter=50,
                                        kernel_size=1)        

        # u3,v3 = angle_mean_check(u3,v3)

        # u3,v3 = correct_by_angle(u3,v3)

        u3,v3 = angle_mean_check(u3,v3)

        #save in the simple ASCII table format        
        tools.save(x, y, u3, v3, sig2noise,mask, os.path.join(results_path,'Stream_%05d.txt'%index_a))        

        quiver_and_contour(x,y,u3,v3,index_a,results_path)
        
        if ns.image_check == True:
            fig,ax = plt.subplots(2,1,figsize=(24,12))
            ax[0].imshow(img_a)
            ax[1].imshow(img_b)

        io.imwrite(os.path.join(results_path,ns.figure_export_name),img_a)

        if ns.show_result == True:
            fig, ax = plt.subplots(figsize=(24,12))
            tools.display_vector_field( os.path.join(results_path,'Stream_%05d.txt'%index_a), 
                                        ax=ax, scaling_factor= ns.pixel_density, 
                                        scale=ns.scale_factor, # scale defines here the arrow length
                                        width=ns.arrow_width, # width is the thickness of the arrow
                                        on_img=True, # overlay on the image
                                        image_name= os.path.join(results_path,ns.figure_export_name))
            fig.savefig(os.path.join(results_path,ns.figure_export_name))

        if ns.show_vertical_profiles:
            field_shape = pyprocess.get_field_shape(image_size=img_a.shape,search_area_size=ns.searchsize,overlap=ns.overlap)
            vertical_profiles(ns.text_export_name,field_shape)
        
        print('Mean of u: %.3f' %np.mean(u3))
        print('Std of u: %.3f' %np.std(u3))        
        print('Mean of v: %.3f' %np.mean(v3))
        print('Std of v: %.3f' %np.std(v3))

        output = np.array([np.mean(u3),np.std(u3),np.mean(v3),np.std(v3)])
        # if np.absolute(np.mean(v3)) < 50:
        #     output = self.quick_piv(search_dict,index_a = index_a + 1, index_b = index_b + 1)

        return x,y,u3,v3,sig2noise

        # return np.std(u3)        

    def stitch_images(self,update = False):
        entire_image_path = os.path.join('_entire_image.png')

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

            num_entire_col = voffset_unit*num_voffset*num_pos + (num_col-voffset_unit)

            entire_image = np.zeros((num_row,num_entire_col))
            print("entire image shape:",entire_image.shape)

            for pos in pos_list:
                for voffset in voffset_list:
                    sd = {'pos': pos, 'VOFFSET': voffset}
                    img_a, img_b = self.read_two_images(sd)

                    xl = int(pos-pos_list[0]) * num_voffset * voffset_unit + int(voffset)
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
    winsize = 16, # pixels, interrogation window size in frame A
    searchsize = 20,  # pixels, search in image B
    overlap = 8, # pixels, 50% overlap
    dt = 0.0001, # sec, time interval between pulses
    image_check = False,
    show_vertical_profiles = False,
    figure_export_name = '_results.png',
    text_export_name =  "_results.txt",
    scale_factor = 1,
    pixel_density = 36.74,
    arrow_width = 0.001,
    show_result = True,
    u_bounds = (-100,100),
    v_bounds = (-10000,10000)
    ):
           
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                        frame_b.astype(np.int32), 
                                                        window_size=winsize, 
                                                        overlap=overlap, 
                                                        dt=dt, 
                                                        search_area_size=searchsize, 
                                                        sig2noise_method='peak2peak')

    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, 
                                    search_area_size=searchsize,                                    
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

    

    #save in the simple ASCII table format    
    if np.std(u3) < 480:
        tools.save(x, y, u3, v3, sig2noise,mask, text_export_name)
    
    if image_check == True:
        fig,ax = plt.subplots(2,1,figsize=(24,12))
        ax[0].imshow(frame_a)
        ax[1].imshow(frame_b)

    io.imwrite(figure_export_name,frame_a)

    if show_result == True:
        fig, ax = plt.subplots(figsize=(24,12))
        tools.display_vector_field(text_export_name, 
                                    ax=ax, scaling_factor= pixel_density, 
                                    scale=scale_factor, # scale defines here the arrow length
                                    width=arrow_width, # width is the thickness of the arrow
                                    on_img=True, # overlay on the image
                                    image_name= figure_export_name)
        fig.savefig(figure_export_name)       

    if show_vertical_profiles:
        field_shape = pyprocess.get_field_shape(image_size=frame_a.shape,search_area_size=searchsize,overlap=overlap)
        vertical_profiles(text_export_name,field_shape)
    
    print('Std of u3: %.3f' %np.std(u3))
    print('Mean of u3: %.3f' %np.mean(u3))

    return np.std(u3)


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
            if abs(Mean_vel[i,j]/vel[i,j]-1)>0.1:
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

def quiver_and_contour(x,y,Ux,Vy,img_a_count,results_path):
        fig = plt.figure(figsize=(15, 3), dpi= 400, constrained_layout=True)
        ax = fig.add_subplot(1,1,1)
        CS = ax.contourf(y,x,(Ux**2+Vy**2)**0.5, 50, vmin = 0.00, vmax=np.max(np.absolute(Vy)), cmap = cm.coolwarm)
        m = plt.cm.ScalarMappable(cmap = cm.coolwarm)
        m.set_array((Ux**2+Vy**2)**0.5)
        m.set_clim(0,0.6)
        ax.set_aspect('auto')

        plt.colorbar(m, orientation = 'vertical')
        ax.quiver(y,x,-Vy,-Ux, color = 'black',
                angles='xy', scale_units='xy', scale=5000, width = 0.003,
                headlength = 2, headwidth = 2, headaxislength = 2, pivot = 'tail')

        ax.set_title('Frame = %0.5f s' %img_a_count)

        pic = 'Stream_%05d.png' %img_a_count

        plt.savefig(os.path.join(results_path,pic), dpi=400, facecolor='w', edgecolor='w')
        #plt.show()
        plt.close()