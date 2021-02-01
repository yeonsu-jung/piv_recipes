# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import imageio

import re
import os

from PIL import Image
import imageio as io

# import matplotlib
# matplotlib.use('Qt5Agg')

# from PyQt5 import QtCore, QtWidgets

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.figure import Figure

# %%
class ParticleImage:    
    def __init__(self, folder_path):
        self.path = folder_path
        self.param_string_list = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(folder_path,x)) and not x.startswith('_')]

        # for x in self.param_string_list:
        #     print(x)
        self.param_dict_list = []

        for x in self.param_string_list:
            # self.param_dict_list.append(self.dummy(x))
            self.param_dict_list.append(self.param_string_to_dictionary(x))

        self.param_dict_list = sorted(self.param_dict_list, key = lambda i: (i['pos'],i['VOFFSET']))
        self.param_string_list = sorted(self.param_string_list, key = lambda i: (self.param_string_to_dictionary(i)['pos'],self.param_string_to_dictionary(i)['VOFFSET']))
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
        }


    def param_string_to_dictionary(self,pstr):
        running_parameter = re.findall("_[a-z]+[0-9]+[.]*[0-9]*", pstr, re.IGNORECASE)
        sample_parameter = pstr.replace("img_","")

        for k in running_parameter:
            sample_parameter = sample_parameter.replace(k,"")

        sample_parameter = sample_parameter.replace('_01-18',"")

        param_dict = {'sample': sample_parameter}
        for k in running_parameter:
            kk = re.findall('[a-x]+', k,re.IGNORECASE)
            vv = re.findall('[0-9]+[.]*[0-9]*', k,re.IGNORECASE)
            param_dict[kk[0]] = float(vv[0])

        param_dict['path'] = pstr

        return param_dict    

    def read_two_images(self,camera_position,sensor_position,index_a = 100,index_b = 101):

        location_path = [x['path'] for x in self.param_dict_list if x['pos'] == camera_position and x['VOFFSET'] == (sensor_position-1)*80]        

        file_a_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_a)
        file_b_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_b)

        # exception handling needed
        # img_a = io.imread(file_a_path)
        # img_b = io.imread(file_b_path)

        img_a = Image.open(file_a_path)
        img_b = Image.open(file_b_path)

        # print(np.std(np.array(img_b) - np.array(img_a)))

        # if np.mean(np.array(img_b) - np.array(img_a)) < 80:
        #     index_a = index_a + 1
        #     index_b = index_b + 1

        #     file_a_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_a)
        #     file_b_path = os.path.join(self.path,*location_path,'frame_%06d.tiff' %index_b)

        #     img_a = Image.open(file_a_path)
        #     img_b = Image.open(file_b_path)
        
        return img_a, img_b

    def check_image_pair():
        return True

    def open_two_images(self,camera_position,sensor_position,index_a = 100,index_b = 101):
        
        im1, im2 = self.read_two_images(camera_position,sensor_position,index_a=index_a,index_b=index_b)

        im1 = Image.open(file_a_path)
        im2 = Image.open(file_b_path)

        im1.show()
        im2.show()

        # plt.ion()
        # fig,ax = plt.subplots(2,1,figsize=(15,4))
        # ax[0].imshow(img_a)
        # ax[1].imshow(img_b)
        # ax[0].axis('off')
        # ax[1].axis('off')        

    def quick_piv(self,camera_position,sensor_position,index_a = 100, index_b = 101):
        img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a,index_b=index_b)

        img_a = np.array(img_a).T
        img_b = np.array(img_b).T

        u_std = run_piv(img_a,img_b,**self.piv_param)
        
        if u_std > 500:
            img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a+1,index_b=index_b+1)
            img_a = np.array(img_a).T
            img_b = np.array(img_b).T
            
            u_std = run_piv(img_a,img_b,**self.piv_param)


    def stitch_images(self):
        entire_image_path = os.path.join(self.path,'_entire_image.png')

        try:
            im = Image.open(entire_image_path)        
            im.show(entire_image_path)            

        except:            
            garo = 1408
            sero = (1296)*11
            entire_image = np.zeros((sero,garo))
            for param_string in self.param_string_list:
                # CHECK IMAGE VALIDITY ?
                img_a_name = 'frame_000102.tiff'
                img_b_name = 'frame_000103.tiff'

                param_dict = self.param_string_to_dictionary(param_string)

                try:
                    file_path_a = os.path.join(folder_path,param_string,img_a_name)
                    file_path_b = os.path.join(folder_path,param_string,img_b_name)
                    img_a = io.imread(file_path_a)
                    # img_b = io.imread(file_path_b)
                except:
                    file_path_a = os.path.join(folder_path,'img_' + param_string,img_a_name)
                    file_path_b = os.path.join(folder_path,'img_' + param_string,img_b_name)

                    img_a = io.imread(file_path_a)
                    # img_b = io.imread(file_path_b)
                    
                position = int(param_dict['pos'])
                v_offset = int(param_dict['VOFFSET'])

                start_index = (position-1)*1296 + (v_offset//80)*108
                end_index = start_index + 108

                entire_image[start_index:end_index,:] = img_a

            io.imwrite(entire_image_path,entire_image.T)
            im = Image.open(entire_image_path)
            im.show(entire_image_path)

        return im

    def piv_upper_region(self,camera_position,sensor_position,index_a = 100, index_b = 101,surface_index = 360):
        
        img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a,index_b=index_b)              

        # img_a = np.array(img_a.rotate(-1.364))
        # img_b = np.array(img_b.rotate(-1.364))

        img_a = np.array(img_a)
        img_b = np.array(img_b)
        
        img_a = img_a[:,0:surface_index].T
        img_b = img_b[:,0:surface_index].T        

        u_std = run_piv(img_a,img_b,**self.piv_param)
        if u_std > 500:
            img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a+1,index_b=index_b+1)
            img_a = np.array(img_a)
            img_b = np.array(img_b)

            img_a = img_a[:,0:surface_index].T
            img_b = img_b[:,0:surface_index].T
            u_std = run_piv(img_a,img_b,**self.piv_param)

    def piv_lower_region(self,camera_position,sensor_position,index_a = 100, index_b = 101,surface_index = 360):
        
        img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a,index_b=index_b)

        # img_a = np.array(img_a.rotate(-1.364))
        # img_b = np.array(img_b.rotate(-1.364))

        img_a = np.array(img_a)
        img_b = np.array(img_b)

        img_a = img_a[:,-surface_index:-1].T
        img_b = img_b[:,-surface_index:-1].T

        u_std = run_piv(img_a,img_b,**self.piv_param)
        if u_std > 500:
            img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a+1,index_b=index_b+1)
            img_a = np.array(img_a)
            img_b = np.array(img_b)

            img_a = img_a[:,-surface_index:-1].T
            img_b = img_b[:,-surface_index:-1].T
            u_std = run_piv(img_a,img_b,**self.piv_param)

    def fast_piv():
        

    def get_entire_vector_field(self,index_a=100,index_b=101):
        self.set_piv_param({"show_result":False})
        # pos_list = [x['pos'] for x in self.param_dict_list]
        # voffset_list = [x['VOFFSET'] for x in self.param_dict_list]

        entire_x = np.empty((49,3))
        entire_y = np.empty((49,3))
        entire_u = np.empty((49,3))
        entire_v = np.empty((49,3))

        for pos in range(1,7,1):
            for voffset in range(1,13,1):
                
                self.fast_piv(pos,voffset)

                xx,yy,uu,vv = convert_xyuv()

                entire_x = np.hstack((entire_x,xx))
                entire_y = np.hstack((entire_y,yy))
                entire_u = np.hstack((entire_u,uu))
                entire_v = np.hstack((entire_v,vv))

        np.savetxt('_entire_x.txt',entire_x.T)
        np.savetxt('_entire_y.txt',entire_y.T)
        np.savetxt('_entire_u.txt',entire_u.T)
        np.savetxt('_entire_v.txt',entire_v.T)

        self.set_piv_param({"show_result":True})

    def set_piv_param(self,param):
        for k in param:
            self.piv_param[k] = param[k]        
        
    def my_argsort(lis):
        return sorted(range(len(lis)),key=lis.__getitem__)

    def open_two_images(self,camera_position,sensor_position,index_a = 100,index_b = 101):
        location = camera_position * sensor_position
        location_info = self.param_dict_list[location]
        location_name = self.param_string_list[location]
        location_path = os.path.join(self.path, location_name)

        file_a_path = os.path.join(location_path,'frame_%06d.tiff' %index_a)
        file_b_path = os.path.join(location_path,'frame_%06d.tiff' %index_b)        

        im1 = Image.open(file_a_path)
        im2 = Image.open(file_b_path)

        im1.show()
        im2.show()    

# plt.rcParams['animation.ffmpeg_path'] = '/Users/yeonsu/opt/anaconda3/envs/piv/share/ffmpeg'

def dummy(a,b):
    print('I have done nothing.')

def run_piv(
    frame_a,
    frame_b,
    winsize = 64, # pixels, interrogation window size in frame A
    searchsize = 66,  # pixels, search in image B
    overlap = 32, # pixels, 50% overlap
    dt = 0.0001, # sec, time interval between pulses
    image_check = False,
    show_vertical_profiles = False,
    figure_export_name = '_results.png',
    text_export_name =  "_results.txt",
    scale_factor = 1,
    pixel_density = 36.74,
    arrow_width = 0.02,
    show_result = True,
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

    u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                            sig2noise, 
                                            threshold = 1.05 )

    u2, v2 = filters.replace_outliers( u1, v1,
                                    method='localmean',
                                    max_iter=10,
                                    kernel_size=3)

    x, y, u3, v3 = scaling.uniform(x, y, u2, v2,
                                scaling_factor = pixel_density) # no. pixel per distance

    #save in the simple ASCII table format    
    tools.save(x, y, u3, v3, sig2noise,mask, text_export_name)
    
    if image_check == True:
        fig,ax = plt.subplots(2,1,figsize=(24,12))
        ax[0].imshow(frame_a)
        ax[1].imshow(frame_b)

    imageio.imwrite(figure_export_name,frame_a)

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
    
    print(np.std(u3))

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

# %%
folder_path = '/Volumes/Backup Plus /ROWLAND/piv-data/2021-01-19'
pi = ParticleImage(folder_path)
# %%
# pi.param_dict_list = [x for x in pi.param_dict_list if x['sample'] == 'Flat_10' and x['motor'] == 25.0]
pi.param_dict_list = [x for x in pi.param_dict_list if x['sample'] == '1_1_1_10' and x['motor'] == 25.0]
# print(pi.param_dict_list)

# %%
pi.get_entire_vector_field()

# %%
entire_x = np.loadtxt('_entire_x.txt').T
entire_y = np.loadtxt('_entire_y.txt').T
entire_u = np.loadtxt('_entire_u.txt').T
entire_v = np.loadtxt('_entire_v.txt').T


# %%
left_u = entire_u[:,1]
right_u = entire_u[:,-1]

top_u = entire_u[2,:]
bottom_u = entire_u[-2,:]

# %%
fig,ax = plt.subplots(2)
ax[0].plot(left_u)
ax[1].plot(right_u)
# %%
fig,ax = plt.subplots(2)
ax[0].plot(top_u)
ax[1].plot(bottom_u)

# %%
pi.quick_piv(1,1,index_a=100,index_b=101)
# %%
pi.quick_piv(3,3,index_a=aa,index_b=bb)
# pi.quick_piv(3,2,index_a=100,index_b=101)
# %%
pi.quick_piv(1,2,index_a=aa,index_b=bb)


# %%
pi.quick_piv(6,5,index_a=aa,index_b=bb)


# %%
pi.set_piv_param({"show_result": False})
pi.quick_piv(6,5,index_a=aa,index_b=bb)

# %%


# # %%
# folder_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-20'

# piv_param = {
#         "winsize": 48,
#         "searchsize": 50,
#         "overlap": 24,
#         "dt": 0.0001,
#         "image_check": False,    
#         "show_vertical_profiles": False,            
#         "figure_export_name": '_quick_piv.tiff',
#         "text_export_name": '_quick_piv.txt',        
#         "scale_factor": 1e4,
#         "pixel_density": 36.74,
#         "arrow_width": 0.02,
# }

# pi = ParticleImage(folder_path)
# pi.set_piv_param(piv_param)

# # %%
# # pi.set_piv_param({"winsize": 35})

# pi.piv_upper_region(9,7,surface_index=285)
# # %%
# pi.piv_lower_region(11,1,surface_index=1000)
# # %%
# pi.piv_lower_region(9,11,surface_index=714)
# # %%
# pi.set_piv_param({"scale_factor": 1e1,"arrow_width":0.02,"pixel_density":1400/0.0381})
# pi.quick_piv(10,5)


# # %%
# im = entire_image = pi.stitch_images()
# # %%
# im_rotated = im.rotate(-1.364)
# # %%
# im_array = np.array(im.rotate(-1.364))

# im_array_cropped = im_array[200:1200,:]
# plt.imshow(im_array_cropped)

# # %%

# width, height = im.size

# length_to_cut = 10

# # left = 0
# # right = width
# # bottom = length_to_cut
# # top = height - length_to_cut

# left = 20
# right = 20
# bottom = 200
# top = 200

# im_cropped = im_rotated.crop((left, top, width - right, height - bottom)) 

# # %%
# im_cropped.show()

# im_cropped_array = np.array(im_cropped)

# im_cropped.save('_rotated_and_cropped.png')

# # 370, 594

# # %%
# im_cropped_array = np.array(im_cropped)


# # np.savetxt('_rotated_and_cropped.csv',im_cropped_array)
# # %%

# ii = 102

# img_a,img_b = pi.read_two_images(10,9,index_a=ii,index_b=ii+1)
# img_b,img_c = pi.read_two_images(10,9,index_a=ii+1,index_b=ii+2)

# ia = np.array(img_a)
# ib = np.array(img_b)
# ic = np.array(img_c)

# print(np.mean(ia-ib), np.mean(ib-ic))

# # %%
# a = np.loadtxt('_quick_piv.txt')

# print(a.shape)

# xy_array = a[:,0:2]
# print(xy_array)
# # %%
# x_array = a[:,0]
# y_array = a[:,1]
# u_array = a[:,2]

# # %%
# xx = x_array.reshape(53,3)
# yy = y_array.reshape(53,3)
# uu = u_array.reshape(53,3)
# print(uu)
# # %%
# print(yy)

# # %%
# fig,ax = plt.subplots()
# ax.plot(uu[:,0],yy[:,0],'o-')



# ax.invert_yaxis()
# # %%
# num_rows = field_shape[0] # 11
#     num_cols = field_shape[1] # 100

#     U_array = np.sqrt(uv_array[:,0]**2+uv_array[:,0]**2).reshape(num_rows,num_cols).T        
#     x_coord = x_array.reshape(num_rows,num_cols)[0,:]
#     y_coord = np.flipud(y_array.reshape(num_rows,num_cols)[:,0])

# xx = x
# %%

entire_x = np.array([])
entire_y = np.array([])


# %%
tttt = np.zeros((5,2))

tttt.shape

# %%
tttt2 = np.vstack((tttt,entire_x))

tttt2.shape
# %%

convert_xyuv()

# %%
np.loadtxt('_quick_piv.txt')

# %%
entire_x = np.empty((49,3))
aa = np.ones((49,3))
# %%
bb = np.hstack((entire_x,aa))

bb.shape


# %%

aa = np.loadtxt('_quick_piv.txt')

# %%
aa[:,0]
# %%

entire_x = np.loadtxt('_entire_x.txt')
entire_y = np.loadtxt('_entire_y.txt')
entire_u = np.loadtxt('_entire_u.txt')
entire_v = np.loadtxt('_entire_v.txt')


# %%
left_u = entire_u[:,2]
right_u = entire_u[:,-1]

top_u = entire_u[2,:]
bottom_u = entire_u[-2,:]
# %%

# %%
fig,ax = plt.subplots(2)
ax[0].plot(left_u)
ax[1].plot(right_u)
# %%
fig,ax = plt.subplots(2)
ax[0].plot(top_u)
ax[1].plot(bottom_u)

# %%
