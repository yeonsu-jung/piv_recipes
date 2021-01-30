# %%
import re
import imageio as io
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from matplotlib import pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling
from scipy.ndimage.morphology import binary_hit_or_miss 
import openpiv_recipes

from PIL import Image

import skimage
import skimage.feature
import skimage.viewer

import importlib
importlib.reload(openpiv_recipes)

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

    def param_string_to_dictionary(self,pstr):
        running_parameter = re.findall("_[a-z]+[0-9]+[.]*[0-9]*", pstr, re.IGNORECASE)
        sample_parameter = pstr.replace("img_","")

        for k in running_parameter:
            sample_parameter = sample_parameter.replace(k,"")

        param_dict = {'sample': sample_parameter}
        for k in running_parameter:
            kk = re.findall('[a-x]+', k,re.IGNORECASE)
            vv = re.findall('[0-9]+[.]*[0-9]*', k,re.IGNORECASE)
            param_dict[kk[0]] = float(vv[0])

        return param_dict    

    def read_two_images(self,camera_position,sensor_position,index_a = 100,index_b = 101):
        location = camera_position * sensor_position
        location_info = self.param_dict_list[location]
        location_name = self.param_string_list[location]
        location_path = os.path.join(self.path, location_name)

        file_a_path = os.path.join(location_path,'frame_%06d.tiff' %index_a)
        file_b_path = os.path.join(location_path,'frame_%06d.tiff' %index_b)

        # exception handling needed
        img_a = io.imread(file_a_path)
        img_b = io.imread(file_b_path)

        # plt.ion()
        # fig,ax = plt.subplots(2,1,figsize=(15,4))
        # ax[0].imshow(img_a)
        # ax[1].imshow(img_b)
        # ax[0].axis('off')
        # ax[1].axis('off')

        return img_a, img_b

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

        # plt.ion()
        # fig,ax = plt.subplots(2,1,figsize=(15,4))
        # ax[0].imshow(img_a)
        # ax[1].imshow(img_b)
        # ax[0].axis('off')
        # ax[1].axis('off')

        return img_a, img_b

    def quick_piv(self,camera_position,sensor_position,index_a = 100, index_b = 101):
        img_a, img_b = self.read_two_images(camera_position,sensor_position,index_a=index_a,index_b=index_b)

        figure_path = '_quick_piv.tiff'
        text_path = '_quick_piv.txt'

        openpiv_recipes.run_piv(img_a,img_b,
            winsize=48,
            searchsize=50,
            overlap=24,
            show_vertical_profiles=False,
            image_check=False,
            figure_export_name=figure_path,
            text_export_name=text_path)
        
    def my_argsort(lis):
        return sorted(range(len(lis)),key=lis.__getitem__)
# %%
folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-20'
# folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-18'

pi = ParticleImage(folder_path)
# pi_instance.read_two_images(1,1)
pi.quick_piv(2,1) # camera (up to 11), sensor (up to 12)

pi.open_two_images(1,1)
# %%
pi.quick_piv(3,11) # camera (up to 11), sensor (up to 12)
# pi.quick_piv(5,11) # camera (up to 11), sensor (up to 12)

# %%
pi.open_two_images(3,11)
# %%
from PIL import Image

img_a,img_b = pi.read_two_images(3,11) # camera (up to 11), sensor (up to 12)

im = Image.open()

im = Image.open(os.path.join(exp_path_in,'_inspection.png'))    
        im.show()

# %%


pi = ParticleImage(folder_path)
pi.param_dict_list

pi.param_string_list



# %%
# %%
search_size = 50
overlap = 24
# %%
def param_string_to_dictionary(pstr):      
    running_parameter = re.findall("_[a-z]+[0-9]+[.]*[0-9]*", pstr, re.IGNORECASE)
    sample_parameter = pstr.replace("img_","")

    for k in running_parameter:
        sample_parameter = sample_parameter.replace(k,"")

    param_dict = {'sample': sample_parameter}
    for k in running_parameter:
        kk = re.findall('[a-x]+', k,re.IGNORECASE)
        vv = re.findall('[0-9]+[.]*[0-9]*', k,re.IGNORECASE)
        param_dict[kk[0]] = vv[0]

    return param_dict

def split_image(img):
    vertically_averaged = np.mean(img,axis=0)
    filtered = gaussian_filter1d(vertically_averaged,10)
    center_by_minimum = np.argmin(vertically_averaged)

    left_max = np.argmax(filtered[0:center_by_minimum]) - 20
    right_max = np.argmax(filtered[center_by_minimum:-1]) +center_by_minimum + 20
    
    img_left = img_a[:,0:left_max]
    img_right = img_a[:,right_max:-1]

    return img_left,img_right

def split_image_arg(img):
    vertically_averaged = np.mean(img,axis=0)
    filtered = gaussian_filter1d(vertically_averaged,10)
    center_by_minimum = np.argmin(vertically_averaged)

    left_max = np.argmax(filtered[0:center_by_minimum]) - 20
    right_max = np.argmax(filtered[center_by_minimum:-1]) +center_by_minimum + 20
    
    return left_max, right_max

def match_images(img_a,img_b):
    span_a = img_a.shape[1]
    span_b = img_b.shape[1]

    if span_a > span_b:
        return img_a[:,span_a-span_b:-1], img_b
    else:
        return img_a, img_b[:,span_b-span_a:-1]

# %%
# folder_path = 'D:/Rowland/piv-data/2021-01-20'
# folder_path = "/Volumes/Backup Plus /ROWLAND/piv-data/2021-01-20"
folder_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-01-18'

result_folder_path = '_results/2021-01-20'

param_string_list = os.listdir(folder_path)
param_string_to_dictionary(param_string_list[5])

# %%
garo = 1408
sero = (1296)*11
entire_image = np.zeros((sero,garo))
# %%
param_string_5 = [param_string_list[5]]

for param_string in param_string_list:
    img_a_name = 'frame_000102.tiff'
    img_b_name = 'frame_000103.tiff'

    param_dict = param_string_to_dictionary(param_string)
    # print(param_dict)  
    try:
        file_path_a = os.path.join(folder_path,param_string,img_a_name)
        file_path_b = os.path.join(folder_path,param_string,img_b_name)        
        img_a = io.imread(file_path_a)
        img_b = io.imread(file_path_b)        
    except:
        file_path_a = os.path.join(folder_path,'img_' + param_string,img_a_name)
        file_path_b = os.path.join(folder_path,'img_' + param_string,img_b_name)

        img_a = io.imread(file_path_a)
        img_b = io.imread(file_path_b)          
        
    position = int(param_dict['pos'])
    v_offset = int(param_dict['VOFFSET'])    

    start_index = (position-1)*1296 + (v_offset//80)*108
    end_index = start_index + 108   

    entire_image[start_index:end_index,:] = img_a
    
    # a_left, a_right = split_image_arg(img_a)
    # b_left, b_right = split_image_arg(img_b)

    # right_edge_index = max(a_right,b_right)
    
    # img_a_right = img_a[:,right_edge_index:-1]
    # img_b_right = img_b[:,right_edge_index:-1]

    result_folder_path = '_results/2021-01-20'

    text_path = os.path.join(result_folder_path,param_string+'.txt')
    figure_path = os.path.join(result_folder_path,param_string+'.png')
    
    openpiv_recipes.run_piv(img_a,img_b,
        winsize=48,
        searchsize=50,
        overlap=24,
        show_vertical_profiles=False,
        image_check=False,
        figure_export_name=figure_path,
        text_export_name=text_path)

# %%
txt_list = [x for x in os.listdir(result_folder_path) if 'txt' in x]

entire_u_array = np.zeros((53,3*12*11))
entire_v_array = np.zeros((53,3*12*11))

for k in txt_list:
    param_dict = param_string_to_dictionary(k)
    # print(param_dict)

    result_path = os.path.join(result_folder_path,k)
    df = pd.read_csv(result_path,comment='#',delimiter='\t',names=['x','y','u','v','s2n','mask'])

    x_array = df.iloc[:,0].to_numpy()
    y_array = df.iloc[:,1].to_numpy()
    u_array = df.iloc[:,2].to_numpy()
    v_array = df.iloc[:,3].to_numpy()
    
    field_shape = pyprocess.get_field_shape(image_size=img_a.shape,search_area_size=search_size,overlap=overlap)

    num_rows = field_shape[0] # 11
    num_cols = field_shape[1] # 100
    
    # x_coord = x_array.reshape(num_rows,num_cols)[0,:]
    # y_coord = np.flipud(y_array.reshape(num_rows,num_cols)[:,0])
    u_array_reshaped = u_array.reshape(num_rows,num_cols).T
    v_array_reshaped = v_array.reshape(num_rows,num_cols).T

    position = int(param_dict['pos'])
    v_offset = int(param_dict['VOFFSET'])

    start_index = (position-1)*36 + (v_offset//80)*3
    end_index = start_index + 3

    entire_u_array[:,start_index:end_index] = u_array_reshaped
    entire_v_array[:,start_index:end_index] = v_array_reshaped

    
# %%
plt.figure(figsize=(8,8))
plt.quiver(entire_v_array[:,150:170],entire_u_array[:,150:170])
# %%
plt.figure(figsize=(20,3))
plt.quiver(entire_v_array,entire_u_array)

# %%
plt.figure(figsize=(15,4))
plt.imshow(entire_image.T)
plt.axis('off')
# plt.savefig('_stitched.png', bbox_inches='tight', pad_inches = 0)

# %%
search_size = 50
overlap = 24
# %%
openpiv_recipes.run_piv(img_a,img_b,
    winsize=48,
    searchsize=search_size,
    overlap=24,
    show_vertical_profiles=False,
    image_check=True,
    figure_export_name='results.png')

# %%

df = pd.read_csv('results.txt',comment='#',delimiter='\t',names=['x','y','u','v','s2n','mask'])

x_array = df.iloc[:,0].to_numpy()
y_array = df.iloc[:,1].to_numpy()
u_array = df.iloc[:,2].to_numpy()

field_shape = pyprocess.get_field_shape(image_size=img_a.shape,search_area_size=search_size,overlap=overlap)

num_rows = field_shape[0] # 11
num_cols = field_shape[1] # 100

# U_array = np.sqrt(uv_array[:,0]**2+uv_array[:,0]**2).reshape(num_rows,num_cols).T        
x_coord = x_array.reshape(num_rows,num_cols)[0,:]
y_coord = np.flipud(y_array.reshape(num_rows,num_cols)[:,0])
U_array = u_array.reshape(num_rows,num_cols).T

# %%

fig, ax = plt.subplots()
ax.plot(U_array[:,0],x_coord,'o-')
ax.plot(U_array[:,1],x_coord,'o-')
ax.plot(U_array[:,2],x_coord,'o-')


# %%
a = np.loadtxt('results.txt')
window_size = 48
scaling_factor = 1

xmax = np.amax(a[:, 0]) + window_size / (2 * scaling_factor)
ymax = np.amax(a[:, 1]) + window_size / (2 * scaling_factor)

ax.imshow(img_a, origin="lower", cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

# %%
fig,axes = plt.subplots(4,3,figsize=(16,16))

axes_flatten = np.ravel(axes)
ii = 0  
for ax in axes_flatten:            
    for i in range(ii*10,(ii+1)*10):
        if i >= num_rows:
            break
        ax.plot(U_array[:,i],x_coord,'o-',label='%.2f'%x_coord[i])
        ax.legend()
        ax.axis([0,np.max(U_array),0,np.max(y_coord)])        
    ax.set_xlabel('Velocity (mm/s)')
    ax.set_ylabel('y-coordinate (mm)')
    ii = ii + 1        
# fig.savefig('vertical_profile.png')

# %%

def stitch_images(folder_path):
    param_string_list = os.listdir(folder_path)
    for param_string in param_string_list:
        # CHECK IMAGE VALIDITY ?
        img_a_name = 'frame_000102.tiff'
        img_b_name = 'frame_000103.tiff'

        param_dict = param_string_to_dictionary(param_string)
        # print(param_dict)
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

        return entire_image

entire_image = stitch_images(folder_path)

# %%

def show_entire_image(entire_image):
    plt.figure(figsize=(15,4))
    plt.imshow(entire_image.T)
    plt.axis('off')
    # plt.savefig('_stitched.png', bbox_inches='tight', pad_inches = 0)

show_entire_image(entire_image)

# %%
def stitch_images(folder_path):
    param_string_list = os.listdir(folder_path)
    for param_string in param_string_list:
        # CHECK IMAGE VALIDITY ?
        img_a_name = 'frame_000102.tiff'
        img_b_name = 'frame_000103.tiff'

        param_dict = param_string_to_dictionary(param_string)
        # print(param_dict)
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

        return entire_image


openpiv_recipes.run_piv(img_a,img_b,
        winsize=48,
        searchsize=50,
        overlap=24,
        show_vertical_profiles=False,
        image_check=False,
        figure_export_name=figure_path,
        text_export_name=text_path)




# %%

openpiv_recipes.run_piv(img_a,img_b,
        winsize=48,
        searchsize=50,
        overlap=24,
        show_vertical_profiles=False,
        image_check=False,
        figure_export_name=figure_path,
        text_export_name=text_path)
