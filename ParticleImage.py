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