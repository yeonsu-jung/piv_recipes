# %%
from re import A
import cv2
from skimage.measure import regionprops
import piv_class as pi
from importlib import reload
import numpy as np
from matplotlib import pyplot as plt

from skimage import measure
from scipy import ndimage
from scipy.optimize import curve_fit

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
# filtering, etc etc, ...
class preprocess_class:
    def __init__(self,img) -> None: # img or img path

        self.bw = img[:,:] > 100
        self.labels = measure.label(self.bw)
        n = np.max(self.labels)

        im_area = np.zeros(n)
        for i in range(1,n):
            im_area[i] = np.sum(self.labels == i)

        two_max = np.argsort(im_area)[-2:]

        wall_1 = self.labels == two_max[0]
        wall_2 = self.labels == two_max[1]
        wall = wall_1 | wall_2

        

        plt.imshow(wall.astype(np.uint8),cmap=plt.cm.gray)

    def locate_line(self,wall_img):
        nonzeros = np.where(wall_img)
        nz_y = nonzeros[0]
        nz_x = nonzeros[1]    

        data = np.hstack((nz_x[:,np.newaxis],nz_y[:,np.newaxis]))
        datamean = data.mean(axis=0)

        uu, dd, vv = np.linalg.svd(data - datamean)

        linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]
        # shift by the mean to get the line in the right place
        linepts += datamean    

        # plt.plot(nz_x,nz_y,'.')
        # plt.axis('equal')

        plt.scatter(*data.T)
        plt.plot(*linepts.T,'k-')
        # plt.plot(*linepts2.T,'k--')
        plt.axis('equal')
        plt.xlim([0,1280])
        plt.ylim([0,240])    

        print('Centroid:', datamean)
        print('Angle:', np.arctan(vv[0,0]/vv[0,1]) * 180/np.pi)

    def locate_lines2(self,wall_img):
        def foo(x,a,b):
            return a*x + b

        nonzeros = np.where(wall_img)
        nz_y = nonzeros[0]
        nz_x = nonzeros[1]    

        popt, pcov = curve_fit(foo,nz_y,nz_x)
        
        # xtmp = np.mgrid[0:240:1j]
        ytmp = np.arrange(240)
        print(ytmp)
        plt.plot(ytmp,foo(ytmp,*popt),'o')       
        plt.show()

    def fill_sample(self):
        pass    

# %%
