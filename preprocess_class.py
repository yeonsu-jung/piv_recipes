# %%
from re import A
# import cv2
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

    def get_wall_pixels(self):
        
        n = np.max(self.labels)

        im_area = np.zeros(n)
        for i in range(1,n):
            im_area[i] = np.sum(self.labels == i)

        two_max = np.argsort(im_area)[-2:]

        wall_1 = self.labels == two_max[0]
        wall_2 = self.labels == two_max[1]
        wall = wall_1 | wall_2
        
        return wall_1, wall_2        


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
        ytmp = np.arange(240).astype(np.uint16)        
        plt.plot(ytmp,foo(ytmp,*popt),'o')

        # print(popt,pcov)        
        print(popt)

        xtmp = foo(ytmp,*popt)
        
        idx = (ytmp,xtmp.astype(np.uint16))
        # print(idx)

        out = np.zeros(wall_img.shape,dtype='uint16')
        out[idx] = 1

        plt.imshow(out,cmap=plt.get_cmap('gray'))        
        
        return idx, out

    def fill_sample_region(self,idx1,idx2):        
        A = np.ones(self.bw.shape)
        N = idx1[0].shape[0]

        for i in range(N):    
            left = np.min((idx1[1][i],idx2[1][i]))
            right = np.max((idx1[1][i],idx2[1][i]))
            A[i, left:right] = 0

        return A

    

    
        
# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06'
parent_path = parent_path.replace('C:/Users/yj/','/Users/yeonsu/')
piv = pi.piv_class(parent_path)

# %%
path_a='C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7'
path_a = path_a.replace('C:/Users/yj/','/Users/yeonsu/')
path_b='C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7'
path_b = path_a.replace('C:/Users/yj/','/Users/yeonsu/')

img_a = piv.read_image_from_path(path_a,index=100)
img_a = np.array(img_a)
img_b = piv.read_image_from_path(path_b,index=101)
img_b = np.array(img_b)
# %%
pp = preprocess_class(img_a)
w1,w2 = pp.get_wall_pixels()
idx1,w1_new = pp.locate_lines2(w1)
idx2,w2_new = pp.locate_lines2(w2)

A = pp.fill_sample_region(idx1,idx2)
final_a = img_a*A

# pp2 = preprocess_class(img_b)
# w1,w2 = pp2.get_wall_pixels()
# idx1,w1_new = pp2.locate_lines2(w1)
# idx2,w2_new = pp2.locate_lines2(w2)

# A = pp.fill_sample_region(idx1,idx2)
final_b = img_b*A


# %%
plt.subplot(2,1,1)
plt.imshow(final_a)
plt.subplot(2,1,2)
plt.imshow(final_b)
# %%
a = np.empty((3,3,))
a[:] = np.nan

cv2.imwrite('_test/test.tiff',a)
# %%
im = cv2.imread('_test/test.tiff')
print(im)
# %%
import piv_class
reload(piv_class)

pth_a = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7_filled/frame_000100.tiff'
pth_b = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7_filled/frame_000101.tiff'
piv_class.run_piv(pth_a,pth_b)
# %%
pth_a = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/frame_000100.tiff'
pth_b = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/frame_000101.tiff'
piv_class.run_piv(pth_a,pth_b)

# %%
piv.piv_over_time(start_index=3,N=95)

# %%
import re
import imageio as io
import cv2
lis = sorted(os.listdir(path_a),key = lambda x: re.findall('\d+',x)[0])

for impth in lis:
    pth = os.path.join(path_a,impth)
    im = np.array(io.imread(pth))
    im_out = im*A
    pth_out = os.path.join(path_a + '_filled',impth)
    try:        
        os.makedirs(path_a + '_filled')
    except:
        pass
    cv2.imwrite(pth_out,im_out.astype(np.uint16))

# %%
im_new = io.imread(pth_out)

plt.imshow(im_new)
# %%
path_a

# %%
re.findall('\d+','frame_000601.tiff')[0]
# %%
os.listdir(path_a)
# %%
# %%

# %%
def foo(path):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    for i in range(u.shape[0]):
        plt.plot(-v[i,0,:],x[0,:],'b.')
    os.chdir(owd)


# %%
def foo2(path):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owd = os.getcwd()
    os.chdir(path)
    
    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")
    u = pi.load_nd_array('u.txt')
    v = pi.load_nd_array('v.txt')

    i = 0
    
    u_avg = np.mean(u,axis=0)
    v_avg = np.mean(v,axis=0)

    plt.plot(-v_avg[0,:],x[0,:],'o-')
    os.chdir(owd)
# %%
foo('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')
plt.ylim([20,30])
# %%
foo('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')
plt.ylim([0,18])

# %%
idx1
# %%
idx1[1].mean().astype(np.uint16)
idx2[1].mean().astype(np.uint16)
idx1[1].mean(),idx2[1].mean()
# %%


# %%
x_wall = idx1[1].mean()/40

foo2('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')
plt.plot([0,500],[x_wall, x_wall],'r-')
plt.ylim([x_wall,30])

# %%
x_wall = idx2[1].mean()/40

foo2('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95')
plt.plot([0,500],[x_wall, x_wall],'r-')
plt.ylim([0,x_wall])
# %%
