# %%
from re import A
import cv2
from skimage.measure import regionprops
import piv_class as pi
from importlib import reload
import numpy as np
from matplotlib import pyplot as plt

from skimage import measure

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06'
ins = pi.piv_class(parent_path)

# %%
for lst in os.listdir(parent_path):
    pth = os.path.join(parent_path,lst)
    if os.path.isdir(pth):
        print(pth)

# %%

def get_edge(img, th = 110):
    bw = img > th
    n, labels = cv2.connectedComponents(bw.astype(np.uint8))
    edge = labels==1

    return edge

def get_edge_removed_img(img,edge):
    img_wo_edge = img * (np.invert(edge)).astype(np.uint8)
    return img_wo_edge

def get_rightmost(row,col):
    rightmost = np.zeros(np.max(row))
    idx = 0
    for i in range(np.max(row)):
        idx = idx + len([x for x in row if x == i])-1   
        rightmost[i] = col[idx]
    return rightmost

def get_img_wo_sample(img,rightmost):
    img_wo_sample = img
    for i in range(len(rightmost)):        
        # print(i,int(rightmost[i]))
        img_wo_sample[i,:int(rightmost[i])] = 0
    return img_wo_sample


# %%
img_a = ins.read_image_from_path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06\Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7',index=100)
img_b = ins.read_image_from_path('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06\Flat_10 (black)_motor10_particle4_hori1280_laser1-4_nd0p7',index=101)
# %%
bw = img_a[:,:] > 100
plt.imshow(bw)

# %%

labels = measure.label(bw, background=0)
n = np.max(labels)

im_area = np.zeros(n)
for i in range(1,n):
    im_area[i] = np.sum(labels == i)

two_max = np.argsort(im_area)[-2:]

wall_1 = labels == two_max[0]
wall_2 = labels == two_max[1]
wall = wall_1 | wall_2

plt.imshow(wall.astype(np.uint8),cmap=plt.cm.gray)
# %%
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
plt.imshow(img_a[:,600:])
plt.subplot(2,1,2)
plt.imshow(img_a[:,600:]*~wall)
# %%

def locate_line(wall_img):
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

# %%
locate_line(wall_2)
# %%
from scipy import ndimage

wall_1_rot = ndimage.rotate(wall_1,0.4149)
wall_2_rot = ndimage.rotate(wall_2,0.4760)
# %%
# locate_line(wall_1_rot)
locate_line(wall_2_rot)

# %%


# %%

nonzeros = np.where(wall_2)
nz_y = nonzeros[0]
nz_x = nonzeros[1]

plt.plot(nz_x,nz_y,'.')
plt.axis('equal')
# %%

# %%
nz_y[:,np.newaxis].shape
# %%
data = np.hstack((nz_x[:,np.newaxis],nz_y[:,np.newaxis]))
datamean = data.mean(axis=0)

uu, dd, vv = np.linalg.svd(data - datamean)

linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]
# shift by the mean to get the line in the right place
linepts += datamean

plt.scatter(*data.T)
plt.plot(*linepts.T,'k-')
# plt.plot(*linepts2.T,'k--')
plt.axis('equal')
plt.xlim([0,1280])
plt.ylim([0,240])
# %%
datamean
# %%
vv[0,0]
vv[0,1]

print(np.arctan(vv[0,0]/vv[0,1]) * 180/np.pi)

# %%
row = np.floor(linepts.T[0]).astype(np.uint16)
col = np.floor(linepts.T[1]).astype(np.uint16)
row,col
# %%
np.where(wall_1)
# %%
ln = np.zeros(bw.shape)

idx = ( (col,row) )
idx
# %%
ln[idx] = 1

plt.imshow(ln)

# %%



# %%

vv[0] * np.mgrid[0:250:2j][:, np.newaxis]

# %%
def foo(ln):
    print(ln)

# %%
print(*linepts.T)



# np.sum(vv[0]**2)

# %%
linepts

# %%



# %%
vv[0]

# %%
from numpy.linalg import svd

u, s, vh = svd(wall_1)
# %%

x = np.mgrid[-2:5:120j]
y = np.mgrid[1:9:120j]
z = np.mgrid[-5:3:120j]

x,y,z
x.shape
# %%
x[:, np.newaxis].shape
# %%
data = np.concatenate((x[:, np.newaxis], 
                       y[:, np.newaxis], 
                       z[:, np.newaxis]), 
                      axis=1)

data += np.random.normal(size=data.shape) * 0.4
# %%

data = wall_1
datamean = data.mean(axis=0)

# Do an SVD on the mean-centered data.
uu, dd, vv = np.linalg.svd(data - datamean)

# Now vv[0] contains the first principal component, i.e. the direction
# vector of the 'best fit' line in the least squares sense.

# Now generate some points along this best fit line, for plotting.

# I use -7, 7 since the spread of the data is roughly 14
# and we want it to have mean 0 (like the points we did
# the svd on). Also, it's a straight line, so we only need 2 points.

linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

# shift by the mean to get the line in the right place
linepts += datamean

# Verify that everything looks right.

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

# ax = m3d.Axes3D(plt.figure())
plt.scatter(data.T)
plt.plot(linepts.T)
plt.show()
# %%
np.mgrid[-7:7:2j][:, np.newaxis]
# %%
# np.mgrid[-7:7:2j][:, np.newaxis]
plt.plot(uu[0])

# %%
vv[0].shape # first ROW of vv matrix

# %%
# np.mgrid[-7:7:4j]


# %%


# %%
wall_1.dtype
# %%

fig = plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.imshow(bw)
plt.subplot(1,2,2)
plt.imshow(labels == 0)
# %%
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np

bw = img_a[:,800:] > 110
labels = measure.label(bw)
# plt.imshow(labels)

# n = 12
# l = 256
# np.random.seed(1)
# im = np.zeros((l, l))
# points = l * np.random.random((2, n ** 2))

# im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1

# im = filters.gaussian(im, sigma= l / (4. * n))
# blobs = im > 0.7 * im.mean()

all_labels = measure.label(bw)
blobs_labels = measure.label(bw, background=0)

plt.figure(figsize=(20, 3.5))
plt.subplot(131)
plt.imshow(bw, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels, cmap='nipy_spectral')
plt.axis('off')
plt.subplot(133)
plt.imshow(blobs_labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

# %%

edge = get_edge(img_a)
plt.imshow(edge)

# %%
img_a_new = get_edge_removed_img(img_a,get_edge(img_a,th = 150))
img_b_new = get_edge_removed_img(img_b,get_edge(img_b,th = 150))

# %%

row, col = np.where(img_a_new == 0)
rightmost_a = get_rightmost(row,col)
row, col = np.where(img_b_new == 0)
rightmost_b = get_rightmost(row,col)

iws_a = get_img_wo_sample(img_a,rightmost_a)
iws_b = get_img_wo_sample(img_b,rightmost_a)

fig,ax = plt.subplots(2)
ax[0].imshow(iws_a)
ax[1].imshow(iws_b)
# %%
import preprocess_class as pc
reload(pc)
# %%
pp = pc.preprocess_class(img_a)

# %%
pp.locate_lines2(wall_1)
# %%
