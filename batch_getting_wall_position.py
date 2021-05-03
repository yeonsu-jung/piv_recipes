# %%
import numpy as np
import os
import re
import yaml
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from get_wall_position import get_wall_pos2

# import pathlib
from pathlib import Path
# %%
base_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/')
out_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/')

date = '2021-04-06'
# %%
def base_to_out(path_in):    
    assert isinstance(path_in,Path)        
    return out_path.joinpath(path_in.relative_to(base_path))

# %%
date_path = base_path.joinpath(date)
# %%
first_image_path_list = date_path.glob('**/frame_000001.tiff')
first_image_path_list = sorted(first_image_path_list)
# %%
for first_image in first_image_path_list:
    print(first_image.parent.name)

# %%

# %%
for first_image in first_image_path_list[5:]:
    # print(first_image.parent)
    # print( len( [x for x in first_image.parent.iterdir() if 'frame' in x.name]) )
    containing_dir = first_image.parent
    img_a = mpimg.imread(containing_dir.joinpath('frame_000010.tiff'))
    img_b = mpimg.imread(containing_dir.joinpath('frame_000011.tiff'))

    wa1_pos, wa2_pos, wa1, wa2 = get_wall_pos2(img_a, check_img=True)
    wb1_pos, wb2_pos, wb1, wb2 = get_wall_pos2(img_b, check_img=True)

    wpos_a_path = base_to_out(containing_dir).joinpath('wall_a_position.txt')
    wpos_b_path = base_to_out(containing_dir).joinpath('wall_b_position.txt')

    np.savetxt(wpos_a_path,[wa1_pos,wa2_pos],fmt='%d')
    np.savetxt(wpos_b_path,[wb1_pos,wb2_pos],fmt='%d')

# %%
from scipy.signal import find_peaks

def get_wall_pos3(img, check_img = False): # img or img path
    num_col = img.shape[1]
    center = num_col//2

    row_averaged = np.zeros(num_col)
    for i in range(num_col):
        row_averaged[i] = np.sum(img[:,i])

    plt.plot(row_averaged,'o-')
    plt.show()

    peaks, _ = find_peaks(row_averaged,distance=150)
    # peaks = sorted(peaks,key = lambda x: -row_averaged[x])
    peaks = sorted(peaks,key = lambda x: (center-x)**2)
    
    print(peaks)

    plt.plot(row_averaged[peaks],'o-')
    plt.show()

    idx1 = min(peaks[0],peaks[1])
    idx2 = max(peaks[0],peaks[1])

    w1_new = np.zeros(img.shape,dtype=np.bool)
    w2_new = np.zeros(img.shape,dtype=np.bool)

    w1_new[:,idx1] = 1    
    w2_new[:,idx2] = 1

    if check_img:
        row_dim = img.shape[0]
        plt.imshow(img)
        plt.plot([idx1,idx1],[0,row_dim],'r-',linewidth=0.5)
        plt.plot([idx2,idx2],[0,row_dim],'r-',linewidth=0.5)
        plt.show()

    return idx1, idx2, w1_new, w2_new

# %%
d = get_wall_pos3(img_a, check_img=True)

# %%
lst = [Path('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-06/Flat_10 (black)_motor10_particle4_hori1024_laser1-4_nd0p7/frame_000010.tiff')]
for first_image in lst:
    # print(first_image.parent)
    # print( len( [x for x in first_image.parent.iterdir() if 'frame' in x.name]) )
    containing_dir = first_image.parent
    img_a = mpimg.imread(containing_dir.joinpath('frame_000010.tiff'))
    img_b = mpimg.imread(containing_dir.joinpath('frame_000011.tiff'))

    wa1_pos, wa2_pos, wa1, wa2 = get_wall_pos3(img_a, check_img=True)
    wb1_pos, wb2_pos, wb1, wb2 = get_wall_pos3(img_b, check_img=True)

    wpos_a_path = base_to_out(containing_dir).joinpath('wall_a_position.txt')
    wpos_b_path = base_to_out(containing_dir).joinpath('wall_b_position.txt')

    np.savetxt(wpos_a_path,[wa1_pos,wa2_pos],fmt='%d')
    np.savetxt(wpos_b_path,[wb1_pos,wb2_pos],fmt='%d')
# %%
tmp = np.loadtxt('/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-results/2021-04-06/Flat_10 (black)_motor10_particle4_hori1024_laser1-4_nd0p7/0,0,0,0_32,26,50/series_003_95/x.txt')
# %%
tmp.shape
# %%
import numpy as np
# from matplotlib import pyplot as plt
from skimage import measure
from scipy import ndimage
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def get_wall_pos(img): # img or img path
    bw = img[:,:] > 120
    labels = measure.label(bw)

    w1, w2 = get_wall_pixels(labels)

    idx1, w1_new = locate_lines2(w1)
    idx2, w2_new = locate_lines2(w2)

    w1_pos = min([np.min(idx1[1]), np.min(idx2[1])])
    w2_pos = max([np.min(idx1[1]), np.min(idx2[1])])

    return w1_pos, w2_pos, w1, w2    

def get_wall_pixels(labels):
    
    n = np.max(labels)

    im_area = np.zeros(n)
    for i in range(1,n):
        im_area[i] = np.sum(labels == i)

    two_max = np.argsort(im_area)[-2:]

    wall_1 = labels == two_max[0]
    wall_2 = labels == two_max[1]
    wall = wall_1 | wall_2
    
    return wall_1, wall_2        


    plt.imshow(wall.astype(np.uint8),cmap=plt.cm.gray)

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

def locate_lines2(wall_img):
    def foo(x,a,b):
        return a*x + b

    nonzeros = np.where(wall_img)
    nz_y = nonzeros[0]
    nz_x = nonzeros[1]    

    popt, pcov = curve_fit(foo,nz_y,nz_x)
    
    # xtmp = np.mgrid[0:240:1j]
    ytmp = np.arange(240).astype(np.uint16)        
    # plt.plot(ytmp,foo(ytmp,*popt),'o')

    # print(popt,pcov)        
    # print(popt)

    xtmp = foo(ytmp,*popt)    
    idx = (ytmp,xtmp.astype(np.uint16))
    # print(idx)

    out = np.zeros(wall_img.shape,dtype='uint16')
    out[idx] = 1

    plt.imshow(out,cmap=plt.get_cmap('gray'))        
    
    return idx, out

def fill_sample_region(bw,idx1,idx2):        
    A = np.ones(bw.shape)
    N = idx1[0].shape[0]

    for i in range(N):    
        left = np.min((idx1[1][i],idx2[1][i]))
        right = np.max((idx1[1][i],idx2[1][i]))
        A[i, left:right] = 0

    return A
# %%
_,_,w1,w2 = get_wall_pos(img_b)
# %%

label_b = measure.label(img_b>80)
w1,w2 = get_wall_pixels(label_b)
plt.imshow(w1)
# %%
plt.hist(img_b.flatten())

# %%

import numpy as np
import warnings
 
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    """
    Anisotropic diffusion.
 
    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)
 
    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration
 
    Returns:
            imgout   - diffused image.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
 
    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in range(niter):
 
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        imgout += gamma*(NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return imgout
# %%
filt_b = anisodiff(img_b)
# %%
plt.subplot(2,1,1)
plt.imshow(filt_b)
plt.subplot(2,1,2)
plt.imshow(img_b)
# %%