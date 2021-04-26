import numpy as np
# from matplotlib import pyplot as plt
from skimage import measure
from scipy import ndimage
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def get_wall_pos2(img, check_img = False): # img or img path
    num_col = img.shape[1]
    row_averaged = np.zeros(num_col)
    for i in range(num_col):
        row_averaged[i] = np.sum(img[:,i])

    # plt.plot(row_averaged,'o-')
    # plt.show()

    peaks, _ = find_peaks(row_averaged,distance=100)
    peaks = sorted(peaks,key = lambda x: -row_averaged[x])
    
    # print(peaks)

    # plt.plot(row_averaged[peaks],'o-')
    # plt.show()

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