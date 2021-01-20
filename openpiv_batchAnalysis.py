# %%
from openpiv import tools, pyprocess, validation, filters, scaling 
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
# import cv2
import time
from scipy import ndimage
import math

# %% Functions
def validity_check(frame_a,frame_b):
    flag = np.sqrt( np.sum((frame_a - frame_b)**2) / np.sum(frame_a-frame_b) )
    if flag > 0.9:
        return True
    else:
        return False
       
def run_piv(frame_a,frame_b,):
    winsize = 64 # pixels, interrogation window size in frame A
    searchsize = 68  # pixels, search in image B
    overlap = 32 # pixels, 50% overlap
    dt = 0.0005 # sec, time interval between pulses

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                        frame_b.astype(np.int32), 
                                                        window_size=winsize, 
                                                        overlap=overlap, 
                                                        dt=dt, 
                                                        search_area_size=searchsize, 
                                                        sig2noise_method='peak2peak')

    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, 
                                    search_area_size=searchsize, 
                                    window_size = winsize,
                                    overlap=overlap)

    u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                            sig2noise, 
                                            threshold = 1.05 )

    u2, v2 = filters.replace_outliers( u1, v1, 
                                    method='localmean', 
                                    max_iter=10, 
                                    kernel_size=3)

    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                                scaling_factor = 41.22 ) # 41.22 microns/pixel

    mean_u = np.mean(u3)
    mean_v = np.mean(v3)

    deficit_u = u3 - mean_u
    deficit_v = v3 - mean_v
    
    u_prime = np.mean(np.sqrt(0.5*(deficit_u**2 + deficit_v**2)))
    u_avg = np.mean(np.sqrt(0.5*(mean_u**2 + mean_v**2 )))

    turbulence_intensity = u_prime / u_avg    

    #save in the simple ASCII table format
    fname =  "./Tables/" + exp_string + ".txt"
    # tools.save(x, y, u3, v3, mask, fname)

    out = np.vstack([m.ravel() for m in [x, y, u3, v3]])    
    # print(out)
    # np.savetxt(fname,out.T)

    with open(fname, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, out.T)

    return turbulence_intensity


# %%

folder_path = '../../PIV(raw)/2020-11-24/'
path_list = os.listdir(folder_path)

# %%
for exp_string in path_list:
    path = '../../PIV(raw)/2020-11-24/' + exp_string + '/'
    try:
        path = '../../PIV(raw)/2020-11-24/' + exp_string + '/'
        frame_a = tools.imread(path + 'frame_000301.tiff')
        frame_b = tools.imread(path + 'frame_000302.tiff')        
    except:
        print('No proper image pairs in the directory. Skipped.')

    frame_a = tools.imread(path + 'frame_000301.tiff')
    frame_b = tools.imread(path + 'frame_000302.tiff')

    if validity_check(frame_a,frame_b):
        frame_numbers = range(201,401,1)
    else:
        frame_numbers = range(200,400,1)

    ti_array = np.zeros(100)


    t = time.time()
    # print(frame_numbers)
    j = 0
    for i in frame_numbers[::2]:
        frame_a = tools.imread(path + 'frame_000%d.tiff' %i)
        frame_b = tools.imread(path + 'frame_000%d.tiff' %(i+1))
        ti_array[j] = run_piv(frame_a,frame_b)
        j = j+1
    elapsed = time.time() - t
    print('Elapsed time = %.2f' %elapsed)

    np.savetxt('ti_' + exp_string + '.txt',ti_array)
    #plt.plot(ti_array,'o-')
# %% Test run_piv
# i = 17
# frame_a = tools.imread(path + 'frame_000%d.tiff' %frame_numbers[i])
# frame_b = tools.imread(path + 'frame_000%d.tiff' %frame_numbers[i+1])

# ti = run_piv(frame_a,frame_b)

# fig, ax = plt.subplots(figsize=(20,20*240/1920))
# tools.display_vector_field('test.txt', 
#                                ax=ax, scaling_factor=41.22, 
#                                scale=0.5e5, # scale defines here the arrow length
#                                width=0.001, # width is the thickness of the arrow
#                                on_img=False, # overlay on the image
#                                image_name= path + 'frame_000%d.tiff' %frame_numbers[16]);


    








# %%



