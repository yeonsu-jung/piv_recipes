from openpiv import tools, pyprocess, validation, filters, scaling 
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import cv2
import imageio

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
    figure_export_name = 'results.png',
    text_export_name =  "results.txt",
    scale_factor = 1
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
                                scaling_factor = 41.22 ) # 41.22 microns/pixel        

    #save in the simple ASCII table format    
    tools.save(x, y, u3, v3, sig2noise,mask, text_export_name)
    
    if image_check == True:
        fig,ax = plt.subplots(2,1,figsize=(24,12))
        ax[0].imshow(frame_a)
        ax[1].imshow(frame_b)

    imageio.imwrite('test_img.tiff',frame_a)

    # fig, ax = plt.subplots(figsize=(24,12))
    # tools.display_vector_field(text_export_name, 
    #                                ax=ax, scaling_factor=41.22, 
    #                                scale=2e5*scale_factor, # scale defines here the arrow length
    #                                width=0.001, # width is the thickness of the arrow
    #                                on_img=True, # overlay on the image
    #                                image_name= 'test_img.tiff');            
    # fig.savefig(figure_export_name)       

    if show_vertical_profiles:
        field_shape = pyprocess.get_field_shape(image_size=frame_a.shape,search_area_size=searchsize,overlap=overlap)
        vertical_profiles(text_export_name,field_shape)


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
    

