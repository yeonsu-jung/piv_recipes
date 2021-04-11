# %%
from openpiv import tools, process, validation, filters, scaling, pyprocess
import yaml

# %%
class piv_class():
    def __init__(self):
        with open('piv_setting.yaml') as f:
            piv_param = yaml.safe_load(f)
            for k,v in piv_cond.items():
                print(k,v)

    def show_piv_param(self):
        print("- PIV parameters -")
        for x, y in self.piv_param.items():
            print(x +":", y)

    def 

    def run_piv(self, path, index_a = 100, index_b = 101, folder = None):
        self.show_piv_param()
        ns = Namespace(**self.piv_param)

        if folder == None:
            img_a, img_b = self.read_two_images(search_dict,index_a=index_a,index_b=index_b,raw=ns.raw_or_cropped)

            location_path = [x['path'] for x in self.piv_dict_list if search_dict.items() <= x.items()]
            results_path = os.path.join(self.results_path,*location_path)
            try:
                os.makedirs(results_path)
            except FileExistsError:
                pass
        else:
            try:
                file_a_path = os.path.join(self.path,folder,'frame_%06d.tiff' %index_a)
                file_b_path = os.path.join(self.path,folder,'frame_%06d.tiff' %index_b)

                img_a = np.array(Image.open(file_a_path))
                img_b = np.array(Image.open(file_b_path))                
            except:
                return None
        
        # crop
        img_a = ndimage.rotate(img_a, ns.rotate)
        img_b = ndimage.rotate(img_b, ns.rotate)

        img_a = img_a[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]
        img_b = img_b[ns.crop[0]:-ns.crop[1]-1,ns.crop[2]:-ns.crop[3]-1]        
            
        u0, v0, sig2noise = process.extended_search_area_piv(img_a.astype(np.int32),
                                                            img_b.astype(np.int32),
                                                            window_size=ns.winsize,
                                                            overlap=ns.overlap, 
                                                            dt=ns.dt, 
                                                            search_area_size=ns.searchsize, 
                                                            sig2noise_method='peak2peak')

        x, y = process.get_coordinates(image_size=img_a.shape, 
                                        window_size=ns.winsize,                                    
                                        overlap=ns.overlap)

        x, y, u0, v0 = scaling.uniform(x, y, u0, v0, scaling_factor = ns.pixel_density) # no. pixel per distance

        u0, v0, mask = validation.global_val(u0,v0,ns.u_bound,ns.v_bound)        

        x,y,u0,v0,mask,sig2noise = peel_off_edges((x,y,u0,v0,mask,sig2noise))       

        print('Number of invalid vectors:',np.sum(np.isnan(mask)))

        u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                                sig2noise, 
                                                threshold = ns.sn_threshold)

        u3, v3 = filters.replace_outliers( u1, v1,
                                        method='localmean',
                                        max_iter=50,
                                        kernel_size=1)
        
        if ns.check_angle:
            # u3, v3 = angle_mean_check(u3,v3)           
            u3, v3 = correct_by_angle(u3,v3)

        #save in the simple ASCII table format        
        if ns.save_result is True:
            tools.save(x, y, u3, v3, mask, os.path.join(results_path,'Stream_%05d.txt'%index_a))            
            io.imwrite(os.path.join(results_path,ns.figure_export_name),img_a)
            quiver_and_contour(x,y,u3,v3,index_a,results_path,show_result = ns.show_result)
        
        if ns.image_check == True:
            fig,ax = plt.subplots(2,1,figsize=(24,12))
            ax[0].imshow(img_a)
            ax[1].imshow(img_b)       

        if ns.show_result == True:
            fig, ax = plt.subplots(figsize=(24,12))
            tools.display_vector_field( os.path.join(results_path,'Stream_%05d.txt'%index_a), 
                                        ax=ax,
                                        scaling_factor= ns.pixel_density, 
                                        scale=ns.scale_factor, # scale defines here the arrow length
                                        width=ns.arrow_width, # width is the thickness of the arrow
                                        on_img=True, # overlay on the image
                                        image_name= os.path.join(results_path,ns.figure_export_name))
            fig.savefig(os.path.join(results_path,ns.figure_export_name))                    
        
        print('Mean of u: %.3f' %np.mean(u3))
        print('Std of u: %.3f' %np.std(u3))        
        print('Mean of v: %.3f' %np.mean(v3))
        print('Std of v: %.3f' %np.std(v3))

        output = np.array([np.mean(u3),np.std(u3),np.mean(v3),np.std(v3)])
        # if np.absolute(np.mean(v3)) < 50:
        #     output = self.quick_piv(search_dict,index_a = index_a + 1, index_b = index_b + 1)

        return x,y,u3,v3

# %%
pi = piv_class()
# %%
