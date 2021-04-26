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
    # print(first_image.parent)
    # print( len( [x for x in first_image.parent.iterdir() if 'frame' in x.name]) )
    containing_dir = first_image.parent
    img_a = mpimg.imread(containing_dir.joinpath('frame_000010.tiff'))
    img_b = mpimg.imread(containing_dir.joinpath('frame_000011.tiff'))

    wa1_pos, wa2_pos, wa1, wa2 = get_wall_pos2(img_a, check_img=False)
    wb1_pos, wb2_pos, wb1, wb2 = get_wall_pos2(img_b, check_img=False)

    wpos_a_path = base_to_out(containing_dir).joinpath('wall_a_position.txt')
    wpos_b_path = base_to_out(containing_dir).joinpath('wall_b_position.txt')

    np.savetxt(wpos_a_path,[wa1_pos,wa2_pos],fmt='%d')
    np.savetxt(wpos_b_path,[wb1_pos,wb2_pos],fmt='%d')

# %%


    





# %%
