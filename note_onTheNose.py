# %%
import piv_class as pi
from importlib import reload

reload(pi)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-07/Flat_10 (black)_stitching process'
ins = pi.piv_class(parent_path)
# %%
d = ins.quick_piv(index = 100) #
# %%
ins.piv_over_time(start_index=3,N=95)
# %%
ins.piv_over_sample(start_index=3,N=95)

# %%
parent_path = 'C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-08/Flat_10 (black)_motor10_stitching'
ins = pi.piv_class(parent_path)
# %%
d = ins.quick_piv(index = 100) #

# %%
