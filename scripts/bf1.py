#!/usr/bin/env python
# coding: utf-8

# In[1]:

import import_ipynb
from pathlib import Path
import BurstFit_paper_template as paper_fit

from burstfit.fit import real_time_burstfit
import numpy as np


#import sys
#sys.path.insert(1, '/home/ubuntu/gechen/software/burstfit/examples/')

# In[ ]:
# python bf1.py 220330aaan 2022_3_29_23_4_54 164 corr09 12.9 467.8 32  
# candidate datestring beam corr snr_heimdall dm_heimdall width_heimdall

# candidate = str(sys.argv[1])
# datestring = str(sys.argv[2])
# beam = int(sys.argv[3])
# corr = str(sys.argv[4])
# snr_heimdall = float(sys.argv[5])
# dm_heimdall = float(sys.argv[6])
# width_heimdall = int(sys.argv[7])


# real_time_burstfit(candidate, datestring, beam, corr, snr_heimdall, dm_heimdall, width_heimdall, save_plot = True, plot = False)

# python bf1.py candidate date_string SNR DM width 

# works: 
# python bf1.py 220330aaan /data/dsa110/T1/corr09/2022_3_29_23_4_54/fil_220330aaan/220330aaan_164.fil 12.9 467.8 32  
# python bf1.py 220204aaai /data/dsa110/T1/corr13/2022_2_4_3_17_59/fil_220204aaai/220204aaai_209.fil 16.2 612.6 4

# bad fitting result: 
# python bf1.py 220319aaeb /data/dsa110/T1/corr09/2022_3_18_4_44_53/fil_220319aaeb/220319aaeb_172.fil 41.7 111 1



if __name__ == "__main__":
    import sys 
    
    candidate = str(sys.argv[1])
    path_to_fil_file = str(sys.argv[2])
    snr_heimdall = float(sys.argv[3])
    dm_heimdall = float(sys.argv[4])
    width_heimdall = int(sys.argv[5])


    real_time_burstfit(candidate, path_to_fil_file, snr_heimdall, dm_heimdall, width_heimdall, save_plot = True, plot = False)


