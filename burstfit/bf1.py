#!/usr/bin/env python
# coding: utf-8

# In[1]:

import import_ipynb
from pathlib import Path
import BurstFit_paper_template as paper_fit

from burstfit.utils.functions import pulse_fn, sgram_fn_vec, sgram_fn, gauss, gauss_norm, model_free_4, model_free_normalized_4, power_law # pulse_fn_vec  
from burstfit.utils.plotter import plot_fit_results
import numpy as np



def real_time_burstfit(candidate, path_to_fil_file, snr_heimdall, dm_heimdall, width_heimdall,
                       voltage = False, dedisperse = True, mask_chans=[], nfreq = 4, save_plot = True,
                       plot = False
                      ): # datestring, beam, corr

    fil_file_dedispersed = '/home/ubuntu/gechen/software/burstfit/fil_files_dedispersed/%s_%d_%s.txt'%(candidate, nfreq, str(voltage))

    bd_heimdall, fil_file_heimdall  = paper_fit.prepare_bd(candidate, dm_heimdall, width_heimdall, snr_heimdall, voltage=False, path_to_fil_file = path_to_fil_file) # datestring=datestring, beam=beam, corr=corr

    data_burst, i_low, i_high = paper_fit.prepare_burst_data_heimdall_only(fil_file_heimdall, fil_file_dedispersed, candidate, bd_heimdall, t_chop_width = 5, dedisperse=dedisperse, plot = plot)

    # fit 
    profile_bounds=[(0, 0.5 / bd_heimdall.tsamp - i_low - bd_heimdall.width, 0, 0),(np.inf, 0.5 / bd_heimdall.tsamp - i_low + bd_heimdall.width, 5 * bd_heimdall.width, np.inf)]

    bf_S1T2_c1 = paper_fit.fit_paper_curvefit(bd_heimdall, data_burst, ['S', 'mu_t', 'sigma_t', 'tau'], pulse_fn, profile_bounds, ['c0', 'c1', 'c2'], model_free_normalized_4, fix_ncomp = True, ncomp = 1, plot = plot)

    # dm from fit 
    dm_fit, dm_fit_err = bf_S1T2_c1.sgram_params['all'][1]['popt'][-1], bf_S1T2_c1.sgram_params['all'][1]['perr'][-1]

    width_samp, width_samp_fine_pow2 = paper_fit.bin_best(bf_S1T2_c1, bd_heimdall)
    bin_num = paper_fit.bin_number(bf_S1T2_c1, bd_heimdall, i_low, width_samp_fine_pow2)
    #print("width_samp, width_samp_fine_pow2, bin_num = ", width_samp, width_samp_fine_pow2, bin_num)

    # dedisperse data using the new dm and compare SNR using adjecent bin widths
    data_new_dm, dm_fit = paper_fit.Dedisperse_data_profile(bf_S1T2_c1, fil_file_heimdall, width_samp)
    best_bin_snr = paper_fit.Compare_bins(data_new_dm, bd_heimdall, width_samp_fine_pow2, plot = plot)

    # save optimal bin width and fit results. 
    dict_burstfit = {
        "bf1_width_bins": width_samp_fine_pow2, # already converted to voltage resolution 
        "bf1_start_bins": bin_num, # already converted to voltage resolution
        "bf1_dm":dm_fit, 
        "bf1_dm_stddev": dm_fit_err,
        "bf1_reduced_chisq": bf_S1T2_c1.reduced_chi_sq,
        "bf1_pvalue": bf_S1T2_c1.p_value
    }
    
    print(dict_burstfit)

    if save_plot:
        save_plot_dir = "/dataz/dsa110/operations/candidates/" + candidate + "/other"
        Path(save_plot_dir).mkdir(parents=True, exist_ok=True)

        plot_fit_results(bf_S1T2_c1.sgram, bf_S1T2_c1.sgram_model.evaluate, bf_S1T2_c1.sgram_params['all'][1]['popt'], 
                          bf_S1T2_c1.tsamp, bf_S1T2_c1.fch1, bf_S1T2_c1.foff, show = False, save=True, 
                         outname = 'bf1_' + candidate, 
                         outdir = save_plot_dir)        

    return dict_burstfit


    

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


