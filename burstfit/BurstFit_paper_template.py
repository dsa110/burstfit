#!/usr/bin/env python
# coding: utf-8

# In[2]:


from burstfit.fit import BurstFit
from burstfit.data import BurstData
from burstfit.model import Model, SgramModel
from burstfit.utils.plotter import plot_me
from burstfit.utils.functions import pulse_fn, sgram_fn_vec, sgram_fn, gauss, gauss_norm, model_free_4, model_free_normalized_4, power_law # pulse_fn_vec  
from burstfit.utils.plotter import plot_fit_results, plot_2d_fit 
from burstfit.io import BurstIO
import logging
import numpy as np
import math
import matplotlib.pyplot as plt
# use Liam's function to read in filterbank files 
import sys
sys.path.insert(1, '/home/ubuntu/gechen/software')
import filplot_funcs_gc as ff


def prepare_bd(candidate, dm_heimdall, width_heimdall, snr_heimdall, mask_chans=[], 
               datestring=None, beam=None, corr=None, fil_file=None, voltage = False, 
               path_to_fil_file = None):
    if voltage: 
        fil_file = '/home/ubuntu/vikram/scratch/' + candidate + '.fil'
    elif path_to_fil_file is not None:
        fil_file = path_to_fil_file 
    else : 
        fil_file ='/data/dsa110/T1/'+corr + '/' + datestring + '/fil_'+candidate+'/'+candidate+'_'+str(beam)+'.fil'

    logging_format = "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format,
    )
    
    bd = BurstData(
    fp=fil_file,
    dm=dm_heimdall ,
    tcand=0.5, # pulse starting time in s.
    width=width_heimdall,  
    snr=snr_heimdall
    )

    bd.prepare_data(mask_chans = mask_chans)
    print("using filterbank ", fil_file)
    
    return bd, fil_file 


def Bin_profile(data_t, bin_size):
    print('data_t', data_t)

    data_t_binned = np.array([])
    
    if bin_size == 1:
        data_t_binned = data_t
    elif bin_size > 1: 
        for i in range(0, len(data_t), bin_size):
            bin_value = np.sum(data_t[i : i + bin_size])
            data_t_binned = np.append(data_t_binned, bin_value)
    else: 
        print("bin_size is negative.")
    
    return data_t_binned, bin_size 



def prepare_burst_data(filterbank, fil_file_dedispersed, candidate, bd_heimdall, bd, t_chop_center_s = 0.5, 
                       voltage = True, bin = True, t_chop_width = 50, dedisperse=False, nfreq = 4, plot = True):
    """
    de-disperse data (or read from file) and chop near the burst time
    """
    # save the de-dispersed data to file (de-dispersion takes long time)
    if dedisperse:
        data = ff.proc_cand_fil(filterbank, bd.dm, bd.width, nfreq_plot=nfreq, ndm=64)[0]
        np.save(fil_file_dedispersed, data, allow_pickle=False)
    else: 
        data = np.load(fil_file_dedispersed + ".npy") 
    
    # heimdall burst starts at 500 ms, voltage removed edge channels so burst dedispersed to a later time
    if voltage:
        t_chop_center_s = t_chop_center_s + 1e3 * (4.15 * bd.dm * (1 / bd.fch1 ** 2 - 1 / bd_heimdall.fch1 ** 2)) 
        bd.tcand = t_chop_center_s
    
    i_low = int(t_chop_center_s / bd.tsamp - t_chop_width * bd.width)
    i_high = int(t_chop_center_s / bd.tsamp + t_chop_width * bd.width)
    
    t_burst = [i * bd.tsamp * 1e3 for i in range(i_low, i_high)]
    data_burst = data[:, i_low: i_high] 
    data_t_unbinned = data_burst.mean(0)
    

    bin_size = int(bd_heimdall.tsamp / bd.tsamp) 
    if bin: # bin profile to compare plots 
        print('data_burst.mean(0), data_burst.mean(0)', data_burst.mean(0), len(data_burst.mean(0)))
        data_t_binned, bin_size = Bin_profile(data_burst.mean(0), bin_size) 
    
        if plot:
            fig1, ax1 = plt.subplots(2, 2, figsize=(12, 8)) 
            ax1[0][0].plot(t_burst[::bin_size], data_t_binned)
            ax1[0][0].set_xlabel('Time (ms) (binned %d to match Heimdall resolution)'%bin_size)
            ax1[0][1].plot(data_burst.mean(1))
            ax1[0][1].set_xlabel('Binned channel')
            ax1[0][1].set_title('bin freq channel #%d'%nfreq)   
            ax1[1][0].plot(t_burst, data_t_unbinned)
            ax1[1][0].set_xlabel('Time (ms) (not binned)')
            plt.tight_layout()
    
    else: 
        if plot:
            fig1, ax1 = plt.subplots(2, 1, figsize=(12, 4)) 
            ax1[0].set_title('No binning') 
            ax1[0].plot(t_burst, data_t_unbinned)
            ax1[0].set_xlabel('Time (ms)')
            ax1[1].plot(data_burst.mean(1))
            ax1[1].set_xlabel('Binned channel')
            ax1[1].set_title('bin freq channel #%d'%nfreq)   
            plt.tight_layout()
        
    
    return data_burst


def prepare_burst_data_heimdall_only(filterbank, fil_file_dedispersed, candidate, bd_heimdall, t_chop_center_s = 0.5, 
                       voltage = False, t_chop_width = 5, dedisperse=False, nfreq = 4, plot = True):
    """
    de-disperse data (or read from file) and chop near the burst time
    No voltage filterbank, heimdall only.  For T3 real-time. 
    """
    print("Only using Heimdall filterbank.")
    
    # save the de-dispersed data to file (de-dispersion takes time)
    if dedisperse:
        data = ff.proc_cand_fil(filterbank, bd_heimdall.dm, bd_heimdall.width, nfreq_plot=nfreq, ndm=64)[0]
        np.save(fil_file_dedispersed, data, allow_pickle=False)
    else: 
        data = np.load(fil_file_dedispersed + ".npy") 
    
    # heimdall burst starts at exactly 500 ms.
    # chop indices in time sample, but sec. 
    i_low = int(t_chop_center_s / bd_heimdall.tsamp - t_chop_width * bd_heimdall.width)
    i_high = int(t_chop_center_s / bd_heimdall.tsamp + t_chop_width * bd_heimdall.width)
    
    print(t_chop_center_s, bd_heimdall.tsamp, t_chop_width, bd_heimdall.width, i_low, i_high)
    
    t_burst = [i * bd_heimdall.tsamp * 1e3 for i in range(i_low, i_high)]
    data_burst = data[:, i_low: i_high] 
    data_t_unbinned = data_burst.mean(0)
    
    

    print(i_low, i_high, len(t_burst), np.shape(data_burst))
    
    if plot:
        fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4)) 
        ax1[0].plot(t_burst, data_t_unbinned)
        ax1[0].set_xlabel('Time (ms) (Heimdall resolution)')
        ax1[1].plot(data_burst.mean(1))
        ax1[1].set_xlabel('Binned channel')
        ax1[1].set_title('bin freq channel #%d'%nfreq)   
        plt.tight_layout()
    
    return data_burst, i_low, i_high



def fit_paper_curvefit(bd, data_2d, pnames, pulse_Model, profile_bounds, 
                       snames, spectra_Model, nfreq = 4, 
                       fix_ncomp=False, ncomp=1, plot=True):
    ncomp = ncomp
    pnames = pnames 
    pulseModel = Model(pulse_Model, param_names=pnames)
    spectraModel = Model(spectra_Model, param_names=snames)
    sgram_mask = np.full(np.shape(data_2d), False)
    sgramModel = SgramModel(pulseModel, spectraModel, sgram_fn, 
                        mask=sgram_mask, clip_fac=bd.clip_fac)
    
    bf = BurstFit(
    sgram_model=sgramModel,
    sgram=data_2d, 
    width=bd.width,
    dm=bd.dm,
    foff=bd.foff * bd.nchans / nfreq,
    fch1=bd.fch1,
    tsamp=bd.tsamp,
    clip_fac=bd.clip_fac,
    mask= sgram_mask, 
    mcmcfit=False,
    #comp_num = 1,
    )
    
    bf.validate()
    bf.precalc()
    #bf.initial_profilefit(bounds = profile_bounds, plot = plot)
    #bf.initial_spectrafit(plot = plot)
    #bf.fitcycle(plot=plot, profile_bounds = profile_bounds) # fit for one component.
    bf.fitall(plot=plot, fix_ncomp = fix_ncomp, ncomp = ncomp, profile_bounds = profile_bounds) # fit all componnts 
    print('{bf}.calc_redchisq()=', bf.calc_redchisq())


    # plot_fit_results(bf_S1T2_c1.sgram, bf_S1T2_c1.sgram_model.evaluate, bf_S1T2_c1.sgram_params['all'][1]['popt'], 
    #                  bf_S1T2_c1.tsamp, bf_S1T2_c1.fch1, bf_S1T2_c1.foff, show=True, save=True, outname=save_name+'2d_fit_res_curvfit', outdir=save_dir)
    #dm_fit, dm_fit_err = bf_S1T2_c1.sgram_params['all'][1]['popt'][-1], bf_S1T2_c1.sgram_params['all'][1]['perr'][-1]

    return bf 


def fit_paper_mcmc_bic(bf, mcmc=True, nwalkers = 60, nsteps = 5000, n_param_overlap = 0):
    mcmc_kwargs = {'nwalkers':nwalkers, 'nsteps':nsteps,
               'skip':500, 'ncores':4, 
               'start_pos_dev':0.01,
               'prior_range':0.8, 
               'save_results':True,
               'outname':'test_file'}

    if mcmc:
        bf.run_mcmc(plot=True, **mcmc_kwargs)
    
    n_model_param = bf.comp_num * len(bf.param_names) - n_param_overlap 
    n_data = bf.nt * bf.nf 
    model_param = []
    for i in range(1, bf.comp_num + 1):
        model_param += bf.sgram_params['all'][i]['popt'] 

    lnL = bf.mcmc.lnlk(model_param)
    bf.BIC(lnL, n_model_param, n_data)
    print('{bf}.bic = ', bf.bic) 
    
    return 0


def select_model(bf_name_list):
    bf_with_bic_list = []
    bic_list = []
    
    for name in bf_name_list:

        if name in globals() or name in locals():
            bf = globals()[name]

            if bf.bic is None: 
                print("%s.bic not found"%name)

            else:         
                bf_with_bic_list = np.append(bf_with_bic_list, name)
                bic_list = np.append(bic_list, bf.bic)
        
        else:
            print("%s not used"%name)

    sorted_bf = ([x for _,x in sorted(zip(bic_list, bf_with_bic_list))])  
    sorted_bic = sorted(bic_list)

    print(sorted_bf) 
    print(sorted_bic)
    
    return sorted_bf, sorted_bic


def save_results(file, candidate, bf_name):
    bf_best = globals()[bf_name]
    with open(file, "a") as f:
        print(candidate, file=f)
        print(bf_name, file=f)
        for ncomp in range(bf_best.ncomponents):
            for i in range(len(bf_best.param_names)):
                print("${:.4}^{{+{:.2}}}_{{-{:.2}}}$".format(bf_best.mcmc_params[ncomp + 1]['popt'][i], 
                                                             bf_best.mcmc_params[ncomp + 1]['perr'][i][0], 
                                                             bf_best.mcmc_params[ncomp + 1]['perr'][i][1]),
                     file=f) 
                
        print("\n", file = f)
    
    return 0


def autocorrelation():
    pass

####################################################
####################################################
# for real time 
####################################################
####################################################

def bin_best(bf, bd, mcmc=False, tsamp_fine = 32.768e-6):
    
    sig_idx = np.where(np.array(bf.param_names) == "sigma_t")[0][0]
    tau_idx = np.where(np.array(bf.param_names) == "tau")[0][0]

    if mcmc: 
        sigma = bf.mcmc_params[bf.comp_num]["popt"][sig_idx]
        tau = bf.mcmc_params[bf.comp_num]["popt"][tau_idx]
    else:
        sigma = bf.sgram_params[bf.comp_num]["popt"][sig_idx]
        tau = bf.sgram_params[bf.comp_num]["popt"][tau_idx]
    
    width_samp = 2.335 * sigma + tau 
    width_samp_fine = width_samp * bd.tsamp / tsamp_fine 
    width_samp_fine_pow2 = 2 ** int(np.log2(width_samp_fine))
        
    return int(width_samp), width_samp_fine_pow2 
 

    
def bin_number(bf, bd, i_low, width_samp_fine, mcmc=False, tsamp_fine = 32.768e-6):
    mu_idx = np.where(np.array(bf.param_names) == "mu_t")[0][0]
    
    if mcmc:
        mu = bf.mcmc_params[bf.comp_num]["popt"][mu_idx] # count from the chopped time window
    else: 
        mu = bf.sgram_params[bf.comp_num]["popt"][mu_idx]
    
    mu_ms = bd.tsamp * (i_low + mu) # count from gulp t0  
    mu_samp_fine = mu_ms / tsamp_fine 
    
    return int(mu_samp_fine / width_samp_fine) # count from bin 0 


# loop over adjecent choices to compare SNR

def Dedisperse_data_profile(bf, fil_file, width, nfreq = 4, ndm = 64):
    dm_fit_idx = np.where(np.array(bf.param_names) == "DM")[0][0]
    dm_fit = bf.sgram_params[bf.comp_num]["popt"][dm_fit_idx] 
    data = ff.proc_cand_fil(fil_file, dm_fit, width, nfreq_plot=nfreq, ndm=ndm)[0]
        
    return data, dm_fit


def Snr(data_t_bin):
    signal = np.max(data_t_bin)
    peak_idx = np.argmax(data_t_bin)
    residual = np.concatenate([data_t_bin[0 : peak_idx - 1], data_t_bin[peak_idx + 2 : -1]])
    noise = np.std(residual)

    return signal / noise 


def Compare_bins(data, bd, width_samp_fine_pow2, tsamp_fine = 32.768e-6, plot = True):
    snr_arr = np.array([])
    width_fine_arr = np.array([])
    width_power = int(np.log2(width_samp_fine_pow2))

    for width_pow2 in range(max(int(np.log2(bd.tsamp / tsamp_fine)), width_power - 1), width_power + 4):
        width_fine = 2 ** width_pow2
        print(width_fine)
        width_fine_arr = np.append(width_fine_arr, width_fine)
        width_filterbank = int(width_fine * tsamp_fine / bd.tsamp)

        data_t_binned = Bin_profile(data.mean(0), width_filterbank)
        snr = Snr(data_t_binned)
        snr_arr = np.append(snr_arr, snr)

        if plot:
            plt.figure()
            plt.plot(data_t_binned)
            plt.title("bin size (at 32.768e-6s) = %d, snr = %.2f"%(width_fine, snr))
        
    best_bin_snr_idx = np.argmax(snr_arr)
    best_bin_snr = int(width_fine_arr[best_bin_snr_idx])
    
    print("fine resolution bin size (at 32.768e-6s) that gives the max snr: %d"%best_bin_snr)
    
    return best_bin_snr 


# In[ ]:




