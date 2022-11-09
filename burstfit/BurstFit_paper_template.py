from burstfit.fit import BurstFit
from burstfit.data import BurstData
from burstfit.model import Model, SgramModel
import logging
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
# use Liam's function to read in filterbank files 

from burstfit import filplot_funcs_gc as ff
from pathlib import Path

from burstfit.utils.functions import pulse_fn, sgram_fn, gauss, gauss_norm, model_free_4, model_free_normalized_4, power_law # pulse_fn_vec  
from burstfit.utils.plotter import plot_fit_results


def real_time_burstfit(candidate, path_to_fil_file, snr_heimdall, dm_heimdall, width_heimdall,
                       voltage = False, dedisperse = True, mask_chans=[], nfreq = 4, save_plot = True,
                       plot = False
                      ): # datestring, beam, corr

    fil_file_dedispersed = '/home/ubuntu/gechen/software/burstfit/fil_files_dedispersed/%s_%d_%s.txt'%(candidate, nfreq, str(voltage))

    bd_heimdall, fil_file_heimdall  = prepare_bd(candidate, dm_heimdall, width_heimdall, snr_heimdall, voltage=False, path_to_fil_file = path_to_fil_file) # datestring=datestring, beam=beam, corr=corr

    data_burst, i_low, i_high = prepare_burst_data_heimdall_only(fil_file_heimdall, fil_file_dedispersed, candidate, bd_heimdall, t_chop_width = 5, dedisperse=dedisperse, plot = plot)

    # fit 
    profile_bounds=[(0, 0.5 / bd_heimdall.tsamp - i_low - bd_heimdall.width, 0, 0),(np.inf, 0.5 / bd_heimdall.tsamp - i_low + bd_heimdall.width, 5 * bd_heimdall.width, np.inf)]

    bf_S1T2_c1 = fit_paper_curvefit(bd_heimdall, data_burst, ['S', 'mu_t', 'sigma_t', 'tau'], pulse_fn, profile_bounds, ['c0', 'c1', 'c2'], model_free_normalized_4, fix_ncomp = True, ncomp = 1, plot = plot)

    # dm from fit 
    dm_fit, dm_fit_err = bf_S1T2_c1.sgram_params['all'][1]['popt'][-1], bf_S1T2_c1.sgram_params['all'][1]['perr'][-1]

    width_samp, width_samp_fine_pow2 = bin_best(bf_S1T2_c1, bd_heimdall)
    bin_num = bin_number(bf_S1T2_c1, bd_heimdall, i_low, width_samp_fine_pow2)
    #print("width_samp, width_samp_fine_pow2, bin_num = ", width_samp, width_samp_fine_pow2, bin_num)

    # dedisperse data using the new dm and compare SNR using adjecent bin widths
    data_new_dm, dm_fit = Dedisperse_data_profile(bf_S1T2_c1, fil_file_heimdall, width_samp)
    best_bin_snr = Compare_bins(data_new_dm, bd_heimdall, width_samp_fine_pow2, plot = plot)

    # save optimal bin width and fit results. 
    dict_burstfit = {
        "bf1_width_bins": width_samp_fine_pow2, # already converted to voltage resolution 
        "bf1_start_bins": bin_num, # already converted to voltage resolution
        "bf1_dm":dm_fit, 
        "bf1_dm_stddev": dm_fit_err,
        "bf1_reduced_chisq": float(bf_S1T2_c1.reduced_chi_sq),
        "bf1_pvalue": float(bf_S1T2_c1.p_value)
    }
    
    print(dict_burstfit)

    if save_plot:
        print(f'Making directory for burstfit plot of {candidate}')
        save_plot_dir = "/dataz/dsa110/operations/candidates/" + candidate + "/other"
        Path(save_plot_dir).mkdir(parents=True, exist_ok=True)

        plot_fit_results(bf_S1T2_c1.sgram, bf_S1T2_c1.sgram_model.evaluate, bf_S1T2_c1.sgram_params['all'][1]['popt'], 
                         bf_S1T2_c1.tsamp, bf_S1T2_c1.fch1, bf_S1T2_c1.foff, show = False, save=True, 
                         outname = 'bf1_' + candidate, 
                         outdir = save_plot_dir)        

    return dict_burstfit


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


def prepare_burst_data(filterbank, fil_file_dedispersed, candidate, bd_heimdall, bd, t_chop_center_s = 0.5, 
                       voltage = True, bin = True, t_chop_width = 50, dedisperse=False, nfreq = 4, plot = True):
    """
    de-disperse data (or read from file) and chop near the burst time
    """
    # save the de-dispersed data to file (de-dispersion takes long time)
    if dedisperse:
        print("dedisperse...") 
        data = ff.proc_cand_fil(filterbank, bd.dm, bd.width, nfreq_plot=nfreq, ndm=64, norm=False)[0]
        np.save(fil_file_dedispersed, data, allow_pickle=False)
    else: 
        print("Reading from file" + fil_file_dedispersed + ".npy")
        data = np.load(fil_file_dedispersed + ".npy") 
    
    # heimdall burst starts at 500 ms, voltage removed edge channels so burst dedispersed to a later time
    if voltage:
        t_chop_center_s = t_chop_center_s + 1e3 * (4.15 * bd.dm * (1 / bd.fch1 ** 2 - 1 / bd_heimdall.fch1 ** 2)) 
        bd.tcand = t_chop_center_s
    
    i_low = max(0, int(t_chop_center_s / bd.tsamp - t_chop_width * bd.width))
    i_high = min(int(t_chop_center_s / bd.tsamp + t_chop_width * bd.width), np.shape(data)[1]) 
    
    t = [i * bd.tsamp * 1e3 for i in range(np.shape(data)[1])] 
    t_burst = [i * bd.tsamp * 1e3 for i in range(i_low, i_high)]
    t_off_burst = t[: i_low] + t[i_high : 2 * int(t_chop_center_s / bd.tsamp)]

    data_off_burst = np.concatenate((data[:, : i_low], data[:, i_high : 2 * int(t_chop_center_s / bd.tsamp)]), axis = 1)
    data_off_burst_t_unbinned = data_off_burst.mean(0)
    #print("i_low, i_high, np.shape(data)[1], 2 * int(t_chop_center_s / bd.tsamp)", i_low, i_high, np.shape(data)[1], 2 * int(t_chop_center_s / bd.tsamp))
    
    data_norm = data - np.median(data_off_burst, axis = 1, keepdims=True)
    #flux_noise = np.std(data_off_burst.mean(0)) # flux noise can be calculated from SEFD 
    flux_noise = np.std(data_off_burst)
    data_norm /= flux_noise 
    data_norm_t_unbinned = data_norm.mean(0)
    
    data_burst_norm = data_norm[:, i_low: i_high] 
    data_burst_norm_t_unbinned = data_burst_norm.mean(0)
    
    

    bin_size = int(bd_heimdall.tsamp / bd.tsamp) 
    
    fig1, ax1 = plt.subplots(figsize=(12, 4)) 
    ax1.plot(t_burst, data_burst_norm_t_unbinned, label='chop near burst') 
    ax1.plot(t_off_burst, data_off_burst_t_unbinned, label='off-pulse used for normalization') # t, data_norm_t_unbinned 
    ax1.legend() 
    ax1.set_xlabel('Time (ms) (Raw resolution)')
    ax1.set_title('Off-pulse-std-normalized and Off-pulse-median-removed')


#     fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4)) 
#     ax2[0].plot(t_burst, data_t_unbinned)
#     ax2[0].set_xlabel('Time (ms) (Heimdall resolution)')
#     ax2[1].plot(data_burst.mean(1))
#     ax2[1].set_xlabel('Binned channel')
#     ax2[1].set_title('bin freq channel #%d'%nfreq)   
#     plt.tight_layout()
        
    if bin: # bin profile to compare plots 
        data_t_binned, bin_size = Bin_profile(data_burst_norm_t_unbinned, bin_size) 
    
        if plot:
            #print("len(t_burst[::bin_size]), len(data_t_binned): %d, %d"%(len(t_burst[::bin_size]), len(data_t_binned)))
            #print(np.shape(data_burst), )
            
            fig1, ax1 = plt.subplots(2, 2, figsize=(12, 8)) 
            ax1[0][0].plot(t_burst[::bin_size], data_t_binned)
            ax1[0][0].set_xlabel('Time (ms) (binned %d to match Heimdall resolution)'%bin_size)
            ax1[0][1].plot(data_burst_norm.mean(1))
            ax1[0][1].set_xlabel('Binned channel')
            ax1[0][1].set_title('bin freq channel #%d'%nfreq)   
            ax1[1][0].plot(t_burst, data_burst_norm_t_unbinned)
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
        
    
    return data_burst_norm, i_low, i_high


def prepare_burst_data_heimdall_only(filterbank, fil_file_dedispersed, candidate, bd_heimdall, t_chop_center_s = 0.5, 
                       voltage = False, t_chop_width = 5, dedisperse=False, nfreq = 4, plot = True):
    """
    de-disperse data (or read from file) and chop near the burst time
    No voltage filterbank, heimdall only.  For T3 real-time. 
    """
    print("Only using Heimdall filterbank.")
    
    # save the de-dispersed data to file (de-dispersion takes time)
    if dedisperse:
        data = ff.proc_cand_fil(filterbank, bd_heimdall.dm, bd_heimdall.width, nfreq_plot=nfreq, ndm=64, norm = False)[0]
        np.save(fil_file_dedispersed, data, allow_pickle=False)
    else: 
        data = np.load(fil_file_dedispersed + ".npy") 
    
    # heimdall burst starts at exactly 500 ms.
    # chop indices in time sample, but sec. 
    i_low = int(t_chop_center_s / bd_heimdall.tsamp - t_chop_width * bd_heimdall.width)
    i_high = int(t_chop_center_s / bd_heimdall.tsamp + t_chop_width * bd_heimdall.width)
    
    #print(t_chop_center_s, bd_heimdall.tsamp, t_chop_width, bd_heimdall.width, i_low, i_high)
    
    t = [i * bd_heimdall.tsamp * 1e3 for i in range(np.shape(data)[0])] 
    t_burst = [i * bd_heimdall.tsamp * 1e3 for i in range(i_low, i_high)]

    data_off_burst = np.concatenate((data[:, : i_low], data[:, i_low :]), axis = 1) 
    
    data_norm = data - np.median(data_off_burst, axis = 1, keepdims=True)
    data_norm /= np.std(data_off_burst) 
    data_norm_t_unbinned = data_norm.mean(0)
    
    data_burst_norm = data_norm[:, i_low: i_high] 
    data_burst_norm_t_unbinned = data_burst_norm.mean(0)
    

    #print(i_low, i_high, len(t_burst), np.shape(data_burst))
    
    if plot:
        fig1, ax1 = plt.subplots(fig1, ax1 = plt.subplots(figsize=(12, 4))) 
        ax1.plot(t, data_norm_t_unbinned)
        ax1.plot(t_burst, data_burst_norm_t_unbinned)        
        ax1.set_xlabel('Time (ms) (Heimdall resolution)')

        
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4)) 
        ax2[0].plot(t_burst, data_t_unbinned)
        ax2[0].set_xlabel('Time (ms) (Heimdall resolution)')
        ax2[1].plot(data_burst.mean(1))
        ax2[1].set_xlabel('Binned channel')
        ax2[1].set_title('bin freq channel #%d'%nfreq)   
        plt.tight_layout()
        
        
    
    return data_burst_norm, i_low, i_high



def Bin_profile(data_t, bin_size):

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



def fit_paper_curvefit(bd, data_2d, pnames, pulse_Model, profile_bounds, 
                       snames, spectra_Model, spectra_bounds=None, nfreq = 4, 
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
    sgram_bounds = [(0,0,0) + profile_bounds[0] + (0,), (1,1,1) + profile_bounds[1] + (np.inf,)]
    
    bf.fitall(plot=plot, fix_ncomp = fix_ncomp, ncomp = ncomp, profile_bounds = profile_bounds, spectra_bounds = spectra_bounds, sgram_bounds = sgram_bounds) # fit all componnts 
    print('{bf}.calc_redchisq()=', bf.calc_redchisq())


    # plot_fit_results(bf_S1T2_c1.sgram, bf_S1T2_c1.sgram_model.evaluate, bf_S1T2_c1.sgram_params['all'][1]['popt'], 
    #                  bf_S1T2_c1.tsamp, bf_S1T2_c1.fch1, bf_S1T2_c1.foff, show=True, save=True, outname=save_name+'2d_fit_res_curvfit', outdir=save_dir)
    #dm_fit, dm_fit_err = bf_S1T2_c1.sgram_params['all'][1]['popt'][-1], bf_S1T2_c1.sgram_params['all'][1]['perr'][-1]

    return bf 


def fit_paper_mcmc_bic(bf, outname, date_string, mcmc=True, nwalkers = 60, nsteps = 5000, prior_range = 0.8, prior_c0 = 8, prior_c1 = 8, prior_c2 = 8, prior_S = 5, prior_S1 = 5, prior_S2 = 5, prior_S3 = 5, prior_S4 = 5, prior_DM = 5, n_param_overlap = 0):
    
    mcmc_kwargs = {'nwalkers':nwalkers, 'nsteps':nsteps,
                   'skip':500, 'ncores':4, 
                   'start_pos_dev': 0.01,
                   'save_results': True,
                   'prior_range' : prior_range, 
                   'prior_c0' : prior_c0, 
                   'prior_c1' : prior_c1, 
                   'prior_c2' : prior_c2, 
                   'prior_S' : prior_S, 
                   'prior_S1' : prior_S1, 
                   'prior_S2' : prior_S2, 
                   'prior_S3' : prior_S3, 
                   'prior_S4' : prior_S4, 
                   'prior_DM' : prior_DM, 
                  }
               

    if mcmc:
        bf.run_mcmc(plot = True, outname = outname, fig_title = date_string, **mcmc_kwargs)
    
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


def save_results(file_latex, file_pandas, candidate, bf_best_name, bf_tau_name):
        
    bf_best = globals()[bf_best_name]
    bf_tau = globals()[bf_tau_name]
    
    with open(file_latex, "a") as f:
        print(candidate, end=' & ', file=f)
        print(bf_name, end=' & ', file=f)
        for ncomp in range(bf_best.ncomponents):
            for i in range(len(bf_best.param_names)):
                print("${:.4}^{{+{:.2}}}_{{-{:.2}}}$".format(bf_best.mcmc_params[ncomp + 1]['popt'][i], 
                                                             bf_best.mcmc_params[ncomp + 1]['perr'][i][0], 
                                                             bf_best.mcmc_params[ncomp + 1]['perr'][i][1]),
                      end=' & ',
                      file=f) 
                
            # tau 
            print("${:.4}^{{+{:.2}}}_{{-{:.2}}}$ \\\\".format(bf_tau.mcmc_params[ncomp + 1]['popt'][-2], 
                                                         bf_tau.mcmc_params[ncomp + 1]['perr'][-2][0], 
                                                         bf_tau.mcmc_params[ncomp + 1]['perr'][-2][1]),
                  file=f) 
           
    
        
        with open(file_pandas, "a") as f:
            print(candidate, end=';', file=f)
            print(bf_name, end=';', file=f)
            for ncomp in range(bf_best.ncomponents):
                for i in range(len(bf_best.param_names)):
                    print("{:.4}".format(bf_best.mcmc_params[ncomp + 1]['popt'][i]), end=';', file=f) 
                    print("{:.4}".format(bf_best.mcmc_params[ncomp + 1]['perr'][i][0]), end=';', file=f) 
                    print("{:.4}".format(bf_best.mcmc_params[ncomp + 1]['perr'][i][1]), end=';', file=f) 

                # tau
                print("{:.4}".format(bf_tau.mcmc_params[ncomp + 1]['popt'][-2]), end=';', file=f) 
                print("{:.4}".format(bf_tau.mcmc_params[ncomp + 1]['perr'][-2][0]), end=';', file=f) 
                print("{:.4}".format(bf_tau.mcmc_params[ncomp + 1]['perr'][-2][1]), file=f) 
                                                                   
    return 0


# def save_results(file, candidate, bf_name):
#     bf_best = globals()[bf_name]
#     with open(file, "a") as f:
#         print(candidate, file=f)
#         print(bf_name, file=f)
#         for ncomp in range(bf_best.ncomponents):
#             for i in range(len(bf_best.param_names)):
#                 print("${:.4}^{{+{:.2}}}_{{-{:.2}}}$".format(bf_best.mcmc_params[ncomp + 1]['popt'][i], 
#                                                              bf_best.mcmc_params[ncomp + 1]['perr'][i][0], 
#                                                              bf_best.mcmc_params[ncomp + 1]['perr'][i][1]),
#                      file=f) 
                
#         print("\n", file = f)
    
#     return 0


####################################################
####################################################
# for spectrum
####################################################
####################################################

def find_param_value(bf, param_name, ncomp = 1):
    ind = np.argwhere(np.array(bf.param_names) == param_name)[0][0]
    param_value = bf.mcmc_params[ncomp]['popt'][ind]
    
    return param_value


def find_fwhm(profile_spec):
    """
    return: indices of the fwhm bounds. 
    """
    peak = max(profile_spec)
    peak_t_idx = np.where(profile_spec == peak)

    signs = np.sign(np.add(profile_spec, -0.5 * peak))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    
    return zero_crossings_i


def make_spectra_FWHM(bd, bf, data, i_low, i_high, weight = True):
        """
        Extract the FWHM for spectrum analysis.
        Weight time samples by SNR 
        
        data: already zoom close to the pulse 
        i_low, i_high: indices used in the zoom 

        Returns: spectra library, by peak / component 

        """
        # make spec for each component
        # another case: multi-peak 
        
        spectra_all = {}
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4)) 
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4)) 
        time_idx = np.arange(i_low, i_high)
        ax2.plot(time_idx * bd.tsamp * 1e3, data.mean(0))
        ax2.set_xlabel("Time (ms)") 
        
        for i in range(bf.ncomponents):       
            comp = i + 1
            print("Making spectra using profile fit parameters.")
            
            mu_idx = np.where(np.array(bf.param_names) == "mu_t")[0][0] 
            S_idx = np.where(np.array(bf.param_names) == "S")[0][0]
            sigma_idx = np.where(np.array(bf.param_names) == "sigma_t")[0][0]
            
            mu = bf.mcmc_params[comp]["popt"][mu_idx] + i_low # in terms of the entire gulp 
            S = bf.mcmc_params[comp]["popt"][S_idx]
            sigma = bf.mcmc_params[comp]["popt"][sigma_idx]
            
            
            
            if "tau" not in bf.param_names:
                spec_func = gauss(time_idx, S, mu, sigma)
            
            else:
                tau_idx = np.where(np.array(bf.param_names) == "tau")[0][0]
                tau = bf.mcmc_params[comp]["popt"][tau_idx]
                spec_func = pulse_fn(time_idx, S, mu, sigma, tau)
                

            start, end = find_fwhm(spec_func)
            start = int(start)
            end = int(end) + 1 
            data_peak = data[:, start : end + 1]
            profile_peak = data_peak.mean(0) # time profile included 

            if weight: 
                weights = profile_peak ** 2 # weights length = time sample number 
                spectra = np.average(data_peak, axis = 1, weights = weights)
            else: 
                spectra = np.average(data_peak, axis = 1) 

            print(f"Normalising spectra to unit area.")
            spectra = spectra / np.sum(spectra)
            
            spectra_all[str(comp)] = spectra
            
            time_spec_region = np.arange(start + i_low, end + 1 + i_low) * bd.tsamp * 1e3
            
            ax1.plot(time_spec_region, spec_func[start: end+1], lw=3, label='comp %d'%comp)            
            ax2.plot(time_spec_region, data_peak.mean(0), lw=3, label='comp %d'%comp)
            
            
        
        ax1.set_xlabel("Time (ms)") 
        ax1.legend()
        

        ax2.legend()

        print(spectra_all.keys())
        return spectra_all 
    
def make_spectra_multi_FWHM(bd, bf, data, i_low, i_high, comp_num, weight = True):
        """
        Extract the FWHM for spectrum analysis.
        Weight time samples by SNR 
        
        data: already zoom close to the pulse 
        i_low, i_high: indices used in the zoom 

        Returns: spectra library, by peak / component 

        """
        # make spec for each component
        # another case: multi-peak 
        
        spectra_all = {}
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4)) 
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4)) 
        time_idx = np.arange(i_low, i_high)
        ax2.plot(time_idx * bd.tsamp * 1e3, data.mean(0))
        ax2.set_xlabel("Time (ms)") 
        
        for i in range(comp_num):       
            comp = i + 1
            print("Making spectra using profile fit parameters.")
            
            
            'S1', 'mu_t1', 'sigma_t1',
            mu_idx = np.where(np.array(bf.param_names) == "mu_t" + str(comp))[0][0] 
            S_idx = np.where(np.array(bf.param_names) == "S" + str(comp))[0][0]
            sigma_idx = np.where(np.array(bf.param_names) == "sigma_t" + str(comp))[0][0]
            
            mu = bf.mcmc_params[1]["popt"][mu_idx] + i_low # in terms of the entire gulp 
            S = bf.mcmc_params[1]["popt"][S_idx]
            sigma = bf.mcmc_params[1]["popt"][sigma_idx]
            
            
            
            if "tau" + str(comp) not in bf.param_names:
                spec_func = gauss(time_idx, S, mu, sigma)
            
            else:
                tau_idx = np.where(np.array(bf.param_names) == "tau" + str(comp))[0][0]
                tau = bf.mcmc_params[1]["popt"][tau_idx]
                spec_func = pulse_fn(time_idx, S, mu, sigma, tau)
                

            start, end = find_fwhm(spec_func)
            start = int(start)
            end = int(end) + 1 
            data_peak = data[:, start : end + 1]
            profile_peak = data_peak.mean(0) # time profile included 

            if weight: 
                weights = profile_peak ** 2 # weights length = time sample number 
                spectra = np.average(data_peak, axis = 1, weights = weights)
            else: 
                spectra = np.average(data_peak, axis = 1) 

            print(f"Normalising spectra to unit area.")
            spectra = spectra / np.sum(spectra)
            
            spectra_all[str(comp)] = spectra
            
            time_spec_region = np.arange(start + i_low, end + 1 + i_low) * bd.tsamp * 1e3
            
            ax1.plot(time_spec_region, spec_func[start: end+1], lw=3, label='comp %d'%comp)            
            ax2.plot(time_spec_region, data_peak.mean(0), lw=3, label='comp %d'%comp)
            
            
        
        ax1.set_xlabel("Time (ms)") 
        ax1.legend()
        

        ax2.legend()

        print(spectra_all.keys())
        return spectra_all 

    
# modified from Liam's code 

import scipy.stats as stats

def acf(bd, data):
    """
    Liam 
    data: 1-d spectrum 
    """
       
    bw = np.abs(bd.foff * bd.nchans)
    
    nfreq = len(data)
    dnu_arr = range(nfreq)
    data_mean = data.mean()
    acf = np.zeros([nfreq])
    dnu_arr = np.linspace(0, bw, nfreq) # array of \Delta \nu
    data = data - data_mean # mean-removed 

    for dnu, dnus in enumerate(dnu_arr): # dnus is the Delta frequency array 
        norm1 = 0 #np.zeros([nfreq])                                                                 
        norm2 = 0 #np.zeros([nfreq])                                                                 

        counter = 0
        for ii in range(nfreq):
                if ii + dnu >= nfreq:
                        continue

                norm = (np.sum(data[ii]**2))**0.5*(np.sum(data[ii+dnu]**2))**0.5 
                if data[ii] != 0 and data[ii+dnu] != 0:
                        val = data[ii] * data[ii+dnu]
                        #val /= (np.abs(data[ii]-data_mean)*np.abs(data[ii+dnu]-data_mean))         
                        acf[dnu] += val
                        norm1 += data[ii]**2
                        norm2 += data[ii+dnu]**2
                        counter += 1

        if counter != 0:
                acf[dnu] /= np.sqrt(norm1 * norm2)
#                       acf[dnu] /= counter   

    
    return dnu_arr, acf, bw, nfreq



def lorenz_func(x, a, b, dnu_bw):
    return a * (x ** 2 + dnu_bw ** 2) ** -1 + b



def plot_acf_results(dnu, corr, pp, cov, fig_outname, date_string, tmax = 1000, comp = 1):
       
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8)) 
    ax1.set_xlabel(r'Frequency Lag $\Delta \nu$ (MHz)', fontsize = 20)
    ax1.set_ylabel(r'Amplitude', fontsize = 20)
    
    #print("corr[1: tmax]", corr[1: tmax])
    ax1.scatter(dnu[1 : int(2 * tmax)], corr[1: int(2 * tmax)], s=3, color='k')#, label="The autocorrelation function of Spectrum")
    #ax1.scatter(dnu, corr, s=3, color='k')
    ax1.plot(dnu[1:tmax], lorenz_func(dnu[1:tmax], pp[0], pp[1], pp[2]), lw=3, color='r')
    ax1.vlines(x = pp[2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'solid', color = 'b')
    ax1.vlines(x = pp[2] + cov[2][2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'dotted', color = 'b')
    ax1.vlines(x = pp[2] - cov[2][2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'dotted', color = 'b')
    ax1.annotate(date_string + "(" + str(comp) + ")", xy=(0.01, 0.99),xycoords='axes fraction', fontsize=20, horizontalalignment='left', verticalalignment='top')
    ax1.legend()
    fig1.savefig(fig_outname + '_acf_zoom.pdf')
    
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8)) 
    ax2.set_xlabel(r'Frequency Lag $\Delta \nu$ (MHz)', fontsize = 20)
    ax2.set_ylabel(r'Amplitude', fontsize = 20)
    
    ax2.scatter(dnu, corr, s=3, color='k')
    ax2.plot(dnu[1:tmax], lorenz_func(dnu[1:tmax], pp[0], pp[1], pp[2]), lw=3, color='r')
    ax2.vlines(x = pp[2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'solid', color = 'b')
    ax2.vlines(x = pp[2] + cov[2][2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'dotted', color = 'b')
    ax2.vlines(x = pp[2] - cov[2][2], ymin = min(corr[1: tmax]), ymax=max(corr[1: tmax]), linestyles = 'dotted', color = 'b')
    ax2.annotate(date_string + "(" + str(comp) + ")", xy=(0.01, 0.99),xycoords='axes fraction', fontsize=20, horizontalalignment='left', verticalalignment='top')
    ax2.legend()
    fig2.savefig(fig_outname + '_acf_all_freq.pdf')
    
    return 0



def fit_acf(bw, nfreq, dnu, corr, fig_outname, save_outdir, date_string, comp = 1, nu_max = 15, plot = True, save = True):
    """
    data: 1-d spectrum, single peak 
    tmax: only want to fit the central peak 
    
    Return: 
    dnu: delta frequency in MHz 
    pp[2]: HWHM, decorrelation frequency 
    
    """
    tmax = int(nfreq * nu_max / bw) 
    
    plt.figure()
    plt.scatter(dnu[1:tmax], corr[1:tmax])
    plt.ylabel("ACF")
    plt.xlabel(r"$\Delta$ $\nu$ (MHZ)")
    
    pp, cov = curve_fit(lorenz_func, dnu[1:tmax], corr[1:tmax], 
                        p0=[50.0, 0, 20.0], bounds=[(-np.inf, 0, -np.inf), (np.inf,np.inf,np.inf)]
                       )
    print("Decorrelation bandwidth %.2f \pm %.2f MHz"%(pp[2], cov[2,2]**0.5))
    print(pp)
    
 
    model = lorenz_func(dnu, pp[0], pp[1], pp[2])
    chisq = np.sum((corr[: tmax] - model[: tmax]) ** 2) # acf error bar?  
    dof = tmax - len(pp) 
    p_value = stats.distributions.chi2.sf(chisq, dof) # p-value(x) = 1 - cdf(x) 

    
    if plot: 
        plot_acf_results(dnu, corr, pp, cov, fig_outname, date_string, tmax = tmax, comp = comp)
    
    if save:
        with open(save_outdir + 'acf.txt', "a") as f:
            print("{}: ${:.4} \pm {:.2} MHz$".format(date_string, pp[2], cov[2][2]), end=' & ', file=f)
            print("{:.2} & {:d} & {:f} \\\\".format(chisq, dof, p_value),  file=f)
    
    return pp, cov, chisq, dof, p_value 

    

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

        data_t_binned = Bin_profile(data.mean(0), width_filterbank)[0]
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
# def make_spectra_after_fitting(bf, data, FWHM = True, weight = True):
#         """
#         Make the spectra by using the profile fitting parameters.
#         Weight time samples by SNR 

#         Returns: spectra

#         """
#         tau_width = 0
        
#         # make spec for each component
#         # another case: multi-peak 
#         try:
#             print("Making spectra using profile fit parameters.")
#             mu_idx = np.where(np.array(bf.profile_param_names) == "mu_t")[0]
#             sig_idx = np.where(np.array(bf.profile_param_names) == "sigma_t")[0]
#             assert len(mu_idx) == 1, "mu not found in profile parameter names"
#             assert len(sig_idx) == 1, "sigma not found in profile parameter names"
#             bf.i0 = bf.profile_params[bf.comp_num]["popt"][mu_idx[0]]
#             width = 2.355 * bf.profile_params[bf.comp_num]["popt"][sig_idx[0]]
#             if "tau" in bf.profile_param_names:
#                 t_idx = np.where(np.array(bf.profile_param_names) == "tau")[0]
#                 assert len(t_idx) == 1, "tau not found in profile parameter names"
#                 tau_width += bf.profile_params[bf.comp_num]["popt"][t_idx[0]]
#             width = int(width)
#             bf.i0 = int(bf.i0)
#         except (KeyError, AssertionError) as e:
#             print(f"{e}")
#             width = bf.width
#             if bf.comp_num == 1:
#                 print(
#                     f"Making spectra using center bins. Could be inaccurate."
#                 )
#                 bf.i0 = bf.nt // 2
#             else:
#                 print(
#                     f"Making spectra using profile argmax. Could be inaccurate."
#                 )
#                 bf.i0 = np.argmax(bf.ts)

#         if width > 2:
#             start = bf.i0 - width // 2
#             end = bf.i0 + width // 2
#         else:
#             start = bf.i0 - 1
#             end = bf.i0 + 1
#         if start < 0:
#             start = 0
#         if end > bf.nt:
#             end = bf.nt
#         end += int(tau_width)
#         print(f"Generating spectra from sample {start} to {end}")
        
#         data_chop = data[:, start : end + 1]
#         #print("data shape", np.shape(data_chop))
        
#         if weight: 
#             weights = data[:, start : end + 1].mean(0) ** 2 # weights length = time sample number 
#             #print("weights length = time sample length ", np.shape(weights))
#             spectra = np.average(data_chop, axis = 1, weights = weights)
#         else: 
#             spectra = np.average(data_chop, axis = 1) 

#         print(f"Normalising spectra to unit area.")
#         spectra = spectra / np.trapz(spectra)
        
#         plt.figure()
#         plt.plot(spectra)
#         plt.xlabel("Channel")
#         plt.title("Normalized Spectrum (raw resolution); weighted=%r"%weight)
#         plt.show()
        
#         return spectra 




