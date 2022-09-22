#!/usr/bin/env python3

import logging
from contextlib import closing
from multiprocessing import Pool

import emcee
import numpy as np

from burstfit.utils.plotter import plot_mcmc_results, autocorr_plot

logger = logging.getLogger(__name__)


class MCMC:
    """
    Class to run MCMC on the burst model.

    Args:
        model_function: Function to create the model
        sgram: 2D spectrogram data
        initial_guess: Initial guess of parameters for MCMC (can be a dictionary or list)
        param_names: Names of parameters
        nwalkers: Number of walkers to use in MCMC
        nsteps: Number of iterations to use in MCMC
        skip: Number of samples to skip for burn-in
        start_pos_dev: Percent deviation for start position of the samples
        prior_range: Percent of initial guess to set as prior range
        prior_c: Upper limit of initial guess for relative spectral amplitude in each channel 
        prior_S: Upper limit of initial guess for fluence in each component 
        ncores: Number of CPUs to use
        outname: Name of output files
        save_results: Save MCMC samples to a file
    """

    def __init__(
        self,
        model_function,
        sgram,
        initial_guess,
        param_names,
        nwalkers=30,
        nsteps=1000,
        skip=3000,
        start_pos_dev=0.01,
        prior_range=0.2,
        prior_c0=10, 
        prior_c1=10, 
        prior_c2=10, 
        prior_S = 5, 
        prior_S1 = 5,
        prior_S2 = 5, 
        prior_S3 = 5, 
        prior_S4 = 5, 
        prior_DM = 5, 
        ncores=10,
        outname="mcmc_res",
        fig_title=None,
        save_results=True,
    ):
        self.model_function = model_function
        if isinstance(initial_guess, dict):
            cf_params = []
            cf_errors = []
            for i, value in initial_guess.items():
                cf_params += value["popt"]
                cf_errors += list(value["perr"])
            initial_guess = np.array(cf_params)
        elif isinstance(initial_guess, list):
            initial_guess = np.array(initial_guess)
            cf_errors = None
        else:
            cf_errors = None

        self.cf_errors = cf_errors
        self.initial_guess = np.array(initial_guess)
        assert len(param_names) == len(initial_guess)
        self.param_names = param_names
        self.prior_range = prior_range
        self.prior_c0 = prior_c0
        self.prior_c1 = prior_c1
        self.prior_c2 = prior_c2
        self.prior_S = prior_S
        self.prior_S1 = prior_S1
        self.prior_S2 = prior_S2
        self.prior_S3 = prior_S3
        self.prior_S4 = prior_S4
        self.prior_DM = prior_DM
        self.sgram = sgram
        self.std = np.std(sgram)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.skip = skip
        self.start_pos_dev = start_pos_dev
        self.ncores = ncores
        self.sampler = None
        self.samples = None
        self.outname = outname
        self.fig_title = fig_title
        self.save_results = save_results
        self.autocorr = None
        self.pos = None
        self.max_prior = None
        self.min_prior = None
        self.set_initial_pos()
        self.set_priors()

    @property
    def ndim(self):
        """
        Returns the number of dimensions.

        Returns:
            number of dimensions

        """
        return len(self.initial_guess)

    def lnprior(self, params):
        """
        Prior function. Priors are uniform from (1-prior_range)*initial_guess to (1+prior_range)*initial_guess.
        Minimum prior for tau is set to 0.

        Args:
            params: Parameters to check.

        Returns:

        """
        m1 = params <= self.max_prior
        m2 = params >= self.min_prior

        if m1.sum() + m2.sum() == 2 * len(self.initial_guess):
            return 0
        else:
            return -np.inf

    def lnprob(self, params):
        """
        Log probability function.

        Args:
            params: Parameters to evaluate at.

        Returns:
            Prior + log likelihood at the inputs.

        """
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlk(inps=params)

    def lnlk(self, inps):
        """
        Log likelihood function. Uses the model_function to generate the model.

        Args:
            inps: Parameters to evaluate at.

        Returns:
            Log likelihood.

        """
        model = self.model_function([0], *inps)
        return -0.5 * np.sum(((self.sgram.ravel() - model) / self.std) ** 2)

    def set_initial_pos(self):
        """
        Function to set the initial values of walkers and prior ranges.
        Minimum prior for tau is set to 0.

        Returns:

        """
        logging.info("Setting initial positions for MCMC.")
        logger.info(f"Initial guess for MCMC is: {self.initial_guess}")

        if self.nwalkers < 2 * self.ndim:
            logger.warning(
                "Number of walkers is less than 2*ndim. Setting nwalkers to 2*ndim+1."
            )
            self.nwalkers = 2 * self.ndim + 1

        pos = [
            np.array(self.initial_guess)
            * np.random.uniform(
                1 - self.start_pos_dev, 1 + self.start_pos_dev, size=self.ndim
            )
            for i in range(self.nwalkers)
        ]
        self.pos = np.array(pos)
        return self

    
    def set_priors(self):
        """
        Set priors for MCMC

        Returns:

        """
        logger.info("Setting priors for MCMC. Use prior_range = {self.prior_range}")
        self.max_prior = (1 + self.prior_range) * self.initial_guess
        self.min_prior = (1 - self.prior_range) * self.initial_guess
      

        tau_idx = [i for i, t in enumerate(self.param_names) if "tau" in t]
        if len(tau_idx):
            max_tau = np.max(np.take(self.max_prior, tau_idx))

        sig_t_idx = [i for i, t in enumerate(self.param_names) if "sigma_t" in t]
        if len(sig_t_idx):
            max_sigma_t = np.max(np.take(self.max_prior, sig_t_idx))

        S_idx = [i for i, t in enumerate(self.param_names) if "S" in t]

        mu_f_idx = [i for i, t in enumerate(self.param_names) if "mu_f" in t]
        sigma_f_idx = [i for i, t in enumerate(self.param_names) if "sigma_f" in t]
        
        DM_idx = [i for i, t in enumerate(self.param_names) if "DM" in t]
        c0_idx = [i for i, t in enumerate(self.param_names) if "c0" in t]
        c1_idx = [i for i, t in enumerate(self.param_names) if "c1" in t]
        c2_idx = [i for i, t in enumerate(self.param_names) if "c2" in t]
        S1_idx = [i for i, t in enumerate(self.param_names) if "S1" in t]
        S2_idx = [i for i, t in enumerate(self.param_names) if "S2" in t]
        S3_idx = [i for i, t in enumerate(self.param_names) if "S1" in t]
        S4_idx = [i for i, t in enumerate(self.param_names) if "S1" in t]


        nf, nt = self.sgram.shape

        if len(DM_idx) > 0:
            DM_range = 3
            logger.info(
                "Found DM in param_names. Setting its prior range to +- {self.prior_DM}"
            )
            self.min_prior[DM_idx] = self.initial_guess[DM_idx] - self.prior_DM
            self.max_prior[DM_idx] = self.initial_guess[DM_idx] + self.prior_DM
        
        if len(c0_idx) > 0:
            logger.info(
                "Found c0 in param_names. Setting its min value of prior to 0." 
                "Setting its max value to max(0, min(1.0, i)) for i in self.initial_guess[c0_idx] * self.prior_c0)])"
            )
            self.min_prior[c0_idx] = 0
            print("c0_idx, self.initial_guess[c0_idx], self.prior_c0:", c0_idx, self.initial_guess[c0_idx], self.prior_c0)
            print("self.initial_guess[c0_idx] * self.prior_c0: ", [self.initial_guess[c0_idx] * (1 + self.prior_c0)])
            self.max_prior[c0_idx] = [max(0, min(1.0, i)) for i in self.initial_guess[c0_idx] * self.prior_c0]
        
        
        if len(c1_idx) > 0:
            logger.info(
                "Found c1 in param_names. Setting its min value of prior to 0."
                "Setting its max value to max(0, min(1.0, i)) for i in self.initial_guess[c1_idx] * self.prior_c1)])"
            )
            self.min_prior[c1_idx] = 0
            self.max_prior[c1_idx] = [max(0, min(1.0, i)) for i in self.initial_guess[c1_idx] * self.prior_c1]
        
        
        if len(c2_idx) > 0:
            logger.info(
                "Found c2 in param_names. Setting its min value of prior to 0."
                "Setting its max value to max(0, min(1.0, i)) for i in self.initial_guess[c2_idx] * self.prior_c2)])"
            )
            self.min_prior[c2_idx] = 0
            self.max_prior[c2_idx] = [max(0, min(1.0, i)) for i in self.initial_guess[c2_idx] * self.prior_c2]
        
        
        if len(S1_idx) > 0:
            logger.info(
                "Found S1 in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[S1_idx] = 0
            self.max_prior[S1_idx] = self.initial_guess[S1_idx] * max(1, self.prior_S1)
 
        if len(S2_idx) > 0:
            logger.info(
                "Found S2 in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[S2_idx] = 0
            self.max_prior[S2_idx] = self.initial_guess[S2_idx] * max(1, self.prior_S2) 

        if len(S3_idx) > 0:
            logger.info(
                "Found S3 in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[S3_idx] = 0
            self.max_prior[S3_idx] = self.initial_guess[S3_idx] * max(1, self.prior_S3) 

            
        if len(S4_idx) > 0:
            logger.info(
                "Found S4 in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[S4_idx] = 0
            self.max_prior[S4_idx] = self.initial_guess[S4_idx] * max(1, self.prior_S4)  
                
        
        if len(tau_idx) > 0:
            logger.info(
                "Found tau in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[tau_idx] = 0

        if len(sig_t_idx) > 0:
            logger.info(
                "Found sigma_t in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[sig_t_idx] = 0

        if len(sig_t_idx) > 0 and len(tau_idx) > 0:
            logger.info(
                f"Found sigma_t and tau in param_names. Setting its max value of prior to "
                f"2*(max_tau_prior({max_tau}) + max_sigma_t_prior({max_sigma_t}))"
            )
            self.max_prior[tau_idx] = 2 * (max_sigma_t + max_tau)
            self.max_prior[sig_t_idx] = 2 * (max_sigma_t + max_tau)

        if len(S_idx) > 0 and len(sig_t_idx) > 0:
            logger.info(
                f"Found S and sigma_t in param_names. Setting its max value of prior to "
                f"500*max(ts)*max_sigma_t_prior. Setting its min value of prior to 0."
            )
            self.max_prior[S_idx] = 800 * np.max(self.sgram.sum(0)) * max_sigma_t
            self.min_prior[S_idx] = 0
            
        

        if len(mu_f_idx) > 0:
            logger.info(
                f"Found mu_f in param_names. Setting its priors to (-2*nf, 3*nf)"
            )
            self.min_prior[mu_f_idx] = -2 * nf
            self.max_prior[mu_f_idx] = 3 * nf

        if len(sigma_f_idx) > 0:
            logger.info(
                f"Found sigma_f in param_names. Setting its priors to (0, 5*nf)"
            )
            self.min_prior[sigma_f_idx] = 0
            self.max_prior[sigma_f_idx] = 5 * nf
        

        return self

    def run_mcmc(self):
        """
        Runs the MCMC.

        Returns:
            Sampler object

        """
        logger.debug(
            f"Range of initial positions of walkers (min, max): ({self.pos.min(0)}, {self.pos.max(0)})"
        )
        logger.debug(
            f"Range of priors (min, max): ({(1 - self.prior_range) * self.initial_guess},"
            f"{(1 + self.prior_range) * self.initial_guess})"
        )

        if self.save_results:
            backend = emcee.backends.HDFBackend(f"{self.outname}_samples.h5")
            backend.reset(self.nwalkers, self.ndim)
        else:
            backend = None

        index = 0
        autocorr = np.zeros(self.nsteps)
        old_tau = np.inf

        logger.info(
            f"Running MCMC with the following parameters: nwalkers={self.nwalkers}, "
            f"nsteps={self.nsteps}, start_pos_dev={self.start_pos_dev}, ncores={self.ncores}, "
            f"skip={self.skip}"
        )

        logger.info("Priors used in MCMC are:")
        for j, p in enumerate(self.param_names):
            logger.info(f"{p}: [{self.min_prior[j]}, {self.max_prior[j]}]")

        with closing(Pool(self.ncores)) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, self.lnprob, pool=pool, backend=backend
            )
            for sample in sampler.sample(
                self.pos, iterations=self.nsteps, progress=True, store=True
            ):
                if sampler.iteration % 100:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

            pool.terminate()
        self.autocorr = autocorr
        self.sampler = sampler
        return self.sampler

    def get_chain(self, skip=None):
        """
        Returns the chanins from sampler object after removing some samples for burn-in.

        Args:
            skip: Number of steps to skip for burn-in.

        Returns:
            Sample chain.

        """
        if not skip:
            skip = self.skip
        logger.info("Using autocorrelation time to estimate burnin and thin values.")
        tau = self.sampler.get_autocorr_time(tol=0)
        if np.isnan(tau).sum() == 0:
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            if burnin < skip:
                skip = burnin
            logger.info(f"Burnin is: {burnin}, thin is: {thin}")
        else:
            thin = 0
            logger.warning(
                "Autocorrelation time is nan. Not using tau for burn-in calculation."
            )

        logger.info(f"Discarding {skip} steps/iterations.")
        if skip > self.sampler.iteration:
            logger.warning(f"Not enough steps in chain to skip. Not removing burn-in.")
            skip = 0
        if thin == 0:
            self.samples = self.sampler.get_chain(flat=True, discard=skip)
        else:
            self.samples = self.sampler.get_chain(flat=True, discard=skip, thin=thin)
        return self.samples

    def print_results(self):
        """
        Prints the results of MCMC analysis. It uses median values with 1-sigma errors based on MCMC posteriors.

        Returns:

        """
        if not np.any(self.samples):
            self.get_chain()
        qs = np.quantile(self.samples, [0.16, 0.5, 0.84], axis=0)
        logger.info(f"MCMC fit results are:")
        e1 = qs[1] - qs[0]
        e2 = qs[2] - qs[1]
        p = qs[1]
        for i, param in enumerate(self.param_names):
            logger.info(f"{self.param_names[i]}: {p[i]} + {e2[i]:.3f} - {e1[i]:.3f}")

    def plot(self, save=False):
        """
        Plot the samples and corner plot of MCMC posteriors.

        Args:
            save: To save the corner plot.

        Returns:

        """
        logger.info("Plotting MCMC results.")
        plot_mcmc_results(
            self.samples, self.outname, self.fig_title, self.initial_guess, self.param_names, save
        )

    def make_autocorr_plot(self, save=False):
        """
        Make autocorrelation plot for MCMC (i.e autocorrelation  time scale vs iteration)
        see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

        Args:
            save: To save the plot

        Returns:

        """
        index = (self.autocorr > 0).sum()
        if index > 2:
            n = 100 * np.arange(1, index + 1)
            y = self.autocorr[:index]
            autocorr_plot(n, y, self.outname, save)
        else:
            logger.warning(
                f"Not enough valid autocorrelation values to plot. Not making autocorrelation plot."
            )
        
        
    def BIC(self, k, n):
        """
        Bayesian information criterion 
        L: likelihood
        k: model parameter number
        n: data sample size 
        """
        params = [self.sgram_params['all'][i]['popt'] for i in range(1, bf_S1T2_c1.comp_num + 1)][0]
        lnL = self.lnlk(params)
        return -2 * lnL + k * np.log(n) 
        #return -2*np.log(L) + k * np.log(n) 
        
