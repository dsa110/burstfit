import logging

import numpy as np
from scipy import special
from burstfit.utils.astro import dedisperse, finer_dispersion_correction

logger = logging.getLogger(__name__)


def gauss(x, S, mu, sigma):
    """
    Gaussian function with area S

    Args:
        x: input array to evaluate the function
        S: Area of the gaussian
        mu: mean of the gaussian
        sigma: sigma of the gaussian

    Returns:

    """
    return (S / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(1 / 2) * ((x - mu) / sigma) ** 2
    )

# def gauss2(x, S1, mu1, sigma1, S2, mu2, sigma2):
#     """
#     Gaussian function with area S

#     Args:
#         x: input array to evaluate the function
#         S: Area of the gaussian
#         mu: mean of the gaussian
#         sigma: sigma of the gaussian

#     Returns:

#     """
#     return ((S1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(
#         -(1 / 2) * ((x - mu1) / sigma1) ** 2) + 
#         (S2 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(
#         -(1 / 2) * ((x - mu2) / sigma2) ** 2))
    
    




def gauss_2d(x, amplitude, x0, y0, sigma_x, sigma_y, offset, theta):
    """ 
    Args:
        x: input array, 2d position (x, y)
        
    """
    #theta=0.
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((x[0]-x0)**2) + 2*b*(x[0]-x0)*(x[1]-y0) 
                            + c*((x[1]-y0)**2)))
    
    return g.ravel()



def gauss_norm(x, mu, sig):
    """
    Gaussian function of unit area

    Args:
        x: input array
        mu: center of the gaussian
        sig: sigma of gaussian

    Returns:

    """
    return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(1 / 2) * ((x - mu) / sig) ** 2)


def gauss_norm2(x, mu1, sig1, mu2, sig2, amp1):
    """
    Two gaussian functions of unit total area

    Args:
        x: input array
        mu1: mean of gaussian1
        sig1: sigma of gaussian1
        mu2: mean of gaussian2
        sig2: sigma of gaussian2
        amp1: amplitude of gaussian1

    Returns:

    """
    return amp1 * gauss_norm(x, mu1, sig1) + (1 - amp1) * gauss_norm(x, mu2, sig2)


def gauss_norm3(x, mu1, sig1, mu2, sig2, mu3, sig3, amp1, amp2):
    """
    Three gaussian functions of unit total area

    Args:
        x: input array
        mu1: mean of gaussian1
        sig1: sigma of gaussian1
        mu2: mean of gaussian2
        sig2: sigma of gaussian2
        mu3: mean of gaussian3
        sig3: sigma of gaussian3
        amp1: amplitude of gaussian1
        amp2: amplitude of gaussian2

    Returns:

    """
    return (
        amp1 * gauss_norm(x, mu1, sig1)
        + amp2 * gauss_norm(x, mu2, sig2)
        + (1 - amp1 - amp2) * gauss_norm(x, mu3, sig3)
    )


def pulse_fn(t, S, mu, sigma, tau):
    """

    Function of the pulse profile: Gaussian convolved with an exponential tail
    (see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

    Args:
        t: input array
        S: Area of the pulse (fluence)
        mu: mean of gaussian
        sigma: sigma of gaussian
        tau: scattering timescale

    Returns:

    """
    if (np.array([S, mu, sigma, tau]) < 0).sum() > 0:
        return np.zeros(len(t))
    if sigma / tau > 6:
        p = gauss(t, S, mu, sigma)
        print("sigma / tau > 6, will use gauss profile gauss(t, S, mu, sigma) without tau.")
    else:
        A = S / (2 * tau)
        B = np.exp((1 / 2) * (sigma / tau) ** 2)
        ln_C = -1 * (t - mu) / tau
        D = 1 + special.erf((t - (mu + (sigma ** 2) / tau)) / (sigma * np.sqrt(2)))
        m0 = D == 0
        ln_C[m0] = 0
        p = A * D * B * np.exp(ln_C)
    return p


def pulse_fn2(t, S1, mu1, sigma1, tau1, S2, mu2, sigma2, tau2):
    """

    Function of the pulse profile: Gaussian convolved with an exponential tail
    (see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

    Args:
        t: input array
        S: Area of the pulse (fluence)
        mu: mean of gaussian
        sigma: sigma of gaussian
        tau: scattering timescale

    Returns:

    """
    if (np.array([S, mu1, sigma1, tau1, S2, mu2, sigma2, tau2]) < 0).sum() > 0:
        return np.zeros(len(t))
    if sigma1 / tau1 > 6:
        p1 = gauss(t, S1, mu1, sigma1)
        print("sigma / tau > 6, will use gauss profile gauss(t, S, mu, sigma) without tau.")
    else:
        A = S1 / (2 * tau1)
        B = np.exp((1 / 2) * (sigma1 / tau1) ** 2)
        ln_C = -1 * (t - mu1) / tau1
        D = 1 + special.erf((t - (mu1 + (sigma1 ** 2) / tau1)) / (sigma1 * np.sqrt(2)))
        m0 = D == 0
        ln_C[m0] = 0
        p1 = A * D * B * np.exp(ln_C)
        
    if sigma2 / tau2 > 6:
        p2 = gauss(t, S2, mu2, sigma2)
        print("sigma / tau > 6, will use gauss profile gauss(t, S, mu, sigma) without tau.")
    else:
        A = S2 / (2 * tau2)
        B = np.exp((1 / 2) * (sigma2 / tau2) ** 2)
        ln_C = -1 * (t - mu2) / tau2
        D = 1 + special.erf((t - (mu2 + (sigma2 ** 2) / tau2)) / (sigma2 * np.sqrt(2)))
        m0 = D == 0
        ln_C[m0] = 0
        p2 = A * D * B * np.exp(ln_C)
        
    return p1 + p2 


def pulse_fn_vec(t, S, mu, sigma, tau):
    """

    Vectorized implementation of pulse profile function: Gaussian convolved with an exponential tail
    (see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

    Args:
        t: input array
        S: Area of the pulse (fluence)
        mu: means of gaussians for each channel
        sigma: sigma of gaussian
        tau: scattering timescale for each channel

    Returns:
        2D spectrogram with pulse profiles

    """
    if not isinstance(tau, np.ndarray):
        tau = np.array([tau])

    if not isinstance(mu, np.ndarray):
        mu = np.array([mu])

    pulse = np.empty(shape=(len(tau), len(t)))

    mask = (sigma / tau) > 6
    gauss_idx = np.where(mask)[0]
    scat_idx = np.where(~mask)[0]

    if len(scat_idx) > 0:
        tau = tau[scat_idx]
        mu_scat = mu[scat_idx]
        A = S / (2 * tau)
        B = np.exp((1 / 2) * (sigma / tau) ** 2)
        ln_C = -1 * (t - mu_scat) / tau
        D = 1 + special.erf((t - (mu_scat + (sigma ** 2) / tau)) / (sigma * np.sqrt(2)))
        m0 = D == 0
        ln_C[m0] = 0
        pulse[scat_idx] = A * B * np.exp(ln_C) * D

    if len(gauss_idx) > 0:
        gauss_pulse = gauss(t, S, mu[gauss_idx], sigma)
        pulse[gauss_idx] = gauss_pulse
    pulse = np.squeeze(pulse)
    return pulse


def no_model(x, c):
    """
    No model.
    returns a constant for each channel x.
    c: array of length(x) 
    """
    return c 
        

def sgram_fn(
    metadata,
    pulse_function,
    spectra_function,
    spectra_params,
    pulse_params,
    other_params,
):
    """
    Spectrogram function

    Args:
        metadata: Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff)
        pulse_function: Function to model pulse
        spectra_function: Function to model spectra
        spectra_params: Dictionary with spectra parameters
        pulse_params: Dictionary with pulse parameters
        other_params: list of other params needed for this function (eg: [dm])

    Returns:

    """
    nt, nf, dispersed_at_dm, tsamp, fstart, foff = metadata
    #     dm, tau_idx = other_params
    [dm] = other_params
    tau_idx = 4
    nt = int(nt)
    nf = int(nf)
    freqs = fstart + foff * np.linspace(0, nf - 1, nf)
    chans = np.arange(nf)
    times = np.arange(nt)
    spectra_from_fit = spectra_function(chans, **spectra_params)  # nu_0, nu_sig)

    model = np.zeros(shape=(nf, nt))
    if "tau" in pulse_params.keys():
        tau = pulse_params["tau"]
        p_params = pulse_params
        for i, freq in enumerate(freqs):
            tau_f = tau * (freq / fstart) ** (-1 * tau_idx)  # tau is defined at fstart
            p_params["tau"] = tau_f
            p = pulse_function(times, **p_params)
            model[i, :] += p
    else:
        for i, freq in enumerate(freqs):
            p = pulse_function(times, **pulse_params)
            model[i, :] += p

    model_dm = dispersed_at_dm - dm

    dedispersed_model, delay_bins, delay_time = dedisperse(
        model, model_dm, tsamp, freqs
    )

    dedispersed_model_corrected = finer_dispersion_correction(
        dedispersed_model, delay_time, delay_bins, tsamp
    )
    model_final = dedispersed_model_corrected * spectra_from_fit[:, None]
    return model_final


def sgram_fn_vec(
    metadata,
    pulse_function,
    spectra_function,
    spectra_params,
    pulse_params,
    other_params,
):
    """
    Vectorized implementation of spectrogram function. Assumes the following input names for pulse_function:
    S, mu, sigma, tau

    Args:
        metadata: Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff)
        pulse_function: Function to model pulse
        spectra_function: Function to model spectra
        spectra_params: Dictionary with spectra parameters
        pulse_params: Dictionary with pulse parameters
        other_params: list of other params needed for this function (eg: [dm])

    Returns:

    """

    nt, nf, dispersed_at_dm, tsamp, fstart, foff = metadata
    [dm] = other_params
    tau_idx = 4
    nt = int(nt)
    nf = int(nf)
    freqs = fstart + foff * np.linspace(0, nf - 1, nf)
    chans = np.arange(nf)
    times = np.arange(nt)
    spectra_from_fit = spectra_function(chans, **spectra_params)  # nu_0, nu_sig)

    model_dm = dispersed_at_dm - dm

    assert "tau" in pulse_params.keys()
    assert "mu" in pulse_params.keys()
    assert "S" in pulse_params.keys()
    assert "sigma" in pulse_params.keys()

    tau = pulse_params["tau"]
    taus = tau * (freqs / fstart) ** (-1 * tau_idx)

    mu_t = pulse_params["mu"]
    mus = (
        mu_t
        + 4148808.0 * model_dm * (1 / (freqs[0]) ** 2 - 1 / (freqs) ** 2) / 1000 / tsamp
    )

    mus = np.expand_dims(mus, -1)
    taus = np.expand_dims(taus, -1)

    l = pulse_function(times, pulse_params["S"], mus, pulse_params["sigma"], taus)
    model = l * spectra_from_fit[:, None]

    return model


# gechen 
# functions for DM fit
# gechen: for DM fit 

# def gauss_dm(x, S, delta_dm, t0, sigma):
#     """
#     Gaussian function spcific for DM fit 

#     Args:
#         x: input array to evaluate the function
#         S: Area of the gaussian
#         mu: mean of the gaussian
#         sigma: sigma of the gaussian

#     Returns: 1D gauss 
#     """
#     mu = t0 + 4.149e3 * delta_dm / (nu_c ** 2) 
    
#     return (S / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
#         -(1 / 2) * ((x - mu) / sigma) ** 2
#     )


# def pulse_fn_vec_dm(t, S, delta_dm, t0, sigma, tau):
#     """

#     Vectorized implementation of pulse profile function: Gaussian convolved with an exponential tail
#     (see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

#     Args:
#         t: input array
#         S: Area of the pulse (fluence)
#         mu: means of gaussians for each channel
#         sigma: sigma of gaussian
#         tau: scattering timescale for each channel

#     Returns:
#         2D spectrogram with pulse profiles

#     """
#     mu = t0 + 4.149e3 * delta_dm / (nu_c ** 2) 
    
#     if not isinstance(tau, np.ndarray):
#         tau = np.array([tau])

#     if not isinstance(mu, np.ndarray):
#         mu = np.array([mu])

#     pulse = np.empty(shape=(len(tau), len(t)))

#     mask = (sigma / tau) > 6
#     gauss_idx = np.where(mask)[0]
#     scat_idx = np.where(~mask)[0]

#     if len(scat_idx) > 0:
#         tau = tau[scat_idx]
#         mu_scat = mu[scat_idx]
#         A = S / (2 * tau)
#         B = np.exp((1 / 2) * (sigma / tau) ** 2)
#         ln_C = -1 * (t - mu_scat) / tau
#         D = 1 + special.erf((t - (mu_scat + (sigma ** 2) / tau)) / (sigma * np.sqrt(2)))
#         m0 = D == 0
#         ln_C[m0] = 0
#         pulse[scat_idx] = A * B * np.exp(ln_C) * D

#     if len(gauss_idx) > 0:
#         gauss_pulse = gauss_dm(t, S, delta_dm, sigma, t0)#(t, S, mu[gauss_idx], sigma)
#         pulse[gauss_idx] = gauss_pulse
#     pulse = np.squeeze(pulse)
#     return pulse
