# -*- coding: utf-8 -*-
"""
Module: anlffr.spectral

A collection of spectral analysis functions for FFR data, curated for
correctness. :)

Includes functions to estimate frequency content / phase locking of
single-channel or individual data channels, as well as functions that produce
estimates by combining across channels (via the cPCA method described in [1]).
The accompanying bootsrap module provides support for bootstrapping any of the
analysis functions in this module.

Function Listing
================
Per-channel functions:
--------------------------------------

    mtplv

    mtspec

    mtphase

    mtppc

    mtspecraw

    mtpspec


Multichannel functions utilizing cPCA:
--------------------------------------

    mtcplv (alias for mtcpca)

    mtcspec

    mtcpca_timeDomain

NOTE: Due to the poor SNR of individual trials typical in FFR datasets,
cPCA-based methods implemented in this module first compute the parameter
of interest on a per-channel basis, then computes the cross-spectral
densities over channels. We point out that this is different from what a
strict interpretation of the notation in the equations in [1] suggests.
Computation of the cross-spectral density on a per-trial basis will
emphasize features that are phase-locked across channels (e.g., noise). For
FFRs, the contributions of activity phase locked over channels but not over
trials will swamp peaks in the resulting output metric, particularly at low
frequencies.


References:
=======================================

[1] Bharadwaj, H and Shinn-Cunningham, BG (2014).
      "Rapid acquisition of auditory subcortical steady state responses using
      multichannel recordings".
      J Clin Neurophys 125 1878--1898.
      http://dx.doi.org/10.1016/j.clinph.2014.01.011

@author: Hari Bharadwaj

"""

import numpy as np
from math import ceil
import scipy as sci
from scipy import linalg

# use nitime if available:
try:
    from nitime.algorithms import dpss_windows
    print '\nnitime detected. Using nitime.algorithms.dpss_windows\n'
except ImportError:
    from .dpss import dpss_windows
    print '\nnitime not detected. Using anlffr.dpss.dpss_windows\n'

# rename verbose to make pep8/pylint checkers stop complainig
from .utils import logger, deprecated, verbose as verbose_decorator


@verbose_decorator
def mtplv(x, params, verbose=None, bootstrapMode=False):
    """Multitaper Phase-Locking Value

    Parameters
    ----------
    x - NumPy Array
        Input Data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        (plvtap, f): Tuple
           plvtap - Multitapered phase-locking estimate (channel x frequency)

    In bootstrap mode:
        Dictionary with the following keys:
         mtplv - Multitapered phase-locking estimate (channel x frequency)i

         f - Frequency vector matching plv

    """

    logger.info('Running Multitaper PLV Estimation')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        nchans = x.shape[0]
        ntrials = x.shape[trialdim]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry, The data should be a 2 or 3 dimensional array')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result

    plvtap = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        if(params['itc'] == 0):
            plvtap[k, :, :] = abs((xw/abs(xw)).mean(axis=trialdim))**2
        else:
            plvtap[k, :, :] = ((abs(xw.mean(axis=trialdim))**2) /
                               ((abs(xw) ** 2).mean(axis=trialdim)))

    plvtap = plvtap.mean(axis=0)

    plvtap = plvtap[:, fInd].squeeze()

    if bootstrapMode:
        out = {}
        out['mtplv'] = plvtap
        out['f'] = f
    else:
        return (plvtap, f)

    return out


@verbose_decorator
def mtspec(x, params, verbose=None, bootstrapMode=False):
    """Multitaper Spectrum and SNR estimate

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['noisefloortype'] - (optional) 1: random phase,
      0 (default): flip-phase on half the trials

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:

        (S, N ,f): Tuple
          S - Multitapered spectrum (channel x frequency)

          N - Noise floor estimate

          f - Frequency vector matching S and N

    In bootstrap mode:
        Dictionary with the following keys:
         mtspec - Multitapered spectrum (channel x frequency)

         mtspec_* - Noise floor estimate, where * is 'randomPhase' if
         params['noisefloortype'] == 1, and 'noiseFloorViaPhaseFlip' otherwise

         f - Frequency vector matching plv

    """

    logger.info('Running Multitaper Spectrum and Noise-floor Estimation')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    S = np.zeros((ntaps, nchans, nfft))
    N = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        S[k, :, :] = abs(xw.mean(axis=trialdim))

        if ('noisefloortype' in params) and (params['noisefloortype'] == 1):
            randph = sci.rand(nchans, ntrials, nfft) * 2 * sci.pi
            N[k, :, :] = abs((xw*sci.exp(1j*randph)).mean(axis=trialdim))
            noiseTag = 'noiseFloorViaRandomPhase'
            logger.info('using random phase for noise floor estimate')
        else:
            randsign = np.ones((nchans, ntrials, nfft))

            # reflects fix to bootstrapmode parameter
            if bootstrapMode and 'bootstrapTrialsSelected' in params:
                flipTheseTrials = np.where(
                    (params['bootstrapTrialsSelected'] % 2) == 0)
            else:
                flipTheseTrials = np.arange(0, ntrials, 2)

            randsign[:, flipTheseTrials, :] = -1
            N[k, :, :] = abs((xw*(randsign.squeeze())).mean(axis=trialdim))
            noiseTag = 'noiseFloorViaPhaseFlip'
            logger.info('flipping phase of half of the trials ' +
                        'for noise floor estimate')

    # Average over tapers and squeeze to pretty shapes
    S = S.mean(axis=0)
    N = N.mean(axis=0)
    S = S[:, fInd].squeeze()
    N = N[:, fInd].squeeze()

    if bootstrapMode:
        out = {}
        out['mtspec'] = S
        out['mtspec_' + noiseTag] = N
        out['f'] = f

        return out
    else:
        return (S, N, f)


@verbose_decorator
def mtphase(x, params, verbose=None, bootstrapMode=False):
    """Multitaper phase estimation

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns:
    -------
    In normal mode:
        (Ph, f): Tuple
          Ph - Multitapered phase spectrum (channel x frequency)

          f - Frequency vector matching S and N

    In bootstrap mode:
        Dictionary with the following keys:

        Ph - Multitapered phase spectrum (channel x frequency)

        f - Frequency vector matching plv

    """

    logger.info('Running Multitaper Spectrum and Noise-floor Estimation')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    Ph = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        Ph[k, :, :] = np.angle(xw.mean(axis=trialdim))

    # Average over tapers and squeeze to pretty shapes
    Ph = Ph[:, :, fInd].mean(axis=0).squeeze()

    if bootstrapMode:
        out = {}
        out['mtphase'] = Ph
        out['f'] = f

        return out
    else:
        return (Ph, f)


@verbose_decorator
def mtcpca(x, params, verbose=None, bootstrapMode=False):
    """Multitaper complex PCA and PLV

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        Tuple (plv, f):
          plv - Multitapered PLV estimate using cPCA

          f - Frequency vector matching plv

    In bootstrap mode:
        Dictionary with the following keys:
          mtcplv - Multitapered PLV estimate using cPCA

          f - Frequency vector matching plv

    """

    logger.info('Running Multitaper Complex PCA based PLV Estimation')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    plv = np.zeros((ntaps, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        if params['itc']:
            C = (xw.mean(axis=trialdim) /
                 (abs(xw).mean(axis=trialdim))).squeeze()
        else:
            C = (xw / abs(xw)).mean(axis=trialdim).squeeze()

        for fi in np.arange(0, nfft):
            Csd = np.outer(C[:, fi], C[:, fi].conj())
            vals = linalg.eigh(Csd, eigvals_only=True)
            plv[k, fi] = vals[-1] / nchans

    # Average over tapers and squeeze to pretty shapes
    plv = (plv.mean(axis=0)).squeeze()
    plv = plv[fInd]
    if bootstrapMode:
        out = {}
        out['mtcplv'] = plv
        out['f'] = f

        return out
    else:
        return (plv, f)


mtcplv = mtcpca


@verbose_decorator
def mtcspec(x, params, verbose=None, bootstrapMode=False):
    """Multitaper complex PCA and power spectral estimate

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        Tuple (cspec, f):

          cspec - Multitapered PLV estimate using cPCA

          f - Frequency vector matching plv

    In bootstrap mode:
        Dictionary with the following keys:

          cspec - Multitapered PLV estimate using cPCA

          f - Frequency vector matching plv
    """

    logger.info('Running Multitaper Complex PCA based power estimation!')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result

    cspec = np.zeros((ntaps, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        C = (xw.mean(axis=trialdim)).squeeze()
        for fi in np.arange(0, nfft):
            Csd = np.outer(C[:, fi], C[:, fi].conj())
            vals = linalg.eigh(Csd, eigvals_only=True)
            cspec[k, fi] = vals[-1] / nchans

    # Average over tapers and squeeze to pretty shapes
    cspec = (cspec.mean(axis=0)).squeeze()
    cspec = cspec[fInd]

    if bootstrapMode:
        out = {}
        out['mtcspec'] = cspec
        out['f'] = f

        return out
    else:
        return (cspec, f)


@verbose_decorator
def mtcpca_timeDomain(x, params, verbose=None, bootstrapMode=False):
    """Multitaper complex PCA and regular time-domain PCA and return time
    domain waveforms.

    Note of caution
    ---------------
    The cPCA method is not really suited to extract fast transient features of
    the time domain waveform. This is because, the frequency domain
    representation of any signal (when you think of it as random process) is
    interpretable only when the signal is stationary, i.e., in steady-state.
    Practically speaking, the process of transforming short epochs to the
    frequency domain necessarily involves smoothing in frequency. This
    leakage is minimized by tapering the original signal using DPSS windows,
    also known as Slepian sequences. The effect of this tapering would be
    present when going back to the time domain. Note that only a single taper
    is used here as combining tapers with different symmetries in the time-
    domain leads to funny cancellations.

    Also, for transient features, simple time-domain PCA is likely
    to perform better as the cPCA smoothes out transient features. Thus
    both regular time-domain PCA and cPCA outputs are returned.

    Note that for sign of the output is indeterminate (you may need to flip
    the output to match the polarity of signal channel responses)

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        Tuple (y_cpc, y_pc):
          'y_cpc' - Multitapered cPCA estimate of time-domain waveform

          'y_pc' - Regular time-domain PCA

    In bootstrap mode:
        Dictionary with the following keys:
          'y_cpc' - Multitapered cPCA estimate of time-domain waveform

          'y_pc' - Regular time-domain PCA

    """

    logger.info('Running Multitaper Complex PCA to extract time waveform!')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    w, conc = dpss_windows(x.shape[timedim], 1, 1)
    w = w.squeeze() / w.max()

    cpc_freq = np.zeros(nfft, dtype=np.complex)
    cspec = np.zeros(nfft)
    xw = sci.fft(w * x, n=nfft, axis=timedim)
    C = (xw.mean(axis=trialdim)).squeeze()
    Cnorm = C / ((abs(xw).mean(axis=trialdim)).squeeze())
    for fi in np.arange(0, nfft):
        Csd = np.outer(Cnorm[:, fi], Cnorm[:, fi].conj())
        vals, vecs = linalg.eigh(Csd, eigvals_only=False)
        cspec[fi] = vals[-1]
        cwts = vecs[:, -1] / (np.abs(vecs[:, -1]).sum())
        cpc_freq[fi] = (cwts.conjugate() * C[:, fi]).sum()

    # Filter through spectrum, do ifft.
    cscale = cspec ** 0.5
    cscale = cscale / cscale.max()  # Maxgain of filter = 1
    y_cpc = sci.ifft(cpc_freq * cscale)[:x.shape[timedim]]

    # Do time domain PCA
    x_ave = x.mean(axis=trialdim)
    C_td = np.cov(x_ave)
    vals, vecs = linalg.eigh(C_td, eigvals_only=False)
    y_pc = np.dot(vecs[:, -1].T, x_ave) / (vecs[:, -1].sum())

    if bootstrapMode:
        out = {}
        out['y_cpc'] = y_cpc
        out['y_pc'] = y_pc

        return out
    else:
        return (y_cpc, y_pc)


@deprecated('Please use the anlffr.bootstrap module for all bootstrap '
            'functions. bootfunc() will be removed in future releases')
@verbose_decorator
def bootfunc(x, nPerDraw, nDraws, params, func='cpca', verbose=None):
    """Run spectral functions with bootstrapping over trials

    Parameters
    ----------
    x - Numpy Array
        Input data (channel x trials x time) or (trials x time)
    nPerDraw - int
        Number of trials for each draw
    nDraws - int
        Number of draws
    params - dict
        Dictionary of parameters to use when calling chosen function
    func - str
        'cpca' or 'plv' or 'itc' or 'spec' or 'ppc' or 'pspec'
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    (mu_func, v_func, f): Tuple
        For everything except when func == 'spec'
    (S, N, vS, vN, f): Tuple
        When func == 'spec'
    A 'v' prefix denotes variance estimate and a prefix 'mu' denotes mean.

    See help for mtcpca(), mtplv() and mtspec() for more details.

    Notes
    -----

    This is not a particularly parallelized piece of code and hence slow.
    It is provided just so the functionality is there.

    """

    logger.info('Running a bootstrapped version of function: %s', func)
    x = x.squeeze()
    if(len(x.shape) == 3):
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        trialdim = 0
        ntrials = x.shape[trialdim]
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    if(func == 'spec'):
        S = 0
        N = 0
        vS = 0
        vN = 0
    else:
        mu_func = 0
        v_func = 0

    for drawnum in np.arange(0, nDraws):
        inds = np.random.randint(0, ntrials, nPerDraw)

        logger.debug('Doing Draw #%d / %d', drawnum + 1, nDraws)

        if(trialdim == 1):
            xdraw = x[:, inds, :]
        elif(trialdim == 0):
            xdraw = x[inds, :]
        else:
            logger.error('Data not in the right formmat!')

        if(func == 'spec'):
            (tempS, tempN, f) = mtspec(xdraw, params, verbose=False)
            S = S + tempS
            N = N + tempN
            vS = vS + tempS ** 2
            vN = vN + tempN ** 2
        elif(func == 'cpca'):
            (temp_func, f) = mtcpca(xdraw, params, verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func ** 2
        elif(func == 'itc'):
            params['itc'] = 1
            (temp_func, f) = mtplv(xdraw, params, verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func ** 2
        elif(func == 'plv'):
            params['plv'] = 0
            (temp_func, f) = mtplv(xdraw, params, verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func ** 2
        elif(func == 'ppc'):
            (temp_func, f) = mtppc(xdraw, params, verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func ** 2
        elif(func == 'pspec'):
            (temp_func, f) = mtpspec(xdraw, params, verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func ** 2
        else:
            logger.error('Unknown func argument!')
            return

    if(func == 'spec'):
        vS = (vS - (S ** 2) / nDraws) / (nDraws - 1)
        vN = (vN - (N ** 2) / nDraws) / (nDraws - 1)
        S = S / nDraws
        N = N / nDraws
        return (S, N, vS, vN, f)
    else:
        v_func = (v_func - (mu_func ** 2) / nDraws) / (nDraws - 1)
        mu_func = mu_func / nDraws
        return (mu_func, v_func, f)


@deprecated('Please use the anlffr.bootstrap module for all bootstrap '
            'functions. indivboot() will be removed in future releases')
@verbose_decorator
def indivboot(x, nPerDraw, nDraws, params, func='cpca', verbose=None):
    """Run spectral functions with bootstrapping over trials
    This also returns individual draw results, unlike bootfunc()

    DEPRECATION WARNING: recommend use of anlffr.bootstrap module

    Parameters
    ----------
    x - Numpy array
        Input data (channel x trials x time) or (trials x time)
    nPerDraw - int
        Number of trials for each draw
    nDraws - int
        Number of draws
    params - dict
        Dictionary of parameters to use when calling chosen function
    func - str
        'cpca' or 'plv' or 'itc' or 'spec' or 'ppc', i.e. which to call?
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    (plv, f) - Tuple
        For everything except when func == 'spec' (including 'ppc')
    (S, N, f)  - Tuple
        When func == 'spec'
    plv, S and N arrays will have an extra dimension spanning the draws.

    See help for mtcpca(), mtplv() and mtspec() for more details.

    Notes
    -----

    This is not a particularly parallelized piece of code and hence slow.
    It is provided just so the functionality is there. Mostly untested.

    """

    logger.info('Running a bootstrapped version of function: %s', func)
    x = x.squeeze()
    if(len(x.shape) == 3):
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Running 1 draw to get the right sizes
    if(func == 'spec'):
        (S, N, f) = mtspec(x, params, verbose=False)
        S = np.zeros(S.shape + (nDraws,))
        N = np.zeros(N.shape + (nDraws,))

    elif((func == 'plv') or (func == 'itc') or (func == 'ppc')):
        (plv, f) = mtplv(x, params, verbose=False)
        plv = np.zeros(plv.shape + (nDraws,))

    elif(func == 'cpca'):
        (plv, f) = mtcpca(x, params, verbose=False)
        plv = np.zeros(plv.shape + (nDraws,))

    for drawnum in np.arange(0, nDraws):
        inds = np.random.randint(0, ntrials, nPerDraw)

        logger.debug('Doing Draw #%d / %d', drawnum + 1, nDraws)

        if(nchans > 1):
            xdraw = x[:, inds, :]
        elif(nchans == 1):
            xdraw = x[inds, :]
        else:
            logger.error('Data not in the right formmat!')

        if(func == 'spec'):
            if(nchans > 1):
                (S[:, :, drawnum], N[:, :, drawnum], f) = mtspec(xdraw,
                                                                 params,
                                                                 verbose=False)
            else:
                (S[:, drawnum], N[:, drawnum], f) = mtspec(xdraw,
                                                           params,
                                                           verbose=False)

        elif(func == 'cpca'):
            (plv[:, drawnum], f) = mtcpca(xdraw, params, verbose=False)

        elif(func == 'itc'):
            params['itc'] = 1
            if(nchans > 1):
                (plv[:, :, drawnum], f) = mtplv(xdraw, params, verbose=False)
            else:
                (plv[:, drawnum], f) = mtplv(x, params, verbose=False)

        elif(func == 'plv'):
            params['plv'] = 0
            if(nchans > 1):
                (plv[:, :, drawnum], f) = mtplv(xdraw, params, verbose=False)
            else:
                (plv[:, drawnum], f) = mtplv(x, params, verbose=False)

        elif(func == 'ppc'):
            if(nchans > 1):
                (plv[:, :, drawnum], f) = mtppc(xdraw, params, verbose=False)
            else:
                (plv[:, drawnum], f) = mtppc(x, params, verbose=False)

        else:
            logger.error('Unknown func argument!')
            return

    if(func == 'spec'):

        return (S, N, f)
    else:
        return (plv, f)


@verbose_decorator
def mtppc(x, params, verbose=None, bootstrapMode=False):
    """Multitaper Pairwise Phase Consistency

    Parameters
    ----------
    x - Numpy array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['Npairs'] - Number of pairs for PPC analysis

      params['itc'] - If True, normalize after mean like ITC instead of PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        Tuple (ppc, f):
          ppc - Multitapered PPC estimate (channel x frequency)

          f - Frequency vector matching plv

    In bootstrap mode:
        Dictionary with the following keys:
          mtppc - Multitapered PPC estimate (channel x frequency)

          f - Frequency vector matching plv

    """

    logger.info('Running Multitaper Pairwise Phase Consistency Estimate')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the result

    ppc = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        npairs = params['nPairs']
        trial_pairs = np.random.randint(0, ntrials, (npairs, 2))

        if(nchans == 1):
            if(not params['itc']):
                xw_1 = xw[trial_pairs[:, 0], :]/abs(xw[trial_pairs[:, 0], :])
                xw_2 = xw[trial_pairs[:, 1], :]/abs(xw[trial_pairs[:, 1], :])
                ppc[k, :, :] = np.real((xw_1*xw_2.conj()).mean(axis=trialdim))
            else:
                xw_1 = xw[trial_pairs[:, 0]]
                xw_2 = xw[trial_pairs[:, 1]]
                ppc_unnorm = np.real((xw_1 * xw_2.conj()).mean(axis=trialdim))
                ppc[k, :, :] = (ppc_unnorm /
                                (abs(xw_1).mean(trialdim) *
                                 abs(xw_2).mean(trialdim)))

        else:
            if(not params['itc']):
                xw_1 = (xw[:, trial_pairs[:, 0], :] /
                        abs(xw[:, trial_pairs[:, 0], :]))
                xw_2 = (xw[:, trial_pairs[:, 1], :] /
                        abs(xw[:, trial_pairs[:, 1], :]))
                ppc[k, :, :] = np.real((xw_1*xw_2.conj()).
                                       mean(axis=trialdim))
            else:
                xw_1 = xw[:, trial_pairs[:, 0], :]
                xw_2 = xw[:, trial_pairs[:, 1], :]
                ppc_unnorm = np.real((xw_1 * xw_2.conj()).mean(axis=trialdim))
                ppc[k, :, :] = (ppc_unnorm /
                                (abs(xw_1).mean(trialdim) *
                                 abs(xw_2).mean(trialdim)))

    ppc = ppc.mean(axis=0)
    ppc = ppc[:, fInd].squeeze()

    if bootstrapMode:
        out = {}
        out['mtppc'] = ppc
        out['f'] = f

        return out
    else:
        return (ppc, f)


@verbose_decorator
def mtspecraw(x, params, verbose=None, bootstrapMode=False):
    """Multitaper Spectrum (of raw signal)

    Parameters
    ----------
    x - Numpy array
        Input data numpy array (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Normal mode:
        Tuple (mtspecraw, f)
          mtspecraw - multitapered spectrum

          f - Frequency vector matching plv

    In bootstrap mode:
        Dictionary with the following keys:
          mtspecraw - Multitapered spectrum (channel x frequency)

          f - Frequency vector matching plv

    """

    logger.info('Running Multitaper Raw Spectrum Estimation')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        nchans = x.shape[0]
        ntrials = x.shape[trialdim]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        nchans = 1
        ntrials = x.shape[trialdim]
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the results

    Sraw = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        Sraw[k, :, :] = (abs(xw)**2).mean(axis=trialdim)

    # Average over tapers and squeeze to pretty shapes
    Sraw = Sraw.mean(axis=0)
    Sraw = Sraw[:, fInd].squeeze()

    if bootstrapMode:
        out = {}
        out['mtspecraw'] = Sraw
        out['f'] = f

        return out
    else:
        return (Sraw, f)


@verbose_decorator
def mtpspec(x, params, verbose=None, bootstrapMode=False):
    """Multitaper Pairwise Power Spectral estimate

    Parameters
    ----------
    x - Numpy Array
        Input data numpy array (channel x trial x time) or (trials x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['Npairs'] - Number of pairs for pairwise analysis
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
      Tuple (pspec, f):
          pspec -  Multitapered Pairwise Power estimate (channel x frequency)

          f - Frequency vector matching plv

    In bootstrap mode:

      Dictionary with following keys:
          pspec -  Multitapered Pairwise Power estimate (channel x frequency)

          f - Frequency vector matching plv

    """
    logger.info('Running Multitaper Pairwise Power Estimate')
    x = x.squeeze()
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        logger.info('The data is of format %d trials x time (single channel)',
                    ntrials)
    else:
        logger.error('Sorry! The data should be a 2 or 3 dimensional array!')

    # Calculate the tapers
    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result

    pspec = np.zeros((ntaps, nchans, nfft))

    for ch in np.arange(0, nchans):
        for k, tap in enumerate(w):
            logger.debug('Running Channel # %d, taper #%d', ch, k)
            xw = sci.fft(tap * x, n=nfft, axis=timedim)
            npairs = params['Npairs']
            trial_pairs = np.random.randint(0, ntrials, (npairs, 2))

            # For unbiasedness, pairs should be made of independent trials!
            trial_pairs = trial_pairs[np.not_equal(trial_pairs[:, 0],
                                                   trial_pairs[:, 1])]
            if(nchans == 1):
                xw_1 = xw[trial_pairs[:, 0]]
                xw_2 = xw[trial_pairs[:, 1]]
                pspec[k, ch, :] = np.real((xw_1*xw_2.conj()).mean(axis=0))
            else:
                xw_1 = xw[ch, trial_pairs[:, 0], :]
                xw_2 = xw[ch, trial_pairs[:, 1], :]
                pspec[k, ch, :] = np.real((xw_1*xw_2.conj()).mean(axis=0))

    pspec = pspec.mean(axis=0)
    pspec = pspec[:, fInd].squeeze()

    if bootstrapMode:
        out = {}
        out['pspec'] = pspec
        out['f'] = f

        return out
    else:
        return (pspec, f)


@verbose_decorator
def _mtcpca_complete(x, params, verbose=None, bootstrapMode=False):
    """
    Internal convenience function to obtain plv and spectrum with cpca and
    multitaper.  Equivalent to calling:

    spectral.mtcpca(data, params, ...)
    spectral.mtcspec(data, params, ...)

    With the exception that this function returns a dictionary for S + N, each
    of which have keys "plv_*" and "spectrum_*", where * is "normalPhase" or
    "noiseFloorViaPhaseFlip".

    Gets power spectra and plv on the same set of data using multitaper and
    complex PCA. Returns a noise floor esimate of each by running the same
    computations on the original data, as well as the original data with the
    phase of half of the trials flipped. For a large number of trials, the
    spectra of the data and the half-trials-phase-flipped data should be
    similar, while the PLV values for the half-trials-flipped data should be
    hovering near the PLV value of off-frequency components in the original
    data.

    Primarily useful when debugging, bootstrapping, or when using scripts that
    for some reason randomizes data in between calls to mtcpca and mtcspec.


    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - dictionary. Must contain the following fields:
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['itc'] - If True, normalize after mean like ITC instead of PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    In normal mode:
        Tuple (S, N, f)

        Where S and N are data for signal and for noise floor, respectively,
        each as a dictionary with the following keys:

          mtcpcaSpectrum - Multitapered power spectral estimate using cPCA

          mtcpcaPLV- Multitapered PLV using cPCA

        f - frequency vector

    In bootstrap mode:
        dictionary with keys:
          mtcpcaSpectrum_* - Multitapered power spectral estimate using cPCA

          mtcpcaPLV_*- Multitapered PLV using cPCA

          f - frequency vector

     where * in the above is the type of noise floor
    """

    out = {}

    logger.info('Running Multitaper Complex PCA based ' +
                'plv and power estimation.')
    x = x.squeeze()
    if len(x.shape) == 3:
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    # Calculate the tapers

    nfft, f, fInd = _get_freq_stuff(x, params, timedim)
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]

    w, conc = dpss_windows(x.shape[timedim], TW, ntaps)

    plv = np.zeros((ntaps, len(f)))
    cspec = np.zeros((ntaps, len(f)))

    phaseTypes = list(['normalPhase', 'noiseFloorViaPhaseFlip'])

    for thisType in phaseTypes:
        if thisType == 'noiseFloorViaPhaseFlip':
            # flip the phase of every other trial
            phaseFlipper = np.ones(x.shape)

            # important change: when noise floor is computed, it will
            # only select trials that were originally labeled as even-numbered
            # this way, bootstrapped noise floors are where they would be
            # expected to be rather than artificially low
            if bootstrapMode and 'bootstrapTrialsSelected' in params:
                flipTheseTrials = np.where(
                    (params['bootstrapTrialsSelected'] % 2) == 0)
            else:
                flipTheseTrials = np.arange(0, x.shape[1], 2)

            phaseFlipper[:, flipTheseTrials, :] = -1.0

        elif thisType == 'normalPhase':
            phaseFlipper = 1.0

        useData = x * phaseFlipper

        for k, tap in enumerate(w):
            logger.info(thisType + 'Doing Taper #%d', k)

            xw = sci.fft((tap * useData), n=nfft, axis=timedim)

            # no point keeping everything if fpass was already set
            xw = xw[:, :, fInd]

            C = xw.mean(axis=trialdim).squeeze()

            if params['itc']:
                plvC = (xw.mean(axis=trialdim) /
                        (abs(xw).mean(axis=trialdim))).squeeze()
            else:
                plvC = (xw / abs(xw)).mean(axis=trialdim).squeeze()

            for fi in np.arange(0, len(f)):
                powerCsd = np.outer(C[:, fi], C[:, fi].conj())
                powerEigenvals = linalg.eigh(powerCsd, eigvals_only=True)
                cspec[k, fi] = powerEigenvals[-1] / nchans

                plvCsd = np.outer(plvC[:, fi], plvC[:, fi].conj())
                plvEigenvals = linalg.eigh(plvCsd, eigvals_only=True)
                plv[k, fi] = plvEigenvals[-1] / nchans

        # Avage over tapers and squeeze to pretty shapes
        mtcpcaSpectrum = (cspec.mean(axis=0)).squeeze()
        mtcpcaPhaseLockingValue = (plv.mean(axis=0)).squeeze()

        if mtcpcaSpectrum.shape != mtcpcaPhaseLockingValue.shape:
            logger.error('internal error: shape mismatch between PLV ' +
                         ' and magnitude result arrays')

        out['mtcpcaSpectrum_' + thisType] = mtcpcaSpectrum
        out['mtcpcaPLV_' + thisType] = mtcpcaPhaseLockingValue

    if bootstrapMode:
        out['f'] = f
        return out
    else:
        S = {}
        S['spectrum'] = ['mtcpcaSpectrum_normalPhase']
        S['plv'] = ['mtcpcaPLV_normalPhase']

        N = {}
        N['spectrum'] = out['mtcpcaSpectrum_noiseFloorViaPhaseFlip']
        N['plv'] = out['mtcpcaPLV_noiseFloorViaPhaseFlip']

        return (S, N, f)


@verbose_decorator
def _get_freq_stuff(x, params, timeDim=2, verbose=None):
    '''
    internal function, not really meant to be called/viewed by the end user
    (unless end user is curious).

    computes nfft based on x.shape.
    '''
    badNfft = False
    if 'nfft' in params:
        if params['nfft'] < x.shape[timeDim]:
            badNfft = True
            logger.warn(
                'nfft should be >= than number of time points. Reverting' +
                'to default setting of nfft = 2**ceil(log2(nTimePts))\n')

    if 'nfft' not in params or badNfft:
        nfft = int(2.0 ** ceil(sci.log2(x.shape[timeDim])))
    else:
        nfft = int(params['nfft'])

    f = (np.arange(0.0, nfft, 1.0) * params['Fs'] / nfft)
    fInd = ((f >= params['fpass'][0]) & (f <= params['fpass'][1]))

    f = f[fInd]

    return (nfft, f, fInd)


@verbose_decorator
def generate_parameters(verbose=True, **kwArgs):

    """
    Generates some default parameter values using keyword arguments!

    See documentation for each individual function to see which keywords are
    required. Samplerate (Fs= ) is always required.

    Without keyword arguments, the following parameter structure is generated:

    params['tapers'] = [2, 3]
    params['Npairs'] = 0
    params['itc'] = False
    params['threads'] = 4
    params['nDraws'] = 100
    params['nPerDraw'] = 200
    params['returnIndividualBootstrapResults'] = False
    params['debugMode'] = False

    Change any key value by using it as an keyword argument; e.g.,

    generate_params(Fs = 16384, threads = 8)

    would result in the parameter values associated with Fs and threads only,
    without changing the other default values.

    Returns
    ---------
    Dictionary of parameters.

    """

    params = {}
    params['tapers'] = [2, 3]
    params['Npairs'] = 0
    params['itc'] = False
    params['threads'] = 4
    params['nDraws'] = 100
    params['nPerDraw'] = 200
    params['returnIndividualBootstrapResults'] = False
    params['debugMode'] = False

    userKeys = kwArgs.keys()

    for kw in userKeys:
        if kw.lower() == 'fs' or kw.lower() == 'samplerate':
            params['Fs'] = int(kwArgs[kw])

        elif kw.lower() == 'nfft':
            params['nfft'] = int(kwArgs[kw])

        elif kw.lower() == 'tapers':
            params['tapers'] = list(kwArgs[kw])
            params['tapers'][1] = int(params['tapers'][1])

        elif kw.lower() == 'fpass':
            params['fpass'] = list(kwArgs[kw])

        elif kw.lower() == 'npairs':
            params['Npairs'] = int(kwArgs[kw])

        elif kw.lower() == 'itc':
            params['itc'] = bool(kwArgs[kw])

        elif kw.lower() == 'threads':
            params['threads'] = int(kwArgs[kw])

        elif kw.lower() == 'ndraws':
            params['nDraws'] = int(kwArgs[kw])

        elif kw.lower() == 'nperdraw':
            params['nPerDraw'] = int(kwArgs[kw])

        elif kw.lower() == 'debugmode':
            params['debugMode'] = bool(kwArgs[kw])

        elif kw.lower() == 'returnindividualbootstrapresults':
            params['returnIndividualBootstrapResults'] = bool(
                kwArgs[kw])
        else:
            params[kw] = kwArgs[kw]
            logger.info((kw + ' = {}').format(kwArgs[kw]))

    _validate_parameters(params)

    logger.info('Current parameters:')
    logger.info('sampleRate (Fs) = {} Hz'.format(params['Fs']))
    if 'nfft' in params:
        logger.info('nfft = {}'.format(params['nfft']))
    else:
        logger.info('nfft = 2**ceil(log2(data.shape[timeDimension]))')

    logger.info('Number of tapers = {} '.format(params['tapers'][1]))
    logger.info('Taper TW = {} '.format(params['tapers'][0]))
    logger.info('fpass = [{}, {}]'.format(params['fpass'][0],
                                          params['fpass'][1]))
    logger.info('itc = {}'.format(params['itc']))
    logger.info('NPairs = {}'.format(params['Npairs']))
    logger.info('nPerDraw = {}'.format(params['nPerDraw']))
    logger.info('nDraws = {}'.format(params['nDraws']))
    logger.info('threads = {}'.format(params['threads']))
    logger.info('debugMode = {}'.format(params['debugMode']))
    logger.info('returnIndividualBootstrapResults = {}'.format(
        params['returnIndividualBootstrapResults']))

    print '\n'

    return params


@verbose_decorator
def _validate_parameters(params, verbose=True):
    '''
    internal function, not really meant to be called/viewed by the end user
    (unless end user is curious).

    validates parameters
    '''

    if 'Fs' not in params:
        logger.error('params[''Fs''] must be specified')

    if 'nfft' not in params:
        logger.warn('params[''nfft''] defaulting to ' +
                    '2**ceil(log2(data.shape[timeDimension]))')

    # check/fix taper input
    if len(params['tapers']) != 2:
        logger.error('params[''tapers''] must be a list/tuple of '
                     'form [TW,taps]')
    if params['tapers'][0] <= 0:
        logger.error('params[''tapers''][0] (TW) must be positive')

    if params['tapers'][1] <= 0:
        logger.error('params[''tapers''][1] (ntaps) must be a ' +
                     'positive integer')

    # check/fix fpass
    if 'fpass' in params:
        if 2 != len(params['fpass']):
            logger.error('fpass must have two values')

        if params['fpass'][0] < 0:
            logger.error('params[''fpass[0]''] should be >= 0')

        if params['fpass'][1] < 0:
            logger.error('params[''fpass[1]''] should be >= 0')

        if params['fpass'][0] > params['Fs'] / 2.0:
            logger.error('params[''fpass''][0] should be <= ' +
                         'params[''Fs'']/2')

        if params['fpass'][1] > params['Fs'] / 2.0:
            logger.error('params[''fpass''][1] should be <= ' +
                         'params[''Fs'']/2')

        if params['fpass'][0] >= params['fpass'][1]:
            logger.error('params[''fpass''][0] should be < ' +
                         'params[''fpass''][1]')
    else:
        params['fpass'] = [0.0, params['Fs'] / 2.0]
        logger.warn('params[''fpass''] defaulting to ' +
                    '[0, (params[''Fs'']/2.0)]')

    return params
