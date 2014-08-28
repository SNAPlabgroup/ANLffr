# -*- coding: utf-8 -*-
"""
Spectral analysis functions for FFR data

@author: Hari Bharadwaj
"""
import nitime.algorithms as alg
import numpy as np
from math import ceil
import scipy as sci
from scipy import linalg
from .utils import logger
# stops warnings about scope redefinition
from .utils import verbose as verbose_decorator


@verbose_decorator
def mtplv(x, params, verbose=None):
    """Multitaper Phase-Locking Value

    Parameters
    ----------
    x - NumPy Array
        Input Data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'mtplv' - Multitapered phase-locking estimate (channel x frequency)

    """

    logger.info('Running Multitaper PLV Estimation')
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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

    plvtap = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        if(params['itc'] == 0):
            plvtap[k, :, :] = abs((xw/abs(xw)).mean(axis=trialdim))**2
        else:
            plvtap[k, :, :] = ((abs(xw.mean(axis=trialdim))**2) /
                               ((abs(xw) ** 2).mean(axis=trialdim)))

    plvtap = plvtap.mean(axis=0).squeeze()
    plvtap = plvtap[:, params['fInd']]

    out = {}
    out['mtplv'] = plvtap

    return out


@verbose_decorator
def mtspec(x, params, verbose=None):
    """Multitaper Spectrum and SNR estimate

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['noisefloortype'] - (optional) 1: random phase,
      0 (default): flip-phase

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'mtspec' - Multitapered spectrum (channel x frequency)

      'mtspec_noise' - Noise floor estimate

    """

    logger.info('Running Multitaper Spectrum and Noise-floor Estimation')
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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the results
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

    S = np.zeros((ntaps, nchans, nfft))
    N = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)

        S[k, :, :] = abs(xw.mean(axis=trialdim))

        if ('noisefloortype' in params) and (params['noisefloortype'] == 1):
            randph = sci.rand(nchans, ntrials, nfft) * 2 * sci.pi
            N[k, :, :] = abs((xw*sci.exp(1j*randph)).mean(axis=trialdim))
        else:
            randsign = np.ones((nchans, ntrials, nfft))
            randsign[:, np.arange(0, ntrials, 2), :] = -1
            N[k, :, :] = abs((xw*(randsign.squeeze())).mean(axis=trialdim))

    # Average over tapers and squeeze to pretty shapes
    S = S.mean(axis=0).squeeze()
    N = N.mean(axis=0).squeeze()
    S = S[:, params['fInd']]
    N = N[:, params['fInd']]

    out = {}
    out['mtspec'] = S
    out['mtspec_noise'] = N

    return out


@verbose_decorator
def mtphase(x, params, verbose=None):
    """Multitaper phase estimation

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'Ph' - Multitapered phase spectrum (channel x frequency)

    """

    logger.info('Running Multitaper Spectrum and Noise-floor Estimation')
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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the results
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater '
                         'than number of time points.')

    Ph = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        Ph[k, :, :] = np.angle(xw.mean(axis=trialdim))

    # Average over tapers and squeeze to pretty shapes
    Ph = Ph[:, :, params['fInd']].mean(axis=0).squeeze()

    out = {}
    out['mtphase'] = Ph

    return out


@verbose_decorator
def mtcpca(x, params, verbose=None):
    """Multitaper complex PCA and PLV

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'mtcplv' - Multitapered PLV estimate using cPCA

    """

    logger.info('Running Multitaper Complex PCA based PLV Estimation')
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

    plv = np.zeros((ntaps, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        C = (xw.mean(axis=trialdim) / (abs(xw).mean(axis=trialdim))).squeeze()
        for fi in np.arange(0, nfft):
            Csd = np.outer(C[:, fi], C[:, fi].conj())
            vals = linalg.eigh(Csd, eigvals_only=True)
            plv[k, fi] = vals[-1] / nchans

    # Average over tapers and squeeze to pretty shapes
    plv = (plv.mean(axis=0)).squeeze()
    plv = plv[params['fInd']]

    out = {}
    out['mtcplv'] = plv

    return out


@verbose_decorator
def mtcspec(x, params, verbose=None):
    """Multitaper complex PCA and power spectral estimate

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['itc'] - 1 for ITC, 0 for PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Tuple (cspec, f):
      'mtcspec' - Multitapered PLV estimate using cPCA

    """

    logger.info('Running Multitaper Complex PCA based power estimation!')
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

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
    cspec = cspec[params['fInd']]

    out = {}
    out['mtcspec'] = cspec

    return out


@verbose_decorator
def mtcpca_timeDomain(x, params, verbose=None):
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

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'y_cpc' - Multitapered cPCA estimate of time-domain waveform

      'y_pc' - Regular time-domain PCA

    """

    logger.info('Running Multitaper Complex PCA to extract time waveform!')
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        logger.info('The data is of format %d channels x %d trials x time',
                    nchans, ntrials)
    else:
        logger.error('Sorry! The data should be a 3 dimensional array!')

    _validate_parameters(params)

    # Calculate the tapers
    w, conc = alg.dpss_windows(x.shape[timedim], 1, 1)
    w = w.squeeze() / w.max()

    # Make space for the CPCA resutls
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

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

    out = {}
    out['y_cpc'] = y_cpc
    out['y_pc'] = y_pc

    return out


@verbose_decorator
def bootfunc(x, nPerDraw, nDraws, params, func='cpca', verbose=None):
    """Run spectral functions with bootstrapping over trials

    DEPRECATION WARNING: recommend use of anlffr.bootstrap module

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
def mtppc(x, params, verbose=None):
    """Multitaper Pairwise Phase Consisttency

    Parameters
    ----------
    x - Numpy array
        Input data (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['Npairs'] - Number of pairs for PPC analysis

      params['itc'] - If True, normalize after mean like ITC instead of PLV

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'mtppc' - Multitapered PPC estimate (channel x frequency)

    """

    logger.info('Running Multitaper Pairwise Phase Consistency Estimate')
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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

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

    ppc = ppc.mean(axis=0).squeeze()
    ppc = ppc[:, params['fInd']]

    out = {}
    out['mtppc'] = ppc

    return out


@verbose_decorator
def mtspecraw(x, params, verbose=None):
    """Multitaper Spectrum (of raw signal)

    Parameters
    ----------
    x - Numpy array
        Input data numpy array (channel x trial x time) or (trials x time)

    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with the following keys:
      'mtspecraw' - Multitapered spectrum (channel x frequency)

    """

    logger.info('Running Multitaper Raw Spectrum Estimation')

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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the results
    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

    Sraw = np.zeros((ntaps, nchans, nfft))

    for k, tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap * x, n=nfft, axis=timedim)
        Sraw[k, :, :] = (abs(xw)**2).mean(axis=trialdim)

    # Average over tapers and squeeze to pretty shapes
    Sraw = Sraw.mean(axis=0).squeeze()
    Sraw = Sraw[:, params['fInd']]

    out = {}
    out['mtspecraw'] = Sraw

    return out


@verbose_decorator
def mtpspec(x, params, verbose=None):
    """Multitaper Pairwise Power Spectral estimate

    Parameters
    ----------
    x - Numpy Array
        Input data numpy array (channel x trial x time) or (trials x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations (default: next
        power of 2 greater than length of time dimension)

      params['Npairs'] - Number of pairs for pairwise analysis
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    Dictionary with following keys:
      'pspec' -  Multitapered Pairwise Power estimate (channel x frequency)

    """
    logger.info('Running Multitaper Pairwise Power Estimate')

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

    _validate_parameters(params)

    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    # Make space for the PLV result

    if 'nfft' not in params:
        nfft = int(2 ** ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error(
                'nfft really should be greater than number of time points.')

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

    pspec = pspec.mean(axis=0).squeeze()
    pspec = pspec[:, params['fInd']]

    out = {}
    out['pspec'] = pspec

    return out


@verbose_decorator
def mtcpca_complete(x, params, verbose=True):
    """Gets power spectra and plv using multitaper with complex PCA applied.

    Parameters
    ----------
    x - NumPy Array
        Input data (channel x trial x time)

    params - dictionary. Must contain the following fields:
      params['Fs'] - sampling rate

      params['tapers'] - [TW, Number of tapers]

      params['fpass'] - Freqency range of interest, e.g. [5, 1000]

      params['nfft'] - length of FFT used for calculations

    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    dictionary with keys:
         mtcpcaSpectrum_* - Multitapered power spectral estimate using cPCA

         mtcpcaPLV_*- Multitapered PLV using cPCA

         mtcpcaSpectrumEigenvalues_* - Eigenvalues from cpca on spectrum (taper
         x frequency)

         mtcpcaPLVEigenvalues_* - Eigenvalues from cpca on PLV (taper x
         frequency)

     where * in the above is one or more of the noise floor types specified
     or 'normalPhase'
    """
    _validate_parameters(params)

    out = {}

    logger.info('Running Multitaper Complex PCA based power estimation!')
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
    nfft = params['nfft']
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w, conc = alg.dpss_windows(x.shape[timedim], TW, ntaps)

    plv = np.zeros((ntaps, nfft))
    cspec = np.zeros((ntaps, nfft))

    phaseTypes = list(params['noiseFloorType'])
    phaseTypes.append('normalPhase')

    for thisType in phaseTypes:
        if thisType == 'phaseFlipHalfTrials':
            # flip half the trials
            phaseShifter = np.ones(x.shape)
            permutedTrialOrder = np.random.permutation(range(x.shape[1]))
            flipTheseTrials = permutedTrialOrder[0:int(x.shape[1] / 2)]
            phaseShifter[:, flipTheseTrials, :] = -1.0
        elif thisType == 'normalPhase':
            phaseShifter = 1.0
        else:
            logger.error('unrecognized selection for trial phase type')

        useData = x * phaseShifter

        for k, tap in enumerate(w):
            logger.info(thisType + 'Doing Taper #%d', k)

            xw = sci.fft((tap * useData), n=nfft, axis=timedim)

            # power spectrum
            C = (xw.mean(axis=trialdim)).squeeze()
            # phase locking value
            plvC = (xw.mean(axis=trialdim) /
                    (abs(xw).mean(axis=trialdim))).squeeze()

            for fi in np.arange(0, nfft):
                powerCsd = np.outer(C[:, fi], C[:, fi].conj())
                powerEigenvals = linalg.eigh(powerCsd, eigvals_only=True)
                cspec[k, fi] = powerEigenvals[-1] / nchans

                plvCsd = np.outer(plvC[:, fi], plvC[:, fi].conj())
                plvEigenvals = linalg.eigh(plvCsd, eigvals_only=True)
                plv[k, fi] = plvEigenvals[-1] / nchans

        # Average over tapers and squeeze to pretty shapes
        cpcaSpectrum = (cspec.mean(axis=0)).squeeze()
        cpcaPhaseLockingValue = (plv.mean(axis=0)).squeeze()

        if cpcaSpectrum.shape != cpcaPhaseLockingValue.shape:
            logger.error(
                'shape mismatch between PLV and magnitude result arrays')

        out['mtcpcaSpectrum_' + thisType] = cpcaSpectrum[params['fInd']]
        out['mtcpcaPLV_' + thisType] = cpcaPhaseLockingValue[params['fInd']]
        out['mtcpcaSpectrumEigenvalues_' + thisType] = cspec[:, params['fInd']]
        out['mtcpcaPLVEigenvalues_' + thisType] = plv[:, params['fInd']]

    return out


@verbose_decorator
def generate_parameters(
        sampleRate=4096,
        nfft=4096,
        tapers=None,
        fpass=None,
        nPairs=0,
        itc=False,
        threads=2,
        nDraws=100,
        nPerDraw=200,
        returnIndividualBootstrapResults=False,
        noiseFloorType=None,
        debugMode=False,
        verbose=True):
    """
    Generates some default parameter values.

    Inputs
    ----------
    sampleRate - sample rate (default: 4096)

    nfft - fft length (default: 4096)

    tapers - list of [TW, number of tapers] (default: [2,3])

    fpass - list of [f_low, f_high] (default: [70.0, 1000.0]

    npairs - number of trial pairs for pairwise analysis (default: 0)

    itc - whether to compute ITC (default: False)

    threads - number of threads to spawn for multiprocess
    bootstrap (default: 1)

    nDraw - number total draws of data for multiprocess bootstrap
    (default: 100)

    nPerDraw - number of trials to use per draw of data for multiprocess
    bootstrap (default: 200)

    returnIndividualBootstrapResults - whether to return the individual draw
    results when running bootstrapped calculations (default: False)

    noiseFloorType - a list or string type of noise floor assumption(s) to use.
        valid choices are:
        'phaseFlipHalfTrials'

        can be equivalently specified using a list of strings or a string
        separated with whitespace. e.g.:
        ['phaseFlipHalfTrials']
        or
        noiseFloorType = 'phaseFlipHalfTrials randomPhaseAcrossTrials'

    debugMode - flag to set debugging on (to fix random seeds)

    Returns
    ---------
    Dictionary with reasonable example/default values for params.

    Note: this will create a frequency axis based on the inputs to
    nfft and Fs, as well as based on fpass. There will also be a key
    containing the logical indexes needed so that the functions in
    this module truncate their output to match the frequency vector.

    I should really switch this to some kind of kwarg function...
    """

    # define this here because apparently a list is a "dangerous"
    # default argument as per PyLint
    if tapers is None:
        tapers = [2, 3]

    if fpass is None:
        fpass = [70.0, 1000.0]

    # removes whitespace characters and converts to list
    if isinstance(noiseFloorType, str):
        noiseFloorType = noiseFloorType.split()

    if noiseFloorType is None:
        noiseFloorType = ['phaseFlipHalfTrials']

    params = {}

    params['Fs'] = int(sampleRate)
    params['nfft'] = int(nfft)
    params['tapers'] = list(tapers)
    params['fpass'] = list(fpass)
    params['nPairs'] = int(nPairs)
    params['itc'] = bool(itc)
    params['threads'] = int(threads)
    params['nDraws'] = int(nDraws)
    params['nPerDraw'] = int(nPerDraw)
    params['noiseFloorType'] = noiseFloorType
    params['debugMode'] = debugMode
    params['returnIndividualBootstrapResults'] = (
        returnIndividualBootstrapResults)

    logger.info('\n sampleRate (Fs) = {} Hz'.format(params['Fs']))
    logger.info('nfft = {}'.format(params['nfft']))
    logger.info('Taper TW = {} '.format(params['tapers'][0]))
    logger.info('Number of tapers = {} '.format(params['tapers'][1]))
    logger.info('fpass = [{}, {}] '.format(params['fpass'][0],
                                           params['fpass'][1]))
    logger.info('nPairs = {}'.format(params['nPairs']))
    logger.info('itc = {}'.format(params['itc']))
    logger.info('threads = {}'.format(params['threads']))
    logger.info('nDraws = {}'.format(params['nDraws']))
    logger.info('nPerDraw = {}'.format(params['nPerDraw']))
    logger.info('noiseFloorType = {}'.format(params['noiseFloorType']))
    logger.info('debugMode = {}'.format(params['debugMode']))
    logger.info('returnIndividualBootstrapResults = {}'.format(
        params['returnIndividualBootstrapResults']))

    _validate_parameters(params)

    return params


@verbose_decorator
def _validate_parameters(params, verbose=True):
    '''
    internal function to validate parameters
    '''

    notValidated = (('function_params_validated' not in params) or
                    (params['function_params_validated'] is False))

    if notValidated:

        if 'Fs' not in params:
            logger.error('params[''Fs''] must be specified')
        if 'nfft' not in params:
            logger.error('params[''nfft''] must be specified')
        if 'tapers' not in params:
            logger.error('params[''tapers''] must be specified')

        # check/fix taper input
        if len(params['tapers']) != 2:
            logger.error('params[''tapers''] must be a list/tuple of '
                         'form [TW,taps]')
        if params['tapers'][0] <= 0:
            logger.error('params[''tapers''][0] (TW) must be positive')

        if int(params['tapers'][1]) <= 0:
            logger.error('params[''tapers''][1] (ntaps) must be a ' +
                         'positive integer')

        if 'noiseFloorType' in params:
            if not isinstance(params['noiseFloorType'], list):
                logger.error('noiseFloorType should be a list')

            validNoiseFloors = ['phaseFlipHalfTrials']

            for k in params['noiseFloorType']:
                if k not in validNoiseFloors:
                    logger.error('unrecognized input for noiseFloorType')

        if 'fpass' in params:
            if 2 != len(params['fpass']):
                logger.error('fpass must have two values')

            if params['fpass'][0] < 0:
                logger.error('params[''fpass[0]''] should be >= 0')

            if params['Fs'] / 2.0 < params['fpass'][1]:
                logger.error('params[''fpass''][1] should be <= ' +
                             'params[''Fs'']/2')

            if params['fpass'][0] >= params['fpass'][1]:
                logger.error('params[''fpass''][0] should be < ' +
                             'params[''fpass''][1]')
        else:
            logger.info('Using default params[''fpass''] of ' +
                        '[0, params[''Fs'']/2]')
            params['fpass'] = [0.0, params['Fs'] / 2.0]

        # take care of frequency stuff
        if 'f' not in params:
            if 'fInd' in params:
                logger.error('f is not specified, but fInd is...' +
                             'something is wrong')

            params['f'] = (np.arange(0.0, params['nfft'], 1.0) *
                           params['Fs'] / params['nfft'])

            if 'fpass' in params:
                params['fInd'] = ((params['f'] >= params['fpass'][0]) &
                                  (params['f'] <= params['fpass'][1]))
            else:
                params['fInd'] = range(params['f'].shape[0])

            params['f'] = params['f'][params['fInd']]

        params['function_params_validated'] = True

    return params
