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
from .utils import logger, verbose
    
@verbose
def mtplv(x, params, verbose = None):
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
    (plvtap, f): Tuple
        plvtap - Multitapered phase-locking estimate (channel x frequency)
        
        f - Frequency vector matching plvtap
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')
            
    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    plvtap = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        
        if(params['itc'] == 0):
            plvtap[k,:,:] = abs((xw/abs(xw)).mean(axis = trialdim))**2
        else:
            plvtap[k,:,:] = ((abs(xw.mean(axis = trialdim))**2)/
            ((abs(xw)**2).mean(axis = trialdim)))
            
    plvtap = plvtap.mean(axis = 0).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    plvtap = plvtap[:,ind]
    f = f[ind]
    return (plvtap,f)

@verbose
def mtspec(x,params, verbose = None):
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
    (S, N ,f): Tuple
        S - Multitapered spectrum (channel x frequency)
        
        N - Noise floor estimate
        
        f - Frequency vector matching S and N
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the results
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    S = np.zeros((ntaps,nchans,nfft))
    N = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d',k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        
        S[k,:,:] = abs(xw.mean(axis = trialdim))
            
        if ('noisefloortype' in params) and (params['noisefloortype'] == 1):
            randph = sci.rand(nchans,ntrials,nfft)*2*sci.pi
            N[k,:,:] = abs((xw*sci.exp(1j*randph)).mean(axis = trialdim))
        else:
            randsign = np.ones((nchans,ntrials,nfft))
            randsign[:,np.arange(0,ntrials,2),:] = -1
            N[k,:,:] = abs((xw*(randsign.squeeze())).mean(axis = trialdim))
            
    # Average over tapers and squeeze to pretty shapes        
    S = S.mean(axis = 0).squeeze() 
    N = N.mean(axis = 0).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    S = S[:,ind]
    N = N[:,ind]
    f = f[ind]
    return (S,N,f)

@verbose      
def mtcpca(x,params, verbose = None):
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
    Tuple (plv, f):
        plv - Multitapered PLV estimate using cPCA
        
        f - Frequency vector matching plv
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    plv = np.zeros((ntaps,nfft))
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        C = (xw.mean(axis = trialdim)/(abs(xw).mean(axis = trialdim))).squeeze()
        for fi in np.arange(0,nfft):
            Csd = np.outer(C[:,fi],C[:,fi].conj())
            vals = linalg.eigh(Csd,eigvals_only = True)
            plv[k,fi] = vals[-1]/nchans
                        
            
    # Average over tapers and squeeze to pretty shapes        
    plv = (plv.mean(axis = 0)).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    plv = plv[ind]
    f = f[ind]
    return (plv,f)

@verbose      
def mtcspec(x,params, verbose = None):
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
        cspec - Multitapered PLV estimate using cPCA
        
        f - Frequency vector matching plv
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    cspec = np.zeros((ntaps,nfft))
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d', k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        C = (xw.mean(axis = trialdim)).squeeze()
        for fi in np.arange(0,nfft):
            Csd = np.outer(C[:,fi],C[:,fi].conj())
            vals = linalg.eigh(Csd,eigvals_only = True)
            cspec[k,fi] = vals[-1]/nchans
                        
            
    # Average over tapers and squeeze to pretty shapes        
    cspec = (cspec.mean(axis = 0)).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    cspec = cspec[ind]
    f = f[ind]
    return (cspec,f)
    
    
@verbose    
def bootfunc(x,nPerDraw,nDraws, params, func = 'cpca', verbose = None):
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
    
    logger.info('Running a bootstrapped version of function: %s',func)
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

    for drawnum in np.arange(0,nDraws):
        inds = np.random.randint(0,ntrials,nPerDraw)
        
        logger.debug('Doing Draw #%d / %d',drawnum+1, nDraws)
        
        if(trialdim == 1):
            xdraw = x[:,inds,:]
        elif(trialdim == 0):
            xdraw = x[inds,:]
        else:
            logger.error('Data not in the right formmat!')
               
        if(func == 'spec'):
            (tempS,tempN,f) = mtspec(xdraw,params,verbose=False)
            S = S + tempS
            N = N + tempN
            vS = vS + tempS**2
            vN = vN + tempN**2
        elif(func == 'cpca'):
            (temp_func,f)  = mtcpca(xdraw,params,verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func**2
        elif(func == 'itc'):
            params['itc'] = 1
            (temp_func,f) = mtplv(xdraw,params,verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func**2
        elif(func == 'plv'):
            params['plv'] = 0
            (temp_func,f) = mtplv(xdraw,params,verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func**2
        elif(func == 'ppc'):
            (temp_func,f) = mtppc(xdraw,params,verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func**2
        elif(func == 'pspec'):
            (temp_func,f) = mtpspec(xdraw,params,verbose=False)
            mu_func = mu_func + temp_func
            v_func = v_func + temp_func**2
        else:
            logger.error('Unknown func argument!')
            return
            
        
    if(func == 'spec'):
        vS = (vS - (S**2)/nDraws)/(nDraws - 1)
        vN = (vN - (N**2)/nDraws)/(nDraws - 1)
        S = S/nDraws
        N = N/nDraws
        return (S,N,vS,vN,f)
    else:
        v_func = (v_func - (mu_func**2)/nDraws)/(nDraws - 1)
        mu_func = mu_func/nDraws
        return (mu_func,v_func,f)

@verbose        
def indivboot(x,nPerDraw,nDraws, params, func = 'cpca', verbose = None):
    """Run spectral functions with bootstrapping over trials 
    This also returns individual draw results, unlike bootfunc()
    
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
    
    logger.info('Running a bootstrapped version of function: %s',func)
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
        (S,N,f) = mtspec(x,params,verbose=False)       
        S = np.zeros(S.shape + (nDraws,))
        N = np.zeros(N.shape + (nDraws,))
        
    elif((func == 'plv') or (func == 'itc') or (func == 'ppc')):
        (plv,f) = mtplv(x,params,verbose=False)
        plv =  np.zeros(plv.shape + (nDraws,))
        
    elif(func == 'cpca'):
        (plv,f) = mtcpca(x, params,verbose=False)
        plv =  np.zeros(plv.shape + (nDraws,))

    for drawnum in np.arange(0,nDraws):
        inds = np.random.randint(0,ntrials,nPerDraw)
        
        logger.debug('Doing Draw #%d / %d',drawnum+1, nDraws)
        
        if(nchans > 1):
            xdraw = x[:,inds,:]
        elif(nchans == 1):
            xdraw = x[inds,:]
        else:
            logger.error('Data not in the right formmat!')
               
        if(func == 'spec'):
            if(nchans > 1):
                (S[:,:,drawnum],N[:,:,drawnum],f) = mtspec(xdraw,params,
                                                           verbose=False)
            else:
                (S[:,drawnum],N[:,drawnum],f) = mtspec(xdraw,params,
                                                       verbose=False)
           
        elif(func == 'cpca'):
            (plv[:,drawnum],f)  = mtcpca(xdraw,params,verbose=False)
           
        elif(func == 'itc'):
            params['itc'] = 1
            if(nchans > 1):
                (plv[:,:,drawnum],f) = mtplv(xdraw,params,verbose=False)
            else:
                (plv[:,drawnum],f) = mtplv(x,params,verbose=False)
            
        elif(func == 'plv'):
            params['plv'] = 0
            if(nchans > 1):
                (plv[:,:,drawnum],f) = mtplv(xdraw,params,verbose=False)
            else:
                (plv[:,drawnum],f) = mtplv(x,params,verbose=False)
                
        elif(func == 'ppc'):
            if(nchans > 1):
                (plv[:,:,drawnum],f) = mtppc(xdraw,params,verbose=False)
            else:
                (plv[:,drawnum],f) = mtppc(x,params,verbose=False)
            
        else:
            logger.error('Unknown func argument!')
            return
            
        
    if(func == 'spec'):
        
        return (S,N,f)
    else:
        return (plv,f)                

@verbose            
def mtppc(x,params,verbose=None):
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
    (ppc, f): Tuple
        ppc - Multitapered PPC estimate (channel x frequency)
        
        f - Frequency vector matching ppc
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    ppc = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d',k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        
        npairs = params['Npairs']
        trial_pairs = np.random.randint(0,ntrials,(npairs,2))
        
        if(nchans == 1):
            if(not params['itc']):
                xw_1 = xw[trial_pairs[:,0],:]/abs(xw[trial_pairs[:,0],:])
                xw_2 = xw[trial_pairs[:,1],:]/abs(xw[trial_pairs[:,1],:])
                ppc[k,:,:] = np.real((xw_1*xw_2.conj()).mean(axis = trialdim))
            else:
                xw_1 = xw[trial_pairs[:,0]]
                xw_2 = xw[trial_pairs[:,1]]
                ppc_unnorm = np.real((xw_1*xw_2.conj()).mean(axis = trialdim))
                ppc[k,:,:] = (ppc_unnorm/
                  (abs(xw_1).mean(trialdim)*abs(xw_2).mean(trialdim)))
                
        else:
            if(not params['itc']):
                xw_1 = xw[:,trial_pairs[:,0],:]/abs(xw[:,trial_pairs[:,0],:])
                xw_2 = xw[:,trial_pairs[:,1],:]/abs(xw[:,trial_pairs[:,1],:])
                ppc[k,:,:] = np.real((xw_1*xw_2.conj()).mean(axis = trialdim))
            else:
                xw_1 = xw[:,trial_pairs[:,0],:]
                xw_2 = xw[:,trial_pairs[:,1],:]
                ppc_unnorm = np.real((xw_1*xw_2.conj()).mean(axis = trialdim))
                ppc[k,:,:] = (ppc_unnorm/
                  (abs(xw_1).mean(trialdim)*abs(xw_2).mean(trialdim)))
              
    ppc = ppc.mean(axis = 0).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    ppc = ppc[:,ind]
    f = f[ind]
    return (ppc,f)

@verbose        
def mtspecraw(x,params,verbose = None):
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
    (Sraw, f): Tuple
        Sraw - Multitapered spectrum (channel x frequency)
        
        f - Frequency vector matching Sraw
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
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the results
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    Sraw = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        logger.info('Doing Taper #%d',k)
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        Sraw[k,:,:] = (abs(xw)**2).mean(axis = trialdim)
         
    # Average over tapers and squeeze to pretty shapes        
    Sraw = Sraw.mean(axis = 0).squeeze() 
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    Sraw = Sraw[:,ind]
    f = f[ind]
    return (Sraw,f)

@verbose    
def mtpspec(x,params,verbose = None):
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
    (pspec, f): Tuple
        pspec - Multitapered Pairwise Power estimate (channel x frequency)
        
        f - Frequency vector matching ppc
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
        
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    if 'nfft' not in params:
        nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    else:
        nfft = int(params['nfft'])
        if nfft < x.shape[timedim]:
            logger.error('nfft really should be greater than number of time points.')

    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    pspec = np.zeros((ntaps,nchans,nfft))
    
    for ch in np.arange(0,nchans):
        for k,tap in enumerate(w):
            logger.debug('Running Channel # %d, taper #%d', ch,k)
            xw = sci.fft(tap*x,n = nfft, axis = timedim)
            npairs = params['Npairs']
            trial_pairs = np.random.randint(0,ntrials,(npairs,2))
            
            # For unbiasedness, pairs should be made of independent trials!
            trial_pairs = trial_pairs[np.not_equal(trial_pairs[:,0],
                                                   trial_pairs[:,1])]
            if(nchans == 1):
                xw_1 = xw[trial_pairs[:,0]]
                xw_2 = xw[trial_pairs[:,1]]
                pspec[k,ch,:] = np.real((xw_1*xw_2.conj()).mean(axis = 0))                         
            else:
                xw_1 = xw[ch,trial_pairs[:,0],:]
                xw_2 = xw[ch,trial_pairs[:,1],:]
                pspec[k,ch,:] = np.real((xw_1*xw_2.conj()).mean(axis = 0))
               
              
    pspec = pspec.mean(axis = 0).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    pspec = pspec[:,ind]
    
    f = f[ind]
    return (pspec,f)    
