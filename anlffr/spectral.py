# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:22:58 2013

@author: hari 
"""
import nitime.algorithms as alg
import numpy as np
from math import ceil
import scipy as sci
from scipy import linalg

def mtplv(x,params):
    """Multitaper Phase-Locking Value
    
    Parameters
    ----------
    x - Input data numpy array (channel x trial x time) or (trials x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate
      params['tapers'] - [TW, Number of tapers]
      params['fpass'] - Freqency range of interest, e.g. [5, 1000]
      params['pad'] - 1 or 0, to pad to the next power of 2 or not
      params['itc'] - 1 for ITC, 0 for PLV
      
    Returns
    -------
    Tuple (plvtap, f):
        plvtap - Multitapered phase-locking estimate (channel x frequency)
        f - Frequency vector matching plvtap
    """
    
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        nchans = x.shape[0]
        print 'The data is of format (channels x trials x time)'
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        nchans = 1
        print 'The data is of format (trials x time) i.e. single channel'
    else:
        print 'Sorry! The data should be a 2 or 3 dimensional array!'
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    plvtap = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        print 'Doing Taper #',k
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

def mtspec(x,params):
    """Multitaper Spectrum and SNR estimate
    
    Parameters
    ----------
    x - Input data numpy array (channel x trial x time) or (trials x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate
      params['tapers'] - [TW, Number of tapers]
      params['fpass'] - Freqency range of interest, e.g. [5, 1000]
      params['pad'] - 1 or 0, to pad to the next power of 2 or not
      
    Returns
    -------
    Tuple (S, N ,f):
        S - Multitapered spectrum (channel x frequency)
        N - Noise floor estimate
        f - Frequency vector matching S and N
    """
    
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        print 'The data is of format (channels x trials x time)'
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        print 'The data is of format (trials x time) i.e. single channel'
    else:
        print 'Sorry! The data should be a 2 or 3 dimensional array!'
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the results
    Fs = params['Fs']
    nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    S = np.zeros((ntaps,nchans,nfft))
    N = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        print 'Doing Taper #',k
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        # randph = sci.rand(nchans,ntrials,nfft)*2*sci.pi
        randsign = np.ones((nchans,ntrials,nfft))
        randsign[:,np.arange(0,ntrials,2),:] = -1
        S[k,:,:] = abs(xw.mean(axis = trialdim))
        # N[k,:,:] = abs((xw*sci.exp(1j*randph)).mean(axis = trialdim))
        N[k,:,:] = abs((xw*randsign).mean(axis = trialdim))
            
    # Average over tapers and squeeze to pretty shapes        
    S = S.mean(axis = 0).squeeze() 
    N = N.mean(axis = 0).squeeze()
    ind = (f > params['fpass'][0]) & (f < params['fpass'][1])
    S = S[:,ind]
    N = N[:,ind]
    f = f[ind]
    return (S,N,f)
      
def mtcpca(x,params):
    """Multitaper complex PCA and PLV
    
    Parameters
    ----------
    x - Input data numpy array (channel x trial x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate
      params['tapers'] - [TW, Number of tapers]
      params['fpass'] - Freqency range of interest, e.g. [5, 1000]
      params['pad'] - 1 or 0, to pad to the next power of 2 or not
      params['itc'] - 1 for ITC, 0 for PLV
      
    Returns
    -------
    Tuple (plv, f):
        plv - Multitapered PLV estimate using cPCA
        f - Frequency vector matching plv
    """
    
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        print 'The data is of format (channels x trials x time)'
        print nchans, 'Channels,', ntrials, 'Trials'
    else:
        print 'Sorry! The data should be a 3 dimensional array!'
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    plv = np.zeros((ntaps,nfft))
    
    for k,tap in enumerate(w):
        print 'Doing Taper #', k
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
    
def bootfunc(x,nPerDraw,nDraws, params, func = 'cpca'):
    """Run spectral functions with bootstrapping over trials
    
    Parameters
    ----------
    x - Input data (channel x trials x time) or (trials x time)
    nPerDraw - Number of trials for each draw
    nDraws - Number of draws
    params - Dictionary of parameters to use when calling chosen function
    func - 'cpca' or 'plv' or 'itc' or 'spec' or 'ppc', i.e. which to call?
    
    Returns
    -------
    (plv, vplv, f) for everything except when func == 'spec'
    (S, N, vS, vN, f) when func == 'spec'
    A 'v' prefix denotes variance estimate
    
    See help for mtcpca(), mtplv() and mtspec() for more details.
    
    Notes
    -----
    
    This is not a particularly parallelized piece of code and hence slow.
    It is provided just so the functionality is there.
    
    """
    
    if(len(x.shape) == 3):
        trialdim = 1
        ntrials = x.shape[trialdim]
        print 'The data is of format (channels x trials x time)'
    elif(len(x.shape) == 2):
        trialdim = 0
        ntrials = x.shape[trialdim]
        print 'The data is of format (trials x time) i.e. single channel'
    else:
        print 'Sorry! The data should be a 2 or 3 dimensional array!'
        
    if(func == 'spec'):
        S = 0
        N = 0
        vS = 0
        vN = 0
    else:
        plv = 0
        vplv = 0

    for drawnum in np.arange(0,nDraws):
        inds = np.random.randint(0,ntrials,nPerDraw)
        
        print 'Doing Draw #',drawnum+1, '/', nDraws
        
        if(trialdim == 1):
            xdraw = x[:,inds,:]
        elif(trialdim == 0):
            xdraw = x[inds,:]
        else:
            print 'Data not in the right formmat!'
               
        if(func == 'spec'):
            (tempS,tempN,f) = mtspec(xdraw,params)
            S = S + tempS
            N = N + tempN
            vS = vS + tempS**2
            vN = vN + tempN**2
        elif(func == 'cpca'):
            (tempplv,f)  = mtcpca(xdraw,params)
            plv = plv + tempplv
            vplv = vplv + tempplv**2
        elif(func == 'itc'):
            params['itc'] = 1
            (tempplv,f) = mtplv(xdraw,params)
            plv = plv + tempplv
            vplv = vplv + tempplv**2
        elif(func == 'plv'):
            params['plv'] = 0
            (tempplv,f) = mtplv(xdraw,params)
            plv = plv + tempplv
            vplv = vplv + tempplv**2
        elif(func == 'ppc'):
            (tempplv,f) = mtppc(xdraw,params)
            plv = plv + tempplv
            vplv = vplv + tempplv**2
        else:
            print 'Unknown func argument!'
            return
            
        
    if(func == 'spec'):
        vS = (vS - (S**2)/nDraws)/(nDraws - 1)
        vN = (vN - (N**2)/nDraws)/(nDraws - 1)
        S = S/nDraws
        N = N/nDraws
        return (S,N,vS,vN,f)
    else:
        vplv = (vplv - (plv**2)/nDraws)/(nDraws - 1)
        plv = plv/nDraws
        return (plv,vplv,f)
        
def indivboot(x,nPerDraw,nDraws, params, func = 'cpca'):
    """Run spectral functions with bootstrapping over trials 
    This also returns individual draw results, unlike bootfunc()
    
    Parameters
    ----------
    x - Input data (channel x trials x time) or (trials x time)
    nPerDraw - Number of trials for each draw
    nDraws - Number of draws
    params - Dictionary of parameters to use when calling chosen function
    func - 'cpca' or 'plv' or 'itc' or 'spec', i.e. which to call?
    
    Returns
    -------
    (plv, f) for everything except when func == 'spec'
    (S, N, f) when func == 'spec'
    plv, S and N arrays will have an extra dimension spanning the draws.
    
    See help for mtcpca(), mtplv() and mtspec() for more details.
    
    Notes
    -----
    
    This is not a particularly parallelized piece of code and hence slow.
    It is provided just so the functionality is there. Mostly untested.
    
    """
    
    if(len(x.shape) == 3):
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        print 'The data is of format (channels x trials x time)'
    elif(len(x.shape) == 2):
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        print 'The data is of format (trials x time) i.e. single channel'
    else:
        print 'Sorry! The data should be a 2 or 3 dimensional array!'
        
    
    # Running 1 draw to get the right sizes
    if(func == 'spec'):
        (S,N,f) = mtspec(x,params)       
        S = np.zeros(S.shape + (nDraws,))
        N = np.zeros(N.shape + (nDraws,))
        
    elif((func == 'plv') or (func == 'itc')):
        (plv,f) = mtplv(x,params)
        plv =  np.zeros(plv.shape + (nDraws,))
        
    elif(func == 'cpca'):
        (plv,f) = mtcpca(x, params)
        plv =  np.zeros(plv.shape + (nDraws,))

    for drawnum in np.arange(0,nDraws):
        inds = np.random.randint(0,ntrials,nPerDraw)
        
        print 'Doing Draw #',drawnum+1, '/', nDraws
        
        if(nchans > 1):
            xdraw = x[:,inds,:]
        elif(nchans == 0):
            xdraw = x[inds,:]
        else:
            print 'Data not in the right formmat!'
               
        if(func == 'spec'):
            (S[:,:,drawnum],N[:,:,drawnum],f) = mtspec(xdraw,params)
           
        elif(func == 'cpca'):
            (plv[:,drawnum],f)  = mtcpca(xdraw,params)
           
        elif(func == 'itc'):
            params['itc'] = 1
            if(nchans > 1):
                (plv[:,:,drawnum],f) = mtplv(xdraw,params)
            else:
                (plv[:,drawnum],f) = mtplv(x,params)
            
        elif(func == 'plv'):
            params['plv'] = 0
            if(nchans > 1):
                (plv[:,:,drawnum],f) = mtplv(xdraw,params)
            else:
                (plv[:,drawnum],f) = mtplv(x,params)
            
        else:
            print 'Unknown func argument!'
            return
            
        
    if(func == 'spec'):
        
        return (S,N,f)
    else:
        return (plv,f)                
            
def mtppc(x,params):
    """Multitaper Pairwise Phase Consisttency
    
    Parameters
    ----------
    x - Input data numpy array (channel x trial x time) or (trials x time)
    params - Dictionary of parameter settings
      params['Fs'] - sampling rate
      params['tapers'] - [TW, Number of tapers]
      params['fpass'] - Freqency range of interest, e.g. [5, 1000]
      params['pad'] - 1 or 0, to pad to the next power of 2 or not
      params['ppcpairs'] - Number of pairs for PPC analysis
      params['itc'] - If True, normalize after mean like ITC instead of PLV
      
    Returns
    -------
    Tuple (ppc, f):
        ppc - Multitapered phase-locking estimate (channel x frequency)
        f - Frequency vector matching ppc
    """
    
    if(len(x.shape) == 3):
        timedim = 2
        trialdim = 1
        ntrials = x.shape[trialdim]
        nchans = x.shape[0]
        print 'The data is of format (channels x trials x time)'
    elif(len(x.shape) == 2):
        timedim = 1
        trialdim = 0
        ntrials = x.shape[trialdim]
        nchans = 1
        print 'The data is of format (trials x time) i.e. single channel'
    else:
        print 'Sorry! The data should be a 2 or 3 dimensional array!'
        
    # Calculate the tapers
    ntaps = params['tapers'][1]
    TW = params['tapers'][0]
    w,conc = alg.dpss_windows(x.shape[timedim],TW,ntaps)
    
    # Make space for the PLV result
    Fs = params['Fs']
    nfft = int(2**ceil(sci.log2(x.shape[timedim])))
    f = np.arange(0.0,nfft,1.0)*Fs/nfft
    ppc = np.zeros((ntaps,nchans,nfft))
    
    
    for k,tap in enumerate(w):
        print 'Doing Taper #',k
        xw = sci.fft(tap*x,n = nfft, axis = timedim)
        
        npairs = params['ppcpairs']
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
        
        