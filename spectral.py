# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:22:58 2013

@author: hari 
"""
import nitime.algorithms as alg
import numpy as np
from math import ceil
import scipy as sci

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
        chandim = 0
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
    f = np.arange(0,nfft)*Fs/nfft
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


      
    