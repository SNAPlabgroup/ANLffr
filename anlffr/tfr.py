"""A module which implements the time frequency estimation.

Authors : Hari Bharadwaj <hari@nmr.mgh.harvard.edu>

License : BSD 3-clause

Multitaper wavelet method
"""

import warnings
from math import sqrt
import numpy as np
from scipy import linalg
from scipy.fftpack import fftn, ifftn

from .utils import logger, verbose
from .dpss import dpss_windows


def _dpss_wavelet(sfreq, freqs, n_cycles=7, time_bandwidth=4.0,
                  zero_mean=False):
    """Compute Wavelets for the given frequency range

    Parameters
    ----------
    sfreq : float
        Sampling Frequency.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
        Defaults to 7.
    time_bandwidth : float, (optional)
        Time x Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1).
        Default is 4.0, giving 3 good tapers.

    Returns
    -------
    Ws : list of array
        Wavelets time series
    """
    Ws = list()
    if time_bandwidth < 2.0:
        raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
    n_taps = int(np.floor(time_bandwidth - 1))
    n_cycles = np.atleast_1d(n_cycles)

    if n_cycles.size != 1 and n_cycles.size != len(freqs):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")

    for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles) != 1:
                this_n_cycles = n_cycles[k]
            else:
                this_n_cycles = n_cycles[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0., t_win, 1.0 / sfreq)
            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.))

            # Get dpss tapers
            tapers, conc = dpss_windows(t.shape[0], time_bandwidth / 2.,
                                        n_taps)

            Wk = oscillation * tapers[m]
            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= sqrt(0.5) * linalg.norm(Wk.ravel())

            Wm.append(Wk)

        Ws.append(Wm)

    return Ws


def _centered(arr, newsize):
    """Aux Function to center data"""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _cwt_fft(X, Ws, mode="same"):
    """Compute cwt with fft based convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))

    # precompute FFTs of Ws
    fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('Wavelet is too long for such a short signal. '
                             'Reduce the number of cycles.')
        fft_Ws[i] = fftn(W, [fsize])

    for k, x in enumerate(X):
        if mode == "full":
            tfr = np.zeros((n_freqs, fsize), dtype=np.complex128)
        elif mode == "same" or mode == "valid":
            tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)

        fft_x = fftn(x, [fsize])
        for i, W in enumerate(Ws):
            ret = ifftn(fft_x * fft_Ws[i])[:n_times + W.size - 1]
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = _centered(ret, sz)
            else:
                tfr[i, :] = _centered(ret, n_times)
        yield tfr


def _cwt_convolve(X, Ws, mode='same'):
    """Compute time freq decomposition with temporal convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    # Compute convolutions
    for x in X:
        tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)
        for i, W in enumerate(Ws):
            ret = np.convolve(x, W, mode=mode)
            if len(W) > len(x):
                raise ValueError('Wavelet is too long for such a short '
                                 'signal. Reduce the number of cycles.')
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = ret
            else:
                tfr[i] = ret
        yield tfr


def _time_frequency(X, Ws, use_fft, decim):
    """Aux of time_frequency for parallel computing over channels
    """
    n_epochs, n_times = X.shape
    n_times = n_times // decim + bool(n_times % decim)
    n_frequencies = len(Ws)
    psd = np.zeros((n_frequencies, n_times))  # PSD
    plf = np.zeros((n_frequencies, n_times), np.complex)  # phase lock

    mode = 'same'
    if use_fft:
        tfrs = _cwt_fft(X, Ws, mode)
    else:
        tfrs = _cwt_convolve(X, Ws, mode)

    for tfr in tfrs:
        tfr = tfr[:, ::decim]
        tfr_abs = np.abs(tfr)
        psd += tfr_abs ** 2
        plf += tfr / tfr_abs
    psd /= n_epochs
    plf = np.abs(plf) / n_epochs
    return psd, plf


@verbose
def tfr_multitaper(data, sfreq, frequencies, time_bandwidth=4.0,
                   use_fft=True, n_cycles=7, decim=1,
                   zero_mean=True, verbose=None):
    """Compute time induced power and inter-trial coherence

    The time frequency decomposition is done with DPSS wavelets

    Parameters
    ----------
    data : np.ndarray, shape (n_epochs, n_channels, n_times)
        The input data.
    sfreq : float
        sampling Frequency
    frequencies : np.ndarray, shape (n_frequencies,)
        Array of frequencies of interest
    time_bandwidth : float
        Time x (Full) Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1). Default is 4.0 (3 tapers).
    use_fft : bool
        Compute transform with fft based convolutions or temporal
        convolutions. Defaults to True.
    n_cycles : float | np.ndarray shape (n_frequencies,)
        Number of cycles. Fixed number or one per frequency. Defaults to 7.
    decim: int
        Temporal decimation factor. Defaults to 1.
    zero_mean : bool
        Make sure the wavelets are zero mean. Defaults to True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    power : np.ndarray, shape (n_channels, n_frequencies, n_times)
        Induced power. Squared amplitude of time-frequency coefficients.
    itc : np.ndarray, shape (n_channels, n_frequencies, n_times)
        Phase locking value.
    times : np.ndarray, shape (n_times, )
         Time vector for convenience based on n_times, sfreq and decim

    """
    n_epochs, n_channels, n_times = data[:, :, ::decim].shape
    logger.info('Data is %d trials and %d channels', n_epochs, n_channels)
    n_frequencies = len(frequencies)
    logger.info('Multitaper time-frequency analysis for %d frequencies',
                n_frequencies)

    # Precompute wavelets for given frequency range to save time
    Ws = _dpss_wavelet(sfreq, frequencies, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, zero_mean=zero_mean)
    n_taps = len(Ws)
    logger.info('Using %d tapers', n_taps)
    n_times_wavelets = Ws[0][0].shape[0]
    if n_times <= n_times_wavelets:
        warnings.warn("Time windows are as long or longer than the epoch. "
                      "Consider reducing n_cycles.")
    psd = np.zeros((n_channels, n_frequencies, n_times))
    itc = np.zeros((n_channels, n_frequencies, n_times))

    for m in range(n_taps):
        psd_itc = (_time_frequency(data[:, c, :], Ws[m], use_fft, decim)
                   for c in range(n_channels))
        for c, (psd_c, itc_c) in enumerate(psd_itc):
            psd[c, :, :] += psd_c
            itc[c, :, :] += itc_c
    psd /= n_taps
    itc /= n_taps
    times = np.arange(n_times) / np.float(sfreq)
    return psd, itc, times


@verbose
def rescale(data, times, baseline, mode, verbose=None, copy=True):
    """Rescale i.e., baseline correcting data

    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    times : 1D array
        Time instants is seconds.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used. If None, no correction is applied.
    mode : 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent' | 'zlogratio'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
        logratio is the same an mean but in log-scale, zlogratio is the
        same as zscore but data is rendered in log-scale first.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    copy : bool
        Operate on a copy of the data, or in place.

    Returns
    -------
    data_scaled : array
        Array of same shape as data after rescaling.
    """
    if copy:
        data = data.copy()

    valid_modes = ('logratio', 'ratio', 'zscore', 'mean', 'percent',
                   'zlogratio')
    if mode not in valid_modes:
        raise Exception('mode should be any of : %s' % (valid_modes, ))

    if baseline is not None:
        logger.info("Applying baseline correction ... (mode: %s)" % mode)
        bmin, bmax = baseline
        if bmin is None:
            imin = 0
        else:
            imin = int(np.where(times >= bmin)[0][0])
        if bmax is None:
            imax = len(times)
        else:
            imax = int(np.where(times <= bmax)[0][-1]) + 1

        # avoid potential "empty slice" warning
        if data.shape[-1] > 0:
            mean = np.mean(data[..., imin:imax], axis=-1)[..., None]
        else:
            mean = 0  # otherwise we get an ugly nan
        if mode == 'mean':
            data -= mean
        if mode == 'logratio':
            data /= mean
            data = np.log10(data)  # a value of 1 means 10 times bigger
        if mode == 'ratio':
            data /= mean
        elif mode == 'zscore':
            std = np.std(data[..., imin:imax], axis=-1)[..., None]
            data -= mean
            data /= std
        elif mode == 'percent':
            data -= mean
            data /= mean
        elif mode == 'zlogratio':
            data /= mean
            data = np.log10(data)
            std = np.std(data[..., imin:imax], axis=-1)[..., None]
            data /= std

    else:
        logger.info("No baseline correction applied...")

    return data


def plot_tfr(tfr, times, freqs, ch_idx=0, vmin=None, vmax=None,
             x_label='Time (s)', y_label='Frequency (Hz)',
             colorbar=True, cmap='RdBu_r', title=None):
    """ Basic plotting function to show time-freq

    Parameters
    ----------
    tfr : np.ndarray, shape (n_channels, n_frequencies, n_times)
        Time-frequency data from tfr_multitaper (power or itc)
    times: np.ndarray, shape (n_times, )
        Time array corresponding to tfr, also from tfr_multitaper
    freqs : np.ndarray, shape (n_times, )
        Frequency array over which tfr was calculated
    ch_idx : integer, option, default: 0
        Index of channel to plot
    vmin : scalar, optional, default: tfr.min()
        Minimum of colorbar
    vmax : scalra, optional, default: tfr.max()
        Maximum of colorbar
    x_label : string, optional, default: 'Time (s)'
        Label for x-axis (i.e., time axis)
    y_label : string, optional, default: 'Frequency (Hz)'
        Label for y-axis (i.e., frequency axis)
    colorbar : boolean, optional, default: False
        Whether to show colorbar
    cmap : string, optional, default: 'RdBu_r'
        matplotlib.colors.Colormap object name
    title : string, optional, default: None
        Title for the plot

    Returns
    -------

    """

    if vmin is None:
        vmin = tfr.min()

    if vmax is None:
        vmax = tfr.max()

    import matplotlib.pyplot as plt
    extent = (times[0], times[-1], freqs[0], freqs[-1])
    plt.imshow(tfr[ch_idx], extent=extent, aspect="auto", origin="lower",
               vmin=vmin, vmax=vmax, cmap=cmap)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)
