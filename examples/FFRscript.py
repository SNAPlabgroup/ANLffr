from anlffr.helper import biosemi2mne as bs
import mne
import numpy as np
from anlffr import spectral
import pylab as pl

# Adding Files and locations
# This below is just an example
# You have to edit this to supply your own data
fpath = '/home/hari/Documents/MNEforBiosemi/'
edfname = 'I13_depth_01.bdf'

# Load data and read event channel
(raw,eves) = bs.importbdf(fpath+edfname)

# Filter the data
raw.filter(l_freq = 70, h_freq = 1500, picks = np.arange(0,32,1))

# Here events 1 and 7 represent a particular stimulus in each polarity
selectedEve = dict(up0 = 1, down0 = 7)

# Epoching events of type 1 and 7
epochs = mne.Epochs(raw,eves,selectedEve,tmin = -0.05,
		tmax = 0.45, baseline = (-0.05, 0), reject = dict(eeg=100e-6))

# Combining both polarities so I can get envelope related FFR responses
epochs = mne.epochs.combine_event_ids(epochs,['up0','down0'],dict(all0= 101))

# Getting the epoched data out, this step will also perform rejection
x = epochs.get_data()

# Reshaping to the format needed by spectral.mtcpca() and calling it
x = x.transpose((1,0,2))
x = x[0:32,:,:]
params = dict(Fs=4096,fpass=[5,1000],tapers=[2, 3],itc=1)
(plv,f) = spectral.mtcpca(x,params)


# Plotting results
pl.plot(f,plv,linewidth = 2)
pl.xlabel('Frequency (Hz)')
pl.grid(True)
pl.show()

