from anlffr.helper import biosemi2mne as bs
import mne
import numpy as np
from anlffr import spectral
import pylab as pl

# Adding Files and locations
fpath = '/home/hari/Documents/MNEforBiosemi/'

# List of files stems, each will be appended by run number 
# Use list [] and enumerate over if filenames are weird

namestem = 'I13_depth'

nruns = 5 # Number of files

for k in np.arange(0,nruns):
    # Load data and read event channel
    # You'll need to modify it based on your file naming convention:
    edfname = fpath + namestem + '_0' + str(k+1) + '.bdf'

    (raw,eves) = bs.importbdf(edfname)

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
    xtemp = epochs.get_data()

    # Reshaping to the format needed by spectral.mtcpca() and calling it
    xtemp = xtemp.transpose((1,0,2))
    xtemp = xtemp[0:32,:,:]

    if(k==0):
        x = xtemp
    else:
        x = np.concatenate((x,xtemp),axis = 1)


params = dict(Fs=4096,fpass=[5,1000],tapers=[2, 3],pad=1,itc=1)
nPerDraw = 400
nDraws = 200

(plv,vplv,f) = spectral.bootfunc(x,nPerDraw,nDraws,params,func='cpca')


# Plotting results
pl.plot(f,plv,linewidth = 2)
pl.xlabel('Frequency (Hz)')
pl.grid(True)
pl.show()




