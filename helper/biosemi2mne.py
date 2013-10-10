from subprocess import call
import mne
import numpy as np

def importbdf(edfname,fiffname,hptsname,aliasname,evename,refchans):
    call(["mne_edf2fiff","--edf",edfname,"--fif",fiffname,"--hpts",hptsname])
    call(["mne_rename_channels","--fif",fiffname,"--alias",aliasname])
    call(["mne_process_raw","--raw",fiffname,"--digtrig","Status",
         "--digtrigmask","0xff","--eventsout",evename])
    raw = mne.fiff.Raw(fiffname, preload = True)
    ref = np.mean(raw._data[refchans,:], axis = 0)
    raw._data = raw._data - ref
    eves = mne.read_events(evename)
    return (raw,eves)

	


