from subprocess import call
import mne
import numpy as np

def importbdf(edfname,fiffname,hptsname,aliasname,evename,refchans):
    """ Wrapper around MNE to import Biosemi BDF files
    
    Parameters
    ----------
    edfname - Name of the biosemi .bdf file with full path
    fiffname - Name of fiff file to be generated with full path
    hptsname - Name of electrode position file in .hpts formal (with path)
    aliasname - Alias .txt file to assign electrode types and names correctly
    evename - Name of event file to be written by reading the 'Status' channel
    refchans - Reference channel(s) for rereferencing e.g. [32, 33]
    
    Returns
    -------
    raw - MNE raw object of rereferenced and preloaded data
    eve - Event list (3 column array as required by mne.Epochs)
    
    Requires
    --------
    MNE commandline tools should be in the path of current shell
    
    """
    call(["mne_edf2fiff","--edf",edfname,"--fif",fiffname,"--hpts",hptsname])
    call(["mne_rename_channels","--fif",fiffname,"--alias",aliasname])
    call(["mne_process_raw","--raw",fiffname,"--digtrig","Status",
         "--digtrigmask","0xff","--eventsout",evename])
    raw = mne.fiff.Raw(fiffname, preload = True)
    ref = np.mean(raw._data[refchans,:], axis = 0)
    raw._data = raw._data - ref
    eves = mne.read_events(evename)
    return (raw,eves)

	


