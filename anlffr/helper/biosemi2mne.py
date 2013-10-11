from subprocess import call
import mne
import numpy as np
import os
import sys

def importbdf(edfname, fiffname, evename, refchans,
              hptsname = None, aliasname = None):
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
    MNE commandline tools should be in the path of current shell.
    See the MNE manual for information about the alias and hpts files.
    
    """
    
    # What happens if they don' specify a hpts/alias file: Use default
    if(hptsname == None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        hptsname = os.path.join(anlffr_root,'sysfiles/biosemi32.hpts')
        
    if(aliasname == None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        aliasname = os.path.join(anlffr_root,'sysfiles/biosemi32alias.txt')
        
    call(["mne_edf2fiff","--edf",edfname,"--fif",fiffname,"--hpts",hptsname])
    call(["mne_rename_channels","--fif",fiffname,"--alias",aliasname])
    call(["mne_process_raw","--raw",fiffname,"--digtrig","Status",
         "--digtrigmask","0xff","--eventsout",evename])
    raw = mne.fiff.Raw(fiffname, preload = True)
    ref = np.mean(raw._data[refchans,:], axis = 0)
    raw._data = raw._data - ref
    eves = mne.read_events(evename)
    return (raw,eves)

	


