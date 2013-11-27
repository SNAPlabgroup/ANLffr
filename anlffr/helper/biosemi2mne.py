from subprocess import call
import mne
import numpy as np
import os
import sys
from mne.fiff import edf
def importbdf_old(edfname, fiffname, evename, refchans,
              hptsname = None, aliasname = None):
    """ Wrapper around MNE to import Biosemi BDF files
    This is deprecated since the python native EDF reader was added to mne.
    
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
        hptsname = os.path.join(anlffr_root,'helper/sysfiles/biosemi32.hpts')
        
    if(aliasname == None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        aliasname = os.path.join(anlffr_root,
                                 'helper/sysfiles/biosemi32alias.txt')
        
    call(["mne_edf2fiff","--edf",edfname,"--fif",fiffname,"--hpts",hptsname])
    call(["mne_rename_channels","--fif",fiffname,"--alias",aliasname])
    call(["mne_process_raw","--raw",fiffname,"--digtrig","Status",
         "--digtrigmask","0xff","--eventsout",evename])
    raw = mne.fiff.Raw(fiffname, preload = True)
    ref = np.mean(raw._data[refchans,:], axis = 0)
    raw._data = raw._data - ref
    eves = mne.read_events(evename)
    return (raw,eves)

def importbdf(bdfname, nchans = 34, refchans = ['EXG1','EXG2'],
              hptsname = None):
    """Wrapper around mne-python to import BDF files
    
    Parameters
    ----------
    bdfname - Name of the biosemi .bdf filename with full path
    
    nchans -  Number of EEG channels (including references)
              (Optional) By default, 34 (32 + 2 references)
    refchans - list of strings with reference channel names
               (Optional) By default ['EXG1','EXG2']
    hptsname - Name of the electrode position file in .hpts format with path
               (Optional) By default a 32 channel Biosemi layout is used.
    
    Returns
    -------
    raw - MNE raw data object of rereferences and preloaded data
    eves - Event list (3 column array as required by mne.Epochs)
    
    Requires
    --------
    mne-python module > release 0.7
    """
    
    # Default HPTS file
    if(hptsname == None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        hptsname = os.path.join(anlffr_root,'helper/sysfiles/biosemi32.hpts')
        
    raw = edf.read_raw_edf(bdfname, n_eeg = nchans, preload = True,
                           hpts = hptsname, stim_channel = 'Status')
    
    
    # Rereference
    print 'Re-referencing data to', refchans
    (raw, ref_data) = mne.fiff.set_eeg_reference(raw, refchans, copy = False)
    
    # Once re-referenced, should not use reference channels as EEG channels
    raw.info['bads'] = refchans
    
    # Add average reference operator for possible use later
    ave_ref_operator = mne.fiff.make_eeg_average_ref_proj(raw.info,
                                                          activate = False)
    
    raw = raw.add_proj(ave_ref_operator)
    
    eves = mne.find_events(raw)
    
    return (raw,eves)
    
    
    
    


