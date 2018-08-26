from subprocess import call
import mne
import numpy as np
import os
import sys
from mne import find_events
from mne.io import edf, set_eeg_reference, make_eeg_average_ref_proj
from mne.channels import read_montage
from ..utils import logger, deprecated, verbose


@deprecated('May fail depending on MNE version! Use importbdf(.) instead.')
def importbdf_old(edfname, fiffname, evename, refchans,
                  hptsname=None, aliasname=None):
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

    # What happens if they don't specify a hpts/alias file: Use default
    if(hptsname is None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        hptsname = os.path.join(anlffr_root, 'helper/sysfiles/biosemi32.hpts')

    if(aliasname is None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        aliasname = os.path.join(anlffr_root,
                                 'helper/sysfiles/biosemi32alias.txt')

    call(["mne_edf2fiff", "--edf", edfname, "--fif", fiffname,
         "--hpts", hptsname])
    call(["mne_rename_channels", "--fif", fiffname, "--alias", aliasname])
    call(["mne_process_raw", "--raw", fiffname, "--digtrig", "Status",
          "--digtrigmask", "0xff", "--eventsout", evename])
    raw = mne.fiff.Raw(fiffname, preload=True)
    ref = np.mean(raw._data[refchans, :], axis=0)
    raw._data = raw._data - ref
    eves = mne.read_events(evename)
    return (raw, eves)


@verbose
def importbdf(bdfname, nchans=34, refchans=['EXG1', 'EXG2'],
              hptsname=None, mask=255, extrachans=[], verbose=None):
    """Wrapper around mne-python to import BDF files

    Parameters
    ----------

    bdfname - Name of the biosemi .bdf filename with full path

    nchans -  Number of EEG channels (including references)
              (Optional) By default, 34 (32 + 2 references)
    refchans - list of strings with reference channel names
               (Optional) By default ['EXG1','EXG2'].
               Use None for average reference.
    hptsname - Name of the electrode position file in .hpts format with path
               (Optional) By default a 32 channel Biosemi layout is used. If
               the nchans is >= 64 and < 96, a 64 channel Biosemi layout is
               used. If nchans >= 96, a 96 channel biosemi layout is used.
               Formats other than .hpts will also likely work, but behavior
               may vary.
    mask - Integer mask to use for trigger channel (Default is 255).
    extrachans - Additional channels other than EEG and EXG that may be in the
                 bdf file. These will be marked as MISC in mne-python.
                 Specify as list of names.
    verbose : bool, str, int, or None (Optional)
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    raw - MNE raw data object of rereferences and preloaded data

    eves - Event list (3 column array as required by mne.Epochs)

    Requires
    --------
    mne-python module > release 0.7
    """

    # Default HPTS file
    if(hptsname is None):
        anlffr_root = os.path.dirname(sys.modules['anlffr'].__file__)
        if nchans >= 64 and nchans < 96:
            logger.info('Number of channels is greater than 64.'
                        ' Hence loading a 64 channel montage.')
            hptspath = os.path.join(anlffr_root, 'helper/sysfiles/')
            hptsname = 'biosemi64'
            montage = read_montage(kind=hptsname, path=hptspath,
                                   transform=True)
            misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        else:
            if nchans >= 96:
                logger.info('Number of channels is greater than 96.'
                            ' Hence loading a 96 channel montage.')
                hptspath = os.path.join(anlffr_root, 'helper/sysfiles/')
                hptsname = 'biosemi96'
                montage = read_montage(kind=hptsname, path=hptspath,
                                       transform=True)
                misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
            else:
                if nchans == 2:
                    logger.info('Number of channels is 2.'
                                'Guessing ABR montage.')
                    montage = None
                    misc = []
                else:
                    logger.info('Loading a default 32 channel montage.')
                    hptspath = os.path.join(anlffr_root, 'helper/sysfiles/')
                    hptsname = 'biosemi32'
                    montage = read_montage(kind=hptsname, path=hptspath,
                                           transform=True)
                    misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

    misc += extrachans
    raw = edf.read_raw_edf(bdfname, montage=montage, preload=True,
                           misc=misc, stim_channel='Status')

    # Rereference
    if refchans is not None:
        print 'Re-referencing data to', refchans
        (raw, ref_data) = set_eeg_reference(raw, refchans, copy=False)
        raw.info['bads'] += refchans
    else:
        # Add average reference operator for possible use later
        ave_ref_operator = make_eeg_average_ref_proj(raw.info, activate=False)
        raw = raw.add_proj(ave_ref_operator)

    eves = find_events(raw, shortest_event=1, mask=mask)

    return (raw, eves)
