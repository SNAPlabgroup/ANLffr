import os
import sys
from mne import find_events
from mne.io import read_raw_bdf
from mne.channels import read_dig_hpts
from ..utils import logger, verbose


@verbose
def importbdf(bdfname, nchans=34, refchans=['EXG1', 'EXG2'],
              hptsname=None, mask=255, extrachans=[],
              exclude=[], verbose=None):
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
    exclude - List of channel names to exclude from importing
    verbose - bool, str, int, or None (Optional)
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
            hptsname = 'biosemi64.hpts'
            montage = read_dig_hpts(hptspath + hptsname)
            misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        else:
            if nchans >= 96:
                logger.info('Number of channels is greater than 96.'
                            ' Hence loading a 96 channel montage.')
                hptspath = os.path.join(anlffr_root, 'helper/sysfiles/')
                hptsname = 'biosemi96.hpts'
                montage = read_dig_hpts(hptspath + hptsname)
                misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
            else:
                if nchans == 2:
                    logger.info('Number of channels is 2.'
                                'Guessing ABR montage or saccades.')
                    montage = None
                    misc = []
                else:
                    logger.info('Loading a default 32 channel montage.')
                    hptspath = os.path.join(anlffr_root, 'helper/sysfiles/')
                    hptsname = 'biosemi10_20.hpts'
                    montage = read_dig_hpts(hptspath + hptsname)
                    misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
    else:
        montage = read_dig_hpts(hptsname)  # User-supplied
        misc = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

    misc += extrachans
    raw = read_raw_bdf(bdfname, preload=True, misc=misc, exclude=exclude,
                       stim_channel='auto')
    raw.set_montage(montage, on_missing='warn')

    # Rereference
    if refchans is not None:
        sys.stdout.write('Re-referencing data to: ' + ' '.join(refchans))
        raw.set_eeg_reference(ref_channels=refchans, copy=False)
        raw.info['bads'] += refchans
    else:
        # Add average reference operator for possible use later
        raw.set_eeg_reference(ref_channels='average', copy=False)

    eves = find_events(raw, shortest_event=1, mask=mask)

    return (raw, eves)
