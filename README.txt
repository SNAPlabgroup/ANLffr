assr-tools
==========

A set of tools to analyze and interpret auditory stead-state responses,
  particularly the subcortical kind commonly known as frequency-following
  responses (FFRs). In particular, the package provides code for
  multitaper-analysis of spectra and phase locking along with
  complex-principal component analysis of phase-locking for multichannel FFRs.
Typical usage would begin with::
    
    #!/usr/bin/env python

    from anlffr import spectral
    from anlffr.helper import biosemi2mne


The project homepage is http://github.com/haribharadwaj/assr-tools/
