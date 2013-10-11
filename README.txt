ANLffr
==========

From the Auditory Neuroscience Lab at Boston University (http://www.cns.bu.edu/~shinn/ANL/index.html), a set of tools to analyze and interpret auditory stead-state responses,particularly the subcortical kind commonly known as frequency-following responses (FFRs). In particular, the package provides code for multitaper-analysis of spectra and phase locking along with complex-principal component analysis of phase-locking for multichannel FFRs. Typical usage would begin with::
    
    #!/usr/bin/env python

    from anlffr import spectral
    from anlffr.helper import biosemi2mne


The project homepage is http://github.com/haribharadwaj/ANLffr. Packages
numpy, scipy, nitime are required for the code in the spectral.py module to
work. The mne-python package and commandline tools are required for the
modules in the anlffr.helper package. See the examples directory for a sample
script to get started!


