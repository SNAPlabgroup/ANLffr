ANLffr
==========

From the `Auditory Neuroscience Lab at Boston University <http://www.cns.bu.edu/~shinn/ANL/index.html>`_, 
a set of tools to analyze and interpret auditory steady-state responses, 
particularly the subcortical kind commonly known as frequency-following responses (FFRs). 
In particular, the package provides code for multitaper-analysis of spectra and phase locking 
along with complex-principal component analysis of phase-locking for multichannel FFRs. 
Support for "bootstrapping" any of the included functions is also available. 
The project homepage is http://nmr.mgh.harvard.edu/~hari/ANLffr/.
Typical usage would begin with::
    
    #!/usr/bin/env python

    from anlffr import spectral

    # If using a Biosemi EEG system
    from anlffr.helper import biosemi2mne


Get the latest code
-------------------

To get the latest code using git, simply type::

    git clone git://github.com/haribharadwaj/ANLffr.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/haribharadwaj/ANLffr/archive/master.zip

If you have pip, you may be able to download and install anlffr in one step using (and hence skip the "Install anlffr" steps below)::

    pip install git+https://github.com/haribharadwaj/ANLffr.git --user

Install anlffr
--------------

As with any Python packages, to install ANLffr, go into the ANLffr source
code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user


Dependencies
------------

Packages NumPy >= 1.4, SciPy >= 0.7.2 and nitime >= 0.4 are required for the code in the spectral.py module to work. `MNE-python <http://github.com/mne-tools/mne-python>`_ >= 0.7 is required for the modules in the anlffr.helper package (For importing Biosemi BDF files and preprocessing).

Getting Started
---------------
At the `project homepage <http://nmr.mgh.harvard.edu/~hari/ANLffr/>`_, 
you can find more extensive and growing documentation,
including in particular a complete and searchable function reference. 
The `examples directory <https://github.com/haribharadwaj/ANLffr/tree/master/examples>`_ 
contains a sample script that you could modify for your purposes. 
That would be a good place to get started! 
See `ARO2013 poster #923 <http://nmr.mgh.harvard.edu/~hari/HB_ARO2013_poster923.pdf>`_ 
for details of the multichannel complex-PCA method. 
The manuscript is under review/revision. 
For details of the PLV computation and bootstrapping, 
see `Zhu et al. (2013) <http://www.cns.bu.edu/~shinn/resources/pdfs/2013/2013JASA_Zhu.pdf>`_.

Briefly, consider the following example::

    #!/usr/bin/env python

    # If you already have a data.mat file with data

    # Using scipy to load the data
    from scipy import io
    dat = io.loadmat(path_to_data + 'data.mat')

    # Say the variable containing the data array is 'x'
    x = dat['x']

    # View to check if 'x' is in the right format
    x.shape

    # Import the spectral module to calculate PLV
    from anlffr import spectral

    # Set parameters for the PLV computation
    # Sampling rate is 4096 Hz
    # Frequencies of interest are with 5 and 500 Hz
    # Calculate inter-trial coherence instead of PLV
    # Zero-pad data to the next power of 2 for fast FFT
    # Use 3 tapers with the time-half bandwidth product of 2
    params = dict(Fs = 4096, tapers = [2, 3], fpass = [5, 500], itc = 1,
                  pad = 1)

    # Actually compute the phase-locking measure (ITC here)
    (plv, f) = spectral.mtplv(x, params)

    # Plot results for channel number 31 (index 30)
    # Use pylab for plotting

    import pylab as pl
    pl.plot(f, plv[30,:],linewidth = 2)
    pl.xlabel('Frequency (Hz)', fontsize = 20)
    pl.ylabel('Inter-Trial Coherence')
    pl.show()

Licensing
---------

ANLffr is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2013, authors of ANLffr.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    * Neither the names of ANLffr authors nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.**

