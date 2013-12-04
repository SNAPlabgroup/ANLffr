ANLffr
==========

From the `Auditory Neuroscience Lab at Boston University <http://www.cns.bu.edu/~shinn/ANL/index.html>`_, a set of tools to analyze and interpret auditory stead-state responses, particularly the subcortical kind commonly known as frequency-following responses (FFRs). In particular, the package provides code for multitaper-analysis of spectra and phase locking along with complex-principal component analysis of phase-locking for multichannel FFRs. Support for "bootstrapping" any of the included functions is also available. Typical usage would begin with::
    
    #!/usr/bin/env python

    from anlffr import spectral

    # If using a Biosemi EEG system
    from anlffr.helper import biosemi2mne


The project homepage is http://github.com/haribharadwaj/ANLffr. The `examples directory <https://github.com/haribharadwaj/ANLffr/tree/master/examples>`_ contains a sample script that you could modify for your purposes. That would be a good place to get started! 

Get the latest code
-------------------

To get the latest code using git, simply type::

    git clone git://github.com/haribharadwaj/ANLffr.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/haribharadwaj/ANLffr/archive/master.zip

If you have pip, you may be able to download and install anlffr in one step using::

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


