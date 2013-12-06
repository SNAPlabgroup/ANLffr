
Introduction
===============

Get the latest code
-------------------

To get the latest code using git, simply type::

    git clone git://github.com/haribharadwaj/ANLffr.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/haribharadwaj/ANLffr/archive/master.zip

If you have pip, 
you may be able to download and install anlffr in one step using 
(and hence skip the "Install anlffr" steps below)::

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

Packages NumPy >= 1.4, SciPy >= 0.7.2 and nitime >= 0.4 
are required for the code in the spectral.py module to work. 
`MNE-python <http://github.com/mne-tools/mne-python>`_ >= 0.7 
is required for the modules in the anlffr.helper package 
(For importing Biosemi BDF files and preprocessing).

Getting Started
---------------

Typical usage would begin with::
    
    #!/usr/bin/env python

    from anlffr import spectral

    # If using a Biosemi EEG system
    from anlffr.helper import biosemi2mne


The project homepage is http://github.com/haribharadwaj/ANLffr. The `examples directory <https://github.com/haribharadwaj/ANLffr/tree/master/examples>`_ contains a sample script that you could modify for your purposes. That would be a good place to get started! See `ARO2013 poster #923 <http://nmr.mgh.harvard.edu/~hari/HB_ARO2013_poster923.pdf>`_ for details of the multichannel complex-PCA method. The manuscript is under review/revision. For details of the PLV computation and bootstrapping, see `Zhu et al. (2013) <http://www.cns.bu.edu/~shinn/resources/pdfs/2013/2013JASA_Zhu.pdf>`_.



