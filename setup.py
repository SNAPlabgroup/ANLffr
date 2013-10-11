# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:00:08 2013

@author: hari
"""

from numpy.distutils.core import setup

setup(
    name='ANLffr',
    version='0.1.0',
    author='Hari Bharadwaj',
    author_email='hari@nmr.mgh.harvard.edu',
    packages=['anlffr','anlffr.helper'],
    url='http://github.com/haribharadwaj/assr-tools/',
    license='BSD (3 Clause)',
    description='Auditory Neuroscience Lab (ANL) at Boston University',
    long_description=open('README.txt').read(),
)