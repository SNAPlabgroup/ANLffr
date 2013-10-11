# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:00:08 2013

@author: hari
"""

from distutils.core import setup

setup(
    name='ANLffr',
    version='0.1.0',
    author='Hari Bharadwaj',
    author_email='hari@nmr.mgh.harvard.edu',
    packages=['anlffr'],
    url='http://github.com/haribharadwaj/assr-tools/',
    license='BSD (3 Clause)',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "nitime >= 0.4",
        "mne >= 0.6",
        "numpy >= 1.7.1"
    ],
)