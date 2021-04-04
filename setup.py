# -*- coding: utf-8 -*-
"""
Setup details for ANLffr
Created on Thu Oct 10 19:00:08 2013

@author: hari
"""

from setuptools import setup, find_packages

setup(
    name='ANLffr',
    version='0.3.0a1',
    author='Hari Bharadwaj',
    author_email='hari.bharadwaj@gmail.com',
    packages=find_packages(include=['anlffr', 'anlffr.*']),
    python_requires='>=3',
    requires=['joblib'],
    package_data={'anlffr.helper': ['sysfiles/*']},
    url='https://github.com/SNAPsoftware/ANLffr',
    license='BSD (3 Clause)',
    description=('Useful functions for processing and analysis of'
                 'mass-potentials and other electrophysiological data from'
                 'SNAPlab at Purdue University. Provides frequency, and'
                 'time-frequency analysis capabilities'),
    long_description=open('README.rst').read(),
)
