#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats

"""QuantStats: Portfolio analytics for quants
QuantStats performs portfolio profiling, to allow quants and
portfolio managers to understand their performance better,
by providing them with in-depth analytics and risk metrics.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
import quantstats

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='QuantStats',
    version=quantstats.__version__,
    description='Portfolio analytics for quants',
    long_description=long_description,
    url='https://github.com/ranaroussi/quantstats',
    author='Ran Aroussi',
    author_email='ran@aroussi.com',
    license='Apache Software License',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',

        'Operating System :: OS Independent',

        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',

        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',

        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    platforms = ['any'],
    keywords='ezibpy interactive brokers tws, ibgw, ibpy',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)
