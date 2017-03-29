#!/usr/bin/env python

from setuptools import setup


setup(name='noisemaker',
      version='0.0.1',
      description='Noise making thinger for python3',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      # url='',
      packages=['noisemaker'],

      entry_points='''
        [console_scripts]
        noisemaker=noisemaker.cli:main
        ''',

      setup_requires=[
        "numpy==1.12.1",  # Need this to build scikit
        ],

      install_requires=[
        "click==6.7",
        "numpy==1.12.1",
        "scikit-image==0.12.3",
        ]
      )
