#!/usr/bin/env python

from setuptools import setup

setup(name='noisemaker',
      version='0.0.1',
      description='Noise making thinger for python3',
      author='Alex Ayars',
      author_email='aayars',
      # url='',
      packages=['noisemaker'],
      entry_points='''
        [console_scripts]
        noisemaker=noisemaker.cli:main
      ''',
      )
