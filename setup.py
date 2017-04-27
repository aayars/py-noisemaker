#!/usr/bin/env python

from setuptools import setup


setup(name='noisemaker',
      version='0.0.1',
      description='Noise making thinger for python3',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      url='http://noisemaker.io/',
      packages=['noisemaker'],

      entry_points='''
        [console_scripts]
        noisemaker=noisemaker.cli:main
        ''',

      install_requires=[
        "click==6.7",
        "numpy==1.12.1",
        ]
      )
