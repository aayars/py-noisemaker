#!/usr/bin/env python

from setuptools import setup


setup(name='noisemaker',
      version='0.0.1',
      description='Visual noise generator for Python 3',
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
        ]
      )
