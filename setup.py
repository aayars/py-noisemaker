#!/usr/bin/env python

from setuptools import setup


setup(name='noisemaker',
      version='0.0.1',
      description='Classic procedural noise for Python 3 and TensorFlow',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      url='http://noisemaker.io/',
      packages=['noisemaker'],

      entry_points='''
        [console_scripts]
        noisemaker=noisemaker.scripts.noisemaker:main
        ''',

      install_requires=[
        "click==6.7",
        ]
      )
