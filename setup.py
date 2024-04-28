#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='noisemaker',
      version='0.5.0',
      description='Generates procedural noise with Python 3 and TensorFlow',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      url='https://github.com/aayars/py-noisemaker',
      packages=find_packages(),

      entry_points='''
        [console_scripts]
        magic-mashup=noisemaker.scripts.magic_mashup:main
        mood=noisemaker.scripts.mood:main
        noisemaker=noisemaker.scripts.noisemaker:main
        ''',

      install_requires=[
        "absl-py>=0.9,<1.4",
        "click==8.1.7",
        "colorthief==0.2.1",
        "h5py>=3.10.0",
        "loguru==0.7.1",
        "opensimplex==0.3",
        "Pillow==9.5.0",
        "protobuf<4.21,>=3.20.3",
        "requests>=2.4.2",
        "six>=1.15,<1.17",
        "tensorflow-graphics==2021.12.3",
        "wheel==0.41.2",  # Needed by TF
        ],

      tests_require=["pytest==8.2.0"],
      )

