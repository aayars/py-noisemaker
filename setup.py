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
        "absl-py>=0.9,<2.2",
        "click==8.1.7",
        "colorthief==0.2.1",
        "h5py>=3.10.0",
        "loguru==0.7.2",
        "opensimplex==0.3",
        "pillow>=10.0.1,<11",
        "protobuf>=4.25.8,<5",
        "requests>=2.4.2",
        "six>=1.15,<1.17",
        "wheel==0.43.0",  # Needed by TF
        ],

      tests_require=["pytest==8.2.0"],
      )

