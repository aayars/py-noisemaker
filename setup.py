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
        "h5py>=3.1,<3.10",
        "loguru==0.7.1",
        "numpy>=1.19.2,<1.24.0",
        "opensimplex==0.3",
        "Pillow==9.5.0",
        "protobuf>=3.7,<5",
        "requests>=2.4.2",
        "six~=1.15.0",
        "tensorflow-graphics==2021.12.3",
        "wheel==0.41.3",  # Needed by TF
        ],

      setup_requires=["pytest-runner"],
      tests_require=["pytest==7.4.0"],
      )
