#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='noisemaker',
      version='0.2.147',
      description='Generates procedural noise with Python 3 and TensorFlow',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      url='https://github.com/aayars/py-noisemaker',
      packages=find_packages(),

      entry_points='''
        [console_scripts]
        animaker=noisemaker.scripts.animaker:main
        animangler=noisemaker.scripts.animangler:main
        artmaker=noisemaker.scripts.artmaker:main
        artmangler=noisemaker.scripts.artmangler:main
        collagemaker=noisemaker.scripts.collagemaker:main
        glitchmaker=noisemaker.scripts.glitchmaker:main
        magic-mashup=noisemaker.scripts.magic_mashup:main
        noisemaker=noisemaker.scripts.noisemaker:main
        noisemaker-old=noisemaker.scripts.noisemaker_old:main
        worldmaker=noisemaker.scripts.worldmaker:main
        crop=noisemaker.scripts.crop:main
        mood=noisemaker.scripts.mood:main
        ''',

      install_requires=[
        "absl-py<0.11,>=0.9",
        "click==7.1.2",
        "h5py~=2.10.0",
        "loguru==0.5.3",
        "opensimplex==0.3",
        "Pillow==8.0.1",
        "protobuf<4,>=3.7",
        "six~=1.15.0",
        "tensorflow_addons==0.12.1",
        "tensorflow-graphics==2020.5.20",
        "wheel==0.36.2",  # Needed by TF
        ],

      setup_requires=["pytest-runner"],
      tests_require=["pytest==6.2.1"],
      )
