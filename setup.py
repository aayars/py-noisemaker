#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='noisemaker',
      version='0.2.153',
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
        "absl-py>=0.9,<1.5",
        "click==8.1.3",
        "h5py>=3.1,<3.8",
        "loguru==0.6.0",
        "numpy>=1.19.2,<1.25.0",
        "opensimplex==0.3",
        "Pillow==9.4.0",
        "protobuf>=3.7,<5",
        "six~=1.15.0",
        "tensorflow-graphics==2021.12.3",
        "typing-extensions>=4.0.1",
        "wheel==0.38.4",  # Needed by TF
        ],

      setup_requires=["pytest-runner"],
      tests_require=["pytest==7.2.0"],
      )
