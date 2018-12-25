#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='noisemaker',
      version='0.1.46',
      description='Generates procedural noise with Python 3 and TensorFlow',
      author='Alex Ayars',
      author_email='aayars@gmail.com',
      url='https://github.com/aayars/py-noisemaker',
      packages=find_packages(),

      entry_points='''
        [console_scripts]
        artmaker=noisemaker.scripts.artmaker:main
        artmangler=noisemaker.scripts.artmangler:main
        collagemaker=noisemaker.scripts.collagemaker:main
        glitchmaker=noisemaker.scripts.glitchmaker:main
        noisemaker=noisemaker.scripts.noisemaker:main
        worldmaker=noisemaker.scripts.worldmaker:main
        crop=noisemaker.scripts.crop:main
        ''',

      install_requires=[
        "click==6.7",
        "Pillow==4.1.1",
        "wheel==0.26",  # Needed by TF
        ]
      )
