#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='noisemaker',
      version='0.1.317',
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
        worldmaker=noisemaker.scripts.worldmaker:main
        crop=noisemaker.scripts.crop:main
        ''',

      install_requires=[
        "click==7.1.2",
        "Pillow==7.2.0",
        "wheel==0.35.1",  # Needed by TF
        "opensimplex==0.3",
        ]
      )
