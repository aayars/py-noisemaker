#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='noisemaker',
      version='0.2.24',
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
        artmaker-new=noisemaker.scripts.artmaker_new:main
        artmangler=noisemaker.scripts.artmangler:main
        collagemaker=noisemaker.scripts.collagemaker:main
        glitchmaker=noisemaker.scripts.glitchmaker:main
        magic-mashup=noisemaker.scripts.magic_mashup:main
        noisemaker=noisemaker.scripts.noisemaker:main
        worldmaker=noisemaker.scripts.worldmaker:main
        crop=noisemaker.scripts.crop:main
        mood=noisemaker.scripts.mood:main
        ''',

      install_requires=[
        "click==7.1.2",
        "loguru==0.5.3",
        "opensimplex==0.3",
        "Pillow==8.0.1",
        "pyfastnoisesimd==0.4.1",
        "tensorflow_addons==0.11.2",
        "wheel==0.36.2",  # Needed by TF
        ],

      setup_requires=["pytest-runner"],
      tests_require=["pytest==6.1.2"],
      )
