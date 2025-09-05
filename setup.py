#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

def read_requirements():
    reqs_path = Path(__file__).with_name("requirements.txt")
    with reqs_path.open() as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
      name='noisemaker',
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

      install_requires=read_requirements(),
      tests_require=["pytest==8.4.2"],
      )

