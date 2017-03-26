# Noisemaker

**Noisemaker** is a visual noise generator for python3, built with TensorFlow.

## External Requirements

- Python 3.6+
- Tensorflow 1.0.1+

## Developer Installation

Clone Noisemaker, and create a virtualenv.

```
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [Tensorflow](https://www.tensorflow.org/install/) in the virtualenv, using pip.

```
# For now, Mac users can just use the included requirements file.
pip install -r requirements-mac.txt
```

Install Noisemaker in the virtualenv.

```
python setup.py develop
python setup.py install_scripts
```

Start making some noise.

```
noisemaker --shadow --sharpen multires --width 1024 --height 512
```

See the help screens for usage. TODO: Make these better!

```
noisemaker --help

noisemaker gaussian --help

noisemaker multires --help
```