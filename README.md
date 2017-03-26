# Noisemaker

**Noisemaker** is a visual noise generator for python3, built with TensorFlow.

## External Requirements

- Python 3.6+
- TensorFlow 1.0.1+

## Developer Installation

Clone Noisemaker, and create a virtualenv.

```
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The above `pip` command installs scikit-image, which takes a while. Let it happen.

Install [TensorFlow](https://www.tensorflow.org/install/) in the virtualenv, using pip.

```
# For now, Mac users can just use the included requirements file.
pip install -r requirements-mac.txt
```

Install Noisemaker in the virtualenv.

```
python setup.py develop
python setup.py install_scripts
```

## Usage

Start making some noise.

```
noisemaker --shadow --sharpen multires --width 1024 --height 512
```

...do it in Python:

```
from noisemaker.generators import multires

tensor = multires()  # Image tensor with shape (height, width, channels)
```

See the help screens for usage. TODO: Make these better!

```
noisemaker --help

noisemaker gaussian --help

noisemaker multires --help
```

## See also

- [Wikipedia: Value Noise](https://en.wikipedia.org/wiki/Value_noise)
- [Wikipedia: Perlin Noise](https://en.wikipedia.org/wiki/Perlin_noise)
- [Wikipedia: Tensor](https://en.wikipedia.org/wiki/Tensor)

