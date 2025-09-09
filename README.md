# Noisemaker

Noisemaker adapts classic procedural noise generation algorithms for **Python 3** and **TensorFlow**. It provides tools for composing layers of noise, applying image effects, and rendering generative art from the command line or Python.

## Features

- Modern Python API built on TensorFlow
- Command line interface for generating images and animations
- High-level *composer* presets with reusable settings and layered effects
- Low-level generator and effect functions for custom workflows
- Optional Docker image and interactive Colab notebook

## Installation

Noisemaker targets Python 3.5+ and requires TensorFlow. Create a virtual environment and install from GitHub:

```bash
python3 -m venv noisemaker
source noisemaker/bin/activate
pip install git+https://github.com/aayars/py-noisemaker
pip install tensorflow
```

### Development setup

```bash
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
python setup.py install_scripts
```

## Usage

### Command line

The `noisemaker` CLI can generate images, animations, and apply post-processing effects:

```bash
noisemaker generate acid -o acid.png
noisemaker animate 2d-chess -o chess.gif
noisemaker apply glitchin-out input.jpg -o output.jpg
```

Run `noisemaker --help` to see the full command list.

### Python API

```python
from noisemaker.composer import Preset

preset = Preset('acid')
# Save the result directly to disk
preset.render(seed=1, shape=[256, 256, 3], filename='art.png')
```

To work with the generated data as a NumPy array instead of writing to a file:

```python
from noisemaker import generators

tensor = generators.multires(seed=1, shape=[256, 256, 3])
array = tensor.numpy()
```

## Documentation

Full documentation, including the complete API reference and preset guide, is available on [Read the Docs](http://noisemaker.readthedocs.io/).

## Contributing

Issues and pull requests are welcome! Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Run the test suite with `pytest` before submitting changes.

## Javascript Port

This document is for the Python version of Noisedeck. See the [Javascript README](README-JS.md).

## License

Noisemaker is released under the [Apache 2.0 License](LICENSE).

