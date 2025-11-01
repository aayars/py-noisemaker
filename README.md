# Noisemaker

Noisemaker is a collection of creative coding effects for Python or JavaScript. It provides tools for composing layers of noise, applying image effects, and rendering generative art.

## Documentation

Full documentation, including the complete API reference and preset guide, is available on [Read the Docs](http://noisemaker.readthedocs.io/).

## Features

- Unified noise and effects pipeline driven by a shared preset DSL
- Command-line interface for generating images, animations, and post-processing workflows
- Programmatic APIs for Python and vanilla JavaScript consumers
- Modular generator and effect building blocks for custom compositions
- Experimental WebGPU playground for GPU-native effects
- Prebuilt Docker image [`aayars/noisemaker`](https://hub.docker.com/r/aayars/noisemaker) and an interactive Colab notebook

## Quick Start

### Python CLI or API

Noisemaker requires Python 3.10+ and TensorFlow. Create a virtual environment and install from GitHub:

```bash
python3 -m venv noisemaker
source noisemaker/bin/activate
pip install git+https://github.com/aayars/noisemaker
```

### Local development

```bash
git clone https://github.com/aayars/noisemaker
cd noisemaker
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run tests:

```bash
pytest
```

Run linting and type checking:

```bash
black noisemaker
ruff check noisemaker
mypy noisemaker
```

For browser usage, see the vanilla JavaScript guide in `js/README-JS.md`. Docker usage is documented in `README-DOCKER.md`.

## Usage

### Command line

The `noisemaker` CLI can generate images, animations, and apply post-processing effects:

```bash
noisemaker generate acid --filename acid.png
noisemaker animate 2d-chess --filename chess.gif
noisemaker apply glitchin-out input.jpg --filename output.jpg
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

## Contributing

Issues and pull requests are welcome! Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Run the test suite with `pytest` before submitting changes.

## Ports

Additional platform-specific guides:

- Python development and API details live in the docs linked above
- Browser integration is covered in the [JavaScript README](js/README-JS.md)
- WebGPU shaders are documented in the [Shaders README](shaders/README-SHADERS.md)
- Container workflows appear in the [Docker README](README-DOCKER.md)

## License

Noisemaker is released under the [Apache 2.0 License](LICENSE).
