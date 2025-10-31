# Noisemaker in Docker

## Building

To build the Docker image from source:

```bash
docker build -t noisemaker .
```

The Dockerfile is located at the root of the repository and uses a multi-stage build for optimal image size.

## Usage

Getting noisemaker output out of Docker requires mounting a volume for output files. You must:
- Mount a local directory to `/output` in the container (`-v /your/local/output:/output`)
- Prefix the output filename with `output/` (`--filename output/noise.png`)

### Docker on Linux, Mac

Make sure your output directory exists (`mkdir output`). To generate basic noise, run:

```bash
docker run -v `pwd`/output:/output noisemaker \
    noisemaker generate multires --filename output/noise.png
```

Or use the published image:

```bash
docker run -v `pwd`/output:/output aayars/noisemaker \
    noisemaker generate multires --filename output/noise.png
```

### Docker on Windows

Make sure that disk sharing is enabled in your local Docker settings, and provide the full local path to `-v`.

Replace {{YOUR-USERNAME-HERE}} and {{PATH-TO-OUTPUT}} in the example below:

```bash
docker run -v c:/Users/{{YOUR-USERNAME-HERE}}/{{PATH-TO-OUTPUT}}:/output noisemaker \
    noisemaker generate multires --filename output/noise.png
```

### Commands

Noisemaker has several commands available:

#### Generate an image from a preset

```bash
docker run -v `pwd`/output:/output noisemaker \
    noisemaker generate acid --filename output/acid.png
```

#### Create an animation

```bash
docker run -v `pwd`/output:/output noisemaker \
    noisemaker animate 2d-chess --filename output/chess.mp4
```

#### Apply an effect to an existing image

First, make sure your input image is accessible to the container:

```bash
docker run -v `pwd`/output:/output noisemaker \
    noisemaker apply glitchin-out /output/input.jpg --filename output/glitched.jpg
```

### Additional Options

Each command supports various options for customization. For example:

```bash
docker run -v `pwd`/output:/output noisemaker \
    noisemaker generate voronoi --width 2048 --height 2048 --seed 12345 --filename output/large.png
```

For all available options, run:

```bash
docker run noisemaker noisemaker --help
docker run noisemaker noisemaker generate --help
docker run noisemaker noisemaker animate --help
docker run noisemaker noisemaker apply --help
```

### See also

- [source](https://github.com/aayars/noisemaker)
- [readthedocs](http://noisemaker.readthedocs.io/en/latest/)
