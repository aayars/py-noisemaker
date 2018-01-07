# Noisemaker in a Docker

## Usage

Getting noisemaker output out of Docker is clunky. You must mount a
volume for noisemaker's output (`-v /your/local/output:/output`), and
specify this directory as the output location for noisemaker commands
(`--name output/noise.png`).

### Docker on Linux

Make sure out output directory exists (`mkdir output`). To generate basic noise, run:

```
docker run -v output:/output aayars/noisemaker noisemaker --name output/noise.png
```

Noise in noisemaker is composed by providing additional arguments. For example, for multi-octave noise, add an `--octaves` argument:

```
docker run -v output:/output aayars/noisemaker noisemaker --name output/noise.png --octaves 8
```

See *Help*.

### Docker on Windows

Make sure that disk sharing is enabled in your local Docker settings, and provide the full local path to `-v` when using noisemaker.

```
docker run -v c:/Users/you/stuff/output:/output aayars/noisemaker noisemaker --name output/noise.png
```

## Help

```
docker run aayars/noisemaker noisemaker --help
```

Other noisemaker commands, such as `artmaker`, are included in the image.

```
docker run aayars/noisemaker artmaker --help
docker run aayars/noisemaker artmaker
```

See also: [readthedocs](http://noisemaker.readthedocs.io/en/latest/)
