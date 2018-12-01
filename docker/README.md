# Noisemaker in a Docker

## Usage

Getting noisemaker output out of Docker is clunky. You must tell Docker to
mount a volume for noisemaker's output (`-v /your/local/output:/output`),
and prefix noisemaker's output filename with `output/` (`--name
output/noise.png`).

### Docker on Linux, Mac

Make sure your output directory exists (`mkdir output`). To generate basic noise, run:

```
    docker run -v output:/output aayars/py-noisemaker \
        noisemaker --name output/noise.png
```

### Docker on Windows

Make sure that disk sharing is enabled in your local Docker settings, and provide the full local path to `-v` when using noisemaker.

Replace {{YOUR-USERNAME-HERE}} and {{PATH-TO-OUTPUT}} in the below example:

```
    docker run -v c:/Users/{{YOUR-USERNAME-HERE}}/{{PATH-TO-OUTPUT}}:/output aayars/py-noisemaker noisemaker --name output/noise.png
```

### Parameters

Noise in noisemaker is composed by providing additional arguments. For example, for multi-octave noise, add an `--octaves` argument:

```
    docker run -v output:/output aayars/py-noisemaker \
        noisemaker --name output/noise.png --octaves 4
```

There are lots of optional parameters. For all options, run:

```
    docker run aayars/py-noisemaker noisemaker --help
```

### Other commands

Additional noisemaker entrypoints, such as `artmaker` and `artmangler`, are included in the image.

```
    docker run aayars/py-noisemaker artmaker --help
    docker run aayars/py-noisemaker artmangler --help
```

### See also

- [source](https://github.com/aayars/py-noisemaker)
- [readthedocs](http://noisemaker.readthedocs.io/en/latest/)
