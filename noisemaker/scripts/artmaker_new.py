import random

import click

from noisemaker.composer import Preset
from noisemaker.util import logger
from noisemaker.presets import PRESETS

import noisemaker.cli as cli
import noisemaker.value as value

GENERATORS = {}
EFFECTS = {}

for preset_name in PRESETS:
    preset = Preset(preset_name, PRESETS)

    if preset.is_generator():
        GENERATORS[preset_name] = preset

    if preset.is_effect():
        EFFECTS[preset_name] = preset


@click.command(help="""
        Artmaker - Presets for Noisemaker

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.width_option(default=2048)
@cli.height_option()
@cli.channels_option()
@cli.time_option()
@cli.clut_option()
@cli.seed_option()
@cli.filename_option(default='art.png')
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATORS)))
@click.pass_context
def main(ctx, width, height, channels, time, clut, seed, filename, preset_name):
    if not seed:
        seed = random.randint(1, 2**32)

    value.set_seed(seed)

    if preset_name == "random":
        preset_name = list(GENERATORS)[random.randint(0, len(GENERATORS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = GENERATORS[preset_name]

    shape = [height, width, channels]

    try:
        preset.render(shape=shape, name=filename)

    except Exception as e:
        logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise
