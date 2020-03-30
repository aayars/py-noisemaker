import json
import subprocess
import tempfile

import click

from noisemaker.constants import ValueDistribution

import noisemaker.cli as cli
import noisemaker.presets as presets


@click.command(help="""
        Animaker - Animated Noisemaker loops

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.clut_option()
@cli.seed_option()
@cli.name_option(default='ani.gif')
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, clut, seed, name, preset_name):
    if preset_name == 'random':
        preset_name = 'random-preset'

    kwargs = presets.preset(preset_name)

    preset_name = kwargs['name']
    print(preset_name)

    # Override defaults to animate better
    overrides = {}

    distrib = kwargs.get('distrib')

    if distrib in (ValueDistribution.exp, 'exp'):
        overrides['distrib'] = 'simplex_exp'

    elif distrib not in (ValueDistribution.ones, 'ones', ValueDistribution.simplex_exp, 'simplex_exp'):
        overrides['distrib'] = 'simplex'

    if not kwargs.get('lattice_drift'):
        overrides['lattice_drift'] = 0.5

    if not kwargs.get('point_drift'):
        overrides['point_drift'] = 0.5

    frames = 30

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            subprocess.check_call(['artmaker', preset_name,
                                  '--seed', str(seed or 1),
                                  '--overrides', json.dumps(overrides),
                                  '--height', str(height),
                                  '--width', str(width),
                                  '--time', f'{i/frames:0.4f}',
                                  '--name', f'{tmp}/{i:04d}.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.check_call(['convert', '-delay', '5', f'{tmp}/*png', name])
