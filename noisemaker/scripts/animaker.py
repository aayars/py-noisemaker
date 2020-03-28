import subprocess
import tempfile

import click

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

    frames = 24

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            subprocess.check_call(['artmaker', preset_name,
                                  '--seed', str(seed or 1),
                                  '--overrides', '{"distrib": "simplex"}',
                                  '--height', str(height),
                                  '--width', str(width),
                                  '--time', f'{i/frames:0.4f}',
                                  '--name', f'{tmp}/{i:04d}.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.check_call(['convert', '-delay', '6', f'{tmp}/*png', name])
