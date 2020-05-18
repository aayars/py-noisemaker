import json
import shutil
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
@cli.option('--effect-preset', type=click.Choice(["random"] + sorted(presets.EFFECTS_PRESETS)))
@cli.name_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, clut, seed, effect_preset, name, save_frames, preset_name):
    if preset_name == 'random':
        preset_name = 'random-preset'

    if effect_preset == 'random':
       effect_preset = 'random-effect'

    kwargs = presets.preset(preset_name)

    preset_name = kwargs['name']

    if effect_preset:
        effect_kwargs = presets.preset(effect_preset)

        effect_preset = effect_kwargs['name']

        print(preset_name + " vs. " + effect_preset)
    else:
        print(preset_name)

    # Override defaults to animate better
    overrides = {}

    distrib = kwargs.get('distrib')

    if distrib in (ValueDistribution.exp, 'exp'):
        overrides['distrib'] = 'simplex_exp'

    elif distrib not in (
        # ValueDistribution.ones, 'ones',
        ValueDistribution.simplex_exp, 'simplex_exp',
        ValueDistribution.column_index, 'column_index',
        ValueDistribution.row_index, 'row_index',
    ):
        overrides['distrib'] = 'simplex'

    if 'point_drift' not in kwargs:
        overrides['point_drift'] = 0.25

    if 'speed' not in kwargs:
        overrides['speed'] = 0.25

    frames = 30

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            filename = f'{tmp}/{i:04d}.png'

            common_params = ['--seed', str(seed or 1),
                             '--overrides', json.dumps(overrides),
                             '--time', f'{i/frames:0.4f}',
                             '--name', filename]

            subprocess.check_call(['artmaker', preset_name,
                                   '--height', str(height),
                                   '--width', str(width)] + common_params,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if effect_preset:
                subprocess.check_call(['artmangler', effect_preset, filename, '--no-resize'] + common_params,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if save_frames:
                shutil.copy(filename, save_frames)

        subprocess.check_call(['convert', '-delay', '5', f'{tmp}/*png', name])
