import json
import os
import shutil
import tempfile

import click

from noisemaker.constants import ValueDistribution
from noisemaker.util import magick

import noisemaker.cli as cli
import noisemaker.presets as presets
import noisemaker.util as util


@click.command(help="""
        Animangler - Apply post-processing to a dir full of frames

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.seed_option()
@cli.name_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option('--frame-count', type=int, default=30, help="How many frames total")
@cli.input_dir_option(required=True)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.EFFECTS_PRESETS)))
@click.pass_context
def main(ctx, seed, name, save_frames, frame_count, input_dir, preset_name):
    if preset_name == 'random':
        preset_name = 'random-effect'

    kwargs = presets.preset(preset_name)

    preset_name = kwargs['name']

    print(preset_name)

    # Override defaults to animate better
    overrides = {}

    # XXX TODO factor this shared animaker logic out into a library
    distrib = kwargs.get('distrib')

    if distrib in (ValueDistribution.exp, 'exp'):
        overrides['distrib'] = 'simplex_exp'

    elif distrib not in (
        ValueDistribution.ones, 'ones',
        ValueDistribution.simplex_exp, 'simplex_exp',
        ValueDistribution.column_index, 'column_index',
        ValueDistribution.row_index, 'row_index',
    ):
        overrides['distrib'] = 'simplex'

    if 'point_drift' not in kwargs:
        overrides['point_drift'] = 0.25

    if 'speed' in kwargs:
        # Adjust speed for length of clip. A "normal" length is 30 frames.
        kwargs['speed'] *= frame_count / 30.0

    else:
        overrides['speed'] = 0.25 * (frame_count / 30.0)

    filenames = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.png')]

    if frame_count != len(filenames):
        print(f"Warning! I want to render {frame_count} frames, but found {len(filenames)} input images. This might look weird.")

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(min(frame_count, len(filenames))):
            filename = f'{tmp}/{i:04d}.png'

            common_params = ['--seed', str(seed or 1),
                             '--overrides', json.dumps(overrides),
                             '--time', f'{i/frame_count:0.4f}',
                             '--name', filename]

            util.check_call(['artmangler', preset_name, os.path.join(input_dir, filenames[i]), '--no-resize'] + common_params)

            if save_frames:
                shutil.copy(filename, save_frames)

        magick(f'{tmp}/*png', name)
