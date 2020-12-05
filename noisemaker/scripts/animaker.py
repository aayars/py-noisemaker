import json
import shutil
import tempfile

import click

from noisemaker.constants import ValueDistribution

import noisemaker.cli as cli
import noisemaker.presets as presets
import noisemaker.util as util


@click.command(help="""
        Animaker - Animated Noisemaker loops

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.seed_option()
@cli.option('--effect-preset', type=click.Choice(["random"] + sorted(presets.EFFECTS_PRESETS)))
@cli.name_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option('--frame-count', type=int, default=30, help="How many frames total")
@cli.option('--watermark', type=str)
@cli.option('--overrides', type=str, help='A JSON dictionary containing preset overrides')
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, seed, effect_preset, name, save_frames, frame_count, watermark, overrides, preset_name):
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

    if overrides:
        overrides = json.loads(overrides)

    else:
        overrides = {}

    # Override defaults to animate better
    distrib = overrides.get('distrib', kwargs.get('distrib'))

    if distrib in (ValueDistribution.exp, 'exp'):
        overrides['distrib'] = 'simplex_exp'

    if distrib in (ValueDistribution.lognormal, 'lognormal'):
        overrides['distrib'] = 'simplex_pow_inv_1'

    elif distrib not in (
        # ValueDistribution.ones, 'ones',
        ValueDistribution.simplex_exp, 'simplex_exp',
        ValueDistribution.column_index, 'column_index',
        ValueDistribution.row_index, 'row_index',
        ValueDistribution.fastnoise, 'fastnoise',
        ValueDistribution.fastnoise_exp, 'fastnoise_exp',
    ):
        overrides['distrib'] = 'simplex'

    if 'point_drift' not in kwargs:
        overrides['point_drift'] = 0.25

    speed = overrides.get("speed", kwargs.get("speed", 0.25))

    # Adjust speed for length of clip. A "normal" length is 30 frames.
    overrides['speed'] = speed * (frame_count / 30.0)

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frame_count):
            filename = f'{tmp}/{i:04d}.png'

            common_params = ['--seed', str(seed or 1),
                             '--overrides', json.dumps(overrides),
                             '--time', f'{i/frame_count:0.4f}',
                             '--name', filename]

            util.check_call(['artmaker', preset_name,
                             '--height', str(height),
                             '--width', str(width)] + common_params)

            if effect_preset:
                util.check_call(['artmangler', effect_preset, filename, '--no-resize'] + common_params)

            if save_frames:
                shutil.copy(filename, save_frames)

            if watermark:
                util.watermark(watermark, filename)

        if name.endswith(".mp4"):
            # when you want something done right
            util.check_call(['ffmpeg',
                             '-y',  # overwrite existing
                             '-framerate', '30',
                             '-i', f'{tmp}/%04d.png',
                             '-c:v', 'libx264',  # because this is what twitter wants
                             '-pix_fmt', 'yuv420p',  # because this is what twitter wants
                             '-b:v', '1700000',  # maximum allowed bitrate for 720x720 (2048k), minus some encoder overhead
                             '-s', '720x720',  # a twitter-recommended size
                             name])

        else:
            util.magick(f'{tmp}/*png', name)
