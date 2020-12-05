import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.cli as cli
import noisemaker.generators as generators
import noisemaker.value as value


@click.command(help="""
        Noisemaker - Visual noise generator

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.freq_option()
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.time_option()
@cli.octaves_option()
@cli.octave_blending_option()
@cli.ridges_option()
@cli.post_ridges_option()
@cli.convolve_option()
@cli.deriv_option()
@cli.deriv_alpha_option()
@cli.post_deriv_option()
@cli.interp_option()
@cli.sin_option()
@cli.distrib_option()
@cli.corners_option()
@cli.mask_option()
@cli.mask_inverse_option()
@cli.glyph_map_option()
@cli.glyph_map_colorize_option()
@cli.glyph_map_zoom_option()
@cli.glyph_map_alpha_option()
@cli.composite_option()
@cli.composite_zoom_option()
@cli.lattice_drift_option()
@cli.vortex_option()
@cli.warp_option()
@cli.warp_octaves_option()
@cli.warp_interp_option()
@cli.warp_freq_option()
@cli.warp_map_option()
@cli.post_reflect_option()
@cli.reflect_option()
@cli.post_refract_option()
@cli.post_refract_y_from_offset_option()
@cli.refract_option()
@cli.refract_y_from_offset_option()
@cli.ripple_option()
@cli.ripple_freq_option()
@cli.ripple_kink_option()
@cli.reindex_option()
@cli.post_reindex_option()
@cli.reverb_option()
@cli.reverb_iterations_option()
@cli.clut_option()
@cli.clut_range_option()
@cli.clut_horizontal_option()
@cli.voronoi_option()
@cli.voronoi_metric_option()
@cli.voronoi_nth_option()
@cli.voronoi_alpha_option()
@cli.voronoi_refract_option()
@cli.voronoi_refract_y_from_offset_option()
@cli.voronoi_inverse_option()
@cli.dla_option()
@cli.dla_padding_option()
@cli.point_freq_option()
@cli.point_distrib_option()
@cli.point_corners_option()
@cli.point_generations_option()
@cli.point_drift_option()
@cli.wormhole_option()
@cli.wormhole_stride_option()
@cli.wormhole_kink_option()
@cli.worms_option()
@cli.worms_density_option()
@cli.worms_drunkenness_option()
@cli.worms_duration_option()
@cli.worms_stride_option()
@cli.worms_stride_deviation_option()
@cli.worms_kink_option()
@cli.worms_alpha_option()
@cli.erosion_worms_option()
@cli.sobel_option()
@cli.outline_option()
@cli.normals_option()
@cli.posterize_option()
@cli.bloom_option()
@cli.glitch_option()
@cli.vhs_option()
@cli.crt_option()
@cli.scan_error_option()
@cli.snow_option()
@cli.dither_option()
@cli.aberration_option()
@cli.light_leak_option()
@cli.vignette_option()
@cli.vignette_brightness_option()
@cli.pop_option()
@cli.shadow_option()
@cli.rgb_option()
@cli.hue_range_option()
@cli.hue_rotation_option()
@cli.post_hue_rotation_option()
@cli.saturation_option()
@cli.hue_distrib_option()
@cli.saturation_distrib_option()
@cli.post_saturation_option()
@cli.brightness_distrib_option()
@cli.input_dir_option()
@cli.wavelet_option()
@cli.density_map_option()
@cli.palette_option()
@cli.seed_option()
@cli.name_option()
@click.pass_context
def main(ctx, freq, width, height, channels, time, octaves, octave_blending,
         ridges, post_ridges, sin, wavelet, lattice_drift, vortex,
         warp, warp_octaves, warp_interp, warp_freq, warp_map,
         reflect, refract, refract_y_from_offset,
         reindex, reverb, reverb_iterations, post_reindex,
         post_reflect, post_refract, post_refract_y_from_offset,
         clut, clut_horizontal, clut_range, ripple, ripple_freq,
         ripple_kink, worms, worms_density, worms_drunkenness, worms_duration,
         worms_stride, worms_stride_deviation, worms_alpha, worms_kink,
         wormhole, wormhole_kink, wormhole_stride, sobel, outline,
         normals, post_deriv, deriv, deriv_alpha, interp, distrib,
         corners, mask, mask_inverse, glyph_map, glyph_map_colorize,
         glyph_map_zoom, glyph_map_alpha, composite, composite_zoom,
         posterize, erosion_worms, voronoi, voronoi_metric, voronoi_nth,
         voronoi_alpha, voronoi_refract, voronoi_refract_y_from_offset,
         voronoi_inverse, glitch, vhs, crt, scan_error, snow, dither,
         aberration, light_leak, vignette, vignette_brightness,
         pop, convolve, shadow, bloom, rgb, hue_range, hue_rotation,
         saturation, hue_distrib, saturation_distrib, post_hue_rotation,
         post_saturation, brightness_distrib, input_dir, dla, dla_padding,
         point_freq, point_distrib, point_corners, point_generations,
         point_drift, density, palette, seed, name):

    value.set_seed(seed)

    shape = [height, width, channels]

    tensor = generators.multires(freq=freq, shape=shape, time=time, octaves=octaves, octave_blending=octave_blending,
                                 ridges=ridges, post_ridges=post_ridges, sin=sin, wavelet=wavelet,
                                 lattice_drift=lattice_drift, reflect_range=reflect, refract_range=refract, reindex_range=reindex,
                                 refract_y_from_offset=refract_y_from_offset,
                                 with_reverb=reverb, reverb_iterations=reverb_iterations,
                                 post_reindex_range=post_reindex, post_reflect_range=post_reflect, post_refract_range=post_refract,
                                 post_refract_y_from_offset=post_refract_y_from_offset,
                                 ripple_range=ripple, ripple_freq=ripple_freq, ripple_kink=ripple_kink,
                                 clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                 with_worms=worms, worms_density=worms_density, worms_drunkenness=worms_drunkenness, worms_duration=worms_duration,
                                 worms_stride=worms_stride, worms_stride_deviation=worms_stride_deviation, worms_alpha=worms_alpha, worms_kink=worms_kink,
                                 with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride, with_erosion_worms=erosion_worms,
                                 with_voronoi=voronoi, voronoi_metric=voronoi_metric, voronoi_nth=voronoi_nth,
                                 voronoi_alpha=voronoi_alpha, voronoi_refract=voronoi_refract, voronoi_inverse=voronoi_inverse,
                                 voronoi_refract_y_from_offset=voronoi_refract_y_from_offset,
                                 with_dla=dla, dla_padding=dla_padding, point_freq=point_freq, point_distrib=point_distrib, point_corners=point_corners,
                                 point_generations=point_generations, point_drift=point_drift,
                                 with_outline=outline, with_sobel=sobel, with_normal_map=normals, post_deriv=post_deriv, deriv=deriv, deriv_alpha=deriv_alpha,
                                 spline_order=interp, distrib=distrib, corners=corners, mask=mask, mask_inverse=mask_inverse,
                                 with_glyph_map=glyph_map, glyph_map_colorize=glyph_map_colorize, glyph_map_zoom=glyph_map_zoom,
                                 glyph_map_alpha=glyph_map_alpha, with_composite=composite, composite_zoom=composite_zoom,
                                 warp_range=warp, warp_octaves=warp_octaves, warp_interp=warp_interp, warp_freq=warp_freq, warp_map=warp_map,
                                 posterize_levels=posterize, vortex_range=vortex,
                                 rgb=rgb, hue_range=hue_range, hue_rotation=hue_rotation, saturation=saturation, post_hue_rotation=post_hue_rotation,
                                 post_saturation=post_saturation, hue_distrib=hue_distrib, brightness_distrib=brightness_distrib,
                                 saturation_distrib=saturation_distrib, input_dir=input_dir, with_aberration=aberration, with_bloom=bloom, with_pop=pop,
                                 with_light_leak=light_leak, with_vignette=vignette, vignette_brightness=vignette_brightness, with_density_map=density,
                                 with_convolve=convolve, with_shadow=shadow, with_palette=palette, with_glitch=glitch, with_vhs=vhs,
                                 with_crt=crt, with_scan_error=scan_error, with_snow=snow, with_dither=dither)

    with tf.compat.v1.Session().as_default():
        save(tensor, name)

    print(name)
