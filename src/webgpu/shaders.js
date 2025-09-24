// WebGPU shader stubs. The previous pipeline has been removed and will be
// replaced in a future rewrite.  These placeholders satisfy existing imports
// while ensuring any attempted WebGPU execution falls back to CPU code paths.
const SHADER_PLACEHOLDER = null;

async function loadShaderSource(relativePath) {
  if (typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('fs/promises');
    const { fileURLToPath } = await import('url');
    const { dirname, join } = await import('path');
    const modulePath = fileURLToPath(import.meta.url);
    const shaderPath = join(dirname(modulePath), relativePath);
    return readFile(shaderPath, 'utf8');
  }

  const url = new URL(relativePath, import.meta.url);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load shader: ${url} (${response.status})`);
  }
  return response.text();
}

export const MULTIRES_WGSL = await loadShaderSource('./shaders/multires.wgsl');
export const MULTIRES_NORMALIZE_WGSL = await loadShaderSource('./shaders/multires_normalize.wgsl');
export const DERIVATIVE_WGSL = await loadShaderSource('./shaders/derivative.wgsl');
export const CLOUDS_WGSL = await loadShaderSource('./shaders/clouds.wgsl');
export const GLOWING_EDGES_STAGE1_WGSL = await loadShaderSource(
  './shaders/glowing_edges_stage1.wgsl',
);
export const GLOWING_EDGES_STAGE2_WGSL = await loadShaderSource(
  './shaders/glowing_edges_stage2.wgsl',
);
export const GLOWING_EDGES_STAGE3_WGSL = await loadShaderSource(
  './shaders/glowing_edges_stage3.wgsl',
);
export const GLOWING_EDGES_STAGE4_WGSL = await loadShaderSource(
  './shaders/glowing_edges_stage4.wgsl',
);

export const VALUE_WGSL = SHADER_PLACEHOLDER;
export const RESAMPLE_WGSL = await loadShaderSource('./shaders/resample.wgsl');
export const DOWNSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const BLEND_WGSL = SHADER_PLACEHOLDER;
export const BLEND_CONST_WGSL = SHADER_PLACEHOLDER;
export const SOBEL_WGSL = await loadShaderSource('./shaders/sobel.wgsl');
export const REFRACT_WGSL = await loadShaderSource('./shaders/refract.wgsl');
export const REFRACT_EFFECT_WGSL = await loadShaderSource(
  './shaders/refract_effect.wgsl',
);
export const CONVOLUTION_WGSL = await loadShaderSource(
  './shaders/convolution.wgsl',
);
export const CONVOLVE_WGSL = await loadShaderSource('./shaders/convolve.wgsl');
export const FXAA_WGSL = await loadShaderSource('./shaders/fxaa.wgsl');
export const NORMALIZE_WGSL = await loadShaderSource('./shaders/normalize.wgsl');
export const RGB_TO_HSV_WGSL = SHADER_PLACEHOLDER;
export const HSV_TO_RGB_WGSL = SHADER_PLACEHOLDER;
export const OCTAVE_COMBINE_WGSL = SHADER_PLACEHOLDER;
export const UPSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const VORONOI_WGSL = SHADER_PLACEHOLDER;
export const EROSION_WORMS_WGSL = await loadShaderSource(
  './shaders/erosion_worms.wgsl',
);
export const WORMS_WGSL = await loadShaderSource('./shaders/worms.wgsl');
export const REINDEX_WGSL = await loadShaderSource('./shaders/reindex.wgsl');
export const OFFSET_INDEX_WGSL = await loadShaderSource('./shaders/offset_index.wgsl');
export const RIPPLE_WGSL = await loadShaderSource('./shaders/ripple.wgsl');
export const COLOR_MAP_WGSL = await loadShaderSource('./shaders/color_map.wgsl');
export const VIGNETTE_WGSL = await loadShaderSource('./shaders/vignette.wgsl');
export const DITHER_WGSL = SHADER_PLACEHOLDER;
export const DENSITY_MAP_WGSL = await loadShaderSource(
  './shaders/density_map.wgsl',
);
export const GRAIN_WGSL = await loadShaderSource('./shaders/grain.wgsl');
export const SNOW_WGSL = await loadShaderSource('./shaders/snow.wgsl');
export const BLOOM_WGSL = await loadShaderSource('./shaders/bloom.wgsl');
export const ABERRATION_WGSL = await loadShaderSource(
  './shaders/aberration.wgsl',
);
export const LIGHT_LEAK_SCREEN_WGSL = await loadShaderSource(
  './shaders/light_leak_screen.wgsl',
);
export const ADJUST_BRIGHTNESS_WGSL = await loadShaderSource(
  './shaders/adjust_brightness.wgsl',
);
export const ADJUST_CONTRAST_WGSL = await loadShaderSource(
  './shaders/adjust_contrast.wgsl',
);
export const ADJUST_SATURATION_WGSL = await loadShaderSource(
  './shaders/adjust_saturation.wgsl',
);
export const ADJUST_HUE_WGSL = await loadShaderSource('./shaders/adjust_hue.wgsl');
export const SMOOTHSTEP_WGSL = await loadShaderSource('./shaders/smoothstep.wgsl');
export const ROTATE_WGSL = await loadShaderSource('./shaders/rotate.wgsl');
export const RIDGE_WGSL = await loadShaderSource('./shaders/ridge.wgsl');
export const SINE_WGSL = await loadShaderSource('./shaders/sine.wgsl');
export const GLYPH_MAP_WGSL = SHADER_PLACEHOLDER;
export const WARP_WGSL = await loadShaderSource('./shaders/warp.wgsl');
export const SPATTER_MASK_WGSL = await loadShaderSource(
  './shaders/spatter_mask.wgsl',
);
export const SCRATCHES_MASK_WGSL = await loadShaderSource(
  './shaders/scratches_mask.wgsl',
);
export const SCRATCHES_BLEND_WGSL = await loadShaderSource(
  './shaders/scratches_blend.wgsl',
);
export const GRIME_MASK_WGSL = await loadShaderSource('./shaders/grime_mask.wgsl');
export const GRIME_BLEND_WGSL = await loadShaderSource('./shaders/grime_blend.wgsl');
export const CONV_FEEDBACK_WGSL = await loadShaderSource(
  './shaders/conv_feedback.wgsl',
);
export const CENTER_MASK_WGSL = await loadShaderSource(
  './shaders/center_mask.wgsl',
);
export const PIXEL_SORT_WGSL = await loadShaderSource('./shaders/pixel_sort.wgsl');
export const POSTERIZE_WGSL = await loadShaderSource('./shaders/posterize.wgsl');
export const KALEIDO_WGSL = await loadShaderSource('./shaders/kaleido.wgsl');
export const GLITCH_WGSL = await loadShaderSource('./shaders/glitch.wgsl');
export const SCANLINE_ERROR_WGSL = await loadShaderSource(
  './shaders/scanline_error.wgsl',
);
export const SPOOKY_TICKER_WGSL = await loadShaderSource(
  './shaders/spooky_ticker.wgsl',
);
export const JPEG_DECIMATE_WGSL = await loadShaderSource('./shaders/jpeg_decimate.wgsl');
export const NORMAL_MAP_WGSL = await loadShaderSource('./shaders/normal_map.wgsl');
export const CRT_WGSL = await loadShaderSource('./shaders/crt.wgsl');
export const WOBBLE_WGSL = await loadShaderSource('./shaders/wobble.wgsl');
export const VORTEX_WGSL = await loadShaderSource('./shaders/vortex.wgsl');
export const WORMHOLE_WGSL = await loadShaderSource('./shaders/wormhole.wgsl');
export const DLA_WGSL = await loadShaderSource('./shaders/dla.wgsl');
export const REVERB_WGSL = await loadShaderSource('./shaders/reverb.wgsl');
export const VASELINE_BLUR_WGSL = await loadShaderSource(
  './shaders/vaseline_blur.wgsl',
);
export const VASELINE_MASK_WGSL = await loadShaderSource(
  './shaders/vaseline_mask.wgsl',
);
export const LENS_WARP_WGSL = await loadShaderSource('./shaders/lens_warp.wgsl');
export const LENS_DISTORTION_WGSL = await loadShaderSource(
  './shaders/lens_distortion.wgsl',
);
export const DEGAUSS_WGSL = SHADER_PLACEHOLDER;
export const TINT_WGSL = await loadShaderSource('./shaders/tint.wgsl');
export const PALETTE_WGSL = await loadShaderSource('./shaders/palette.wgsl');
export const TEXTURE_WGSL = await loadShaderSource('./shaders/texture.wgsl');
export const VHS_WGSL = await loadShaderSource('./shaders/vhs.wgsl');
export const UNARY_OP_WGSL = await loadShaderSource('./shaders/unary_op.wgsl');
export const BINARY_OP_WGSL = await loadShaderSource('./shaders/binary_op.wgsl');
export const GRAYSCALE_WGSL = await loadShaderSource('./shaders/grayscale.wgsl');
export const EXPAND_CHANNELS_WGSL = await loadShaderSource(
  './shaders/expand_channels.wgsl',
);
export const SOBEL_OPERATOR_FINALIZE_WGSL = await loadShaderSource(
  './shaders/sobel_operator_finalize.wgsl',
);
export const PROPORTIONAL_DOWNSAMPLE_WGSL = await loadShaderSource(
  './shaders/proportional_downsample.wgsl',
);
export const SCALE_TENSOR_WGSL = await loadShaderSource('./shaders/scale_tensor.wgsl');
export const SQUARE_CROP_WGSL = await loadShaderSource('./shaders/square_crop.wgsl');
export const SHADOW_WGSL = await loadShaderSource('./shaders/shadow.wgsl');
export const OUTLINE_WGSL = await loadShaderSource('./shaders/outline.wgsl');
export const SIMPLE_FRAME_WGSL = await loadShaderSource('./shaders/simple_frame.wgsl');
export const FRAME_WGSL = await loadShaderSource('./shaders/frame.wgsl');
export const INNER_TILE_WGSL = await loadShaderSource('./shaders/inner_tile.wgsl');
export const STRAY_HAIR_WGSL = await loadShaderSource('./shaders/stray_hair.wgsl');
export const FIBERS_WGSL = await loadShaderSource('./shaders/fibers.wgsl');
export const NEBULA_WGSL = await loadShaderSource('./shaders/nebula.wgsl');
export const EXPAND_TILE_WGSL = await loadShaderSource('./shaders/expand_tile.wgsl');
