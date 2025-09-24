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
export const REFRACT_WGSL = SHADER_PLACEHOLDER;
export const CONVOLUTION_WGSL = SHADER_PLACEHOLDER;
export const FXAA_WGSL = await loadShaderSource('./shaders/fxaa.wgsl');
export const NORMALIZE_WGSL = await loadShaderSource('./shaders/normalize.wgsl');
export const RGB_TO_HSV_WGSL = SHADER_PLACEHOLDER;
export const HSV_TO_RGB_WGSL = SHADER_PLACEHOLDER;
export const OCTAVE_COMBINE_WGSL = SHADER_PLACEHOLDER;
export const UPSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const VORONOI_WGSL = SHADER_PLACEHOLDER;
export const EROSION_WORMS_WGSL = SHADER_PLACEHOLDER;
export const WORMS_WGSL = SHADER_PLACEHOLDER;
export const REINDEX_WGSL = SHADER_PLACEHOLDER;
export const RIPPLE_WGSL = await loadShaderSource('./shaders/ripple.wgsl');
export const COLOR_MAP_WGSL = SHADER_PLACEHOLDER;
export const VIGNETTE_WGSL = await loadShaderSource('./shaders/vignette.wgsl');
export const DITHER_WGSL = SHADER_PLACEHOLDER;
export const GRAIN_WGSL = await loadShaderSource('./shaders/grain.wgsl');
export const BLOOM_WGSL = await loadShaderSource('./shaders/bloom.wgsl');
export const ABERRATION_WGSL = await loadShaderSource(
  './shaders/aberration.wgsl',
);
export const ADJUST_BRIGHTNESS_WGSL = await loadShaderSource(
  './shaders/adjust_brightness.wgsl',
);
export const ADJUST_CONTRAST_WGSL = await loadShaderSource(
  './shaders/adjust_contrast.wgsl',
);
export const SMOOTHSTEP_WGSL = await loadShaderSource('./shaders/smoothstep.wgsl');
export const ROTATE_WGSL = await loadShaderSource('./shaders/rotate.wgsl');
export const GLYPH_MAP_WGSL = SHADER_PLACEHOLDER;
export const WARP_WGSL = SHADER_PLACEHOLDER;
export const SPATTER_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_BLEND_WGSL = SHADER_PLACEHOLDER;
export const GRIME_MASK_WGSL = await loadShaderSource('./shaders/grime_mask.wgsl');
export const GRIME_BLEND_WGSL = await loadShaderSource('./shaders/grime_blend.wgsl');
export const PIXEL_SORT_WGSL = SHADER_PLACEHOLDER;
export const POSTERIZE_WGSL = await loadShaderSource('./shaders/posterize.wgsl');
export const KALEIDO_WGSL = await loadShaderSource('./shaders/kaleido.wgsl');
export const GLITCH_WGSL = await loadShaderSource('./shaders/glitch.wgsl');
export const NORMAL_MAP_WGSL = SHADER_PLACEHOLDER;
export const CRT_WGSL = SHADER_PLACEHOLDER;
export const WOBBLE_WGSL = await loadShaderSource('./shaders/wobble.wgsl');
export const VORTEX_WGSL = SHADER_PLACEHOLDER;
export const WORMHOLE_WGSL = SHADER_PLACEHOLDER;
export const DLA_WGSL = await loadShaderSource('./shaders/dla.wgsl');
export const REVERB_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_BLUR_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_MASK_WGSL = SHADER_PLACEHOLDER;
export const LENS_DISTORTION_WGSL = await loadShaderSource(
  './shaders/lens_distortion.wgsl',
);
export const DEGAUSS_WGSL = SHADER_PLACEHOLDER;
export const TINT_WGSL = await loadShaderSource('./shaders/tint.wgsl');
export const TEXTURE_WGSL = await loadShaderSource('./shaders/texture.wgsl');
export const VHS_WGSL = SHADER_PLACEHOLDER;
export const UNARY_OP_WGSL = SHADER_PLACEHOLDER;
export const BINARY_OP_WGSL = SHADER_PLACEHOLDER;
export const GRAYSCALE_WGSL = SHADER_PLACEHOLDER;
export const EXPAND_CHANNELS_WGSL = SHADER_PLACEHOLDER;
export const SOBEL_OPERATOR_FINALIZE_WGSL = await loadShaderSource(
  './shaders/sobel_operator_finalize.wgsl',
);
export const PROPORTIONAL_DOWNSAMPLE_WGSL = await loadShaderSource(
  './shaders/proportional_downsample.wgsl',
);
export const SCALE_TENSOR_WGSL = await loadShaderSource('./shaders/scale_tensor.wgsl');
