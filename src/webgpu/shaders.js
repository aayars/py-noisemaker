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

export const VALUE_WGSL = SHADER_PLACEHOLDER;
export const RESAMPLE_WGSL = SHADER_PLACEHOLDER;
export const DOWNSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const BLEND_WGSL = SHADER_PLACEHOLDER;
export const BLEND_CONST_WGSL = SHADER_PLACEHOLDER;
export const SOBEL_WGSL = SHADER_PLACEHOLDER;
export const REFRACT_WGSL = SHADER_PLACEHOLDER;
export const CONVOLUTION_WGSL = SHADER_PLACEHOLDER;
export const FXAA_WGSL = SHADER_PLACEHOLDER;
export const NORMALIZE_WGSL = SHADER_PLACEHOLDER;
export const RGB_TO_HSV_WGSL = SHADER_PLACEHOLDER;
export const HSV_TO_RGB_WGSL = SHADER_PLACEHOLDER;
export const OCTAVE_COMBINE_WGSL = SHADER_PLACEHOLDER;
export const UPSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const VORONOI_WGSL = SHADER_PLACEHOLDER;
export const EROSION_WORMS_WGSL = SHADER_PLACEHOLDER;
export const WORMS_WGSL = SHADER_PLACEHOLDER;
export const REINDEX_WGSL = SHADER_PLACEHOLDER;
export const RIPPLE_WGSL = SHADER_PLACEHOLDER;
export const COLOR_MAP_WGSL = SHADER_PLACEHOLDER;
export const VIGNETTE_WGSL = SHADER_PLACEHOLDER;
export const DITHER_WGSL = SHADER_PLACEHOLDER;
export const GRAIN_WGSL = await loadShaderSource('./shaders/grain.wgsl');
export const ADJUST_BRIGHTNESS_WGSL = await loadShaderSource(
  './shaders/adjust_brightness.wgsl',
);
export const ADJUST_CONTRAST_WGSL = SHADER_PLACEHOLDER;
export const ROTATE_WGSL = SHADER_PLACEHOLDER;
export const GLYPH_MAP_WGSL = SHADER_PLACEHOLDER;
export const WARP_WGSL = SHADER_PLACEHOLDER;
export const SPATTER_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_BLEND_WGSL = SHADER_PLACEHOLDER;
export const GRIME_MASK_WGSL = SHADER_PLACEHOLDER;
export const GRIME_BLEND_WGSL = SHADER_PLACEHOLDER;
export const PIXEL_SORT_WGSL = SHADER_PLACEHOLDER;
export const KALEIDO_WGSL = await loadShaderSource('./shaders/kaleido.wgsl');
export const NORMAL_MAP_WGSL = SHADER_PLACEHOLDER;
export const CRT_WGSL = SHADER_PLACEHOLDER;
export const WOBBLE_WGSL = SHADER_PLACEHOLDER;
export const VORTEX_WGSL = SHADER_PLACEHOLDER;
export const WORMHOLE_WGSL = SHADER_PLACEHOLDER;
export const DLA_WGSL = SHADER_PLACEHOLDER;
export const REVERB_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_BLUR_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_MASK_WGSL = SHADER_PLACEHOLDER;
export const LENS_DISTORTION_WGSL = SHADER_PLACEHOLDER;
export const DEGAUSS_WGSL = SHADER_PLACEHOLDER;
export const TINT_WGSL = SHADER_PLACEHOLDER;
export const TEXTURE_WGSL = await loadShaderSource('./shaders/texture.wgsl');
export const VHS_WGSL = SHADER_PLACEHOLDER;
export const UNARY_OP_WGSL = SHADER_PLACEHOLDER;
export const BINARY_OP_WGSL = SHADER_PLACEHOLDER;
export const GRAYSCALE_WGSL = SHADER_PLACEHOLDER;
export const EXPAND_CHANNELS_WGSL = SHADER_PLACEHOLDER;
