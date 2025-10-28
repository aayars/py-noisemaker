// Import all effect metadata
import multiresMeta from '../../shaders/generators/multires/meta.json' with { type: 'json' };
import aberrationMeta from '../../shaders/effects/aberration/meta.json' with { type: 'json' };
import adjustBrightnessMeta from '../../shaders/effects/adjust_brightness/meta.json' with { type: 'json' };
import adjustContrastMeta from '../../shaders/effects/adjust_contrast/meta.json' with { type: 'json' };
import adjustHueMeta from '../../shaders/effects/adjust_hue/meta.json' with { type: 'json' };
import adjustSaturationMeta from '../../shaders/effects/adjust_saturation/meta.json' with { type: 'json' };
import blurMeta from '../../shaders/effects/blur/meta.json' with { type: 'json' };
import bloomMeta from '../../shaders/effects/bloom/meta.json' with { type: 'json' };
import cloudsMeta from '../../shaders/effects/clouds/meta.json' with { type: 'json' };
import colorMapMeta from '../../shaders/effects/color_map/meta.json' with { type: 'json' };
import convFeedbackMeta from '../../shaders/effects/conv_feedback/meta.json' with { type: 'json' };
import convolveMeta from '../../shaders/effects/convolve/meta.json' with { type: 'json' };
import crtMeta from '../../shaders/effects/crt/meta.json' with { type: 'json' };
import degaussMeta from '../../shaders/effects/degauss/meta.json' with { type: 'json' };
import densityMapMeta from '../../shaders/effects/density_map/meta.json' with { type: 'json' };
import derivativeMeta from '../../shaders/effects/derivative/meta.json' with { type: 'json' };
import dlaMeta from '../../shaders/effects/dla/meta.json' with { type: 'json' };
import { additionalPasses as dlaPasses } from '../../shaders/effects/dla/effect.js';
import erosionWormsMeta from '../../shaders/effects/erosion_worms/meta.json' with { type: 'json' };
import { additionalPasses as erosionPasses } from '../../shaders/effects/erosion_worms/effect.js';
import falseColorMeta from '../../shaders/effects/false_color/meta.json' with { type: 'json' };
import fibersMeta from '../../shaders/effects/fibers/meta.json' with { type: 'json' };
import frameMeta from '../../shaders/effects/frame/meta.json' with { type: 'json' };
import fxaaMeta from '../../shaders/effects/fxaa/meta.json' with { type: 'json' };
import glowingEdgesMeta from '../../shaders/effects/glowing_edges/meta.json' with { type: 'json' };
import glyphMapMeta from '../../shaders/effects/glyph_map/meta.json' with { type: 'json' };
import grainMeta from '../../shaders/effects/grain/meta.json' with { type: 'json' };
import grimeMeta from '../../shaders/effects/grime/meta.json' with { type: 'json' };
import jpegDecimateMeta from '../../shaders/effects/jpeg_decimate/meta.json' with { type: 'json' };
import kaleidoMeta from '../../shaders/effects/kaleido/meta.json' with { type: 'json' };
import lensDistortionMeta from '../../shaders/effects/lens_distortion/meta.json' with { type: 'json' };
import lensWarpMeta from '../../shaders/effects/lens_warp/meta.json' with { type: 'json' };
import lightLeakMeta from '../../shaders/effects/light_leak/meta.json' with { type: 'json' };
import lowpolyMeta from '../../shaders/effects/lowpoly/meta.json' with { type: 'json' };
import nebulaMeta from '../../shaders/effects/nebula/meta.json' with { type: 'json' };
import normalMapMeta from '../../shaders/effects/normal_map/meta.json' with { type: 'json' };
import normalizeMeta from '../../shaders/effects/normalize/meta.json' with { type: 'json' };
import { additionalPasses as normalizePasses } from '../../shaders/effects/normalize/effect.js';
import onScreenDisplayMeta from '../../shaders/effects/on_screen_display/meta.json' with { type: 'json' };
import outlineMeta from '../../shaders/effects/outline/meta.json' with { type: 'json' };
import paletteMeta from '../../shaders/effects/palette/meta.json' with { type: 'json' };
import pixelSortMeta from '../../shaders/effects/pixel_sort/meta.json' with { type: 'json' };
import posterizeMeta from '../../shaders/effects/posterize/meta.json' with { type: 'json' };
import refractMeta from '../../shaders/effects/refract/meta.json' with { type: 'json' };
import reindexMeta from '../../shaders/effects/reindex/meta.json' with { type: 'json' };
import reverbMeta from '../../shaders/effects/reverb/meta.json' with { type: 'json' };
import ridgeMeta from '../../shaders/effects/ridge/meta.json' with { type: 'json' };
import rippleMeta from '../../shaders/effects/ripple/meta.json' with { type: 'json' };
import rotateMeta from '../../shaders/effects/rotate/meta.json' with { type: 'json' };
import scanlineErrorMeta from '../../shaders/effects/scanline_error/meta.json' with { type: 'json' };
import scratchesMeta from '../../shaders/effects/scratches/meta.json' with { type: 'json' };
import shadowMeta from '../../shaders/effects/shadow/meta.json' with { type: 'json' };
import simpleFrameMeta from '../../shaders/effects/simple_frame/meta.json' with { type: 'json' };
import sineMeta from '../../shaders/effects/sine/meta.json' with { type: 'json' };
import sketchMeta from '../../shaders/effects/sketch/meta.json' with { type: 'json' };
import smoothstepMeta from '../../shaders/effects/smoothstep/meta.json' with { type: 'json' };
import snowMeta from '../../shaders/effects/snow/meta.json' with { type: 'json' };
import sobelMeta from '../../shaders/effects/sobel/meta.json' with { type: 'json' };
import spatterMeta from '../../shaders/effects/spatter/meta.json' with { type: 'json' };
import spookyTickerMeta from '../../shaders/effects/spooky_ticker/meta.json' with { type: 'json' };
import strayHairMeta from '../../shaders/effects/stray_hair/meta.json' with { type: 'json' };
import textureMeta from '../../shaders/effects/texture/meta.json' with { type: 'json' };
import tintMeta from '../../shaders/effects/tint/meta.json' with { type: 'json' };
import vaselineMeta from '../../shaders/effects/vaseline/meta.json' with { type: 'json' };
import vhsMeta from '../../shaders/effects/vhs/meta.json' with { type: 'json' };
import vignetteMeta from '../../shaders/effects/vignette/meta.json' with { type: 'json' };
import voronoiMeta from '../../shaders/effects/voronoi/meta.json' with { type: 'json' };
import vortexMeta from '../../shaders/effects/vortex/meta.json' with { type: 'json' };
import warpMeta from '../../shaders/effects/warp/meta.json' with { type: 'json' };
import wobbleMeta from '../../shaders/effects/wobble/meta.json' with { type: 'json' };
import wormholeMeta from '../../shaders/effects/wormhole/meta.json' with { type: 'json' };
import { additionalPasses as wormholePasses } from '../../shaders/effects/wormhole/effect.js';
import wormsMeta from '../../shaders/effects/worms/meta.json' with { type: 'json' };
import { additionalPasses as wormsPasses } from '../../shaders/effects/worms/effect.js';

// Convert effect metadata to shader descriptor format
function metaToDescriptor(meta) {
  return {
    id: meta.id,
    label: meta.label || `${meta.id}.wgsl`,
    stage: meta.stage || 'compute',
    entryPoint: meta.shader?.entryPoint || 'main',
    url: meta.shader?.url || `/shaders/effects/${meta.id}/${meta.id}.wgsl`,
    resources: meta.resources || {}
  };
}

// Build manifest from metadata
const SHADER_MANIFEST = {
  multires: metaToDescriptor(multiresMeta),
  aberration: metaToDescriptor(aberrationMeta),
  adjust_brightness: metaToDescriptor(adjustBrightnessMeta),
  adjust_contrast: metaToDescriptor(adjustContrastMeta),
  adjust_hue: metaToDescriptor(adjustHueMeta),
  adjust_saturation: metaToDescriptor(adjustSaturationMeta),
  blur: metaToDescriptor(blurMeta),
  bloom: metaToDescriptor(bloomMeta),
  clouds: metaToDescriptor(cloudsMeta),
  color_map: metaToDescriptor(colorMapMeta),
  conv_feedback: metaToDescriptor(convFeedbackMeta),
  convolve: metaToDescriptor(convolveMeta),
  crt: metaToDescriptor(crtMeta),
  degauss: metaToDescriptor(degaussMeta),
  density_map: metaToDescriptor(densityMapMeta),
  derivative: metaToDescriptor(derivativeMeta),
  dla: metaToDescriptor(dlaMeta),
  ...dlaPasses,
  erosion_worms: metaToDescriptor(erosionWormsMeta),
  ...erosionPasses,
  false_color: metaToDescriptor(falseColorMeta),
  fibers: metaToDescriptor(fibersMeta),
  frame: metaToDescriptor(frameMeta),
  fxaa: metaToDescriptor(fxaaMeta),
  glowing_edges: metaToDescriptor(glowingEdgesMeta),
  glyph_map: metaToDescriptor(glyphMapMeta),
  grain: metaToDescriptor(grainMeta),
  grime: metaToDescriptor(grimeMeta),
  jpeg_decimate: metaToDescriptor(jpegDecimateMeta),
  kaleido: metaToDescriptor(kaleidoMeta),
  lens_distortion: metaToDescriptor(lensDistortionMeta),
  lens_warp: metaToDescriptor(lensWarpMeta),
  light_leak: metaToDescriptor(lightLeakMeta),
  lowpoly: metaToDescriptor(lowpolyMeta),
  nebula: metaToDescriptor(nebulaMeta),
  normal_map: metaToDescriptor(normalMapMeta),
  normalize: metaToDescriptor(normalizeMeta),
  ...normalizePasses,
  on_screen_display: metaToDescriptor(onScreenDisplayMeta),
  outline: metaToDescriptor(outlineMeta),
  palette: metaToDescriptor(paletteMeta),
  pixel_sort: metaToDescriptor(pixelSortMeta),
  posterize: metaToDescriptor(posterizeMeta),
  refract: metaToDescriptor(refractMeta),
  reindex: metaToDescriptor(reindexMeta),
  reverb: metaToDescriptor(reverbMeta),
  ridge: metaToDescriptor(ridgeMeta),
  ripple: metaToDescriptor(rippleMeta),
  rotate: metaToDescriptor(rotateMeta),
  scanline_error: metaToDescriptor(scanlineErrorMeta),
  scratches: metaToDescriptor(scratchesMeta),
  shadow: metaToDescriptor(shadowMeta),
  simple_frame: metaToDescriptor(simpleFrameMeta),
  sine: metaToDescriptor(sineMeta),
  sketch: metaToDescriptor(sketchMeta),
  smoothstep: metaToDescriptor(smoothstepMeta),
  snow: metaToDescriptor(snowMeta),
  sobel: metaToDescriptor(sobelMeta),
  spatter: metaToDescriptor(spatterMeta),
  spooky_ticker: metaToDescriptor(spookyTickerMeta),
  stray_hair: metaToDescriptor(strayHairMeta),
  texture: metaToDescriptor(textureMeta),
  tint: metaToDescriptor(tintMeta),
  vaseline: metaToDescriptor(vaselineMeta),
  vhs: metaToDescriptor(vhsMeta),
  vignette: metaToDescriptor(vignetteMeta),
  voronoi: metaToDescriptor(voronoiMeta),
  vortex: metaToDescriptor(vortexMeta),
  warp: metaToDescriptor(warpMeta),
  wobble: metaToDescriptor(wobbleMeta),
  wormhole: metaToDescriptor(wormholeMeta),
  worms: metaToDescriptor(wormsMeta),
  ...wormholePasses,
  ...wormsPasses,
};

function getShaderDescriptor(shaderId) {
  const descriptor = SHADER_MANIFEST[shaderId];
  if (!descriptor) {
    throw new Error(`Unknown shader identifier: ${shaderId}`);
  }
  return descriptor;
}

function parseWorkgroupSize(code) {
  const regex = /@workgroup_size\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/;
  const match = code.match(regex);
  if (!match) {
    console.warn('No @workgroup_size found in shader. Defaulting to [8, 8, 1].');
    return [8, 8, 1];
  }
  const [, x, y, z] = match;
  const values = [Number.parseInt(x, 10), Number.parseInt(y, 10), Number.parseInt(z, 10)];
  return values.map((value, index) => {
    if (Number.isNaN(value) || value <= 0) {
      return index === 2 ? 1 : 8;
    }
    return value;
  });
}

function normalizeStorageAccess(access) {
  if (!access) {
    return 'write-only';
  }
  const trimmed = access.trim().toLowerCase();
  if (trimmed === 'write') {
    return 'write-only';
  }
  if (trimmed === 'read') {
    return 'read-only';
  }
  if (trimmed === 'read_write' || trimmed === 'read-write') {
    return 'read-write';
  }
  return 'write-only';
}

function classifyBufferResource(qualifiers) {
  if (!qualifiers) {
    return null;
  }
  const parts = qualifiers.split(',').map((part) => part.trim());
  if (parts[0] === 'uniform') {
    return { resource: 'uniformBuffer', access: 'read-only' };
  }
  if (parts[0] === 'storage') {
    const access = normalizeStorageAccess(parts[1] ?? 'read_write');
    if (access === 'read-only') {
      return { resource: 'readOnlyStorageBuffer', access };
    }
    return { resource: 'storageBuffer', access };
  }
  return null;
}

function parseShaderBindings(code) {
  const bindings = [];
  // Normalize: strip comments and collapse whitespace to make regex robust to newlines and inline comments
  const normalized = code
    .replace(/\/\/.*$/gm, '') // remove line comments
    .replace(/\/\*[\s\S]*?\*\//g, '') // remove block comments
    .replace(/\s+/g, ' '); // collapse whitespace

  const regex = /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var(?:\s*<([^>]+)>)?\s+([A-Za-z0-9_]+)\s*:\s*([^;]+);/g;
  let match;
  while ((match = regex.exec(normalized)) !== null) {
    const [, group, binding, qualifiers, name, typeExpression] = match;
    const groupIndex = Number.parseInt(group, 10);
    const bindingIndex = Number.parseInt(binding, 10);
    const rawType = typeExpression.trim();
    const bindingInfo = {
      group: groupIndex,
      binding: bindingIndex,
      name,
      rawType,
      resource: 'unknown',
    };

    if (rawType.startsWith('texture_storage')) {
      const textureMatch = rawType.match(/texture_storage_[a-z0-9_]+<\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_-]+)\s*>/i);
      if (textureMatch) {
        bindingInfo.resource = 'storageTexture';
        bindingInfo.storageTextureFormat = textureMatch[1];
        bindingInfo.storageTextureAccess = normalizeStorageAccess(textureMatch[2]);
      }
    } else if (rawType.startsWith('texture_')) {
      bindingInfo.resource = 'sampledTexture';
    } else if (rawType.startsWith('sampler')) {
      bindingInfo.resource = 'sampler';
    } else {
      const bufferInfo = classifyBufferResource(qualifiers ?? '');
      if (bufferInfo) {
        bindingInfo.resource = bufferInfo.resource;
        bindingInfo.bufferAccess = bufferInfo.access;
      }
    }

    bindings.push(bindingInfo);
  }

  return bindings.sort((a, b) => {
    if (a.group !== b.group) {
      return a.group - b.group;
    }
    return a.binding - b.binding;
  });
}

function parseShaderMetadata(code) {
  return {
    workgroupSize: parseWorkgroupSize(code),
    bindings: parseShaderBindings(code),
  };
}

export {
  SHADER_MANIFEST,
  getShaderDescriptor,
  parseWorkgroupSize,
  parseShaderBindings,
  parseShaderMetadata,
};
