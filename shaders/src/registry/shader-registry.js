import multiresMeta from '../../generators/multires/meta.json' with { type: 'json' };
import aberrationMeta from '../../effects/aberration/meta.json' with { type: 'json' };
import adjustBrightnessMeta from '../../effects/adjust_brightness/meta.json' with { type: 'json' };
import adjustContrastMeta from '../../effects/adjust_contrast/meta.json' with { type: 'json' };
import adjustHueMeta from '../../effects/adjust_hue/meta.json' with { type: 'json' };
import adjustSaturationMeta from '../../effects/adjust_saturation/meta.json' with { type: 'json' };
import blurMeta from '../../effects/blur/meta.json' with { type: 'json' };
import bloomMeta from '../../effects/bloom/meta.json' with { type: 'json' };
import cloudsMeta from '../../effects/clouds/meta.json' with { type: 'json' };
import colorMapMeta from '../../effects/color_map/meta.json' with { type: 'json' };
import convFeedbackMeta from '../../effects/conv_feedback/meta.json' with { type: 'json' };
import convolveMeta from '../../effects/convolve/meta.json' with { type: 'json' };
import crtMeta from '../../effects/crt/meta.json' with { type: 'json' };
import degaussMeta from '../../effects/degauss/meta.json' with { type: 'json' };
import densityMapMeta from '../../effects/density_map/meta.json' with { type: 'json' };
import derivativeMeta from '../../effects/derivative/meta.json' with { type: 'json' };
import dlaMeta from '../../effects/dla/meta.json' with { type: 'json' };
import { additionalPasses as dlaPasses } from '../../effects/dla/effect.js';
import erosionWormsMeta from '../../effects/erosion_worms/meta.json' with { type: 'json' };
import { additionalPasses as erosionPasses } from '../../effects/erosion_worms/effect.js';
import falseColorMeta from '../../effects/false_color/meta.json' with { type: 'json' };
import fibersMeta from '../../effects/fibers/meta.json' with { type: 'json' };
import frameMeta from '../../effects/frame/meta.json' with { type: 'json' };
import fxaaMeta from '../../effects/fxaa/meta.json' with { type: 'json' };
import glowingEdgesMeta from '../../effects/glowing_edges/meta.json' with { type: 'json' };
import glyphMapMeta from '../../effects/glyph_map/meta.json' with { type: 'json' };
import grainMeta from '../../effects/grain/meta.json' with { type: 'json' };
import grimeMeta from '../../effects/grime/meta.json' with { type: 'json' };
import jpegDecimateMeta from '../../effects/jpeg_decimate/meta.json' with { type: 'json' };
import kaleidoMeta from '../../effects/kaleido/meta.json' with { type: 'json' };
import lensDistortionMeta from '../../effects/lens_distortion/meta.json' with { type: 'json' };
import lensWarpMeta from '../../effects/lens_warp/meta.json' with { type: 'json' };
import lightLeakMeta from '../../effects/light_leak/meta.json' with { type: 'json' };
import lowpolyMeta from '../../effects/lowpoly/meta.json' with { type: 'json' };
import nebulaMeta from '../../effects/nebula/meta.json' with { type: 'json' };
import normalMapMeta from '../../effects/normal_map/meta.json' with { type: 'json' };
import normalizeMeta from '../../effects/normalize/meta.json' with { type: 'json' };
import { additionalPasses as normalizePasses } from '../../effects/normalize/effect.js';
import { additionalPasses as convFeedbackPasses } from '../../effects/conv_feedback/effect.js';
import onScreenDisplayMeta from '../../effects/on_screen_display/meta.json' with { type: 'json' };
import outlineMeta from '../../effects/outline/meta.json' with { type: 'json' };
import paletteMeta from '../../effects/palette/meta.json' with { type: 'json' };
import pixelSortMeta from '../../effects/pixel_sort/meta.json' with { type: 'json' };
import posterizeMeta from '../../effects/posterize/meta.json' with { type: 'json' };
import refractMeta from '../../effects/refract/meta.json' with { type: 'json' };
import reindexMeta from '../../effects/reindex/meta.json' with { type: 'json' };
import reverbMeta from '../../effects/reverb/meta.json' with { type: 'json' };
import ridgeMeta from '../../effects/ridge/meta.json' with { type: 'json' };
import rippleMeta from '../../effects/ripple/meta.json' with { type: 'json' };
import rotateMeta from '../../effects/rotate/meta.json' with { type: 'json' };
import scanlineErrorMeta from '../../effects/scanline_error/meta.json' with { type: 'json' };
import scratchesMeta from '../../effects/scratches/meta.json' with { type: 'json' };
import shadowMeta from '../../effects/shadow/meta.json' with { type: 'json' };
import { additionalPasses as shadowPasses } from '../../effects/shadow/effect.js';
import simpleFrameMeta from '../../effects/simple_frame/meta.json' with { type: 'json' };
import sineMeta from '../../effects/sine/meta.json' with { type: 'json' };
import sketchMeta from '../../effects/sketch/meta.json' with { type: 'json' };
import smoothstepMeta from '../../effects/smoothstep/meta.json' with { type: 'json' };
import snowMeta from '../../effects/snow/meta.json' with { type: 'json' };
import sobelMeta from '../../effects/sobel/meta.json' with { type: 'json' };
import { additionalPasses as sobelPasses } from '../../effects/sobel/effect.js?v=2';
import spatterMeta from '../../effects/spatter/meta.json' with { type: 'json' };
import { additionalPasses as spatterPasses } from '../../effects/spatter/effect.js';
import spookyTickerMeta from '../../effects/spooky_ticker/meta.json' with { type: 'json' };
import strayHairMeta from '../../effects/stray_hair/meta.json' with { type: 'json' };
import textureMeta from '../../effects/texture/meta.json' with { type: 'json' };
import tintMeta from '../../effects/tint/meta.json' with { type: 'json' };
import vaselineMeta from '../../effects/vaseline/meta.json' with { type: 'json' };
import vhsMeta from '../../effects/vhs/meta.json' with { type: 'json' };
import vignetteMeta from '../../effects/vignette/meta.json' with { type: 'json' };
import voronoiMeta from '../../effects/voronoi/meta.json' with { type: 'json' };
import vortexMeta from '../../effects/vortex/meta.json' with { type: 'json' };
import warpMeta from '../../effects/warp/meta.json' with { type: 'json' };
import wobbleMeta from '../../effects/wobble/meta.json' with { type: 'json' };
import wormholeMeta from '../../effects/wormhole/meta.json' with { type: 'json' };
import { additionalPasses as wormholePasses } from '../../effects/wormhole/effect.js';
import wormsMeta from '../../effects/worms/meta.json' with { type: 'json' };
import { additionalPasses as wormsPasses } from '../../effects/worms/effect.js';

function metaToDescriptor(meta) {
  return {
    id: meta.id,
    label: meta.label || `${meta.id}.wgsl`,
    stage: meta.stage || 'compute',
    entryPoint: meta.shader?.entryPoint || 'main',
    url: meta.shader?.url || `/shaders/effects/${meta.id}/${meta.id}.wgsl`,
    resources: meta.resources || {},
  };
}

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
  ...convFeedbackPasses,
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
  ...shadowPasses,
  simple_frame: metaToDescriptor(simpleFrameMeta),
  sine: metaToDescriptor(sineMeta),
  sketch: metaToDescriptor(sketchMeta),
  smoothstep: metaToDescriptor(smoothstepMeta),
  snow: metaToDescriptor(snowMeta),
  sobel: metaToDescriptor(sobelMeta),
  ...sobelPasses,
  spatter: metaToDescriptor(spatterMeta),
  ...spatterPasses,
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
  const normalized = code
    .replace(/\/\/.*$/gm, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\s+/g, ' ');

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
