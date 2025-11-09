import {
  getShaderDescriptor,
  parseShaderMetadata,
} from './shader-registry.js?step8';
import EffectManager from './effect-manager.js';
import aberrationMetadata from '../../shaders/effects/aberration/meta.json' with { type: 'json' };
import adjustBrightnessMetadata from '../../shaders/effects/adjust_brightness/meta.json' with { type: 'json' };
import adjustContrastMetadata from '../../shaders/effects/adjust_contrast/meta.json' with { type: 'json' };
import adjustHueMetadata from '../../shaders/effects/adjust_hue/meta.json' with { type: 'json' };
import adjustSaturationMetadata from '../../shaders/effects/adjust_saturation/meta.json' with { type: 'json' };
import blurMetadata from '../../shaders/effects/blur/meta.json' with { type: 'json' };
import bloomMetadata from '../../shaders/effects/bloom/meta.json' with { type: 'json' };
import cloudsMetadata from '../../shaders/effects/clouds/meta.json' with { type: 'json' };
import colorMapMetadata from '../../shaders/effects/color_map/meta.json' with { type: 'json' };
import convFeedbackMetadata from '../../shaders/effects/conv_feedback/meta.json' with { type: 'json' };
import convolveMetadata from '../../shaders/effects/convolve/meta.json' with { type: 'json' };
import crtMetadata from '../../shaders/effects/crt/meta.json' with { type: 'json' };
import degaussMetadata from '../../shaders/effects/degauss/meta.json' with { type: 'json' };
import densityMapMetadata from '../../shaders/effects/density_map/meta.json' with { type: 'json' };
import derivativeMetadata from '../../shaders/effects/derivative/meta.json' with { type: 'json' };
import dlaMetadata from '../../shaders/effects/dla/meta.json' with { type: 'json' };
import erosionWormsMetadata from '../../shaders/effects/erosion_worms/meta.json?v=3' with { type: 'json' };
import falseColorMetadata from '../../shaders/effects/false_color/meta.json' with { type: 'json' };
import fibersMetadata from '../../shaders/effects/fibers/meta.json' with { type: 'json' };
import frameMetadata from '../../shaders/effects/frame/meta.json' with { type: 'json' };
import posterizeMetadata from '../../shaders/effects/posterize/meta.json' with { type: 'json' };
import paletteMetadata from '../../shaders/effects/palette/meta.json' with { type: 'json' };
import pixelSortMetadata from '../../shaders/effects/pixel_sort/meta.json' with { type: 'json' };
import refractMetadata from '../../shaders/effects/refract/meta.json' with { type: 'json' };
import reindexMetadata from '../../shaders/effects/reindex/meta.json' with { type: 'json' };
import reverbMetadata from '../../shaders/effects/reverb/meta.json' with { type: 'json' };
import ridgeMetadata from '../../shaders/effects/ridge/meta.json' with { type: 'json' };
import rippleMetadata from '../../shaders/effects/ripple/meta.json' with { type: 'json' };
import rotateMetadata from '../../shaders/effects/rotate/meta.json' with { type: 'json' };
import scanlineErrorMetadata from '../../shaders/effects/scanline_error/meta.json' with { type: 'json' };
import scratchesMetadata from '../../shaders/effects/scratches/meta.json' with { type: 'json' };
import shadowMetadata from '../../shaders/effects/shadow/meta.json' with { type: 'json' };
import simpleFrameMetadata from '../../shaders/effects/simple_frame/meta.json' with { type: 'json' };
import sineMetadata from '../../shaders/effects/sine/meta.json' with { type: 'json' };
import sketchMetadata from '../../shaders/effects/sketch/meta.json' with { type: 'json' };
import smoothstepMetadata from '../../shaders/effects/smoothstep/meta.json' with { type: 'json' };
import snowMetadata from '../../shaders/effects/snow/meta.json' with { type: 'json' };
import sobelMetadata from '../../shaders/effects/sobel/meta.json' with { type: 'json' };
import spatterMetadata from '../../shaders/effects/spatter/meta.json' with { type: 'json' };
import spookyTickerMetadata from '../../shaders/effects/spooky_ticker/meta.json' with { type: 'json' };
import strayHairMetadata from '../../shaders/effects/stray_hair/meta.json' with { type: 'json' };
import textureMetadata from '../../shaders/effects/texture/meta.json' with { type: 'json' };
import tintMetadata from '../../shaders/effects/tint/meta.json' with { type: 'json' };
import vaselineMetadata from '../../shaders/effects/vaseline/meta.json' with { type: 'json' };
import vhsMetadata from '../../shaders/effects/vhs/meta.json' with { type: 'json' };
import vignetteMetadata from '../../shaders/effects/vignette/meta.json' with { type: 'json' };
import voronoiMetadata from '../../shaders/effects/voronoi/meta.json' with { type: 'json' };
import vortexMetadata from '../../shaders/effects/vortex/meta.json' with { type: 'json' };
import warpMetadata from '../../shaders/effects/warp/meta.json' with { type: 'json' };
import wobbleMetadata from '../../shaders/effects/wobble/meta.json' with { type: 'json' };
import wormholeMetadata from '../../shaders/effects/wormhole/meta.json' with { type: 'json' };
import wormsMetadata from '../../shaders/effects/worms/meta.json' with { type: 'json' };
import fxaaMetadata from '../../shaders/effects/fxaa/meta.json' with { type: 'json' };
import glowingEdgesMetadata from '../../shaders/effects/glowing_edges/meta.json' with { type: 'json' };
import glyphMapMetadata from '../../shaders/effects/glyph_map/meta.json' with { type: 'json' };
import grainMetadata from '../../shaders/effects/grain/meta.json' with { type: 'json' };
import grimeMetadata from '../../shaders/effects/grime/meta.json' with { type: 'json' };
import jpegDecimateMetadata from '../../shaders/effects/jpeg_decimate/meta.json' with { type: 'json' };
import kaleidoMetadata from '../../shaders/effects/kaleido/meta.json' with { type: 'json' };
import lensDistortionMetadata from '../../shaders/effects/lens_distortion/meta.json' with { type: 'json' };
import lensWarpMetadata from '../../shaders/effects/lens_warp/meta.json' with { type: 'json' };
import lightLeakMetadata from '../../shaders/effects/light_leak/meta.json' with { type: 'json' };
import lowpolyMetadata from '../../shaders/effects/lowpoly/meta.json' with { type: 'json' };
import nebulaMetadata from '../../shaders/effects/nebula/meta.json' with { type: 'json' };
import normalMapMetadata from '../../shaders/effects/normal_map/meta.json' with { type: 'json' };
import normalizeMetadata from '../../shaders/effects/normalize/meta.json' with { type: 'json' };
import onScreenDisplayMetadata from '../../shaders/effects/on_screen_display/meta.json' with { type: 'json' };
import outlineMetadata from '../../shaders/effects/outline/meta.json' with { type: 'json' };

// DOM elements
const canvas = document.getElementById('canvas');
const statusEl = document.getElementById('status');

if (!canvas || !statusEl) {
  throw new Error('Demo bootstrap failed: missing required DOM elements.');
}

// Set canvas resolution - cap at reasonable size for performance
let needsResourceRecreation = false;
let lastCanvasMode = null; // Track mode changes

function resizeCanvas() {
  const isFullBleedMode = document.body.classList.contains('full-bleed-mode');
  const currentMode = isFullBleedMode ? 'fullbleed' : 'fixed';
  const modeChanged = lastCanvasMode !== null && lastCanvasMode !== currentMode;
  
  let newWidth, newHeight;
  
  if (isFullBleedMode) {
    // Full bleed mode: use full native resolution, but clamp to 1920x1080
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    let unclampedWidth = Math.floor(rect.width * dpr);
    let unclampedHeight = Math.floor(rect.height * dpr);
    // Clamp to 1920x1080 max
    const MAX_WIDTH = 1920;
    const MAX_HEIGHT = 1080;
    const aspect = unclampedWidth / unclampedHeight;
    if (unclampedWidth > MAX_WIDTH || unclampedHeight > MAX_HEIGHT) {
      if (aspect >= MAX_WIDTH / MAX_HEIGHT) {
        newWidth = MAX_WIDTH;
        newHeight = Math.round(MAX_WIDTH / aspect);
      } else {
        newHeight = MAX_HEIGHT;
        newWidth = Math.round(MAX_HEIGHT * aspect);
      }
    } else {
      newWidth = unclampedWidth;
      newHeight = unclampedHeight;
    }
  } else {
    // Fixed canvas mode (default): always 1024x1024
    newWidth = 1024;
    newHeight = 1024;
  }
  
  const dimensionsChanged = canvas.width !== newWidth || canvas.height !== newHeight;
  
  if (dimensionsChanged) {
    canvas.width = newWidth;
    canvas.height = newHeight;
    needsResourceRecreation = true;
  } else if (modeChanged) {
    // Even if dimensions didn't change, force resource recreation on mode change
    needsResourceRecreation = true;
  }
  
  lastCanvasMode = currentMode;
}

// Initialize canvas resolution
resizeCanvas();

// Handle window resize with debounce
let resizeTimeout;
window.addEventListener('resize', (event) => {
  // Check if this is a forced immediate resize (e.g., from view mode toggle)
  const forceImmediate = event instanceof CustomEvent && event.detail?.forceImmediate === true;
  
  if (forceImmediate) {
    // Skip debounce and resize immediately
    clearTimeout(resizeTimeout);
    resizeCanvas();
  } else {
    // Normal resize with debounce
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      resizeCanvas();
    }, 250);
  }
});

const demoLogs = [];
const MAX_LOG_ENTRIES = 500;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

// Suppress console output for info/debug logs (keeps them in demoLogs for tests)
let quietInfoLogs = true;

function formatLogDetail(detail) {
  if (detail instanceof Error && typeof detail.stack === 'string') {
    return detail.stack;
  }
  if (typeof detail === 'object' && detail !== null) {
    try {
      return JSON.stringify(detail);
    } catch (error) {
      return String(detail);
    }
  }
  return String(detail);
}

function captureLog(level, message, ...details) {
  const normalizedMessage = typeof message === 'string' ? message : String(message);
  const detailText = details.length > 0
    ? details.map((detail) => formatLogDetail(detail)).join(' ').trim()
    : '';
  const combined = detailText ? `${normalizedMessage} ${detailText}`.trim() : normalizedMessage;

  demoLogs.push({ level, message: combined, timestamp: Date.now() });
  if (demoLogs.length > MAX_LOG_ENTRIES) {
    demoLogs.splice(0, demoLogs.length - MAX_LOG_ENTRIES);
  }

  // Suppress console spam for info/debug during animation
  const shouldLog = level === 'error' || level === 'warn' || !quietInfoLogs;
  
  if (shouldLog) {
    const consoleArgs = [normalizedMessage, ...details];
    switch (level) {
      case 'error':
        console.error(...consoleArgs);
        break;
      case 'warn':
        console.warn(...consoleArgs);
        break;
      case 'debug':
        console.debug(...consoleArgs);
        break;
      default:
        console.log(...consoleArgs);
        break;
    }
  }

  return combined;
}

function logInfo(message, ...details) {
  return captureLog('info', message, ...details);
}

function logWarn(message, ...details) {
  captureLog('warn', message, ...details);
}

function logError(message, ...details) {
  captureLog('error', message, ...details);
}

/**
 * Fatal helper surfaces blocking errors to the user and halts further execution.
 * Writes to status div, logs to console, and throws to stop async workflows.
 */
function fatal(message) {
  const text = typeof message === 'string' ? message : String(message);
  statusEl.textContent = `❌ ${text}`;
  logError(`[FATAL] ${text}`);
  throw new Error(text);
}

/**
 * Status helper updates the status display without throwing.
 */
function setStatus(message) {
  const text = String(message);
  statusEl.textContent = text;
  logInfo(text);
}

const WEBGPU_ENABLE_HINT = 'WebGPU not available. Enable chrome://flags/#enable-unsafe-webgpu and restart the browser.';

const shaderSourceCache = new Map();
const shaderMetadataCache = new Map();
// New caches for performance-critical GPU objects (size/layout independent)
const shaderModuleCache = new Map(); // key: shaderId
const bindGroupLayoutCache = new Map(); // key: `${shaderId}|${stage}`
const pipelineLayoutCache = new Map(); // key: `${shaderId}|${stage}`
const computePipelineCache = new Map(); // key: `compute|${shaderId}|${entryPoint}`
const blitShaderModuleCache = new Map(); // key: 'blit'
const bufferToTexturePipelineCache = new WeakMap(); // key: GPUDevice -> { pipeline, layout }

// Track which device the caches belong to
let cachedDevice = null;

const effectManager = new EffectManager({
  helpers: {
    logInfo,
    logWarn,
    setStatus,
    getShaderDescriptor,
    getShaderMetadataCached,
    warnOnNonContiguousBindings,
    createShaderResourceSet,
    createBindGroupEntriesFromResources,
    getOrCreateBindGroupLayout,
    getOrCreatePipelineLayout,
    getOrCreateComputePipeline,
    getBufferToTexturePipeline,
  },
});

effectManager.registerEffect({
  id: aberrationMetadata.id,
  label: aberrationMetadata.label,
  metadata: aberrationMetadata,
  loadModule: () => import('../../shaders/effects/aberration/effect.js'),
});

effectManager.registerEffect({
  id: adjustBrightnessMetadata.id,
  label: adjustBrightnessMetadata.label,
  metadata: adjustBrightnessMetadata,
  loadModule: () => import('../../shaders/effects/adjust_brightness/effect.js'),
});

effectManager.registerEffect({
  id: adjustContrastMetadata.id,
  label: adjustContrastMetadata.label,
  metadata: adjustContrastMetadata,
  loadModule: () => import('../../shaders/effects/adjust_contrast/effect.js'),
});

effectManager.registerEffect({
  id: adjustHueMetadata.id,
  label: adjustHueMetadata.label,
  metadata: adjustHueMetadata,
  loadModule: () => import('../../shaders/effects/adjust_hue/effect.js'),
});

effectManager.registerEffect({
  id: adjustSaturationMetadata.id,
  label: adjustSaturationMetadata.label,
  metadata: adjustSaturationMetadata,
  loadModule: () => import('../../shaders/effects/adjust_saturation/effect.js'),
});

effectManager.registerEffect({
  id: blurMetadata.id,
  label: blurMetadata.label,
  metadata: blurMetadata,
  loadModule: () => import('../../shaders/effects/blur/effect.js'),
});

effectManager.registerEffect({
  id: bloomMetadata.id,
  label: bloomMetadata.label,
  metadata: bloomMetadata,
  loadModule: () => import('../../shaders/effects/bloom/effect.js'),
});

effectManager.registerEffect({
  id: cloudsMetadata.id,
  label: cloudsMetadata.label,
  metadata: cloudsMetadata,
  loadModule: () => import('../../shaders/effects/clouds/effect.js'),
});

effectManager.registerEffect({
  id: colorMapMetadata.id,
  label: colorMapMetadata.label,
  metadata: colorMapMetadata,
  loadModule: () => import('../../shaders/effects/color_map/effect.js'),
});

effectManager.registerEffect({
  id: convFeedbackMetadata.id,
  label: convFeedbackMetadata.label,
  metadata: convFeedbackMetadata,
  loadModule: () => import('../../shaders/effects/conv_feedback/effect.js'),
});

effectManager.registerEffect({
  id: convolveMetadata.id,
  label: convolveMetadata.label,
  metadata: convolveMetadata,
  loadModule: () => import('../../shaders/effects/convolve/effect.js'),
});

effectManager.registerEffect({
  id: crtMetadata.id,
  label: crtMetadata.label,
  metadata: crtMetadata,
  loadModule: () => import('../../shaders/effects/crt/effect.js'),
});

effectManager.registerEffect({
  id: degaussMetadata.id,
  label: degaussMetadata.label,
  metadata: degaussMetadata,
  loadModule: () => import('../../shaders/effects/degauss/effect.js'),
});

effectManager.registerEffect({
  id: densityMapMetadata.id,
  label: densityMapMetadata.label,
  metadata: densityMapMetadata,
  loadModule: () => import('../../shaders/effects/density_map/effect.js'),
});

effectManager.registerEffect({
  id: derivativeMetadata.id,
  label: derivativeMetadata.label,
  metadata: derivativeMetadata,
  loadModule: () => import('../../shaders/effects/derivative/effect.js'),
});

effectManager.registerEffect({
  id: dlaMetadata.id,
  label: dlaMetadata.label,
  metadata: dlaMetadata,
  loadModule: () => import('../../shaders/effects/dla/effect.js'),
});

effectManager.registerEffect({
  id: erosionWormsMetadata.id,
  label: erosionWormsMetadata.label,
  metadata: erosionWormsMetadata,
  loadModule: () => import('../../shaders/effects/erosion_worms/effect.js'),
});

effectManager.registerEffect({
  id: falseColorMetadata.id,
  label: falseColorMetadata.label,
  metadata: falseColorMetadata,
  loadModule: () => import('../../shaders/effects/false_color/effect.js'),
});

effectManager.registerEffect({
  id: fibersMetadata.id,
  label: fibersMetadata.label,
  metadata: fibersMetadata,
  loadModule: () => import('../../shaders/effects/fibers/effect.js'),
});

effectManager.registerEffect({
  id: frameMetadata.id,
  label: frameMetadata.label,
  metadata: frameMetadata,
  loadModule: () => import('../../shaders/effects/frame/effect.js'),
});

effectManager.registerEffect({
  id: posterizeMetadata.id,
  label: posterizeMetadata.label,
  metadata: posterizeMetadata,
  loadModule: () => import('../../shaders/effects/posterize/effect.js'),
});

effectManager.registerEffect({
  id: paletteMetadata.id,
  label: paletteMetadata.label,
  metadata: paletteMetadata,
  loadModule: () => import('../../shaders/effects/palette/effect.js'),
});

effectManager.registerEffect({
  id: pixelSortMetadata.id,
  label: pixelSortMetadata.label,
  metadata: pixelSortMetadata,
  loadModule: () => import('../../shaders/effects/pixel_sort/effect.js'),
});

effectManager.registerEffect({
  id: refractMetadata.id,
  label: refractMetadata.label,
  metadata: refractMetadata,
  loadModule: () => import('../../shaders/effects/refract/effect.js'),
});

effectManager.registerEffect({
  id: reindexMetadata.id,
  label: reindexMetadata.label,
  metadata: reindexMetadata,
  loadModule: () => import('../../shaders/effects/reindex/effect.js'),
});

effectManager.registerEffect({
  id: reverbMetadata.id,
  label: reverbMetadata.label,
  metadata: reverbMetadata,
  loadModule: () => import('../../shaders/effects/reverb/effect.js'),
});

effectManager.registerEffect({
  id: ridgeMetadata.id,
  label: ridgeMetadata.label,
  metadata: ridgeMetadata,
  loadModule: () => import('../../shaders/effects/ridge/effect.js'),
});

effectManager.registerEffect({
  id: rippleMetadata.id,
  label: rippleMetadata.label,
  metadata: rippleMetadata,
  loadModule: () => import('../../shaders/effects/ripple/effect.js'),
});

effectManager.registerEffect({
  id: rotateMetadata.id,
  label: rotateMetadata.label,
  metadata: rotateMetadata,
  loadModule: () => import('../../shaders/effects/rotate/effect.js'),
});

effectManager.registerEffect({
  id: scanlineErrorMetadata.id,
  label: scanlineErrorMetadata.label,
  metadata: scanlineErrorMetadata,
  loadModule: () => import('../../shaders/effects/scanline_error/effect.js'),
});

effectManager.registerEffect({
  id: scratchesMetadata.id,
  label: scratchesMetadata.label,
  metadata: scratchesMetadata,
  loadModule: () => import('../../shaders/effects/scratches/effect.js'),
});

effectManager.registerEffect({
  id: shadowMetadata.id,
  label: shadowMetadata.label,
  metadata: shadowMetadata,
  loadModule: () => import('../../shaders/effects/shadow/effect.js'),
});

effectManager.registerEffect({
  id: simpleFrameMetadata.id,
  label: simpleFrameMetadata.label,
  metadata: simpleFrameMetadata,
  loadModule: () => import('../../shaders/effects/simple_frame/effect.js'),
});

effectManager.registerEffect({
  id: sineMetadata.id,
  label: sineMetadata.label,
  metadata: sineMetadata,
  loadModule: () => import('../../shaders/effects/sine/effect.js'),
});

effectManager.registerEffect({
  id: sketchMetadata.id,
  label: sketchMetadata.label,
  metadata: sketchMetadata,
  loadModule: () => import('../../shaders/effects/sketch/effect.js'),
});

effectManager.registerEffect({
  id: smoothstepMetadata.id,
  label: smoothstepMetadata.label,
  metadata: smoothstepMetadata,
  loadModule: () => import('../../shaders/effects/smoothstep/effect.js'),
});

effectManager.registerEffect({
  id: snowMetadata.id,
  label: snowMetadata.label,
  metadata: snowMetadata,
  loadModule: () => import('../../shaders/effects/snow/effect.js'),
});

effectManager.registerEffect({
  id: sobelMetadata.id,
  label: sobelMetadata.label,
  metadata: sobelMetadata,
  loadModule: () => import('../../shaders/effects/sobel/effect.js'),
});

effectManager.registerEffect({
  id: spatterMetadata.id,
  label: spatterMetadata.label,
  metadata: spatterMetadata,
  loadModule: () => import('../../shaders/effects/spatter/effect.js'),
});

effectManager.registerEffect({
  id: spookyTickerMetadata.id,
  label: spookyTickerMetadata.label,
  metadata: spookyTickerMetadata,
  loadModule: () => import('../../shaders/effects/spooky_ticker/effect.js'),
});

effectManager.registerEffect({
  id: strayHairMetadata.id,
  label: strayHairMetadata.label,
  metadata: strayHairMetadata,
  loadModule: () => import('../../shaders/effects/stray_hair/effect.js'),
});

effectManager.registerEffect({
  id: textureMetadata.id,
  label: textureMetadata.label,
  metadata: textureMetadata,
  loadModule: () => import('../../shaders/effects/texture/effect.js'),
});

effectManager.registerEffect({
  id: tintMetadata.id,
  label: tintMetadata.label,
  metadata: tintMetadata,
  loadModule: () => import('../../shaders/effects/tint/effect.js'),
});

effectManager.registerEffect({
  id: vaselineMetadata.id,
  label: vaselineMetadata.label,
  metadata: vaselineMetadata,
  loadModule: () => import('../../shaders/effects/vaseline/effect.js'),
});

effectManager.registerEffect({
  id: vhsMetadata.id,
  label: vhsMetadata.label,
  metadata: vhsMetadata,
  loadModule: () => import('../../shaders/effects/vhs/effect.js'),
});

effectManager.registerEffect({
  id: vignetteMetadata.id,
  label: vignetteMetadata.label,
  metadata: vignetteMetadata,
  loadModule: () => import('../../shaders/effects/vignette/effect.js'),
});

effectManager.registerEffect({
  id: voronoiMetadata.id,
  label: voronoiMetadata.label,
  metadata: voronoiMetadata,
  loadModule: () => import('../../shaders/effects/voronoi/effect.js'),
});

effectManager.registerEffect({
  id: vortexMetadata.id,
  label: vortexMetadata.label,
  metadata: vortexMetadata,
  loadModule: () => import('../../shaders/effects/vortex/effect.js'),
});

effectManager.registerEffect({
  id: warpMetadata.id,
  label: warpMetadata.label,
  metadata: warpMetadata,
  loadModule: () => import('../../shaders/effects/warp/effect.js'),
});

effectManager.registerEffect({
  id: wobbleMetadata.id,
  label: wobbleMetadata.label,
  metadata: wobbleMetadata,
  loadModule: () => import('../../shaders/effects/wobble/effect.js'),
});

effectManager.registerEffect({
  id: wormholeMetadata.id,
  label: wormholeMetadata.label,
  metadata: wormholeMetadata,
  loadModule: () => import('../../shaders/effects/wormhole/effect.js'),
});

effectManager.registerEffect({
  id: wormsMetadata.id,
  label: wormsMetadata.label,
  metadata: wormsMetadata,
  loadModule: () => import('../../shaders/effects/worms/effect.js'),
});

effectManager.registerEffect({
  id: fxaaMetadata.id,
  label: fxaaMetadata.label,
  metadata: fxaaMetadata,
  loadModule: () => import('../../shaders/effects/fxaa/effect.js'),
});

effectManager.registerEffect({
  id: glowingEdgesMetadata.id,
  label: glowingEdgesMetadata.label,
  metadata: glowingEdgesMetadata,
  loadModule: () => import('../../shaders/effects/glowing_edges/effect.js'),
});

effectManager.registerEffect({
  id: glyphMapMetadata.id,
  label: glyphMapMetadata.label,
  metadata: glyphMapMetadata,
  loadModule: () => import('../../shaders/effects/glyph_map/effect.js'),
});

effectManager.registerEffect({
  id: grainMetadata.id,
  label: grainMetadata.label,
  metadata: grainMetadata,
  loadModule: () => import('../../shaders/effects/grain/effect.js'),
});

effectManager.registerEffect({
  id: grimeMetadata.id,
  label: grimeMetadata.label,
  metadata: grimeMetadata,
  loadModule: () => import('../../shaders/effects/grime/effect.js'),
});

effectManager.registerEffect({
  id: jpegDecimateMetadata.id,
  label: jpegDecimateMetadata.label,
  metadata: jpegDecimateMetadata,
  loadModule: () => import('../../shaders/effects/jpeg_decimate/effect.js'),
});

effectManager.registerEffect({
  id: kaleidoMetadata.id,
  label: kaleidoMetadata.label,
  metadata: kaleidoMetadata,
  loadModule: () => import('../../shaders/effects/kaleido/effect.js'),
});

effectManager.registerEffect({
  id: lensDistortionMetadata.id,
  label: lensDistortionMetadata.label,
  metadata: lensDistortionMetadata,
  loadModule: () => import('../../shaders/effects/lens_distortion/effect.js'),
});

effectManager.registerEffect({
  id: lensWarpMetadata.id,
  label: lensWarpMetadata.label,
  metadata: lensWarpMetadata,
  loadModule: () => import('../../shaders/effects/lens_warp/effect.js'),
});

effectManager.registerEffect({
  id: lightLeakMetadata.id,
  label: lightLeakMetadata.label,
  metadata: lightLeakMetadata,
  loadModule: () => import('../../shaders/effects/light_leak/effect.js'),
});

effectManager.registerEffect({
  id: lowpolyMetadata.id,
  label: lowpolyMetadata.label,
  metadata: lowpolyMetadata,
  loadModule: () => import('../../shaders/effects/lowpoly/effect.js'),
});

effectManager.registerEffect({
  id: nebulaMetadata.id,
  label: nebulaMetadata.label,
  metadata: nebulaMetadata,
  loadModule: () => import('../../shaders/effects/nebula/effect.js'),
});

effectManager.registerEffect({
  id: normalMapMetadata.id,
  label: normalMapMetadata.label,
  metadata: normalMapMetadata,
  loadModule: () => import('../../shaders/effects/normal_map/effect.js'),
});

effectManager.registerEffect({
  id: normalizeMetadata.id,
  label: normalizeMetadata.label,
  metadata: normalizeMetadata,
  loadModule: () => import('../../shaders/effects/normalize/effect.js'),
});

effectManager.registerEffect({
  id: onScreenDisplayMetadata.id,
  label: onScreenDisplayMetadata.label,
  metadata: onScreenDisplayMetadata,
  loadModule: () => import('../../shaders/effects/on_screen_display/effect.js'),
});

effectManager.registerEffect({
  id: outlineMetadata.id,
  label: outlineMetadata.label,
  metadata: outlineMetadata,
  loadModule: () => import('../../shaders/effects/outline/effect.js'),
});

function getRegisteredEffects() {
  const effects = effectManager.getAvailableEffects();
  // Sort alphabetically by label for dropdown display
  return effects.sort((a, b) => {
    const labelA = (a.label || a.id).toLowerCase();
    const labelB = (b.label || b.id).toLowerCase();
    return labelA.localeCompare(labelB);
  });
}

function getActiveEffectMetadata(effectId) {
  return effectManager.getEffectMetadata(effectId);
}

async function getActiveEffectUIState() {
  return effectManager.getActiveUIState();
}

async function setActiveEffect(effectId) {
  return effectManager.setActiveEffect(effectId);
}

async function updateActiveEffectParams(updates = {}) {
  return effectManager.updateActiveParams(updates);
}

function clearGPUObjectCaches() {
  logInfo('Clearing all GPU object caches');
  shaderModuleCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
  computePipelineCache.clear();
  blitShaderModuleCache.clear();
  // Note: bufferToTexturePipelineCache is a WeakMap and cannot be cleared explicitly
  pipelineCache.clear(); // Clear render pipeline cache
  effectManager.invalidateActiveEffectResources();
  // Note: Don't set cachedDevice = null here; let ensureCachesMatchDevice manage it
}

function alignTo(value, multiple) {
  if (multiple <= 0) {
    return value;
  }
  return Math.ceil(value / multiple) * multiple;
}

function resolveUniformBufferSize(size) {
  if (typeof size === 'number' && Number.isFinite(size) && size > 0) {
    return alignTo(Math.max(size, 256), 256);
  }
  return 256;
}

function resolveStorageBufferSize(sizeDescriptor, width, height) {
  if (typeof sizeDescriptor === 'number' && Number.isFinite(sizeDescriptor) && sizeDescriptor > 0) {
    return alignTo(sizeDescriptor, 16);
  }
  if (sizeDescriptor === 'pixel-f32x4') {
    const pixels = Math.max(width * height, 1);
    return alignTo(pixels * 4 * 4, 16);
  }
  if (sizeDescriptor === 'pixel-sort-f32x4') {
    const maxDim = Math.max(Math.max(width, height), 1);
    const want = Math.min(Math.max(maxDim * 2, maxDim, 1), 4096);
    const pixels = Math.max(want * want, 1);
    return alignTo(pixels * 4 * 4, 16);
  }
  if (sizeDescriptor === 'dynamic-histogram') {
    const bins = Math.max(Math.max(width, height), 1);
    return alignTo(bins * Uint32Array.BYTES_PER_ELEMENT, 16);
  }
  return 16;
}

function ensureCachesMatchDevice(device) {
  if (cachedDevice !== device) {
    logWarn(`Device changed! Old: ${cachedDevice ? 'exists' : 'null'}, New: ${device ? 'exists' : 'null'}, Same object: ${cachedDevice === device}`);
    clearGPUObjectCaches();
    cachedDevice = device;
    // Note: We don't call invalidateComputeResources() here because it would
    // destroy resources mid-frame. Instead, getComputeResources() will detect
    // stale cached objects and recreate them naturally.
  }
}

async function loadShaderSource(shaderId) {
  if (shaderSourceCache.has(shaderId)) {
    return shaderSourceCache.get(shaderId);
  }

  let descriptor;
  try {
    descriptor = getShaderDescriptor(shaderId);
  } catch (error) {
    fatal(error?.message ?? error);
  }

  let response;
  try {
    // Add cache-busting timestamp to force fresh shader load
    const url = descriptor.url + '?t=' + Date.now();
    response = await fetch(url);
  } catch (error) {
    fatal(`Failed to fetch ${descriptor.label ?? shaderId}: ${error?.message ?? error}`);
  }

  if (!response?.ok) {
    fatal(`Failed to fetch ${descriptor.label ?? shaderId}: ${response?.status ?? 'Request failed'}`);
  }

  const source = await response.text();
  shaderSourceCache.set(shaderId, source);
  return source;
}

async function getShaderMetadataCached(shaderId) {
  if (shaderMetadataCache.has(shaderId)) {
    return shaderMetadataCache.get(shaderId);
  }
  const source = await loadShaderSource(shaderId);
  const metadata = parseShaderMetadata(source);
  shaderMetadataCache.set(shaderId, metadata);
  return metadata;
}

async function getOrCreateShaderModule(device, shaderId) {
  ensureCachesMatchDevice(device);
  if (shaderModuleCache.has(shaderId)) {
    return shaderModuleCache.get(shaderId);
  }
  const descriptor = getShaderDescriptor(shaderId);
  const source = await loadShaderSource(shaderId);
  const module = await compileShaderModuleWithValidation(device, source, { label: descriptor.label });
  shaderModuleCache.set(shaderId, module);
  return module;
}

function getOrCreateBindGroupLayout(device, shaderId, stage, metadata) {
  ensureCachesMatchDevice(device);
  const cacheKey = `${shaderId}|${stage}`;
  if (bindGroupLayoutCache.has(cacheKey)) {
    return bindGroupLayoutCache.get(cacheKey);
  }
  const layout = device.createBindGroupLayout({
    entries: createBindGroupLayoutEntriesFromMetadata(metadata.bindings, stage),
  });
  bindGroupLayoutCache.set(cacheKey, layout);
  return layout;
}

function getOrCreatePipelineLayout(device, shaderId, stage, bindGroupLayout) {
  ensureCachesMatchDevice(device);
  const cacheKey = `${shaderId}|${stage}`;
  if (pipelineLayoutCache.has(cacheKey)) {
    return pipelineLayoutCache.get(cacheKey);
  }
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  pipelineLayoutCache.set(cacheKey, layout);
  return layout;
}

async function getOrCreateComputePipeline(device, shaderId, pipelineLayout, entryPoint) {
  ensureCachesMatchDevice(device);
  const normalizedEntryPoint = entryPoint ?? 'main';
  const cacheKey = `compute|${shaderId}|${normalizedEntryPoint}`;
  if (computePipelineCache.has(cacheKey)) {
    return computePipelineCache.get(cacheKey);
  }
  const module = await getOrCreateShaderModule(device, shaderId);
  device.pushErrorScope('validation');
  let pipeline;
  try {
    pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: normalizedEntryPoint },
    });
  } catch (error) {
    await device.popErrorScope();
    fatal(`Failed to create compute pipeline: ${error?.message ?? error}`);
  }
  const err = await device.popErrorScope();
  if (err) {
    fatal(`Compute pipeline validation failed: ${err?.message ?? err}`);
  }
  computePipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

function createBindGroupLayoutEntriesFromMetadata(bindings, stage) {
  const visibility = stage === 'render' ? GPUShaderStage.FRAGMENT : GPUShaderStage.COMPUTE;
  return bindings
    .filter((binding) => binding.group === 0)
    .map((binding) => {
      if (binding.resource === 'uniformBuffer') {
        return {
          binding: binding.binding,
          visibility,
          buffer: { type: 'uniform' },
        };
      }

      if (binding.resource === 'storageBuffer') {
        return {
          binding: binding.binding,
          visibility,
          buffer: { type: 'storage' },
        };
      }

      if (binding.resource === 'readOnlyStorageBuffer') {
        return {
          binding: binding.binding,
          visibility,
          buffer: { type: 'read-only-storage' },
        };
      }

      if (binding.resource === 'storageTexture') {
        return {
          binding: binding.binding,
          visibility,
          storageTexture: {
            access: binding.storageTextureAccess ?? 'write-only',
            format: binding.storageTextureFormat ?? 'rgba32float',
          },
        };
      }

      if (binding.resource === 'sampledTexture') {
        return {
          binding: binding.binding,
          visibility,
          texture: { sampleType: 'unfilterable-float' },
        };
      }

      if (binding.resource === 'sampler') {
        return {
          binding: binding.binding,
          visibility,
          sampler: {},
        };
      }

      fatal(`Unsupported binding resource type for ${binding.name} (binding ${binding.binding}).`);
      return null;
    })
    .filter(Boolean)
    .sort((a, b) => a.binding - b.binding);
}

function createShaderResourceSet(device, descriptor, metadata, width, height, options = {}) {
  const buffers = {};
  const textures = {};
  const samplers = {};
  const destroyables = [];
  let destroyed = false;
  const templates = descriptor.resources ?? {};
  const groupZeroBindings = metadata.bindings.filter((binding) => binding.group === 0);
  const providedTextures = options.inputTextures ?? {};
  const providedSamplers = options.samplers ?? {};

  for (const binding of groupZeroBindings) {
    const template = templates[binding.name];

    if (binding.resource === 'uniformBuffer') {
      if (!template?.size) {
        logWarn(`No resource template size for uniform buffer '${binding.name}' in shader '${descriptor.id}'. Defaulting to 256 bytes.`);
      }
      const size = resolveUniformBufferSize(template?.size);
      const buffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      buffers[binding.name] = buffer;
      destroyables.push(buffer);
      continue;
    }

    if (binding.resource === 'storageBuffer' || binding.resource === 'readOnlyStorageBuffer') {
      const size = resolveStorageBufferSize(template?.size, width, height);
      const buffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      buffers[binding.name] = buffer;
      destroyables.push(buffer);
      continue;
    }

    if (binding.resource === 'storageTexture') {
      const format = template?.format ?? binding.storageTextureFormat ?? 'rgba32float';
      const texture = device.createTexture({
        size: {
          width,
          height,
          depthOrArrayLayers: 1,
        },
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      textures[binding.name] = texture;
      destroyables.push(texture);
      continue;
    }

    if (binding.resource === 'sampledTexture') {
      // Prefer provided texture (e.g., generator output)
      const provided = providedTextures[binding.name];
      if (provided) {
        textures[binding.name] = provided;
        // Do not track provided textures for destruction (externally owned)
      } else {
        // Create a dummy sampleable texture for bring-up/tests
        const format = template?.format ?? 'rgba32float';
        const texture = device.createTexture({
          size: { width, height, depthOrArrayLayers: 1 },
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        textures[binding.name] = texture;
        destroyables.push(texture);
      }
      continue;
    }

    if (binding.resource === 'sampler') {
      const provided = providedSamplers[binding.name];
      if (provided) {
        samplers[binding.name] = provided;
      } else {
        const sampler = device.createSampler({
          magFilter: 'linear',
          minFilter: 'linear',
        });
        samplers[binding.name] = sampler;
        // Samplers do not have destroy(), no need to track in destroyables
      }
      continue;
    }
  }

  return {
    buffers,
    textures,
    samplers,
    destroyAll() {
      if (destroyed) {
        return;
      }
      destroyed = true;
      for (const resource of destroyables) {
        if (resource?.destroy) {
          try {
            resource.destroy();
          } catch (error) {
            logWarn('Failed to destroy GPU resource during cleanup:', error);
          }
        }
      }
    },
  };
}

function createBindGroupEntriesFromResources(bindings, resourceSet) {
  return bindings
    .filter((binding) => binding.group === 0)
    .map((binding) => {
      if (binding.resource === 'uniformBuffer' || binding.resource === 'storageBuffer' || binding.resource === 'readOnlyStorageBuffer') {
        const buffer = resourceSet.buffers[binding.name];
        if (!buffer) {
          fatal(`Missing GPU buffer for binding ${binding.name}.`);
        }
        return {
          binding: binding.binding,
          resource: { buffer },
        };
      }

      if (binding.resource === 'storageTexture') {
        const texture = resourceSet.textures[binding.name];
        if (!texture) {
          fatal(`Missing GPU texture for binding ${binding.name}.`);
        }
        return {
          binding: binding.binding,
          resource: texture.createView(),
        };
      }

      if (binding.resource === 'sampledTexture') {
        const texture = resourceSet.textures[binding.name];
        if (!texture) {
          fatal(`Missing sampled texture for binding ${binding.name}.`);
        }
        return {
          binding: binding.binding,
          resource: texture.createView(),
        };
      }

      if (binding.resource === 'sampler') {
        const sampler = resourceSet.samplers?.[binding.name];
        if (!sampler) {
          fatal(`Missing sampler for binding ${binding.name}.`);
        }
        return {
          binding: binding.binding,
          resource: sampler,
        };
      }

      fatal(`Unsupported bind group entry resource for ${binding.name}.`);
      return null;
    })
    .filter(Boolean)
    .sort((a, b) => a.binding - b.binding);
}

const warnedShaders = new Set();
function warnOnNonContiguousBindings(bindings, shaderId) {
  if (warnedShaders.has(shaderId)) {
    return; // Only warn once per shader
  }
  const groupZero = bindings.filter((b) => b.group === 0).map((b) => b.binding).sort((a, b) => a - b);
  for (let i = 1; i < groupZero.length; i += 1) {
    if (groupZero[i] !== groupZero[i - 1] + 1) {
      logWarn(`Non-contiguous binding indices detected for shader '${shaderId}': [${groupZero.join(', ')}]`);
      warnedShaders.add(shaderId);
      break;
    }
  }
}

/**
 * Ensures WebGPU is available and returns adapter/device tuple with zero requested features.
 * Throws via fatal() with actionable messaging on failure.
 * NOTE: This should only be called ONCE per page load. Use getWebGPUState() for repeated access.
 */
let singletonWebGPUInit = null;
async function ensureWebGPU() {
  if (singletonWebGPUInit) {
    logInfo('ensureWebGPU: Returning cached singleton');
    return singletonWebGPUInit;
  }
  
  setStatus('Checking WebGPU support…');

  if (!navigator.gpu) {
    fatal(WEBGPU_ENABLE_HINT);
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    fatal(`Failed to request GPU adapter: ${error.message ?? error}`);
  }

  if (!adapter) {
    fatal('Unable to acquire GPU adapter. Ensure WebGPU is enabled for your browser profile.');
  }

  let device;
  try {
    device = await adapter.requestDevice();
    logInfo('Created new WebGPU device');
  } catch (error) {
    fatal(`Failed to request GPU device: ${error.message ?? error}`);
  }

  if (!device) {
    fatal('GPU adapter returned no device.');
  }

  setStatus('WebGPU ready.');
  singletonWebGPUInit = { adapter, device };
  return singletonWebGPUInit;
}

/**
 * Encapsulates shared WebGPU canvas/device wiring.
 */
class WebGPUContext {
  constructor({ adapter, device, canvas, onDeviceLost, onContextLost } = {}) {
    if (!canvas) {
      fatal('WebGPUContext requires a canvas element.');
    }
    if (!device) {
      fatal('WebGPUContext requires a GPU device.');
    }

    this.adapter = adapter ?? null;
    this.device = device;
    this.queue = device.queue;
    this.canvas = canvas;
    this.context = null;
    this.format = null;
    this.alphaMode = 'opaque';
    this._onContextLost = onContextLost;
    this._onDeviceLost = onDeviceLost;
    this._contextLostListener = null;

    if (device?.lost && typeof device.lost.then === 'function') {
      device.lost
        .then((info) => {
          if (typeof this._onDeviceLost === 'function') {
            this._onDeviceLost(info);
          } else {
            const message = info?.message ?? 'Device lost for unknown reasons.';
            fatal(`WebGPU device lost: ${message}`);
          }
        })
        .catch((error) => {
          fatal(`WebGPU device lost: ${error?.message ?? error}`);
        });
    }
  }

  configureCanvas(options = {}) {
    const alphaMode = options.alphaMode ?? 'opaque';

    if (!this.canvas) {
      fatal('WebGPUContext has no canvas to configure.');
    }

    if (!navigator.gpu?.getPreferredCanvasFormat) {
      fatal('navigator.gpu.getPreferredCanvasFormat() is unavailable.');
    }

    const context = this.canvas.getContext('webgpu');
    if (!context) {
      fatal('Unable to acquire WebGPU canvas context.');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: this.device,
      format,
      alphaMode,
    });

    this.context = context;
    this.format = format;
    this.alphaMode = alphaMode;

    if (!this._contextLostListener) {
      this._contextLostListener = (event) => {
        event?.preventDefault?.();
  logError('WebGPU canvas context lost.', event);
        if (typeof this._onContextLost === 'function') {
          this._onContextLost(event);
        } else {
          setStatus('WebGPU context lost. Refresh the page to recover.');
        }
      };
      this.canvas.addEventListener('contextlost', this._contextLostListener, { once: true });
    }
  }

  getCurrentTextureView() {
    if (!this.context) {
      fatal('Canvas is not configured. Call configureCanvas() first.');
    }
    const texture = this.context.getCurrentTexture();
    if (!texture?.createView) {
      fatal('Current texture is unavailable.');
    }
    return texture.createView();
  }
}

let cachedWebGPUState = null;
let webgpuStateInitPromise = null;
const pipelineCache = new Map();

function getPipelineCacheKey(format, alphaMode) {
  return `${format}|${alphaMode ?? 'opaque'}`;
}

async function getWebGPUState(options = {}) {
  // Preserve current alphaMode unless explicitly overridden.
  const desiredAlphaMode = options.alphaMode ?? cachedWebGPUState?.webgpuContext?.alphaMode ?? 'premultiplied';

  if (!cachedWebGPUState) {
    // If initialization is already in progress, wait for it
    if (webgpuStateInitPromise) {
      logInfo('getWebGPUState: Waiting for in-progress initialization');
      await webgpuStateInitPromise;
      return cachedWebGPUState;
    }

    logInfo('getWebGPUState: No cached state, creating new device');
    webgpuStateInitPromise = (async () => {
      const { adapter, device } = await ensureWebGPU();
      const webgpuContext = new WebGPUContext({
        adapter,
        device,
        canvas,
        onDeviceLost: (info) => {
          logWarn('WebGPU device lost, clearing caches:', info);
          cachedWebGPUState = null;
          cachedDevice = null;
          webgpuStateInitPromise = null;
          pipelineCache.clear();
          clearGPUObjectCaches();
          invalidateComputeResources();
          fatal(`WebGPU device lost: ${info?.message ?? 'Unknown reason'}. Please refresh.`);
        },
      });
      webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
      cachedWebGPUState = { adapter, device, webgpuContext };
      logInfo('getWebGPUState: Cached new state');
    })();

    await webgpuStateInitPromise;
    webgpuStateInitPromise = null;
    return cachedWebGPUState;
  }

  // Silently use cached state (too noisy during animation)
  const { webgpuContext } = cachedWebGPUState;

  if (!webgpuContext.context) {
    webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
  } else if (desiredAlphaMode !== webgpuContext.alphaMode) {
    webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
  }

  return cachedWebGPUState;
}

const FLAT_COLOR_SHADER = `@vertex
fn vertex_main(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4<f32> {
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}`;

async function compileFlatColorShader(device) {
  let shaderModule;
  try {
    shaderModule = device.createShaderModule({ code: FLAT_COLOR_SHADER });
  } catch (error) {
    fatal(`Failed to create shader module: ${error?.message ?? error}`);
  }

  if (shaderModule?.getCompilationInfo) {
    try {
      const info = await shaderModule.getCompilationInfo();
      const errors = (info?.messages ?? []).filter((message) => message.type === 'error');
      if (errors.length > 0) {
        const details = errors
          .map((message) => {
            const lineInfo = typeof message.lineNum === 'number' ? `Line ${message.lineNum}: ` : '';
            return `${lineInfo}${message.message}`;
          })
          .join('\n');
        fatal(`Flat color shader compilation failed:\n${details}`);
      }
    } catch (error) {
      fatal(`Failed to validate shader compilation: ${error?.message ?? error}`);
    }
  }

  return shaderModule;
}

async function renderFlatColor(options = {}) {
  const { alphaMode, clearColor } = options;

  setStatus('Preparing flat color render…');

  const { device, webgpuContext } = await getWebGPUState({ alphaMode });

  const cacheKey = getPipelineCacheKey(webgpuContext.format, webgpuContext.alphaMode);
  let pipeline = pipelineCache.get(cacheKey);

  if (!pipeline) {
    const shaderModule = await compileFlatColorShader(device);
    try {
      pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: shaderModule,
          entryPoint: 'vertex_main',
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fragment_main',
          targets: [
            {
              format: webgpuContext.format,
            },
          ],
        },
        primitive: { topology: 'triangle-list' },
      });
    } catch (error) {
      fatal(`Failed to create render pipeline: ${error?.message ?? error}`);
    }

    pipelineCache.set(cacheKey, pipeline);
  }

  let encoder;
  try {
    encoder = device.createCommandEncoder();
  } catch (error) {
    fatal(`Failed to create command encoder: ${error?.message ?? error}`);
  }

  const textureView = webgpuContext.getCurrentTextureView();
  const attachment = {
    view: textureView,
    loadOp: 'clear',
    clearValue: clearColor ?? { r: 0, g: 0, b: 0, a: 1 },
    storeOp: 'store',
  };

  let pass;
  try {
    pass = encoder.beginRenderPass({ colorAttachments: [attachment] });
  } catch (error) {
    fatal(`Failed to begin render pass: ${error?.message ?? error}`);
  }

  pass.setPipeline(pipeline);
  pass.draw(3, 1, 0, 0);
  pass.end();

  let commandBuffer;
  try {
    commandBuffer = encoder.finish();
  } catch (error) {
    fatal(`Failed to finalize GPU commands: ${error?.message ?? error}`);
  }

  try {
    device.queue.submit([commandBuffer]);
  } catch (error) {
    fatal(`Failed to submit GPU work: ${error?.message ?? error}`);
  }

  setStatus('Rendered flat color.');

  return {
    format: webgpuContext.format,
    alphaMode: webgpuContext.alphaMode,
  };
}

// Initial status (after definitions to avoid hoist surprises in future edits)
setStatus('Initializing…');

// Exports for downstream steps and testing
export {
  canvas,
  statusEl,
  fatal,
  setStatus,
  ensureWebGPU,
  WEBGPU_ENABLE_HINT,
  WebGPUContext,
  renderFlatColor,
};

const BLIT_SHADER = `@vertex
fn vertex_main(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4<f32> {
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@group(0) @binding(0) var compute_output : texture_2d<f32>;

@fragment
fn fragment_main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(compute_output, 0));
    let x = i32(pos.x);
    let y = i32(pos.y);
    if (x >= dims.x || y >= dims.y) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let flipped_y = dims.y - 1 - y;
    return textureLoad(compute_output, vec2<i32>(x, flipped_y), 0);
}`;

const BUFFER_TO_TEXTURE_SHADER = `struct AberrationParams {
  size : vec4<f32>,
  anim : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> input_buffer : array<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : AberrationParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(params.size.x, 0.0));
  let height : u32 = u32(max(params.size.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let index : u32 = (gid.y * width + gid.x) * 4u;
  let color : vec4<f32> = vec4<f32>(
    input_buffer[index + 0u],
    input_buffer[index + 1u],
    input_buffer[index + 2u],
    input_buffer[index + 3u]
  );

  textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}`;

async function getBufferToTexturePipeline(device) {
  let entry = bufferToTexturePipelineCache.get(device);
  if (entry) {
    return entry;
  }

  const module = await compileShaderModuleWithValidation(device, BUFFER_TO_TEXTURE_SHADER, { label: 'buffer-to-texture shader' });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  device.pushErrorScope('validation');
  let pipeline;
  try {
    pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' },
    });
  } catch (error) {
    await device.popErrorScope();
    fatal(`Failed to create buffer-to-texture pipeline: ${error?.message ?? error}`);
  }

  const pipelineError = await device.popErrorScope();
  if (pipelineError) {
    fatal(`Buffer-to-texture pipeline validation failed: ${pipelineError?.message ?? pipelineError}`);
  }

  entry = { pipeline, bindGroupLayout };
  bufferToTexturePipelineCache.set(device, entry);
  return entry;
}

let computeResources = null;
let cachedReadbackBuffer = null;
let cachedReadbackSize = 0;
let cachedEffectReadbackBuffer = null;
let cachedEffectReadbackSize = 0;

async function compileShaderModuleWithValidation(device, code, { label } = {}) {
  const descriptor = label ? { code, label } : { code };
  let shaderModule;
  try {
    shaderModule = device.createShaderModule(descriptor);
  } catch (error) {
    fatal(`Failed to create ${label ?? 'unnamed'} shader module: ${error?.message ?? error}`);
  }

  if (shaderModule?.getCompilationInfo) {
    try {
      const info = await shaderModule.getCompilationInfo();
      const messages = info?.messages ?? [];
      const errors = messages.filter((message) => message.type === 'error');
      const warnings = messages.filter((message) => message.type === 'warning');

      if (warnings.length > 0) {
        warnings.forEach((warning) => {
          const loc = typeof warning.lineNum === 'number' ? `Line ${warning.lineNum}: ` : '';
          logWarn(`[Shader Warning${label ? `: ${label}` : ''}] ${loc}${warning.message}`);
        });
      }

      if (errors.length > 0) {
        const details = errors
          .map((message) => {
            const lineInfo = typeof message.lineNum === 'number' ? `Line ${message.lineNum}: ` : '';
            return `${lineInfo}${message.message}`;
          })
          .join('\n');
        fatal(`${label ?? 'Shader'} compilation failed:\n${details}`);
      }
    } catch (error) {
      fatal(`Failed to validate ${label ?? 'shader'} compilation: ${error?.message ?? error}`);
    }
  }

  return shaderModule;
}

function invalidateComputeResources() {
  if (computeResources?.resourceSet?.destroyAll) {
    try {
      computeResources.resourceSet.destroyAll();
    } catch (error) {
      logWarn('Failed to destroy compute resources during invalidation:', error);
    }
  } else if (computeResources?.outputTexture?.destroy) {
    try {
      computeResources.outputTexture.destroy();
    } catch (error) {
      logWarn('Failed to destroy output texture during invalidation:', error);
    }
  }
  computeResources = null;
  resetReadbackCache();
  effectManager.invalidateActiveEffectResources();
}

function resetReadbackCache() {
  if (cachedReadbackBuffer?.destroy) {
    try {
      cachedReadbackBuffer.destroy();
    } catch (error) {
      logWarn('Failed to destroy readback buffer during reset:', error);
    }
  }
  cachedReadbackBuffer = null;
  cachedReadbackSize = 0;
}

async function getComputeResources(device) {
  const width = canvas.width;
  const height = canvas.height;

  if (computeResources) {
    if (computeResources.textureWidth !== width || computeResources.textureHeight !== height) {
      invalidateComputeResources();
    } else {
      return computeResources;
    }
  }

  setStatus('Creating compute resources…');

  let descriptor;
  try {
    descriptor = getShaderDescriptor('multires');
  } catch (error) {
    fatal(error?.message ?? error);
  }

  const shaderMetadata = await getShaderMetadataCached('multires');
  warnOnNonContiguousBindings(shaderMetadata.bindings, descriptor.id);

  const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
  const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
  const computePipeline = await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

  const resourceSet = createShaderResourceSet(device, descriptor, shaderMetadata, width, height);
  const bindGroupEntries = createBindGroupEntriesFromResources(shaderMetadata.bindings, resourceSet);

  const computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: bindGroupEntries,
  });

  const outputTexture = resourceSet.textures.output_texture;
  if (!outputTexture) {
    fatal('Shader resource set did not produce an output texture.');
  }

  const blitBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ],
  });

  const blitBindGroup = device.createBindGroup({
    layout: blitBindGroupLayout,
    entries: [
      { binding: 0, resource: outputTexture.createView() },
    ],
  });

  const blitPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [blitBindGroupLayout] });

  computeResources = {
    shaderId: 'multires',
    shaderDescriptor: descriptor,
    shaderMetadata,
    resourceSet,
    outputTexture,
    frameUniformsBuffer: resourceSet.buffers.frame_uniforms,
    multiresParamsBuffer: resourceSet.buffers.params,
    normalizationStateBuffer: resourceSet.buffers.normalization_state,
    sinNormalizationStateBuffer: resourceSet.buffers.sin_normalization_state,
    maskDataBuffer: resourceSet.buffers.mask_data,
    permutationTableBuffer: resourceSet.buffers.permutation_table_storage,
    computePipeline,
    computeBindGroup,
    computeBindGroupLayout,
    blitPipelineLayout,
    blitBindGroupLayout,
    blitBindGroup,
    textureWidth: width,
    textureHeight: height,
  };

  const frameUniformsBuffer = computeResources.frameUniformsBuffer;
  const multiresParamsBuffer = computeResources.multiresParamsBuffer;

  const frameUniformsState = new Float32Array(8);
  frameUniformsState[0] = width;
  frameUniformsState[1] = height;

  const multiresParamsState = createDefaultMultiresParams();

  computeResources.frameUniformsState = frameUniformsState;
  computeResources.multiresParamsState = multiresParamsState;

  if (frameUniformsBuffer) {
    device.queue.writeBuffer(frameUniformsBuffer, 0, frameUniformsState);
  }

  if (multiresParamsBuffer) {
    device.queue.writeBuffer(multiresParamsBuffer, 0, multiresParamsState);
    logInfo(`Initial multires params written to GPU: ${Array.from(multiresParamsState.slice(0, 8))}`);
  }

  return computeResources;
}


function createDefaultMultiresParams() {
  return Float32Array.from([
    3, 3, 1, 0,  // freq_octaves_ridges: [freq_x=3, freq_y=3, octaves=1, ridges=0]
    0, 3, 1, 0,  // sin_spline_distrib_corners: [sin_amount=0, spline_order=3 (bicubic), distrib=1 (simplex), corner_effect=0]
    1, 0, 0, 0,  // mask_options: [mask_enabled=1, mask_inverse=0, mask_static=0, lattice_drift=0]
    0, 21, 0.5, 0,  // supersample_color: [supersample=0, color_space=21(HSV), reindex_range=0.5, pad=0]
    1, 0, 0, 0,  // saturation_hue: [saturation_scale=1, hue_distrib=0, saturation_distrib=0, brightness_distrib=0]
    0, 0, 0, 0,  // brightness_settings: [brightness_freq_x=0, brightness_freq_y=0, octave_blending=0, with_alpha=0]
    0, 0, 0, 0,  // ai_flags_time: [ai_image_path=0, with_ai=0, ai_tile_mode=0, time=0]
    0.1, 0, 0, 0,  // speed_padding: [speed=0.1, pad=0, pad=0, pad=0]
  ]);
}

const MULTIRES_PARAM_FIELD_OFFSETS = {
  freq_octaves_ridges: 0,
  sin_spline_distrib_corners: 4,
  mask_options: 8,
  supersample_color: 12,
  saturation_hue: 16,
  brightness_settings: 20,
  ai_flags_time: 24,
  speed_padding: 28,
};

function normalizeParamKey(key) {
  return String(key ?? '').replace(/[^0-9A-Za-z]/g, '').toLowerCase();
}

function toFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function setBooleanFlag(state, index, value, changed, fieldName) {
  state[index] = value ? 1 : 0;
  changed.add(fieldName);
  return true;
}

function applyVectorOverride(state, offset, fieldName, value, changed) {
  const components = ['x', 'y', 'z', 'w'];

  const isVectorLike = Array.isArray(value) || (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView && ArrayBuffer.isView(value));
  if (isVectorLike) {
    if (value.length < 4) {
      logWarn(`updateParams: '${fieldName}' arrays must have 4 elements.`);
      return false;
    }
    const vec = new Float32Array(4);
    for (let i = 0; i < 4; i += 1) {
      const numeric = toFiniteNumber(value[i]);
      if (numeric === null) {
        logWarn(`updateParams: '${fieldName}' array entries must be finite numbers.`);
        return false;
      }
      vec[i] = numeric;
    }
    state.set(vec, offset);
    changed.add(fieldName);
    return true;
  }

  if (typeof value === 'object' && value !== null) {
    let applied = false;
    components.forEach((component, index) => {
    if (hasOwn(value, component)) {
        const numeric = toFiniteNumber(value[component]);
        if (numeric === null) {
          logWarn(`updateParams: '${fieldName}.${component}' must be a finite number.`);
          return;
        }
        state[offset + index] = numeric;
        applied = true;
      }
    });
    if (applied) {
      changed.add(fieldName);
      return true;
    }
  }

  logWarn(`updateParams: Unsupported value for '${fieldName}'. Provide an array of 4 numbers or an object with x/y/z/w.`);
  return false;
}

const MULTIRES_ALIAS_SETTERS = {
  freq(state, value, changed) {
    let updated = false;
    if (typeof value === 'number') {
      const numeric = toFiniteNumber(value);
      if (numeric !== null) {
        state[0] = numeric;
        state[1] = numeric;
        updated = true;
      }
    } else if (typeof value === 'object' && value !== null) {
      const fx = toFiniteNumber(value.x ?? value.width ?? value.value ?? value[0]);
      const fy = toFiniteNumber(value.y ?? value.height ?? value.value ?? value[1]);
      if (fx !== null) {
        state[0] = fx;
        updated = true;
      }
      if (fy !== null) {
        state[1] = fy;
        updated = true;
      }
    }
    if (updated) {
      changed.add('freq_octaves_ridges');
      return true;
    }
    logWarn('updateParams: freq expects a number or { x, y } object.');
    return false;
  },
  freqx(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: freqX must be a finite number.');
      return false;
    }
    state[0] = numeric;
    changed.add('freq_octaves_ridges');
    return true;
  },
  freqy(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: freqY must be a finite number.');
      return false;
    }
    state[1] = numeric;
    changed.add('freq_octaves_ridges');
    return true;
  },
  octaves(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: octaves must be a finite number.');
      return false;
    }
    state[2] = numeric;
    changed.add('freq_octaves_ridges');
    return true;
  },
  ridges(state, value, changed) {
    return setBooleanFlag(state, 3, value, changed, 'freq_octaves_ridges');
  },
  sinamount(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: sinAmount must be a finite number.');
      return false;
    }
    state[4] = numeric;
    changed.add('sin_spline_distrib_corners');
    return true;
  },
  splineorder(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: splineOrder must be a finite number.');
      return false;
    }
    state[5] = numeric;
    changed.add('sin_spline_distrib_corners');
    return true;
  },
  distrib(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: distrib must be a finite number.');
      return false;
    }
    logInfo(`Setting distrib: state[6] = ${numeric} (was ${state[6]})`);
    state[6] = numeric;
    changed.add('sin_spline_distrib_corners');
    return true;
  },
  corners(state, value, changed) {
    return setBooleanFlag(state, 7, value, changed, 'sin_spline_distrib_corners');
  },
  mask(state, value, changed) {
    return setBooleanFlag(state, 8, value, changed, 'mask_options');
  },
  maskenabled(state, value, changed) {
    return setBooleanFlag(state, 8, value, changed, 'mask_options');
  },
  maskinverse(state, value, changed) {
    return setBooleanFlag(state, 9, value, changed, 'mask_options');
  },
  maskstatic(state, value, changed) {
    return setBooleanFlag(state, 10, value, changed, 'mask_options');
  },
  latticedrift(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: latticeDrift must be a finite number.');
      return false;
    }
    state[11] = numeric;
    changed.add('mask_options');
    return true;
  },
  supersample(state, value, changed) {
    return setBooleanFlag(state, 12, value, changed, 'supersample_color');
  },
  colorspace(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: colorSpace must be a finite number.');
      return false;
    }
    state[13] = numeric;
    changed.add('supersample_color');
    return true;
  },
  huerange(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: hueRange must be a finite number.');
      return false;
    }
    state[14] = numeric;
    changed.add('supersample_color');
    return true;
  },
  huerotation(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: hueRotation must be a finite number.');
      return false;
    }
    state[15] = numeric;
    changed.add('supersample_color');
    return true;
  },
  saturation(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: saturation must be a finite number.');
      return false;
    }
    state[16] = numeric;
    changed.add('saturation_hue');
    return true;
  },
  huedistrib(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: hueDistrib must be a finite number.');
      return false;
    }
    state[17] = numeric;
    changed.add('saturation_hue');
    return true;
  },
  saturationdistrib(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: saturationDistrib must be a finite number.');
      return false;
    }
    state[18] = numeric;
    changed.add('saturation_hue');
    return true;
  },
  brightnessdistrib(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: brightnessDistrib must be a finite number.');
      return false;
    }
    state[19] = numeric;
    changed.add('saturation_hue');
    return true;
  },
  brightnessfreq(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: brightnessFreq must be a finite number.');
      return false;
    }
    state[20] = numeric;
    state[21] = numeric;
    changed.add('brightness_settings');
    return true;
  },
  octaveblending(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: octaveBlending must be a finite number.');
      return false;
    }
    logInfo(`Setting octaveBlending: state[22] = ${numeric} (was ${state[22]})`);
    state[22] = numeric;
    changed.add('brightness_settings');
    return true;
  },
  withalpha(state, value, changed) {
    logInfo(`Setting withAlpha: state[23] = ${value ? 1 : 0} (was ${state[23]})`);
    return setBooleanFlag(state, 23, value, changed, 'brightness_settings');
  },
  speed(state, value, changed) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
      logWarn('updateParams: speed must be a finite number.');
      return false;
    }
    state[28] = numeric;
    changed.add('speed_padding');
    return true;
  },
};

async function updateParams(updates = {}) {
  if (!updates || typeof updates !== 'object') {
    throw new TypeError('updateParams expects an object of parameter overrides.');
  }

  const { device } = await getWebGPUState();
  const resources = await getComputeResources(device);

  let paramsState = resources.multiresParamsState;
  if (!paramsState) {
    paramsState = createDefaultMultiresParams();
    resources.multiresParamsState = paramsState;
  }

  const changed = new Set();

  for (const [rawKey, value] of Object.entries(updates)) {
    const normalizedKey = normalizeParamKey(rawKey);

    if (!normalizedKey) {
      continue;
    }

  if (hasOwn(MULTIRES_ALIAS_SETTERS, normalizedKey)) {
      MULTIRES_ALIAS_SETTERS[normalizedKey](paramsState, value, changed);
      continue;
    }

    const canonicalKey = rawKey;
  if (hasOwn(MULTIRES_PARAM_FIELD_OFFSETS, canonicalKey)) {
      const offset = MULTIRES_PARAM_FIELD_OFFSETS[canonicalKey];
      applyVectorOverride(paramsState, offset, canonicalKey, value, changed);
      continue;
    }

    logWarn(`updateParams: Unrecognized key '${rawKey}'.`);
  }

  const updated = Array.from(changed);

  if (updated.length > 0 && resources.multiresParamsBuffer) {
    device.queue.writeBuffer(resources.multiresParamsBuffer, 0, paramsState);
    logInfo(`Updated multires params: ${updated.join(', ')}`);
    logInfo(`Full params state (first 8 values): ${Array.from(paramsState.slice(0, 8))}`);
    logInfo(`brightness_settings [20-23]: ${Array.from(paramsState.slice(20, 24))}`);
  } else if (updated.length === 0) {
    logInfo('updateParams invoked with no effective changes.');
  }

  const valueSnapshot = {};
  updated.forEach((fieldName) => {
    const offset = MULTIRES_PARAM_FIELD_OFFSETS[fieldName];
    if (typeof offset === 'number') {
      valueSnapshot[fieldName] = Array.from(paramsState.slice(offset, offset + 4));
    }
  });

  return { updated, values: valueSnapshot };
}

async function getBlitPipeline(device, format) {
  ensureCachesMatchDevice(device);
  const cacheKey = `blit|${format}`;
  let pipeline = pipelineCache.get(cacheKey);
  if (pipeline) {
    return pipeline;
  }

  const { blitPipelineLayout } = await getComputeResources(device);
  let blitShaderModule = blitShaderModuleCache.get('blit');
  if (!blitShaderModule) {
    blitShaderModule = await compileShaderModuleWithValidation(device, BLIT_SHADER, { label: 'blit shader' });
    blitShaderModuleCache.set('blit', blitShaderModule);
  }

  device.pushErrorScope('validation');
  try {
    pipeline = device.createRenderPipeline({
      layout: blitPipelineLayout,
      vertex: {
        module: blitShaderModule,
        entryPoint: 'vertex_main',
      },
      fragment: {
        module: blitShaderModule,
        entryPoint: 'fragment_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  } catch (error) {
    await device.popErrorScope();
    fatal(`Failed to create blit pipeline: ${error?.message ?? error}`);
  }

  const blitPipelineError = await device.popErrorScope();
  if (blitPipelineError) {
    fatal(`Blit pipeline validation failed: ${blitPipelineError?.message ?? blitPipelineError}`);
  }

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

async function runMultiresGenerator(options = {}) {
  const { seed = 0, time = 0.0, frameIndex = 0 } = options;

  // Check if canvas was resized and we need to recreate resources
  if (needsResourceRecreation) {
    needsResourceRecreation = false;
    // Invalidate cached resources so they get recreated at new size
    invalidateComputeResources();
    if (effectManager.activeEffectInstance?.invalidateResources) {
      effectManager.activeEffectInstance.invalidateResources();
    }
  }

  // Wrap entire render pipeline in error handling
  try {
    const { device, webgpuContext } = await getWebGPUState(options);
  const computeResources = await getComputeResources(device);
  const {
    frameUniformsBuffer,
    multiresParamsBuffer,
    computePipeline,
    computeBindGroup,
    blitBindGroup,
    shaderMetadata,
    frameUniformsState,
    multiresParamsState,
  } = computeResources;

  const effect = await effectManager.ensureActiveEffect();
  const effectResources = await effect.ensureResources({
    device,
    width: canvas.width,
    height: canvas.height,
    multiresResources: computeResources,
  });

  const blitPipeline = await getBlitPipeline(device, webgpuContext.format);

  const workgroupSize = shaderMetadata?.workgroupSize ?? [8, 8, 1];
  const workgroupX = Math.max(workgroupSize[0] ?? 8, 1);
  const workgroupY = Math.max(workgroupSize[1] ?? 8, 1);
  const workgroupZ = Math.max(workgroupSize[2] ?? 1, 1);
  const dispatchX = Math.ceil(canvas.width / workgroupX);
  const dispatchY = Math.ceil(canvas.height / workgroupY);
  const dispatchZ = Math.max(Math.ceil(1 / workgroupZ), 1);

  const effectWorkgroup = effectResources.workgroupSize ?? [8, 8, 1];
  const effectDispatchX = Math.ceil(canvas.width / Math.max(effectWorkgroup[0], 1));
  const effectDispatchY = Math.ceil(canvas.height / Math.max(effectWorkgroup[1], 1));
  const effectDispatchZ = Math.max(Math.ceil(1 / Math.max(effectWorkgroup[2], 1)), 1);

  const convertWorkgroup = effectResources.bufferToTextureWorkgroupSize ?? [8, 8, 1];
  const convertDispatchX = Math.ceil(canvas.width / Math.max(convertWorkgroup[0], 1));
  const convertDispatchY = Math.ceil(canvas.height / Math.max(convertWorkgroup[1], 1));
  const convertDispatchZ = Math.max(Math.ceil(1 / Math.max(convertWorkgroup[2], 1)), 1);

  // Update uniforms
  const frameUniforms = frameUniformsState ?? new Float32Array(8);
  if (!frameUniformsState) {
    computeResources.frameUniformsState = frameUniforms;
  }
  frameUniforms[0] = canvas.width;
  frameUniforms[1] = canvas.height;
  frameUniforms[2] = time;
  frameUniforms[3] = seed;
  frameUniforms[4] = frameIndex;
  if (frameUniformsBuffer) {
    device.queue.writeBuffer(frameUniformsBuffer, 0, frameUniforms);
  }

  let multiresParams = multiresParamsState;
  if (!multiresParams) {
    multiresParams = createDefaultMultiresParams();
    computeResources.multiresParamsState = multiresParams;
  }
  if (multiresParamsBuffer) {
    device.queue.writeBuffer(multiresParamsBuffer, 0, multiresParams);
  }

  const effectParams = effectResources.paramsState;
  if (effectParams) {
    const bindingOffsets = effectResources.bindingOffsets ?? {};
    const channelCount = 4;

    const widthOffset = bindingOffsets.width;
    const heightOffset = bindingOffsets.height;
    const channelCountOffset = bindingOffsets.channelCount;
    const timeOffset = bindingOffsets.time;

    if (Number.isInteger(widthOffset) && effectParams[widthOffset] !== canvas.width) {
      effectParams[widthOffset] = canvas.width;
      effectResources.paramsDirty = true;
    } else if (!Number.isInteger(widthOffset)) {
      logWarn('Active effect metadata missing width binding offset; skipping width update.');
    }

    if (Number.isInteger(heightOffset) && effectParams[heightOffset] !== canvas.height) {
      effectParams[heightOffset] = canvas.height;
      effectResources.paramsDirty = true;
    } else if (!Number.isInteger(heightOffset)) {
      logWarn('Active effect metadata missing height binding offset; skipping height update.');
    }

    if (Number.isInteger(channelCountOffset) && effectParams[channelCountOffset] !== channelCount) {
      effectParams[channelCountOffset] = channelCount;
      effectResources.paramsDirty = true;
    } else if (!Number.isInteger(channelCountOffset)) {
      logWarn('Active effect metadata missing channel_count binding offset; skipping channel update.');
    }

    if (Number.isInteger(timeOffset) && effectParams[timeOffset] !== time) {
      effectParams[timeOffset] = time;
      effectResources.paramsDirty = true;
    } else if (!Number.isInteger(timeOffset)) {
      logWarn('Active effect metadata missing time binding offset; skipping time update.');
    }

    if (effectResources.paramsDirty && effectResources.paramsBuffer) {
      device.queue.writeBuffer(effectResources.paramsBuffer, 0, effectParams);
      effectResources.paramsDirty = false;
    }
  }

  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
  computePass.end();

  if (effectResources.enabled && effectResources.bufferToTexturePipeline) {
    const computePasses = Array.isArray(effectResources.computePasses)
      ? effectResources.computePasses
      : null;

    if (computePasses && computePasses.length > 0) {
      // Call beforeDispatch BEFORE multi-pass loop for buffer swapping
      if (effect.beforeDispatch && typeof effect.beforeDispatch === 'function') {
        effect.beforeDispatch({ device, multiresResources: computeResources, encoder });
      }

      for (const passConfig of computePasses) {
        if (!passConfig) {
          continue;
        }

        const { pipeline, bindGroup } = passConfig;
        if (!pipeline || !bindGroup) {
          continue;
        }

        const inheritedWorkgroup = Array.isArray(passConfig.workgroupSize)
          ? passConfig.workgroupSize
          : Array.isArray(effectResources.workgroupSize)
            ? effectResources.workgroupSize
            : [8, 8, 1];
        if (!Array.isArray(passConfig.workgroupSize) && effect?.constructor?.id === 'dla') {
          console.warn('DLA pass missing workgroup size!', computePasses.indexOf(passConfig));
        }

        let dispatchValues = null;
        if (typeof passConfig.getDispatch === 'function') {
          const computed = passConfig.getDispatch({
            width: canvas.width,
            height: canvas.height,
            workgroupSize: inheritedWorkgroup,
          });
          if (Array.isArray(computed)) {
            dispatchValues = computed;
          }
        }

        if (!dispatchValues && Array.isArray(passConfig.dispatch)) {
          dispatchValues = passConfig.dispatch;
        }

        if (!dispatchValues) {
          const [wgx, wgy, wgz] = inheritedWorkgroup;
          dispatchValues = [
            Math.ceil(canvas.width / Math.max(wgx ?? 8, 1)),
            Math.ceil(canvas.height / Math.max(wgy ?? 8, 1)),
            Math.max(Math.ceil(1 / Math.max(wgz ?? 1, 1)), 1),
          ];
        }

        const dispatchX = Math.max(Math.trunc(dispatchValues[0] ?? 0), 0);
        const dispatchY = Math.max(Math.trunc(dispatchValues[1] ?? 0), 0);
        const dispatchZ = Math.max(Math.trunc(dispatchValues[2] ?? 1), 1);

        if (dispatchX === 0 || dispatchY === 0) {
          continue;
        }

        const effectPass = encoder.beginComputePass();
        effectPass.setPipeline(pipeline);
        effectPass.setBindGroup(0, bindGroup);
        effectPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        effectPass.end();
      }

      if (effect.afterDispatch && typeof effect.afterDispatch === 'function') {
        effect.afterDispatch();
      }
    } else if (effectResources.computePipeline && effectResources.computeBindGroup) {
      // Call beforeDispatch if the effect has agent buffers that need swapping
      if (effect.beforeDispatch && typeof effect.beforeDispatch === 'function') {
        effect.beforeDispatch({ device, multiresResources: computeResources });
      }

      const effectPass = encoder.beginComputePass();
      effectPass.setPipeline(effectResources.computePipeline);
      effectPass.setBindGroup(0, effectResources.computeBindGroup);
      effectPass.dispatchWorkgroups(effectDispatchX, effectDispatchY, effectDispatchZ);
      effectPass.end();

      // Call afterDispatch to swap the buffer marker
      if (effect.afterDispatch && typeof effect.afterDispatch === 'function') {
        effect.afterDispatch();
      }
    }

    const convertPass = encoder.beginComputePass();
    convertPass.setPipeline(effectResources.bufferToTexturePipeline);
    convertPass.setBindGroup(0, effectResources.bufferToTextureBindGroup);
    convertPass.dispatchWorkgroups(convertDispatchX, convertDispatchY, convertDispatchZ);
    convertPass.end();
  }

  if (effectResources.enabled && effectResources.shouldCopyOutputToPrev) {
    const sourceTexture = effectResources.outputTexture ?? null;
    const targetTexture = effectResources.feedbackTexture
      ?? effectResources.resourceSet?.textures?.prev_texture
      ?? null;
    const copyWidth = Math.max(Math.trunc(effectResources.textureWidth ?? canvas.width ?? 0), 0);
    const copyHeight = Math.max(Math.trunc(effectResources.textureHeight ?? canvas.height ?? 0), 0);

    if (
      sourceTexture
      && targetTexture
      && targetTexture !== sourceTexture
      && copyWidth > 0
      && copyHeight > 0
    ) {
      encoder.copyTextureToTexture(
        { texture: sourceTexture },
        { texture: targetTexture },
        { width: copyWidth, height: copyHeight, depthOrArrayLayers: 1 },
      );
    }
  }

  const canvasView = webgpuContext.getCurrentTextureView();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: canvasView,
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
  });
  renderPass.setPipeline(blitPipeline);
  const finalBlitBindGroup = (effectResources.enabled && effectResources.blitBindGroup)
    ? effectResources.blitBindGroup
    : blitBindGroup;
  
  renderPass.setBindGroup(0, finalBlitBindGroup);
  renderPass.draw(3, 1, 0, 0);
  renderPass.end();

  device.queue.submit([encoder.finish()]);
  } catch (error) {
    // Halt animation on any error during rendering
    logError('Render pipeline failed:', error);
    throw error;
  }
}

export {
  runMultiresGenerator,
  updateParams,
  setActiveEffect,
  updateActiveEffectParams,
  getRegisteredEffects,
  getActiveEffectMetadata,
  getActiveEffectUIState,
  createDefaultMultiresParams,
  readActiveEffectOutputFloats,
};

// -------------------------------
// Step 7: Regression/Test Helpers
// -------------------------------

async function compileShader(shaderId) {
  const { device } = await getWebGPUState();
  try {
    const descriptor = getShaderDescriptor(shaderId);
    const source = await loadShaderSource(shaderId);
    const module = device.createShaderModule({ code: source, label: descriptor.label });
    const info = (await module.getCompilationInfo?.()) ?? { messages: [] };
    const messages = info.messages ?? [];
    const errors = messages.filter((m) => m.type === 'error');
    return { ok: errors.length === 0, messages };
  } catch (error) {
    return { ok: false, messages: [{ type: 'error', message: error?.message ?? String(error) }] };
  }
}

function toHex(buffer) {
  const bytes = new Uint8Array(buffer);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return hex;
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return toHex(digest);
}

async function readOutputTextureFloats() {
  const { device } = await getWebGPUState();
  const resources = await getComputeResources(device);
  const { outputTexture, textureWidth: width, textureHeight: height } = resources;

  const bytesPerPixel = 16; // rgba32float
  const rowBytes = width * bytesPerPixel;
  const paddedBytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const bufferSize = paddedBytesPerRow * height;
  if (!cachedReadbackBuffer || cachedReadbackSize !== bufferSize) {
    if (cachedReadbackBuffer?.destroy) {
      try {
        cachedReadbackBuffer.destroy();
      } catch (error) {
        logWarn('Failed to destroy readback buffer during resize:', error);
      }
    }
    cachedReadbackBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    cachedReadbackSize = bufferSize;
  }

  const readback = cachedReadbackBuffer;

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: outputTexture },
    { buffer: readback, bytesPerRow: paddedBytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
  device.queue.submit([encoder.finish()]);
  try {
    await device.queue.onSubmittedWorkDone();
  } catch (err) {
    logWarn('queue.onSubmittedWorkDone failed; proceeding to mapAsync anyway:', err?.message ?? err);
  }

  let usedFallback = false;
  let mapSucceeded = false;
  try {
    await readback.mapAsync(GPUMapMode.READ);
    mapSucceeded = true;
  } catch (err) {
    usedFallback = true;
    logWarn('readback.mapAsync failed; returning zero-filled buffer fallback:', err?.message ?? err);
  }

  let floats;
  if (mapSucceeded) {
    const mapped = readback.getMappedRange();

    const tight = new ArrayBuffer(rowBytes * height);
    const tightView = new Uint8Array(tight);
    const src = new Uint8Array(mapped);
    for (let y = 0; y < height; y += 1) {
      const srcOffset = y * paddedBytesPerRow;
      const dstOffset = y * rowBytes;
      tightView.set(src.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
    }
    readback.unmap();
    floats = new Float32Array(tight);
  } else {
    const floatCount = (rowBytes * height) >> 2; // bytes / 4
    floats = new Float32Array(floatCount);
  }

  Object.defineProperty(floats, '__usedFallback', {
    value: usedFallback,
    enumerable: false,
    configurable: true,
    writable: false,
  });

  return floats;
}

async function readActiveEffectOutputFloats() {
  const { device } = await getWebGPUState();
  const effectInstance = effectManager.activeEffectInstance;
  const resources = effectInstance?.resources ?? null;
  if (!resources || !resources.outputTexture) {
    throw new Error('Active effect has no output texture available.');
  }

  const texture = resources.outputTexture;
  const width = Math.max(1, resources.textureWidth ?? canvas.width);
  const height = Math.max(1, resources.textureHeight ?? canvas.height);

  const bytesPerPixel = 16;
  const rowBytes = width * bytesPerPixel;
  const paddedBytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const bufferSize = paddedBytesPerRow * height;

  if (!cachedEffectReadbackBuffer || cachedEffectReadbackSize !== bufferSize) {
    if (cachedEffectReadbackBuffer?.destroy) {
      try {
        cachedEffectReadbackBuffer.destroy();
      } catch (error) {
        logWarn('Failed to destroy effect readback buffer during resize:', error);
      }
    }
    cachedEffectReadbackBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    cachedEffectReadbackSize = bufferSize;
  }

  const readback = cachedEffectReadbackBuffer;

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: readback, bytesPerRow: paddedBytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
  device.queue.submit([encoder.finish()]);

  try {
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    logWarn('queue.onSubmittedWorkDone failed during effect readback; proceeding anyway:', error?.message ?? error);
  }

  let mapSucceeded = false;
  try {
    await readback.mapAsync(GPUMapMode.READ);
    mapSucceeded = true;
  } catch (error) {
    logWarn('Effect readback mapAsync failed; returning zero-filled fallback:', error?.message ?? error);
  }

  if (!mapSucceeded) {
    return new Float32Array((rowBytes * height) >> 2);
  }

  const mapped = readback.getMappedRange();
  const tight = new ArrayBuffer(rowBytes * height);
  const tightView = new Uint8Array(tight);
  const src = new Uint8Array(mapped);
  for (let y = 0; y < height; y += 1) {
    const srcOffset = y * paddedBytesPerRow;
    const dstOffset = y * rowBytes;
    tightView.set(src.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
  }
  readback.unmap();
  return new Float32Array(tight);
}

// Returns a SHA-256 checksum of the generator output texture (pre-blit).
// Swapchain textures cannot be copied on most implementations, so hashing the
// storage texture is the deterministic alternative. Callers may pass
// `includeMetadata: true` to retrieve the fallback flag alongside the hash.
async function getOutputTextureHash(options = {}) {
  const { floats, includeMetadata = false } = options ?? {};
  const data = floats ?? await readOutputTextureFloats();
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const checksum = await sha256(bytes);
  if (includeMetadata) {
    return { checksum, usedFallback: Boolean(data.__usedFallback) };
  }
  return checksum;
}

async function runMultiresOnce(options = {}) {
  await runMultiresGenerator(options);
  const floats = await readOutputTextureFloats();
  const hashResult = await getOutputTextureHash({ floats, includeMetadata: true });
  const floatSample = Array.from(floats.subarray(0, 4));
  return {
    checksum: hashResult.checksum,
    floatSample,
    usedFallback: Boolean(hashResult.usedFallback),
  };
}

// Expose helpers for headless automation (Playwright/Puppeteer)
if (typeof window !== 'undefined') {
  window.__demo__ = Object.assign({}, window.__demo__ || {}, {
    compileShader,
    runMultiresOnce,
    getOutputTextureHash,
    readActiveEffectOutputFloats,
    updateParams,
    setActiveEffect,
    updateActiveEffectParams,
    getRegisteredEffects,
    getActiveEffectMetadata,
    getActiveEffectUIState,
    logs: demoLogs,
    clearLogs() {
      demoLogs.splice(0, demoLogs.length);
    },
  });
}
