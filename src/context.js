/**
 * WebGPU context helper with CPU fallback and shader utilities.
 */

import { Random, setSeed, getSeed, random } from './rng.js';

export { Random, setSeed, getSeed, random };

// PresentParams packs two vec4<f32> plus a scalar channel count. WGSL uniform
// buffers align structs to 16-byte boundaries, so the GPU requires 64 bytes
// (16 floats) even though only 9 values carry data.
const PRESENT_PARAMS_FLOATS = 16;
const PRESENT_PARAMS_SIZE = PRESENT_PARAMS_FLOATS * 4;
const RANGE_EPSILON = 1e-6;

const PRESENT_STATS_WGSL = `
struct PresentStatsParams {
  width: u32,
  height: u32,
  channels: u32,
  _pad: u32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> params: PresentStatsParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u || gid.y != 0u) {
    return;
  }
  let w = max(params.width, 1u);
  let h = max(params.height, 1u);
  let c = max(params.channels, 1u);
  var minv = vec4<f32>(1e20, 1e20, 1e20, 1e20);
  var maxv = vec4<f32>(-1e20, -1e20, -1e20, -1e20);
  for (var y: u32 = 0u; y < h; y = y + 1u) {
    for (var x: u32 = 0u; x < w; x = x + 1u) {
      let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
      if (c > 0u) {
        let v = col.x;
        minv.x = min(minv.x, v);
        maxv.x = max(maxv.x, v);
      }
      if (c > 1u) {
        let v = col.y;
        minv.y = min(minv.y, v);
        maxv.y = max(maxv.y, v);
      }
      if (c > 2u) {
        let v = col.z;
        minv.z = min(minv.z, v);
        maxv.z = max(maxv.z, v);
      }
      if (c > 3u) {
        let v = col.w;
        minv.w = min(minv.w, v);
        maxv.w = max(maxv.w, v);
      }
    }
  }
  if (c == 1u) {
    minv.y = minv.x;
    maxv.y = maxv.x;
    minv.z = minv.x;
    maxv.z = maxv.x;
    minv.w = 0.0;
    maxv.w = 1.0;
  } else if (c == 2u) {
    minv.z = minv.x;
    maxv.z = maxv.x;
    minv.w = minv.y;
    maxv.w = maxv.y;
  } else if (c == 3u) {
    minv.w = 0.0;
    maxv.w = 1.0;
  }
  stats[0] = minv.x;
  stats[1] = maxv.x;
  stats[2] = minv.y;
  stats[3] = maxv.y;
  stats[4] = minv.z;
  stats[5] = maxv.z;
  stats[6] = minv.w;
  stats[7] = maxv.w;
}
`;

function inferChannelsFromFormat(format) {
  switch (format) {
    case 'r8unorm':
    case 'r16float':
    case 'r32float':
      return 1;
    case 'rg8unorm':
    case 'rg16float':
    case 'rg32float':
      return 2;
    case 'rgba8unorm':
    case 'rgba8unorm-srgb':
    case 'bgra8unorm':
    case 'bgra8unorm-srgb':
    case 'rgba16float':
    case 'rgba32float':
      return 4;
    default:
      return 4;
  }
}

export class Context {
  constructor(canvas, debug = false, powerPreference = 'high-performance') {
    this.canvas = canvas;
    this.debug = debug;
    this.powerPreference = powerPreference;
    this.gpu = canvas && canvas.getContext ? canvas.getContext('webgpu') : null;
    this.device = null;
    this.queue = null;
    this.presentationFormat = null;
    this.isCPU = !this.gpu;
    this.currentTarget = null;
    this._renderPipeline = null;
    this._renderSampler = null;
    this._renderParamsBuffer = null;
    this._renderParamsArray = new Float32Array(PRESENT_PARAMS_FLOATS);
    this._pipelineCache = new Map();
    this._pendingDispatch = false;
    this._encoder = null;
    this._workDonePromise = null;
  }

  async initWebGPU() {
    if (this.device || !this.gpu || typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: this.powerPreference,
    });
    if (!adapter) return false;
    this.device = await adapter.requestDevice();
    this.queue = this.device.queue;
    this.presentationFormat =
      navigator.gpu.getPreferredCanvasFormat ?
        navigator.gpu.getPreferredCanvasFormat() :
        this.gpu.getPreferredFormat?.(adapter) || 'bgra8unorm';
    this.gpu.configure({ device: this.device, format: this.presentationFormat });
    this.isCPU = false;
    return true;
  }

  createShaderModule(code) {
    if (!this.device) throw new Error('WebGPU device not initialized');
    return this.device.createShaderModule({ code });
  }

  async createComputePipeline(code) {
    if (!this.device) throw new Error('WebGPU device not initialized');
    const module = this.createShaderModule(code);
    if (module.getCompilationInfo) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) {
        const details = errors
          .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
          .join('\n');
        throw new Error(`Shader compilation failed:\n${details}`);
      }
    }
    return this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  createRenderPipeline(vsCode, fsCode, targets = [{ format: this.presentationFormat }]) {
    if (!this.device) throw new Error('WebGPU device not initialized');
    return this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: this.createShaderModule(vsCode), entryPoint: 'main' },
      fragment: { module: this.createShaderModule(fsCode), entryPoint: 'main', targets },
      primitive: { topology: 'triangle-list' },
    });
  }

  createBindGroup(pipeline, entries, index = 0) {
    if (!this.device) throw new Error('WebGPU device not initialized');
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(index),
      entries,
    });
  }

  beginRenderPass(encoder, view, clearValue = [0, 0, 0, 1]) {
    return encoder.beginRenderPass({
      colorAttachments: [
        { view, loadOp: 'clear', storeOp: 'store', clearValue },
      ],
    });
  }

  beginComputePass(encoder) {
    return encoder.beginComputePass();
  }

  createTexture(width, height, data = null) {
    // WebGPU textures are always RGBA float32. If the caller provides fewer
    // channels, pad the data so each pixel occupies four components. This
    // prevents stride misalignment when uploading to the GPU.
    const padChannels = (arr) => {
      if (!arr) return new Float32Array(width * height * 4);
      if (arr.length === width * height * 4) return new Float32Array(arr);
      const pixels = width * height;
      const srcChannels = arr.length / pixels;
      const out = new Float32Array(pixels * 4);
      for (let i = 0; i < pixels; i++) {
        const src = i * srcChannels;
        const dst = i * 4;
        const r = arr[src];
        const g = srcChannels > 1 ? arr[src + 1] : r;
        const b = srcChannels > 2 ? arr[src + 2] : r;
        const a =
          srcChannels > 3 ? arr[src + 3] : srcChannels === 2 ? arr[src + 1] : 1;
        out[dst] = r;
        out[dst + 1] = g;
        out[dst + 2] = b;
        out[dst + 3] = a;
      }
      return out;
    };

    if (this.isCPU || !this.device) {
      const arr = padChannels(data ? (data instanceof Float32Array ? data : new Float32Array(data)) : null);
      return { width, height, channels: 4, data: arr };
    }
    const texture = this.device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    try {
      texture._noisemakerShape = [height, width, 4];
      texture._noisemakerChannels = 4;
    } catch (_) {
      // ignore if metadata assignment fails
    }
    if (data) {
      const arr = padChannels(data instanceof Float32Array ? data : new Float32Array(data));
      const bytesPerPixel = 4 * 4;
      const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
      let upload = arr;
      if (bytesPerRow !== width * bytesPerPixel) {
        const stride = bytesPerRow / 4;
        const padded = new Float32Array(stride * height);
        for (let y = 0; y < height; y++) {
          padded.set(arr.subarray(y * width * 4, (y + 1) * width * 4), y * stride);
        }
        upload = padded;
      }
      this.queue.writeTexture(
        { texture },
        upload,
        { bytesPerRow },
        { width, height, depthOrArrayLayers: 1 }
      );
    }
    return texture;
  }

  createFBO(texture) {
    if (this.isCPU) return { texture };
    return texture.createView();
  }

  bindFramebuffer(fbo, width, height) {
    if (this.isCPU) return;
    this.currentTarget = { view: fbo, width, height };
  }

  drawQuad(pipeline, bindGroup) {
    if (this.isCPU) return;
    const encoder = this._encoder || this.device.createCommandEncoder();
    const view =
      this.currentTarget?.view || this.gpu.getCurrentTexture().createView();
    const pass = this.beginRenderPass(encoder, view);
    pass.setPipeline(pipeline);
    if (bindGroup) pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    if (!this._encoder) {
      this.queue.submit([encoder.finish()]);
      this._pendingDispatch = true;
    }
  }

  // `target` may be a GPUTexture, GPUTextureView, or a callback returning
  // either, allowing callers to defer swap chain acquisition until the render
  // pass is ready to encode.
  async renderTexture(source, target = null) {
    if (this.isCPU || !this.device) return;
    const info = this._resolveTextureInfo(source);
    if (!info) return;
    const { texture, width, height, channels, normalized } = info;
    const channelCount = Math.min(Math.max(Math.floor(channels), 1), 4);
    const range = await this._computePresentRange(
      texture,
      width,
      height,
      channelCount,
      Boolean(normalized),
    );
    if (!range) return;
    const { minVals, invRange } = range;
    if (!this._renderPipeline) {
      const vs = `
        struct VSOut {
          @builtin(position) pos: vec4<f32>,
        };
        @vertex
        fn main(@builtin(vertex_index) idx : u32) -> VSOut {
          var pos = array<vec2<f32>,6>(
            vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
            vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0)
          );
          var out: VSOut;
          out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
          return out;
        }
      `;
      const fs = `
        struct PresentParams {
          minVals: vec4<f32>,
          invRange: vec4<f32>,
          channels: f32,
          padding: vec3<f32>,
        };
        @group(0) @binding(0) var tex : texture_2d<f32>;
        @group(0) @binding(1) var<uniform> params : PresentParams;
        @fragment
        fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let dims = textureDimensions(tex);
          let width = max(i32(dims.x), 1);
          let height = max(i32(dims.y), 1);
          let pixel = clamp(
            vec2<i32>(floor(fragCoord.xy)),
            vec2<i32>(0, 0),
            vec2<i32>(width - 1, height - 1)
          );
          let color = textureLoad(tex, pixel, 0);
          let adjusted = (color - params.minVals) * params.invRange;
          let channels = params.channels;
          var rgb: vec3<f32>;
          var alpha: f32;
          if (channels < 1.5) {
            let gray = adjusted.x;
            rgb = vec3<f32>(gray, gray, gray);
            alpha = 1.0;
          } else if (channels < 2.5) {
            let gray = adjusted.x;
            rgb = vec3<f32>(gray, gray, gray);
            alpha = adjusted.y;
          } else if (channels < 3.5) {
            rgb = adjusted.xyz;
            alpha = 1.0;
          } else {
            rgb = adjusted.xyz;
            alpha = adjusted.w;
          }
          let clampedRgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
          let clampedAlpha = clamp(alpha, 0.0, 1.0);
          return vec4<f32>(clampedRgb, clampedAlpha);
        }
      `;
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
        ],
      });
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      });
      this._renderPipeline = this.device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: this.createShaderModule(vs), entryPoint: 'main' },
        fragment: {
          module: this.createShaderModule(fs),
          entryPoint: 'main',
          targets: [{ format: this.presentationFormat }],
        },
        primitive: { topology: 'triangle-list' },
      });
    }
    if (
      !this._renderParamsBuffer ||
      (this._renderParamsBuffer.size ?? this._renderParamsBuffer._size ?? 0) <
        PRESENT_PARAMS_SIZE
    ) {
      if (this._renderParamsBuffer?.destroy) {
        try {
          this._renderParamsBuffer.destroy();
        } catch (_) {
          // ignore destroy errors
        }
      }
      this._renderParamsBuffer = this.device.createBuffer({
        size: PRESENT_PARAMS_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._renderParamsBuffer._size = PRESENT_PARAMS_SIZE;
    }
    const paramsData = this._renderParamsArray;
    paramsData.fill(0);
    paramsData.set(minVals, 0);
    paramsData.set(invRange, 4);
    paramsData[8] = channelCount;
    this.queue.writeBuffer(
      this._renderParamsBuffer,
      0,
      paramsData.buffer,
      paramsData.byteOffset,
      paramsData.byteLength,
    );
    const bindGroup = this.device.createBindGroup({
      layout: this._renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: this._renderParamsBuffer } },
      ],
    });
    let resolvedTarget = target;
    if (typeof resolvedTarget === 'function') {
      resolvedTarget = resolvedTarget();
    }
    let targetView = null;
    if (resolvedTarget) {
      if (typeof resolvedTarget.createView === 'function') {
        targetView = resolvedTarget.createView();
      } else {
        targetView = resolvedTarget;
      }
    }
    const encoder = this._encoder || this.device.createCommandEncoder();
    const view = targetView || this.gpu.getCurrentTexture().createView();
    const pass = this.beginRenderPass(encoder, view);
    pass.setPipeline(this._renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    if (!this._encoder) {
      this.queue.submit([encoder.finish()]);
      this._pendingDispatch = true;
    }
  }

  _resolveTextureInfo(source) {
    let texture = source;
    let width = 0;
    let height = 0;
    let channels = 4;
    let normalized = false;
    const flag = '_noisemakerPresentationNormalized';
    if (source && typeof source === 'object') {
      if (source[flag]) normalized = Boolean(source[flag]);
      const maybeShape = source.shape;
      const maybeHandle = source.handle;
      if (
        Array.isArray(maybeShape) &&
        maybeShape.length >= 3 &&
        maybeHandle
      ) {
        height = maybeShape[0];
        width = maybeShape[1];
        channels = maybeShape[2];
        texture = maybeHandle;
      }
      if (
        !normalized &&
        maybeHandle &&
        typeof maybeHandle === 'object' &&
        maybeHandle[flag]
      ) {
        normalized = Boolean(maybeHandle[flag]);
      }
    }
    if (!texture) {
      return null;
    }
    if (!normalized && texture && typeof texture === 'object' && texture[flag]) {
      normalized = Boolean(texture[flag]);
    }
    const metaShape = texture._noisemakerShape;
    if (Array.isArray(metaShape) && metaShape.length >= 3) {
      height = height || metaShape[0];
      width = width || metaShape[1];
      channels = metaShape[2];
    }
    if (!width && typeof texture.width === 'number') width = texture.width;
    if (!height && typeof texture.height === 'number') height = texture.height;
    if (!channels && typeof texture._noisemakerChannels === 'number') {
      channels = texture._noisemakerChannels;
    }
    if (!channels && typeof texture.format === 'string') {
      channels = inferChannelsFromFormat(texture.format);
    }
    channels = Math.min(Math.max(Math.floor(channels || 4), 1), 4);
    if (!width || !height) {
      return null;
    }
    return { texture, width, height, channels, normalized };
  }

  async _computePresentRange(texture, width, height, channels, normalizedHint = false) {
    if (!this.device) return null;
    if (normalizedHint) {
      const minVals = new Float32Array(4);
      const invRange = new Float32Array(4);
      minVals.fill(0);
      invRange.fill(1);
      return { minVals, invRange };
    }
    const statsBuf = this.device.createBuffer({
      size: 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    statsBuf._size = 8 * 4;
    const paramsArr = new Uint32Array([width, height, channels, 0]);
    const paramsBuf = this.createGPUBuffer(
      paramsArr,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    try {
      await this.runCompute(
        PRESENT_STATS_WGSL,
        [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: { buffer: statsBuf } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ],
        1,
        1,
        1,
      );
      const stats = await this.readGPUBuffer(statsBuf, 8 * 4);
      const minVals = new Float32Array(4);
      const invRange = new Float32Array(4);
      const normalized = new Array(4).fill(false);
      for (let i = 0; i < 4; i++) {
        const minVal = stats[i * 2];
        const maxVal = stats[i * 2 + 1];
        const range = maxVal - minVal;
        const finite = Number.isFinite(minVal) && Number.isFinite(maxVal);
        if (!finite || Math.abs(range) <= RANGE_EPSILON) {
          minVals[i] = 0;
          invRange[i] = 1;
        } else {
          minVals[i] = minVal;
          invRange[i] = 1 / range;
        }
        if (finite) {
          const withinUnit = minVal >= -RANGE_EPSILON && maxVal <= 1 + RANGE_EPSILON;
          const touchesZero = minVal <= RANGE_EPSILON;
          const touchesOne = maxVal >= 1 - RANGE_EPSILON;
          const flat = Math.abs(range) <= RANGE_EPSILON;
          // Treat the channel as already normalized only when it stays within the
          // unit interval *and* either spans the [0, 1] endpoints or is a flat
          // line at one of them (e.g. an opaque alpha channel). This avoids
          // skipping normalization for palettes that merely sit inside the unit
          // interval without actually using its extremes.
          if (withinUnit && ((touchesZero && touchesOne) || (flat && (touchesZero || touchesOne)))) {
            normalized[i] = true;
          }
        }
      }
      const activeChannels = Math.min(Math.max(channels, 1), 4);
      const alreadyNormalized = normalized
        .slice(0, activeChannels)
        .every(Boolean);
      if (alreadyNormalized) {
        minVals.fill(0);
        invRange.fill(1);
      }
      return { minVals, invRange };
    } finally {
      if (paramsBuf && paramsBuf.destroy) {
        try {
          paramsBuf.destroy();
        } catch (_) {
          /* ignore */
        }
      }
      if (statsBuf && statsBuf.destroy) {
        try {
          statsBuf.destroy();
        } catch (_) {
          /* ignore */
        }
      }
    }
  }

  createGPUBuffer(array, usage) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    let byteLength = array.byteLength;
    if (usage & GPUBufferUsage.UNIFORM) {
      byteLength = Math.ceil(byteLength / 16) * 16;
    }
    const buf = this.device.createBuffer({
      size: byteLength,
      usage,
      mappedAtCreation: true,
    });
    new array.constructor(buf.getMappedRange()).set(array);
    buf.unmap();
    buf._size = byteLength;
    return buf;
  }

  async flush() {
    if (!this._pendingDispatch || !this.queue?.onSubmittedWorkDone) {
      return;
    }
    if (!this._workDonePromise) {
      const start = this.profile ? performance.now() : 0;
      this._workDonePromise = (async () => {
        try {
          await this.queue.onSubmittedWorkDone();
          if (this.profile) {
            this.profile.webgpu += performance.now() - start;
          }
          this._pendingDispatch = false;
        } finally {
          this._workDonePromise = null;
        }
      })();
    }
    await this._workDonePromise;
  }

  async withEncoder(callback) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    if (this._encoder) {
      return await callback(this._encoder);
    }
    const encoder = this.device.createCommandEncoder();
    this._encoder = encoder;
    try {
      const result = await callback(encoder);
      this.queue.submit([encoder.finish()]);
      this._pendingDispatch = true;
      return result;
    } finally {
      this._encoder = null;
    }
  }

  async runCompute(code, bindEntries, x, y = 1, z = 1) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    const device = this.device;
    if (this.debug) device.pushErrorScope('validation');
    let pipeline = this._pipelineCache.get(code);
    if (!pipeline) {
      pipeline = await this.createComputePipeline(code);
      this._pipelineCache.set(code, pipeline);
    }
    if (this.debug) {
      const compileErr = await device.popErrorScope();
      if (compileErr) {
        throw compileErr;
      }
    }
    if (this.debug) device.pushErrorScope('validation');
    const bindGroup = this.createBindGroup(pipeline, bindEntries);
    const encoder = this._encoder || device.createCommandEncoder();
    const pass = this.beginComputePass(encoder);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    if (!this._encoder) {
      this.queue.submit([encoder.finish()]);
      this._pendingDispatch = true;
    }
    if (this.debug) {
      const err = await device.popErrorScope();
      if (err) {
        throw err;
      }
    }
  }

  async readGPUBuffer(buffer, size) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    const device = this.device;
    const bufSize = buffer.size || buffer._size || size;
    if (bufSize < size) {
      throw new Error(`Buffer too small: ${bufSize} < ${size}`);
    }
    const canMapDirectly = Boolean(buffer.usage & GPUBufferUsage.MAP_READ);
    const scopePushed = this.debug && !canMapDirectly;
    let staging = buffer;
    if (!canMapDirectly) {
      if (scopePushed) device.pushErrorScope('validation');
      staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
      this.queue.submit([encoder.finish()]);
      this._pendingDispatch = true;
    }
    let mapped = false;
    let result;
    try {
      await staging.mapAsync(GPUMapMode.READ, 0, size);
      mapped = true;
      const mappedRange = staging.getMappedRange(0, size);
      result = new Float32Array(mappedRange.slice(0));
    } finally {
      if (mapped) {
        staging.unmap();
      }
      if (!canMapDirectly && staging !== buffer && staging.destroy) {
        try {
          staging.destroy();
        } catch (_) {
          // ignore destroy failures
        }
      }
      if (scopePushed) {
        const err = await device.popErrorScope();
        if (err) {
          throw err;
        }
      }
    }
    return result;
  }

  pingPong(width, height) {
    return new PingPong(this, width, height);
  }
}

export class PingPong {
  constructor(ctx, width, height) {
    this.ctx = ctx;
    this.width = width;
    this.height = height;
    this.readTex = ctx.createTexture(width, height);
    this.writeTex = ctx.createTexture(width, height);
    this.readFbo = ctx.createFBO(this.readTex);
    this.writeFbo = ctx.createFBO(this.writeTex);
  }

  swap() {
    [this.readTex, this.writeTex] = [this.writeTex, this.readTex];
    [this.readFbo, this.writeFbo] = [this.writeFbo, this.readFbo];
  }
}
