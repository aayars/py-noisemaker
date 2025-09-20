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

const GPU_STORAGE_TEXTURE_USAGE =
  typeof GPUTextureUsage !== 'undefined'
    ? GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING
    : 0;

const STORAGE_TEXTURE_FORMAT = 'rgba32float';
const FRAME_UNIFORM_SIZE = 32;

const DATA_VIEW_WRITE_SIZES = {
  setFloat32: 4,
  setInt32: 4,
  setUint32: 4,
  setFloat16: 2,
  setInt16: 2,
  setUint16: 2,
  setInt8: 1,
  setUint8: 1,
  setBigInt64: 8,
  setBigUint64: 8,
  setFloat64: 8,
};

function alignTo(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : value + (alignment - remainder);
}

function alignDown(value, alignment) {
  return value - (value % alignment);
}

function createTrackedDataView(buffer, onWrite) {
  const view = new DataView(buffer);
  const handler = {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (typeof value === 'function') {
        if (Object.prototype.hasOwnProperty.call(DATA_VIEW_WRITE_SIZES, prop)) {
          const byteSize = DATA_VIEW_WRITE_SIZES[prop];
          return function trackedSetter(byteOffset, ...args) {
            onWrite(byteOffset, byteSize);
            return value.call(target, byteOffset, ...args);
          };
        }
        return value.bind(target);
      }
      return value;
    },
  };
  const proxy = new Proxy(view, handler);
  Object.defineProperty(proxy, 'markDirty', {
    value(offset = 0, size = buffer.byteLength) {
      onWrite(offset, size);
    },
  });
  Object.defineProperty(proxy, 'arrayBuffer', {
    get() {
      return buffer;
    },
  });
  return proxy;
}

class GPUUniformBufferEntry {
  constructor(ctx, size) {
    this.ctx = ctx;
    this.size = size;
    this.buffers = [
      ctx.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      ctx.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    ];
    this.arrays = [new ArrayBuffer(size), new ArrayBuffer(size)];
    this.views = this.arrays.map((buffer, index) =>
      createTrackedDataView(buffer, (offset, byteSize) =>
        this.markDirty(index, offset, byteSize),
      ),
    );
    this.dirtyStart = [size, size];
    this.dirtyEnd = [0, 0];
    this.inUse = false;
  }

  acquire() {
    this.inUse = true;
    return this;
  }

  release() {
    this.inUse = false;
  }

  getArrayBuffer(index) {
    return this.arrays[index];
  }

  getView(index) {
    return this.views[index];
  }

  getGPUBuffer(index) {
    return this.buffers[index];
  }

  markDirty(index, offset, byteSize) {
    if (!Number.isFinite(offset) || !Number.isFinite(byteSize)) {
      return;
    }
    const start = Math.max(0, Math.min(this.size, offset));
    const end = Math.max(start, Math.min(this.size, offset + byteSize));
    this.dirtyStart[index] = Math.min(this.dirtyStart[index], start);
    this.dirtyEnd[index] = Math.max(this.dirtyEnd[index], end);
  }

  flush(index) {
    if (!this.ctx?.queue) return;
    const start = this.dirtyStart[index];
    const end = this.dirtyEnd[index];
    if (!(end > start)) {
      return;
    }
    const alignedStart = Math.max(0, alignDown(start, 4));
    const alignedEnd = Math.min(this.size, alignTo(end, 4));
    const size = Math.max(0, alignedEnd - alignedStart);
    if (!size) {
      this.resetDirty(index);
      return;
    }
    this.ctx.queue.writeBuffer(
      this.buffers[index],
      alignedStart,
      this.arrays[index],
      alignedStart,
      size,
    );
    this.resetDirty(index);
  }

  flushAll() {
    for (let i = 0; i < this.buffers.length; i++) {
      this.flush(i);
    }
  }

  resetDirty(index) {
    this.dirtyStart[index] = this.size;
    this.dirtyEnd[index] = 0;
  }

  destroy() {
    for (const buffer of this.buffers) {
      if (buffer && typeof buffer.destroy === 'function') {
        try {
          buffer.destroy();
        } catch (_) {
          /* ignore destroy failures */
        }
      }
    }
    this.buffers = [];
    this.arrays = [];
    this.views = [];
  }
}

class CPUUniformBufferEntry {
  constructor(size) {
    this.size = size;
    this.arrays = [new ArrayBuffer(size), new ArrayBuffer(size)];
    this.views = this.arrays.map((buffer) =>
      createTrackedDataView(buffer, () => {}),
    );
    this.inUse = false;
  }

  acquire() {
    this.inUse = true;
    return this;
  }

  release() {
    this.inUse = false;
  }

  getArrayBuffer(index) {
    return this.arrays[index];
  }

  getView(index) {
    return this.views[index];
  }

  getGPUBuffer() {
    return null;
  }

  flushAll() {}

  flush(index) {
    void index;
  }

  destroy() {
    this.arrays = [];
    this.views = [];
  }
}

class UniformBufferPool {
  constructor(ctx) {
    this.ctx = ctx;
    this.bySize = new Map();
  }

  acquire(size) {
    const alignedSize = alignTo(size, 16);
    let entries = this.bySize.get(alignedSize);
    if (!entries) {
      entries = [];
      this.bySize.set(alignedSize, entries);
    }
    let entry = entries.find((item) => !item.inUse);
    if (!entry) {
      if (this.ctx?.device) {
        entry = new GPUUniformBufferEntry(this.ctx, alignedSize);
      } else {
        entry = new CPUUniformBufferEntry(alignedSize);
      }
      entries.push(entry);
    }
    return entry.acquire();
  }

  release(entry) {
    if (!entry) return;
    entry.release();
    if (entry.resetDirty) {
      entry.resetDirty(0);
      entry.resetDirty(1);
    }
  }

  destroy() {
    for (const entries of this.bySize.values()) {
      for (const entry of entries) {
        if (typeof entry.destroy === 'function') {
          entry.destroy();
        }
      }
    }
    this.bySize.clear();
  }
}

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
    this._storageTextureCache = new Map();
    this._pingPongCache = new Map();
    this._uniformBufferPool = null;
    this._bindGroupLayoutCache = new Map();
    this._pipelineLayoutCache = new Map();
    this._frameUniform = null;
  }

  async initWebGPU() {
    this.device = null;
    this.queue = null;
    this.presentationFormat = null;
    this.isCPU = true;
    if (!this.gpu || typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('WebGPU pipeline has been removed. Falling back to CPU rendering.');
    }
    return false;
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

  _createCPUTexture(width, height) {
    const data = new Float32Array(width * height * 4);
    const texture = { width, height, channels: 4, data };
    try {
      texture._noisemakerShape = [height, width, 4];
      texture._noisemakerChannels = 4;
    } catch (_) {
      /* ignore metadata errors */
    }
    return texture;
  }

  _createStorageTexture(width, height) {
    const usage =
      GPU_STORAGE_TEXTURE_USAGE ||
      (typeof GPUTextureUsage !== 'undefined'
        ? GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING
        : 0);
    const texture = this.device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: STORAGE_TEXTURE_FORMAT,
      usage,
    });
    try {
      texture._noisemakerShape = [height, width, 4];
      texture._noisemakerChannels = 4;
    } catch (_) {
      /* ignore metadata errors */
    }
    return texture;
  }

  _destroyTexture(texture) {
    if (!texture) return;
    if (typeof texture.destroy === 'function') {
      try {
        texture.destroy();
      } catch (_) {
        /* ignore destroy failures */
      }
    }
  }

  ensureStorageTexture(width, height, id = 'default') {
    const key = `${width}x${height}:${id}`;
    let record = this._storageTextureCache.get(key);
    const format = STORAGE_TEXTURE_FORMAT;
    const needsRecreate =
      !record ||
      record.width !== width ||
      record.height !== height ||
      record.format !== format;
    if (needsRecreate) {
      if (record?.texture && !this.isCPU && this.device) {
        this._destroyTexture(record.texture);
      }
      const texture =
        this.isCPU || !this.device
          ? this._createCPUTexture(width, height)
          : this._createStorageTexture(width, height);
      record = { key, width, height, format, texture };
      this._storageTextureCache.set(key, record);
    }
    return record.texture;
  }

  ensurePingPongTextures(width, height) {
    const key = `${width}x${height}`;
    let record = this._pingPongCache.get(key);
    const format = STORAGE_TEXTURE_FORMAT;
    const needsRecreate =
      !record ||
      record.width !== width ||
      record.height !== height ||
      record.format !== format ||
      !Array.isArray(record.textures) ||
      record.textures.length !== 2;
    if (needsRecreate) {
      if (record && record.textures && !this.isCPU && this.device) {
        for (const tex of record.textures) {
          this._destroyTexture(tex);
        }
      }
      const textures = this.isCPU || !this.device
        ? [
            this._createCPUTexture(width, height),
            this._createCPUTexture(width, height),
          ]
        : [
            this._createStorageTexture(width, height),
            this._createStorageTexture(width, height),
          ];
      const views = this.isCPU || !this.device
        ? [null, null]
        : textures.map((texture) => texture.createView());
      record = { key, width, height, format, textures, views };
      this._pingPongCache.set(key, record);
    }
    return record;
  }

  _ensureFrameUniformObject() {
    if (!this.device) return null;
    if (this._frameUniform) return this._frameUniform;
    const buffer = this.device.createBuffer({
      size: FRAME_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const arrayBuffer = new ArrayBuffer(FRAME_UNIFORM_SIZE);
    const float32 = new Float32Array(arrayBuffer);
    const uint32 = new Uint32Array(arrayBuffer);
    const info = {
      buffer,
      arrayBuffer,
      float32,
      uint32,
      dirtyStart: arrayBuffer.byteLength,
      dirtyEnd: 0,
      view: null,
    };
    info.view = createTrackedDataView(arrayBuffer, (offset, byteSize) => {
      const start = Math.max(0, Math.min(arrayBuffer.byteLength, offset));
      const end = Math.max(start, Math.min(arrayBuffer.byteLength, offset + byteSize));
      info.dirtyStart = Math.min(info.dirtyStart, start);
      info.dirtyEnd = Math.max(info.dirtyEnd, end);
    });
    this._frameUniform = info;
    return info;
  }

  ensureFrameUniforms(width, height, seed, time, frameIndex) {
    if (!this.device) {
      return null;
    }
    const info = this._ensureFrameUniformObject();
    if (!info) return null;
    const { float32, uint32, arrayBuffer } = info;
    let dirtyStart = info.dirtyStart;
    let dirtyEnd = info.dirtyEnd;
    const updateRange = (offsetBytes) => {
      dirtyStart = Math.min(dirtyStart, offsetBytes);
      dirtyEnd = Math.max(dirtyEnd, offsetBytes + 4);
    };
    const setFloat = (index, value) => {
      if (!Number.isFinite(value)) value = 0;
      if (float32[index] !== value) {
        float32[index] = value;
        updateRange(index * 4);
      }
    };
    const setUint = (index, value) => {
      const normalized = value >>> 0;
      if (uint32[index] !== normalized) {
        uint32[index] = normalized;
        updateRange(index * 4);
      }
    };

    setFloat(0, Number(width) || 0);
    setFloat(1, Number(height) || 0);
    setFloat(2, Number(time) || 0);
    setUint(3, Number(seed) || 0);
    setUint(4, Number(frameIndex) || 0);

    info.dirtyStart = dirtyStart;
    info.dirtyEnd = dirtyEnd;

    if (this.queue && dirtyEnd > dirtyStart) {
      const alignedStart = Math.max(0, alignDown(dirtyStart, 4));
      const alignedEnd = Math.min(arrayBuffer.byteLength, alignTo(dirtyEnd, 4));
      const byteSize = Math.max(0, alignedEnd - alignedStart);
      if (byteSize) {
        this.queue.writeBuffer(
          info.buffer,
          alignedStart,
          arrayBuffer,
          alignedStart,
          byteSize,
        );
      }
      info.dirtyStart = arrayBuffer.byteLength;
      info.dirtyEnd = 0;
    }

    return {
      buffer: info.buffer,
      arrayBuffer,
      float32,
      uint32,
      view: info.view,
      size: arrayBuffer.byteLength,
    };
  }

  dispose(options = {}) {
    const { keepLookupTextures = true } = options || {};
    if (this._renderParamsBuffer?.destroy) {
      try {
        this._renderParamsBuffer.destroy();
      } catch (_) {
        /* ignore */
      }
    }
    this._renderParamsBuffer = null;
    this._renderPipeline = null;
    if (!keepLookupTextures) {
      this._destroyLookupTextures?.();
    }
    for (const record of this._storageTextureCache.values()) {
      if (!this.isCPU && this.device) {
        this._destroyTexture(record.texture);
      }
    }
    this._storageTextureCache.clear();
    for (const record of this._pingPongCache.values()) {
      if (!this.isCPU && this.device && Array.isArray(record.textures)) {
        for (const tex of record.textures) {
          this._destroyTexture(tex);
        }
      }
    }
    this._pingPongCache.clear();
    if (this._uniformBufferPool) {
      this._uniformBufferPool.destroy();
      this._uniformBufferPool = null;
    }
    if (this._frameUniform?.buffer?.destroy) {
      try {
        this._frameUniform.buffer.destroy();
      } catch (_) {
        /* ignore */
      }
    }
    this._frameUniform = null;
    if (this._pipelineCache?.clear) {
      this._pipelineCache.clear();
    }
    if (this._bindGroupLayoutCache?.clear) {
      this._bindGroupLayoutCache.clear();
    }
    if (this._pipelineLayoutCache?.clear) {
      this._pipelineLayoutCache.clear();
    }
  }

  destroy(options = {}) {
    this.dispose(options);
  }

  getCachedBindGroupLayout(signature, entriesFactory) {
    if (!this.device) return null;
    if (!signature) return null;
    const key = typeof signature === 'string' ? signature : JSON.stringify(signature);
    let record = this._bindGroupLayoutCache.get(key);
    if (!record) {
      if (typeof entriesFactory !== 'function') {
        return null;
      }
      const entries = entriesFactory();
      if (!entries || !entries.length) {
        return null;
      }
      const layout = this.device.createBindGroupLayout({ entries });
      record = { layout, entries };
      this._bindGroupLayoutCache.set(key, record);
    }
    return record.layout;
  }

  getCachedPipelineLayout(shaderId, signature, bindGroupLayouts) {
    if (!this.device) return null;
    const normalizedSignature =
      typeof signature === 'string' ? signature : JSON.stringify(signature);
    const key = `${shaderId || 'generic'}|${normalizedSignature}`;
    let layout = this._pipelineLayoutCache.get(key);
    if (!layout) {
      const layouts = (Array.isArray(bindGroupLayouts) ? bindGroupLayouts : []).filter(Boolean);
      if (!layouts.length) {
        return null;
      }
      layout = this.device.createPipelineLayout({ bindGroupLayouts: layouts });
      this._pipelineLayoutCache.set(key, layout);
    }
    return layout;
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

  _getUniformBufferPool() {
    if (!this._uniformBufferPool) {
      this._uniformBufferPool = new UniformBufferPool(this);
    }
    return this._uniformBufferPool;
  }

  acquireUniformBufferPair(size) {
    return this._getUniformBufferPool().acquire(size);
  }

  releaseUniformBufferPair(entry) {
    if (!entry) return;
    if (this._uniformBufferPool) {
      this._uniformBufferPool.release(entry);
    }
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
    const record = this.ensurePingPongTextures(width, height);
    return new PingPong(this, width, height, record);
  }
}

export class PingPong {
  constructor(ctx, width, height, record) {
    this.ctx = ctx;
    this.width = width;
    this.height = height;
    this._record = record;
    this._textures = record?.textures || [];
    this._views = record?.views || [];
    this.readIndex = 0;
    this.writeIndex = 1;
    this._updateHandles();
  }

  swap() {
    [this.readIndex, this.writeIndex] = [this.writeIndex, this.readIndex];
    this._updateHandles();
  }

  _updateHandles() {
    this.readTex = this._textures[this.readIndex];
    this.writeTex = this._textures[this.writeIndex];
    const readView = this._views[this.readIndex] || null;
    const writeView = this._views[this.writeIndex] || null;
    this.readFbo = readView || this.ctx.createFBO(this.readTex);
    this.writeFbo = writeView || this.ctx.createFBO(this.writeTex);
  }
}
