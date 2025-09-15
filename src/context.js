/**
 * WebGPU context helper with CPU fallback and shader utilities.
 */

import { Random, setSeed, getSeed, random } from './rng.js';

export { Random, setSeed, getSeed, random };

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

  renderTexture(texture, target = null) {
    if (this.isCPU || !this.device) return;
    if (!this._renderPipeline) {
      const vs = `
        struct VSOut {
          @builtin(position) pos: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };
        @vertex
        fn main(@builtin(vertex_index) idx : u32) -> VSOut {
          var pos = array<vec2<f32>,6>(
            vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
            vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0)
          );
          var uv = array<vec2<f32>,6>(
            vec2(0.0,1.0), vec2(1.0,1.0), vec2(0.0,0.0),
            vec2(0.0,0.0), vec2(1.0,1.0), vec2(1.0,0.0)
          );
          var out: VSOut;
          out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
          out.uv = uv[idx];
          return out;
        }
      `;
      const fs = `
        @group(0) @binding(0) var samp : sampler;
        @group(0) @binding(1) var tex : texture_2d<f32>;
        @fragment
        fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
          return textureSample(tex, samp, uv);
        }
      `;
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: 'non-filtering' },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'unfilterable-float' },
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
      this._renderSampler = this.device.createSampler({
        magFilter: 'nearest',
        minFilter: 'nearest',
      });
    }
    const bindGroup = this.device.createBindGroup({
      layout: this._renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._renderSampler },
        { binding: 1, resource: texture.createView() },
      ],
    });
    const encoder = this._encoder || this.device.createCommandEncoder();
    const view = (target || this.gpu.getCurrentTexture()).createView();
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
