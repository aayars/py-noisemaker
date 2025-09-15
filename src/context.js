/**
 * WebGPU context helper with CPU fallback and shader utilities.
 */

import { Random, setSeed, getSeed, random } from './rng.js';

export { Random, setSeed, getSeed, random };

export class Context {
  constructor(canvas) {
    this.canvas = canvas;
    this.gpu = canvas && canvas.getContext ? canvas.getContext('webgpu') : null;
    this.device = null;
    this.queue = null;
    this.presentationFormat = null;
    this.isCPU = !this.gpu;
    this.currentTarget = null;
    this._renderPipeline = null;
    this._renderSampler = null;
  }

  async initWebGPU() {
    if (this.device || !this.gpu || typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }
    const adapter = await navigator.gpu.requestAdapter();
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
    if (this.isCPU || !this.device) {
      const arr = data ? new Float32Array(data) : new Float32Array(width * height * 4);
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
      const arr = data instanceof Float32Array ? data : new Float32Array(data);
      this.queue.writeTexture(
        { texture },
        arr,
        { bytesPerRow: width * 4 * 4 },
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
    const encoder = this.device.createCommandEncoder();
    const view =
      this.currentTarget?.view || this.gpu.getCurrentTexture().createView();
    const pass = this.beginRenderPass(encoder, view);
    pass.setPipeline(pipeline);
    if (bindGroup) pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    this.queue.submit([encoder.finish()]);
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
    const encoder = this.device.createCommandEncoder();
    const view = (target || this.gpu.getCurrentTexture()).createView();
    const pass = this.beginRenderPass(encoder, view);
    pass.setPipeline(this._renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    this.queue.submit([encoder.finish()]);
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

  async runCompute(code, bindEntries, x, y = 1, z = 1) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    const device = this.device;
    device.pushErrorScope('validation');
    const pipeline = await this.createComputePipeline(code);
    const compileErr = await device.popErrorScope();
    if (compileErr) {
      throw compileErr;
    }
    device.pushErrorScope('validation');
    const bindGroup = this.createBindGroup(pipeline, bindEntries);
    const encoder = device.createCommandEncoder();
    const pass = this.beginComputePass(encoder);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    const t0 = performance.now();
    this.queue.submit([encoder.finish()]);
    if (this.queue.onSubmittedWorkDone) {
      await this.queue.onSubmittedWorkDone();
      if (this.profile) {
        this.profile.webgpu += performance.now() - t0;
      }
    }
    const err = await device.popErrorScope();
    if (err) {
      throw err;
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
    if (buffer.usage & GPUBufferUsage.MAP_READ) {
      const t0 = performance.now();
      if (this.queue.onSubmittedWorkDone) {
        await this.queue.onSubmittedWorkDone();
        if (this.profile) {
          this.profile.webgpu += performance.now() - t0;
        }
      }
      await buffer.mapAsync(GPUMapMode.READ);
      const arr = buffer.getMappedRange().slice(0, size);
      buffer.unmap();
      return new Float32Array(arr);
    }
    device.pushErrorScope('validation');
    const readBuf = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuf, 0, size);
    const t0 = performance.now();
    this.queue.submit([encoder.finish()]);
    if (this.queue.onSubmittedWorkDone) {
      await this.queue.onSubmittedWorkDone();
      if (this.profile) {
        this.profile.webgpu += performance.now() - t0;
      }
    }
    await readBuf.mapAsync(GPUMapMode.READ);
    const arr = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    const err = await device.popErrorScope();
    if (err) {
      throw err;
    }
    return new Float32Array(arr);
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
