/**
 * WebGL2 context helper with CPU fallback and shader utilities.
 */

import { Random, setSeed, getSeed, random } from './rng.js';

// Cache compiled shader programs keyed by vertex/fragment source.
const PROGRAM_CACHE = new Map();

export function disposePrograms() {
  for (const { gl, program } of PROGRAM_CACHE.values()) {
    try {
      gl.deleteProgram(program);
    } catch (e) {
      // ignore errors during context loss
    }
  }
  PROGRAM_CACHE.clear();
}

export { Random, setSeed, getSeed, random };

export class Context {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas && canvas.getContext ? canvas.getContext('webgl2') : null;
    this.isCPU = true;
    this.device = null;
    this.queue = null;

    if (this.gl) {
      const gl = this.gl;
      const hasColorBufferFloat = gl.getExtension('EXT_color_buffer_float');
      // Linear filtering for float textures is core in WebGL2 and we only use
      // NEAREST sampling, so `OES_texture_float_linear` isn't required. Some
      // browsers omit this extension which previously forced a CPU fallback
      // and produced glitchy output.
      if (hasColorBufferFloat) {
        this.isCPU = false;
        // Explicitly operate on 32-bit float textures
        this.textureFormat = gl.RGBA32F;
        this.textureType = gl.FLOAT;

        // setup a fullscreen quad
        this.quadVao = gl.createVertexArray();
        gl.bindVertexArray(this.quadVao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          new Float32Array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1
          ]),
          gl.STATIC_DRAW
        );
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
      } else {
        // Lose the WebGL context so a 2D context can be acquired later
        gl.getExtension('WEBGL_lose_context')?.loseContext?.();
        this.gl = null;
      }
    }
  }

  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(info || 'Shader compilation failed');
    }
    return shader;
  }

  createProgram(vsSource, fsSource) {
    const gl = this.gl;
    const prog = gl.createProgram();
    const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'position');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(info || 'Program link failed');
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  getProgram(vsSource, fsSource) {
    const key = `${vsSource}\u0000${fsSource}`;
    const cached = PROGRAM_CACHE.get(key);
    if (cached && cached.gl === this.gl) {
      return cached.program;
    }
    const prog = this.createProgram(vsSource, fsSource);
    PROGRAM_CACHE.set(key, { gl: this.gl, program: prog });
    return prog;
  }

  createTexture(width, height, data = null) {
    if (this.isCPU) {
      const arr = data ? new Float32Array(data) : new Float32Array(width * height * 4);
      return { width, height, channels: 4, data: arr };
    }
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const initData = data || new Float32Array(width * height * 4);
    if (data) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      this.textureFormat,
      width,
      height,
      0,
      gl.RGBA,
      this.textureType,
      initData
    );
    if (data) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  createFBO(texture) {
    if (this.isCPU) return { texture };
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer incomplete: ' + status);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  bindFramebuffer(fbo, width, height) {
    if (this.isCPU) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, width, height);
  }

  drawQuad() {
    if (this.isCPU) return;
    const gl = this.gl;
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  async initWebGPU() {
    if (this.device || typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    this.device = await adapter.requestDevice();
    this.queue = this.device.queue;
    return true;
  }

  createGPUBuffer(array, usage) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    const buf = this.device.createBuffer({
      size: array.byteLength,
      usage,
      mappedAtCreation: true,
    });
    new array.constructor(buf.getMappedRange()).set(array);
    buf.unmap();
    return buf;
  }

  async runCompute(code, bindEntries, x, y = 1, z = 1) {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    const device = this.device;

    // Capture validation errors so callers can fall back to WebGL.
    device.pushErrorScope('validation');

    const module = device.createShaderModule({ code });
    const pipeline = await device.createComputePipelineAsync({
      compute: { module, entryPoint: 'main' },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindEntries,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    this.queue.submit([encoder.finish()]);

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
    if (buffer.size < size) {
      throw new Error(`Buffer too small: ${buffer.size} < ${size}`);
    }
    // Capture validation errors so callers can fall back to WebGL.
    device.pushErrorScope('validation');
    const readBuf = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuf, 0, size);
    this.queue.submit([encoder.finish()]);
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

