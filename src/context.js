/**
 * WebGL2 context helper with CPU fallback and shader utilities.
 */

import { setSeed, getSeed, random } from './rng.js';

export { setSeed, getSeed, random };

export class Context {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas && canvas.getContext ? canvas.getContext('webgl2') : null;
    this.isCPU = !this.gl;

    if (this.gl) {
      const gl = this.gl;
      gl.getExtension('EXT_color_buffer_float');
      gl.getExtension('OES_texture_float_linear');

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

  createTexture(width, height, data = null) {
    if (this.isCPU) {
      const arr = data || new Float32Array(width * height * 4);
      return { width, height, channels: 4, data: arr };
    }
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      width,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      data
    );
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

