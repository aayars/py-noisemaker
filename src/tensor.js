/**
 * Tensor wrapper around WebGL textures or CPU arrays.
 */

export class Tensor {
  constructor(ctx, handle, shape, data = null) {
    this.ctx = ctx;
    this.handle = handle; // WebGL texture or CPU object
    this.shape = shape; // [height, width, channels]
    this.data = data; // CPU data if applicable
  }

  static fromArray(ctx, array, shape) {
    const [h, w, c] = shape;
    if (ctx && !ctx.isCPU) {
      const tex = ctx.createTexture(w, h, array);
      return new Tensor(ctx, tex, shape, null);
    }
    const data = array ? array.slice() : new Float32Array(h * w * c);
    return new Tensor(ctx, null, shape, data);
  }

  read() {
    const [h, w, c] = this.shape;
    if (this.ctx && !this.ctx.isCPU) {
      const gl = this.ctx.gl;
      const fbo = this.ctx.createFBO(this.handle);
      this.ctx.bindFramebuffer(fbo, w, h);
      const out = new Float32Array(h * w * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, out);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return out.subarray(0, h * w * c);
    }
    return this.data.slice();
  }
}

