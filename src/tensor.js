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
      let data = array;
      if (!data) {
        data = new Float32Array(h * w * 4);
      } else if (data.length !== h * w * 4) {
        // WebGL textures are always RGBA; pad incoming data if it has fewer channels
        const padded = new Float32Array(h * w * 4);
        for (let i = 0, j = 0; i < h * w; i++) {
          for (let k = 0; k < c; k++, j++) {
            padded[i * 4 + k] = data[j];
          }
        }
        data = padded;
      }
      const tex = ctx.createTexture(w, h, data);
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
      if (c === 4) {
        return out;
      }
      const arr = new Float32Array(h * w * c);
      for (let i = 0; i < h * w; i++) {
        for (let k = 0; k < c; k++) {
          arr[i * c + k] = out[i * 4 + k];
        }
      }
      return arr;
    }
    return this.data.slice();
  }
}

