/**
 * Tensor wrapper around GPU textures, buffers, or CPU arrays.
 */

export class Tensor {
  constructor(ctx, handle, shape, data = null) {
    this.ctx = ctx;
    this.handle = handle; // GPUTexture, GPUBuffer, or CPU object
    this.shape = shape; // [height, width, channels]
    this.data = data; // CPU data if applicable
  }

  static fromArray(ctx, array, shape) {
    const [h, w, c] = shape;
    if (ctx && ctx.device) {
      const format = c === 1 ? 'r32float' : c === 2 ? 'rg32float' : 'rgba32float';
      const channels = format === 'r32float' ? 1 : format === 'rg32float' ? 2 : 4;
      const tex = ctx.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      if (array) {
        let data = array instanceof Float32Array ? array : new Float32Array(array);
        if (c !== channels) {
          const padded = new Float32Array(h * w * channels);
          for (let i = 0; i < h * w; i++) {
            for (let k = 0; k < c; k++) {
              padded[i * channels + k] = data[i * c + k];
            }
          }
          data = padded;
        }
        ctx.queue.writeTexture(
          { texture: tex },
          data,
          { bytesPerRow: w * channels * 4 },
          { width: w, height: h, depthOrArrayLayers: 1 }
        );
      }
      return new Tensor(ctx, tex, shape, null);
    }
    const data = array ? array.slice() : new Float32Array(h * w * c);
    return new Tensor(ctx, null, shape, data);
  }

  read() {
    const [h, w, c] = this.shape;
    if (
      this.ctx &&
      this.ctx.device &&
      typeof GPUBuffer !== 'undefined' &&
      this.handle instanceof GPUBuffer
    ) {
      const size = h * w * c * 4;
      return this.ctx.readGPUBuffer(this.handle, size);
    }
    if (
      this.ctx &&
      this.ctx.device &&
      typeof GPUTexture !== 'undefined' &&
      this.handle instanceof GPUTexture
    ) {
      const channels = c === 1 ? 1 : c === 2 ? 2 : 4;
      const bytesPerPixel = channels * 4;
      const bytesPerRow = Math.ceil((w * bytesPerPixel) / 256) * 256;
      const size = bytesPerRow * h;
      const buffer = this.ctx.device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = this.ctx.device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: this.handle },
        { buffer, bytesPerRow },
        { width: w, height: h, depthOrArrayLayers: 1 }
      );
      this.ctx.queue.submit([encoder.finish()]);
      return this.ctx.readGPUBuffer(buffer, size).then((arr) => {
        if (bytesPerRow === w * bytesPerPixel && channels === c) {
          return arr.subarray(0, h * w * c);
        }
        const out = new Float32Array(h * w * c);
        const stride = bytesPerRow / 4;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const srcBase = y * stride + x * channels;
            const dstBase = (y * w + x) * c;
            for (let k = 0; k < c; k++) {
              out[dstBase + k] = arr[srcBase + k];
            }
          }
        }
        return out;
      });
    }
    return this.data.slice();
  }

  readSync() {
    const res = this.read();
    if (res && typeof res.then === 'function') {
      throw new Error('tensor.read() returns a Promise under WebGPU; use await tensor.read()');
    }
    return res;
  }
}

