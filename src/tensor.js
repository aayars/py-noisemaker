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
      const bytesPerPixel = channels * 4;
      const bytesPerRow = Math.ceil((w * bytesPerPixel) / 256) * 256;
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
        if (bytesPerRow !== w * bytesPerPixel) {
          const stride = bytesPerRow / 4;
          const padded = new Float32Array(stride * h);
          for (let y = 0; y < h; y++) {
            padded.set(
              data.subarray(y * w * channels, (y + 1) * w * channels),
              y * stride,
            );
          }
          data = padded;
        }
        ctx.queue.writeTexture(
          { texture: tex },
          data,
          { bytesPerRow },
          { width: w, height: h, depthOrArrayLayers: 1 },
        );
      }
      return new Tensor(ctx, tex, shape, null);
    }
    const data = array ? array.slice() : new Float32Array(h * w * c);
    return new Tensor(ctx, null, shape, data);
  }

  static fromGPUBuffer(ctx, buffer, shape, target = null) {
    const [h, w, c] = shape;
    if (!ctx || !ctx.device) {
      throw new Error('GPU context required');
    }
    const channels = c === 1 ? 1 : c === 2 ? 2 : 4;
    const bytesPerPixel = channels * 4;
    const rowStride = w * bytesPerPixel;
    const bytesPerRow = Math.ceil(rowStride / 256) * 256;
    const tex =
      target &&
      target.ctx === ctx &&
      typeof GPUTexture !== 'undefined' &&
      target.handle instanceof GPUTexture
        ? target.handle
        : ctx.device.createTexture({
            size: { width: w, height: h, depthOrArrayLayers: 1 },
            format:
              channels === 1
                ? 'r32float'
                : channels === 2
                ? 'rg32float'
                : 'rgba32float',
            usage:
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.COPY_SRC |
              GPUTextureUsage.COPY_DST |
              GPUTextureUsage.RENDER_ATTACHMENT,
          });
    const encoder = ctx._encoder || ctx.device.createCommandEncoder();
    const submit = !ctx._encoder;
    if (bytesPerRow === rowStride) {
      encoder.copyBufferToTexture(
        { buffer, bytesPerRow },
        { texture: tex },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
    } else {
      const padded = ctx.device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      for (let y = 0; y < h; y++) {
        const src = y * rowStride;
        const dst = y * bytesPerRow;
        encoder.copyBufferToBuffer(buffer, src, padded, dst, rowStride);
      }
      encoder.copyBufferToTexture(
        { buffer: padded, bytesPerRow },
        { texture: tex },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
    }
    if (submit) {
      ctx.queue.submit([encoder.finish()]);
      ctx._pendingDispatch = true;
    }
    if (target && target.ctx === ctx && target.handle === tex) {
      return target;
    }
    return new Tensor(ctx, tex, shape, null);
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

