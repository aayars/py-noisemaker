/**
 * Tensor wrapper around GPU textures, buffers, or CPU arrays.
 */

export class Tensor {
  constructor(ctx, handle, shape, data = null) {
    this.ctx = ctx;
    this.handle = handle; // GPUTexture, GPUBuffer, or CPU object
    this.shape = shape; // [height, width, channels]
    this.data = data; // CPU data if applicable
    if (
      typeof GPUTexture !== 'undefined' &&
      handle instanceof GPUTexture &&
      Array.isArray(shape)
    ) {
      try {
        handle._noisemakerShape = shape.slice();
        handle._noisemakerChannels = shape[2] ?? 4;
      } catch (_) {
        // Silently ignore metadata assignment failures (e.g. frozen objects).
      }
    }
  }

  static storageChannels(c) {
    return c <= 1 ? 1 : c === 2 ? 2 : 4;
  }

  static fromArray(ctx, array, shape) {
    const [h, w, c] = shape;
    if (ctx && ctx.device) {
      const channels = Tensor.storageChannels(c);
      const format =
        channels === 1 ? 'r32float' : channels === 2 ? 'rg32float' : 'rgba32float';
      const bytesPerPixel = channels * 4;
      const bytesPerRow = Math.ceil((w * bytesPerPixel) / 256) * 256;
      const tex = ctx.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.STORAGE_BINDING,
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
    const channels = Tensor.storageChannels(c);
    if (channels !== c) {
      if (!ctx._bufferExpandPipeline) {
        const module = ctx.createShaderModule(BUFFER_EXPAND_WGSL);
        ctx._bufferExpandPipeline = ctx.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      }
      const pipeline = ctx._bufferExpandPipeline;
      const bytesPerPixel = channels * 4;
      const rowStride = w * bytesPerPixel;
      const bytesPerRow = Math.ceil(rowStride / 256) * 256;
      const dstStride = bytesPerRow / 4;
      const srcStride = w * c;
      const expanded = ctx.device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const tex =
        target &&
        target.ctx === ctx &&
        typeof GPUTexture !== 'undefined' &&
        target.handle instanceof GPUTexture
          ? target.handle
          : ctx.device.createTexture({
              size: { width: w, height: h, depthOrArrayLayers: 1 },
              format: 'rgba32float',
              usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.STORAGE_BINDING,
            });
      const paramsArr = new Uint32Array([w, h, c, channels, srcStride, dstStride, 0, 0]);
      const paramsBuf = ctx.createGPUBuffer(paramsArr, GPUBufferUsage.UNIFORM);
      const bindGroup = ctx.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: expanded } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ],
      });
      const encoder = ctx._encoder || ctx.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1);
      pass.end();
      encoder.copyBufferToTexture(
        { buffer: expanded, bytesPerRow },
        { texture: tex },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      if (!ctx._encoder) {
        ctx.queue.submit([encoder.finish()]);
        ctx._pendingDispatch = true;
      }
      if (target && target.ctx === ctx && target.handle === tex) {
        return target;
      }
      return new Tensor(ctx, tex, shape, null);
    }
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
              GPUTextureUsage.RENDER_ATTACHMENT |
              GPUTextureUsage.STORAGE_BINDING,
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

export function markPresentationNormalized(tensor, normalized = true) {
  if (!tensor) {
    return tensor;
  }
  try {
    tensor._noisemakerPresentationNormalized = normalized;
  } catch (_) {
    /* ignore assignment failures */
  }
  const handle = tensor.handle;
  if (handle && typeof handle === 'object') {
    try {
      handle._noisemakerPresentationNormalized = normalized;
    } catch (_) {
      /* ignore assignment failures */
    }
  }
  return tensor;
}

const BUFFER_EXPAND_WGSL = `
struct BufferExpandParams {
  width: u32,
  height: u32,
  srcChannels: u32,
  dstChannels: u32,
  srcStride: u32,
  dstStride: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: BufferExpandParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;
  if (x >= w || y >= h) { return; }
  let srcCh = params.srcChannels;
  let dstCh = params.dstChannels;
  let srcStride = params.srcStride;
  let dstStride = params.dstStride;
  let srcBase = y * srcStride + x * srcCh;
  let dstBase = y * dstStride + x * dstCh;
  var r = 0.0;
  var g = 0.0;
  var b = 0.0;
  var a = 1.0;
  if (srcCh > 0u) { r = src[srcBase]; }
  if (srcCh > 1u) { g = src[srcBase + 1u]; } else { g = r; }
  if (srcCh > 2u) { b = src[srcBase + 2u]; } else { b = r; }
  if (srcCh > 3u) { a = src[srcBase + 3u]; } else if (srcCh == 2u) { a = g; }
  if (dstCh > 0u) { dst[dstBase] = r; }
  if (dstCh > 1u) { dst[dstBase + 1u] = g; }
  if (dstCh > 2u) { dst[dstBase + 2u] = b; }
  if (dstCh > 3u) { dst[dstBase + 3u] = a; }
}
`;

