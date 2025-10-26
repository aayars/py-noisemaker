export class Tensor {
  constructor(ctx, data, shape) {
    if (!Array.isArray(shape) || shape.length !== 3) {
      throw new Error('Tensor shape must be [height, width, channels]');
    }
    
    // Validate shape values before processing
    const [rawHeight, rawWidth, rawChannels] = shape;
    if (!Number.isFinite(rawHeight) || rawHeight <= 0) {
      throw new Error(`Invalid height in shape: ${rawHeight}. Height must be a positive finite number.`);
    }
    if (!Number.isFinite(rawWidth) || rawWidth <= 0) {
      throw new Error(`Invalid width in shape: ${rawWidth}. Width must be a positive finite number.`);
    }
    if (!Number.isFinite(rawChannels) || rawChannels <= 0) {
      throw new Error(`Invalid channels in shape: ${rawChannels}. Channels must be a positive finite number.`);
    }
    
    const [height, width, channels] = shape.map((v) => Math.max(1, Math.floor(v)));
    this.ctx = ctx || null;
    this.shape = [height, width, channels];
    const count = height * width * channels;
    
    // Additional validation for count
    if (!Number.isFinite(count) || count <= 0) {
      throw new Error(`Invalid tensor size: ${count} (height=${height}, width=${width}, channels=${channels}). Total size must be a positive finite number.`);
    }
    
    if (data instanceof Float32Array && data.length === count) {
      this.data = data.slice();
    } else if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      const arr = ArrayBuffer.isView(data) ? data : Array.from(data);
      if (arr.length !== count) {
        throw new Error(`Tensor data length ${arr.length} does not match shape (${count})`);
      }
      this.data = new Float32Array(arr);
    } else if (data == null) {
      this.data = new Float32Array(count);
    } else {
      throw new Error('Tensor data must be null, Float32Array, or array-like');
    }
    this.handle = null;
  }

  static storageChannels(channels) {
    return Math.max(1, Math.floor(channels));
  }

  static fromArray(ctx, array, shape) {
    return new Tensor(ctx, array, shape);
  }

  static fromGPUBuffer() {
    throw new Error('WebGPU support has been removed; GPU buffers are unavailable.');
  }

  read() {
    return this.data.slice();
  }

  readSync() {
    return this.read();
  }
}

export function markPresentationNormalized(tensor, normalized = true) {
  if (!tensor) {
    return tensor;
  }
  try {
    tensor._presentationNormalized = Boolean(normalized);
  } catch (_) {
    /* ignore */
  }
  return tensor;
}
