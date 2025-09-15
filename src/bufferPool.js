export class BufferPool {
  constructor(device) {
    this.device = device;
    this._buffers = [];
  }

  acquire(size, usage) {
    const idx = this._buffers.findIndex(
      (b) => !b.inUse && b.usage === usage && b.size >= size,
    );
    if (idx !== -1) {
      const entry = this._buffers[idx];
      entry.inUse = true;
      return entry.buffer;
    }
    const buffer = this.device.createBuffer({ size, usage });
    this._buffers.push({ buffer, size, usage, inUse: true });
    return buffer;
  }

  release(buffer) {
    const entry = this._buffers.find((b) => b.buffer === buffer);
    if (entry) {
      entry.inUse = false;
    } else if (buffer.destroy) {
      buffer.destroy();
    }
  }

  destroy() {
    for (const { buffer } of this._buffers) {
      try {
        buffer.destroy();
      } catch (_) {
        // ignore
      }
    }
    this._buffers = [];
  }
}

export function getBufferPool(ctx) {
  if (!ctx._bufferPool) {
    ctx._bufferPool = new BufferPool(ctx.device);
  }
  return ctx._bufferPool;
}
