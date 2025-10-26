export class Context {
  constructor(canvas = null, debug = false) {
    this.canvas = canvas;
    this.debug = Boolean(debug);
    this.forceCPU = true;
    this.isCPU = true;
    this.device = null;
    this.queue = null;
  }

  async initWebGPU() {
    if (this.debug) {
      console.warn('[noisemaker] WebGPU support has been removed; CPU mode only');
    }
    return false;
  }

  frame() {
    throw new Error('WebGPU pipelines are no longer available.');
  }

  safeSubmit() {
    throw new Error('WebGPU pipelines are no longer available.');
  }

  flush() {}

  destroy() {}
}

export default Context;
