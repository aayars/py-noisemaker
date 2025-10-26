import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const ITERATION_DEFAULT = 100;

class ConvFeedbackEffect extends SimpleComputeEffect {
  static metadata = metadata;

  constructor(options = {}) {
    super(options);
    this.feedbackTexture = null;
    this.feedbackTextureDevice = null;
    this.feedbackTextureWidth = 0;
    this.feedbackTextureHeight = 0;
  }

  destroy() {
    this.#destroyFeedbackTexture();
    super.destroy();
  }

  getResourceCreationOptions(context = {}) {
    const base = super.getResourceCreationOptions(context) ?? {};
    const inputTextures = { ...(base.inputTextures ?? {}) };
    const feedbackTexture = this.#ensureFeedbackTexture(context);
    if (feedbackTexture) {
      inputTextures.prev_texture = feedbackTexture;
    }

    return {
      ...base,
      inputTextures,
    };
  }

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const finalResources = baseResources ?? resources;

    const offsets = finalResources?.bindingOffsets ?? {};
    const iterationOffset = offsets.iterations;
    if (Number.isInteger(iterationOffset) && finalResources?.paramsState) {
      const params = finalResources.paramsState;
      if (params[iterationOffset] !== ITERATION_DEFAULT) {
        params[iterationOffset] = ITERATION_DEFAULT;
        finalResources.paramsDirty = true;
      }
    }

    finalResources.feedbackTexture = this.feedbackTexture;
    finalResources.shouldCopyOutputToPrev = Boolean(this.feedbackTexture);

    return finalResources;
  }

  #ensureFeedbackTexture(context = {}) {
    const device = context?.device ?? this.feedbackTextureDevice;
    const rawWidth = Number(context?.width);
    const rawHeight = Number(context?.height);
    const width = Number.isFinite(rawWidth) && rawWidth > 0 ? Math.trunc(rawWidth) : this.feedbackTextureWidth || 1;
    const height = Number.isFinite(rawHeight) && rawHeight > 0 ? Math.trunc(rawHeight) : this.feedbackTextureHeight || 1;

    if (!device) {
      return this.feedbackTexture;
    }

    const deviceChanged = this.feedbackTextureDevice && this.feedbackTextureDevice !== device;
    const sizeChanged = this.feedbackTextureWidth !== width || this.feedbackTextureHeight !== height;

    if (!this.feedbackTexture || deviceChanged || sizeChanged) {
      this.#destroyFeedbackTexture();
      try {
        this.feedbackTexture = device.createTexture({
          size: { width, height, depthOrArrayLayers: 1 },
          format: 'rgba32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
        });
        this.feedbackTextureDevice = device;
        this.feedbackTextureWidth = width;
        this.feedbackTextureHeight = height;
      } catch (error) {
        this.helpers.logWarn?.('ConvFeedback: failed to create feedback texture.', error);
        this.feedbackTexture = null;
        this.feedbackTextureDevice = null;
        this.feedbackTextureWidth = 0;
        this.feedbackTextureHeight = 0;
      }
    }

    return this.feedbackTexture;
  }

  #destroyFeedbackTexture() {
    if (this.feedbackTexture?.destroy) {
      try {
        this.feedbackTexture.destroy();
      } catch (error) {
        this.helpers.logWarn?.('ConvFeedback: failed to destroy feedback texture.', error);
      }
    }
    this.feedbackTexture = null;
    this.feedbackTextureDevice = null;
    this.feedbackTextureWidth = 0;
    this.feedbackTextureHeight = 0;
  }
}

export default ConvFeedbackEffect;
