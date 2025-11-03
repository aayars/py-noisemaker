import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  enabled: getDefaultParamValue('enabled', true),
  iterations: getDefaultParamValue('iterations', 0),
  alpha: getDefaultParamValue('alpha', 0.5),
});

class ConvFeedbackEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.device = null;
    this.width = 0;
    this.height = 0;
    this.resources = null;
    this.userState = {
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
      iterations: Number(PARAMETER_DEFAULTS.iterations ?? 0),
      alpha: Number(PARAMETER_DEFAULTS.alpha ?? 0.5),
    };
  }

  destroy() {
    this.invalidateResources();
    this.device = null;
    this.width = 0;
    this.height = 0;
  }

  getUIState() {
    return { ...this.userState };
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('ConvFeedbackEffect.updateParams expects an object.');
    }

    const updated = [];

    if (hasOwn(updates, 'enabled')) {
      const enabled = Boolean(updates.enabled);
      if (enabled !== this.userState.enabled) {
        this.userState.enabled = enabled;
        if (this.resources) {
          this.resources.enabled = enabled;
        }
        updated.push('enabled');
      }
    }

    const numericParams = [
      ['iterations', Number(PARAMETER_DEFAULTS.iterations ?? 0)],
      ['alpha', Number(PARAMETER_DEFAULTS.alpha ?? 0.5)],
    ];

    numericParams.forEach(([name, fallback]) => {
      if (!hasOwn(updates, name)) {
        return;
      }

      const numeric = Number(updates[name]);
      if (!Number.isFinite(numeric)) {
        this.helpers.logWarn?.(`ConvFeedbackEffect.updateParams: ${name} must be finite.`);
        return;
      }

      const clamped = name === 'alpha'
        ? Math.min(Math.max(numeric, 0), 1)
        : numeric;

      if (clamped === this.userState[name]) {
        return;
      }

      this.userState[name] = clamped;
      if (this.resources?.paramsState && Number.isInteger(this.resources.bindingOffsets?.[name])) {
        const offset = this.resources.bindingOffsets[name];
        if (offset >= 0 && offset < this.resources.paramsState.length) {
          this.resources.paramsState[offset] = clamped;
          this.resources.paramsDirty = true;
        }
      }
      updated.push(name);
    });

    if (updated.length > 0) {
      this.helpers.logInfo?.(`ConvFeedbackEffect.updateParams: updated ${updated.join(', ')}`);
    }

    return updated;
  }

  invalidateResources() {
    if (!this.resources) {
      return;
    }

    const { logWarn } = this.helpers;
    const resources = this.resources;

    if (resources.resourceSet?.destroyAll) {
      try {
        resources.resourceSet.destroyAll();
      } catch (error) {
        logWarn?.('Failed to destroy conv_feedback resources during invalidation:', error);
      }
    }

    if (resources.blurredBuffer?.destroy) {
      try {
        resources.blurredBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy blurred buffer:', error);
      }
    }

    if (resources.outputBuffer?.destroy) {
      try {
        resources.outputBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy output buffer:', error);
      }
    }

    if (resources.paramsBuffer?.destroy) {
      try {
        resources.paramsBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy params buffer:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy conv_feedback output texture during invalidation:', error);
      }
    }

    if (resources.feedbackTexture?.destroy) {
      try {
        resources.feedbackTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy conv_feedback feedback texture during invalidation:', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('ConvFeedbackEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('ConvFeedbackEffect.ensureResources requires multires output texture.');
    }

    if (this.device && this.device !== device) {
      this.invalidateResources();
    }

    this.device = device;
    this.width = width;
    this.height = height;

    const existing = this.resources;
    if (existing) {
      const sizeMatches = existing.textureWidth === width && existing.textureHeight === height;
      if (sizeMatches) {
        existing.enabled = this.userState.enabled;
        return existing;
      }
      this.invalidateResources();
    }

    const {
      logInfo,
      logWarn,
      setStatus,
      getShaderDescriptor,
      getShaderMetadataCached,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    setStatus?.('Creating conv_feedback resources…');

    try {
      const blurDescriptor = getShaderDescriptor('conv_feedback/blur');
      const sharpenDescriptor = getShaderDescriptor('conv_feedback/sharpen');
      const blurMetadata = await getShaderMetadataCached('conv_feedback/blur');
      const sharpenMetadata = await getShaderMetadataCached('conv_feedback/sharpen');

      const paramsSize = 32;
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const paramsState = new Float32Array(paramsSize / Float32Array.BYTES_PER_ELEMENT);
      const bindingOffsets = {
        width: 0,
        height: 1,
        channel_count: 2,
        channelCount: 2,
        iterations: 3,
        alpha: 4,
        time: 5,
        speed: 6,
      };
      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channel_count] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.iterations] = this.userState.iterations;
      paramsState[bindingOffsets.alpha] = this.userState.alpha;
      paramsState[bindingOffsets.time] = 0;
      if (Number.isInteger(bindingOffsets.speed)) {
        paramsState[bindingOffsets.speed] = 0;
      }
      if (paramsState.length > 7) {
        paramsState[7] = 0;
      }
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const bufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const blurredBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const outputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const feedbackTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });

      const blurBindGroupLayout = getOrCreateBindGroupLayout(device, blurDescriptor.id, 'compute', blurMetadata);
      const blurPipelineLayout = getOrCreatePipelineLayout(device, blurDescriptor.id, 'compute', blurBindGroupLayout);
      const blurPipeline = await getOrCreateComputePipeline(
        device,
        blurDescriptor.id,
        blurPipelineLayout,
        blurDescriptor.entryPoint ?? 'main',
      );
      const blurBindings = new Map(
        blurMetadata.bindings
          .filter((binding) => binding.group === 0)
          .map((binding) => [binding.name, binding.binding]),
      );
      const blurInputBinding = blurBindings.get('input_texture') ?? blurBindings.get('prev_texture');
      const blurBufferBinding = blurBindings.get('blurred_buffer');
      const blurParamsBinding = blurBindings.get('params');
      if (!Number.isInteger(blurInputBinding) || !Number.isInteger(blurBufferBinding) || !Number.isInteger(blurParamsBinding)) {
        throw new Error('ConvFeedback blur shader metadata missing expected bindings.');
      }
      const blurBindGroup = device.createBindGroup({
        label: 'conv_feedback_blur_bg',
        layout: blurBindGroupLayout,
        entries: [
          { binding: blurInputBinding, resource: feedbackTexture.createView() },
          { binding: blurBufferBinding, resource: { buffer: blurredBuffer } },
          { binding: blurParamsBinding, resource: { buffer: paramsBuffer } },
        ],
      });

      const blurBindingIndices = {
        inputTexture: blurInputBinding,
        outputBuffer: blurBufferBinding,
        params: blurParamsBinding,
      };

      const sharpenBindGroupLayout = getOrCreateBindGroupLayout(device, sharpenDescriptor.id, 'compute', sharpenMetadata);
      const sharpenPipelineLayout = getOrCreatePipelineLayout(device, sharpenDescriptor.id, 'compute', sharpenBindGroupLayout);
      const sharpenPipeline = await getOrCreateComputePipeline(
        device,
        sharpenDescriptor.id,
        sharpenPipelineLayout,
        sharpenDescriptor.entryPoint ?? 'main',
      );
      const sharpenBindings = new Map(
        sharpenMetadata.bindings
          .filter((binding) => binding.group === 0)
          .map((binding) => [binding.name, binding.binding]),
      );
      const sharpenBlurBinding = sharpenBindings.get('blurred_buffer');
      const sharpenOutputBinding = sharpenBindings.get('output_buffer') ?? sharpenBindings.get('sharpen_buffer');
      const sharpenParamsBinding = sharpenBindings.get('params');
      if (!Number.isInteger(sharpenBlurBinding) || !Number.isInteger(sharpenOutputBinding) || !Number.isInteger(sharpenParamsBinding)) {
        throw new Error('ConvFeedback sharpen shader metadata missing expected bindings.');
      }
      const sharpenBindGroup = device.createBindGroup({
        label: 'conv_feedback_sharpen_bg',
        layout: sharpenBindGroupLayout,
        entries: [
          { binding: sharpenBlurBinding, resource: { buffer: blurredBuffer } },
          { binding: sharpenOutputBinding, resource: { buffer: outputBuffer } },
          { binding: sharpenParamsBinding, resource: { buffer: paramsBuffer } },
        ],
      });

      const blurWorkgroupSize = Array.isArray(blurMetadata.workgroupSize)
        ? blurMetadata.workgroupSize.slice()
        : [8, 8, 1];
      const sharpenWorkgroupSize = Array.isArray(sharpenMetadata.workgroupSize)
        ? sharpenMetadata.workgroupSize.slice()
        : blurWorkgroupSize.slice();

      const makeDispatch = (workgroup) => ({ width: frameWidth, height: frameHeight }) => {
        const wgx = Math.max(Math.trunc(workgroup[0] ?? 1), 1);
        const wgy = Math.max(Math.trunc(workgroup[1] ?? 1), 1);
        return [
          Math.ceil(frameWidth / wgx),
          Math.ceil(frameHeight / wgy),
          1,
        ];
      };

      const computePasses = [
        {
          label: 'conv_feedback_blur_pass',
          pipeline: blurPipeline,
          bindGroup: blurBindGroup,
          workgroupSize: blurWorkgroupSize,
          getDispatch: makeDispatch(blurWorkgroupSize),
        },
        {
          label: 'conv_feedback_sharpen_pass',
          pipeline: sharpenPipeline,
          bindGroup: sharpenBindGroup,
          workgroupSize: sharpenWorkgroupSize,
          getDispatch: makeDispatch(sharpenWorkgroupSize),
        },
      ];

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);
      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: outputTexture.createView() },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const blitBindGroup = device.createBindGroup({
        layout: multiresResources.blitBindGroupLayout,
        entries: [{ binding: 0, resource: outputTexture.createView() }],
      });

      const resources = {
        enabled: this.userState.enabled,
        blurPipeline,
        sharpenPipeline,
        blurredBuffer,
        outputBuffer,
        outputTexture,
        feedbackTexture,
        paramsBuffer,
        paramsState,
        paramsDirty: false,
        bindingOffsets,
        computePasses,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        blitBindGroup,
        blurBindGroup,
        blurBindGroupLayout,
        blurBindingIndices,
        feedbackInitialized: false,
        textureWidth: width,
        textureHeight: height,
        inputTexture: feedbackTexture,
        shouldCopyOutputToPrev: true,
      };

      this.resources = resources;
      setStatus?.('');
      logInfo?.('ConvFeedbackEffect: resources created successfully');
      return resources;
    } catch (error) {
      setStatus?.('');
      logWarn?.('Failed to create conv_feedback resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources, encoder }) {
    if (!this.resources || !device) {
      return;
    }

    const resources = this.resources;

    const needsFeedbackUpload = !resources.feedbackInitialized
      && resources.feedbackTexture
      && multiresResources?.outputTexture
      && encoder;

    if (needsFeedbackUpload) {
      try {
        const copyWidth = Math.max(Math.trunc(resources.textureWidth || this.width || 0), 0);
        const copyHeight = Math.max(Math.trunc(resources.textureHeight || this.height || 0), 0);
        if (copyWidth > 0 && copyHeight > 0) {
          encoder.copyTextureToTexture(
            { texture: multiresResources.outputTexture },
            { texture: resources.feedbackTexture },
            { width: copyWidth, height: copyHeight, depthOrArrayLayers: 1 },
          );
          resources.feedbackInitialized = true;
        }
      } catch (error) {
        this.helpers.logWarn?.('ConvFeedback: failed to seed feedback texture.', error);
      }
    }

    if (resources.paramsDirty && resources.paramsBuffer && resources.paramsState) {
      device.queue.writeBuffer(resources.paramsBuffer, 0, resources.paramsState);
      resources.paramsDirty = false;
    }
  }
}

export const additionalPasses = {
  'conv_feedback/blur': {
    id: 'conv_feedback/blur',
    label: 'blur.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/conv_feedback/blur.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      blurred_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
  params: { kind: 'uniformBuffer', size: 32 },
    },
  },
  'conv_feedback/sharpen': {
    id: 'conv_feedback/sharpen',
    label: 'sharpen.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/conv_feedback/sharpen.wgsl',
    resources: {
      blurred_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
  params: { kind: 'uniformBuffer', size: 32 },
    },
  },
};

export default ConvFeedbackEffect;
