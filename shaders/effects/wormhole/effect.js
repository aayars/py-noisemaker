import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  kink: getDefaultParamValue('kink', 1.0),
  input_stride: getDefaultParamValue('input_stride', 0.05),
  alpha: getDefaultParamValue('alpha', 1.0),
  speed: getDefaultParamValue('speed', 1.0),
  enabled: getDefaultParamValue('enabled', true),
});

class WormholeEffect {
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
      kink: PARAMETER_DEFAULTS.kink,
      input_stride: PARAMETER_DEFAULTS.input_stride,
      alpha: PARAMETER_DEFAULTS.alpha,
      speed: PARAMETER_DEFAULTS.speed,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
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
        logWarn?.('Failed to destroy wormhole resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy wormhole output texture during invalidation:', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('WormholeEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('WormholeEffect.ensureResources requires multires output texture.');
    }

    if (this.device && this.device !== device) {
      this.invalidateResources();
    }

    this.device = device;
    this.width = width;
    this.height = height;

    const resources = this.resources;
    if (resources) {
      const sizeMatches = resources.textureWidth === width && resources.textureHeight === height;
      if (resources.computePipeline && sizeMatches) {
        return resources;
      }
      this.invalidateResources();
    }

    return this.#createResources({ device, width, height, multiresResources });
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('WormholeEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    const numericParams = ['kink', 'input_stride', 'alpha', 'speed'];
    numericParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const numeric = Number(updates[name]);
        if (Number.isFinite(numeric)) {
          this.userState[name] = numeric;
          if (this.resources?.paramsState && this.resources?.bindingOffsets?.[name] !== undefined) {
            this.resources.paramsState[this.resources.bindingOffsets[name]] = numeric;
            this.resources.paramsDirty = true;
          }
          updated.push(name);
        } else {
          logWarn?.(`updateWormholeParams: ${name} must be a finite number.`);
        }
      }
    });

    const booleanParams = ['enabled'];
    booleanParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const value = Boolean(updates[name]);
        this.userState[name] = value;
        if (name === 'enabled' && this.resources) {
          this.resources.enabled = value;
        }
        updated.push(name);
      }
    });

    return { updated };
  }

  getUIState() {
    return { ...this.userState };
  }

  destroy() {
    this.invalidateResources();
    this.device = null;
    this.width = 0;
    this.height = 0;
  }

  async #createResources({ device, width, height, multiresResources }) {
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

    setStatus?.('Creating wormhole resources…');

    try {
      const descriptor = getShaderDescriptor('wormhole');
      const shaderMetadata = await getShaderMetadataCached('wormhole');
      
      // Get descriptors for multi-pass shaders
      const clearDescriptor = getShaderDescriptor('wormhole/clear');
      const scatterDescriptor = getShaderDescriptor('wormhole/scatter');
      const normalizeBlendDescriptor = getShaderDescriptor('wormhole/normalize_blend');

      const paramsSize = 64; // 4 vec4s
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const paramsLength = paramsSize / Float32Array.BYTES_PER_ELEMENT;
      const paramsState = new Float32Array(paramsLength);
      const bindingOffsets = {
        width: 0, height: 1, channels: 2, channelCount: 2,
        kink: 4, input_stride: 5, alpha: 6, time: 7,
        speed: 8,
      };

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channels] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.kink] = this.userState.kink;
      paramsState[bindingOffsets.input_stride] = this.userState.input_stride;
      paramsState[bindingOffsets.alpha] = this.userState.alpha;
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = this.userState.speed;

      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
      
      let clearPipeline, scatterPipeline, normalizeBlendPipeline;
      let useMultiPass = true;
      
      try {
        // Pass 0: Clear buffer
        clearPipeline = await getOrCreateComputePipeline(
          device,
          'wormhole/clear',
          computePipelineLayout,
          clearDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 1: Scatter weighted samples
        scatterPipeline = await getOrCreateComputePipeline(
          device, 
          'wormhole/scatter',
          computePipelineLayout, 
          scatterDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 2: Normalize and blend
        normalizeBlendPipeline = await getOrCreateComputePipeline(
          device,
          'wormhole/normalize_blend',
          computePipelineLayout,
          normalizeBlendDescriptor.entryPoint ?? 'main'
        );
        
        logInfo?.('Wormhole: Using optimized multi-pass shaders');
      } catch (error) {
        logWarn?.('Wormhole: Multi-pass pipeline creation failed, falling back to single-pass:', error);
        useMultiPass = false;
      }
      
      // Fallback: use original single-pass shader (if multi-pass fails)
      const computePipeline = useMultiPass 
        ? scatterPipeline 
        : await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
      });

      const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } =
        await getBufferToTexturePipeline(device);
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

      const workgroupSize = [8, 8, 1];
      
      let computePasses = null;
      
      if (useMultiPass) {
        computePasses = [
          // Pass 0: Clear output buffer
          {
            pipeline: clearPipeline,
            bindGroup: computeBindGroup,
            workgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / workgroupSize[0]),
              Math.ceil(height / workgroupSize[1]),
              1
            ],
          },
          // Pass 1: Scatter weighted samples
          {
            pipeline: scatterPipeline,
            bindGroup: computeBindGroup,
            workgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / workgroupSize[0]),
              Math.ceil(height / workgroupSize[1]),
              1
            ],
          },
          // Pass 2: Normalize and blend
          {
            pipeline: normalizeBlendPipeline,
            bindGroup: computeBindGroup,
            workgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / workgroupSize[0]),
              Math.ceil(height / workgroupSize[1]),
              1
            ],
          },
        ];
      }

      this.resources = {
        descriptor,
        shaderMetadata,
        computePipeline,
        computeBindGroup,
        computePasses,
        paramsBuffer,
        paramsState,
        outputBuffer,
        outputTexture,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        blitBindGroup,
        workgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        device,
        bindingOffsets,
      };

      setStatus?.('Wormhole resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create wormhole resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device }) {
    if (!this.resources) return;

    // Update params buffer if dirty
    if (this.resources.paramsDirty) {
      device.queue.writeBuffer(this.resources.paramsBuffer, 0, this.resources.paramsState);
      this.resources.paramsDirty = false;
    }
  }
}

export default WormholeEffect;

// Multi-pass shader descriptors
export const additionalPasses = {
  'wormhole/clear': {
    id: 'wormhole/clear',
    label: 'clear.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/wormhole/clear.wgsl',
    resources: {
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 }
    }
  },
  'wormhole/scatter': {
    id: 'wormhole/scatter',
    label: 'scatter.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/wormhole/scatter.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 }
    }
  },
  'wormhole/normalize_blend': {
    id: 'wormhole/normalize_blend',
    label: 'normalize_blend.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/wormhole/normalize_blend.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 }
    }
  }
};

