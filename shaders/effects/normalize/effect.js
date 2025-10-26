import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  enabled: getDefaultParamValue('enabled', true),
});

class NormalizeEffect {
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
        logWarn?.('Failed to destroy normalize resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy normalize output texture during invalidation:', error);
      }
    }

    if (resources.statsBuffer?.destroy) {
      try {
        resources.statsBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy normalize stats buffer during invalidation:', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('NormalizeEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('NormalizeEffect.ensureResources requires multires output texture.');
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
      throw new TypeError('NormalizeEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    if (hasOwn(updates, 'enabled')) {
      const enabled = Boolean(updates.enabled);
      this.userState.enabled = enabled;
      if (this.resources) {
        this.resources.enabled = enabled;
      }
      updated.push('enabled');
    }

    if (updated.length > 0) {
      this.helpers.logInfo?.(`NormalizeEffect.updateParams: updated ${updated.join(', ')}`);
    }

    return updated;
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

    setStatus?.('Creating normalize resources…');

    try {
      // Get descriptors for all three passes
      const statsDescriptor = getShaderDescriptor('normalize/stats');
      const reduceDescriptor = getShaderDescriptor('normalize/reduce');
      const applyDescriptor = getShaderDescriptor('normalize/apply');
      const shaderMetadata = await getShaderMetadataCached('normalize');

      // Create parameters buffer
      const paramsSize = 32;
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const paramsLength = paramsSize / Float32Array.BYTES_PER_ELEMENT;
      const paramsState = new Float32Array(paramsLength);
      const bindingOffsets = {
        width: 0, height: 1, channel_count: 2, channelCount: 2,
        time: 4, speed: 5,
      };

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channel_count] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = 1.0;

      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      // Create output buffer
      const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Create stats buffer for min/max communication between passes
      // Layout: [final_min, final_max, workgroup0_min, workgroup0_max, ...]
      const numWorkgroupsX = Math.ceil(width / 8);
      const numWorkgroupsY = Math.ceil(height / 8);
      const numWorkgroups = numWorkgroupsX * numWorkgroupsY;
      const statsBufferSize = (2 + numWorkgroups * 2) * Float32Array.BYTES_PER_ELEMENT;
      const statsBuffer = device.createBuffer({
        size: statsBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Initialize stats buffer with extreme values
      const statsInit = new Float32Array(2 + numWorkgroups * 2);
      const F32_MAX = 3.4028235e+38;
      const F32_MIN = -3.4028235e+38;
      statsInit[0] = F32_MAX; // final_min
      statsInit[1] = F32_MIN; // final_max
      for (let i = 0; i < numWorkgroups; i++) {
        statsInit[2 + i * 2] = F32_MAX;
        statsInit[2 + i * 2 + 1] = F32_MIN;
      }
      device.queue.writeBuffer(statsBuffer, 0, statsInit);

      // Create bind group layout and pipeline layout
      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, 'normalize', 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, 'normalize', 'compute', computeBindGroupLayout);

      // Create pipelines for all three passes
      const statsPipeline = await getOrCreateComputePipeline(
        device,
        'normalize/stats',
        computePipelineLayout,
        statsDescriptor.entryPoint ?? 'main'
      );

      const reducePipeline = await getOrCreateComputePipeline(
        device,
        'normalize/reduce',
        computePipelineLayout,
        reduceDescriptor.entryPoint ?? 'main'
      );

      const applyPipeline = await getOrCreateComputePipeline(
        device,
        'normalize/apply',
        computePipelineLayout,
        applyDescriptor.entryPoint ?? 'main'
      );

      // Create bind group
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: { buffer: statsBuffer } },
        ],
      });

      // Create output texture for buffer-to-texture conversion
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

      // Configure three-pass execution
      const workgroupSize = [8, 8, 1];
      const computePasses = [
        // Pass 1: Calculate statistics (min/max) per workgroup
        {
          pipeline: statsPipeline,
          bindGroup: computeBindGroup,
          workgroupSize,
          getDispatch: ({ width, height }) => [
            Math.ceil(width / workgroupSize[0]),
            Math.ceil(height / workgroupSize[1]),
            1
          ],
        },
        // Pass 2: Reduce all workgroup statistics to final global min/max
        {
          pipeline: reducePipeline,
          bindGroup: computeBindGroup,
          workgroupSize: [1, 1, 1],
          getDispatch: () => [1, 1, 1], // Single thread does the reduction
        },
        // Pass 3: Apply normalization
        {
          pipeline: applyPipeline,
          bindGroup: computeBindGroup,
          workgroupSize,
          getDispatch: ({ width, height }) => [
            Math.ceil(width / workgroupSize[0]),
            Math.ceil(height / workgroupSize[1]),
            1
          ],
        },
      ];

      logInfo?.('Normalize: Using three-pass shader (stats + reduce + apply)');

      this.resources = {
        shaderMetadata,
        computePipeline: applyPipeline, // Dummy for compatibility
        computeBindGroup,
        computePasses, // Multi-pass configuration drives execution
        paramsBuffer,
        paramsState,
        outputBuffer,
        outputTexture,
        statsBuffer,
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
        computeBindGroupLayout,
      };

      setStatus?.('Normalize resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create normalize resources:', error);
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

    // Reset stats buffer before each frame
    const numWorkgroupsX = Math.ceil(this.width / 8);
    const numWorkgroupsY = Math.ceil(this.height / 8);
    const numWorkgroups = numWorkgroupsX * numWorkgroupsY;
    const statsInit = new Float32Array(2 + numWorkgroups * 2);
    const F32_MAX = 3.4028235e+38;
    const F32_MIN = -3.4028235e+38;
    statsInit[0] = F32_MAX;
    statsInit[1] = F32_MIN;
    for (let i = 0; i < numWorkgroups; i++) {
      statsInit[2 + i * 2] = F32_MAX;
      statsInit[2 + i * 2 + 1] = F32_MIN;
    }
    device.queue.writeBuffer(this.resources.statsBuffer, 0, statsInit);
  }

  afterDispatch() {
    // No state to update after dispatch
  }
}

// Export additional pass descriptors for shader registry
export const additionalPasses = {
  'normalize/stats': {
    id: 'normalize/stats',
    label: 'stats.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/normalize/stats.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 32 },
      stats_buffer: { kind: 'storageBuffer', size: 'dynamic' }
    }
  },
  'normalize/reduce': {
    id: 'normalize/reduce',
    label: 'reduce.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/normalize/reduce.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 32 },
      stats_buffer: { kind: 'storageBuffer', size: 'dynamic' }
    }
  },
  'normalize/apply': {
    id: 'normalize/apply',
    label: 'apply.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/normalize/apply.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 32 },
      stats_buffer: { kind: 'storageBuffer', size: 'dynamic' }
    }
  }
};

export default NormalizeEffect;
