import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;

class ReindexEffect {
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
      enabled: true,
      displacement: 0.5,
      time: 0.0,
      speed: 1.0,
    };
  }

  invalidateResources() {
    if (!this.resources) {
      return;
    }

    const resources = this.resources;

    if (resources.resourceSet?.destroyAll) {
      try {
        resources.resourceSet.destroyAll();
      } catch (error) {
        this.helpers.logWarn?.('Failed to destroy reindex resources:', error);
      }
    }

    if (resources.statsBuffer?.destroy) {
      try {
        resources.statsBuffer.destroy();
      } catch (error) {
        this.helpers.logWarn?.('Failed to destroy reindex stats buffer:', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('ReindexEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('ReindexEffect.ensureResources requires multires output texture.');
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
      throw new TypeError('ReindexEffect.updateParams expects an object.');
    }

    const updated = [];

    for (const [name, value] of Object.entries(updates)) {
      if (name === 'enabled') {
        const newEnabled = Boolean(value);
        if (newEnabled !== this.userState.enabled) {
          this.userState.enabled = newEnabled;
          updated.push('enabled');
          if (!newEnabled && this.resources) {
            this.invalidateResources();
          }
        }
        continue;
      }

      if (name === 'displacement' || name === 'time' || name === 'speed') {
        const newValue = Number(value);
        if (Number.isFinite(newValue) && newValue !== this.userState[name]) {
          this.userState[name] = newValue;
          updated.push(name);
          if (this.resources?.paramsState) {
            const offset = name === 'displacement' ? 3 : name === 'time' ? 4 : 5;
            this.resources.paramsState[offset] = newValue;
            this.resources.paramsDirty = true;
          }
        }
      }
    }

    return { updated };
  }

  getUIState() {
    return {
      enabled: this.userState.enabled,
      displacement: this.userState.displacement,
      time: this.userState.time,
      speed: this.userState.speed,
    };
  }

  destroy() {
    this.invalidateResources();
    this.device = null;
  }

  async #createResources({ device, width, height, multiresResources }) {
    const { logInfo, logWarn } = this.helpers;

    // Load all three shaders
    const [statsCode, reduceCode, applyCode] = await Promise.all([
      fetch('/shaders/effects/reindex/stats.wgsl').then(r => r.text()),
      fetch('/shaders/effects/reindex/reduce.wgsl').then(r => r.text()),
      fetch('/shaders/effects/reindex/apply.wgsl').then(r => r.text()),
    ]);

    const statsModule = device.createShaderModule({ code: statsCode });
    const reduceModule = device.createShaderModule({ code: reduceCode });
    const applyModule = device.createShaderModule({ code: applyCode });

    // Create uniform buffer for params
    const paramsState = new Float32Array(8);
    paramsState[0] = width;
    paramsState[1] = height;
    paramsState[2] = RGBA_CHANNEL_COUNT;
    paramsState[3] = this.userState.displacement;
    paramsState[4] = this.userState.time;
    paramsState[5] = this.userState.speed;

    const paramsBuffer = device.createBuffer({
      size: paramsState.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'reindex-params',
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsState);

    // Create output buffer
    const outputBuffer = device.createBuffer({
      size: width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'reindex-output',
    });

    // Create stats buffer for workgroup min/max results
    const numWorkgroups = Math.ceil(width / 8) * Math.ceil(height / 8);
    const statsBufferSize = (2 + numWorkgroups * 2) * Float32Array.BYTES_PER_ELEMENT;
    const statsBuffer = device.createBuffer({
      size: statsBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'reindex-stats',
    });

    // Create bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
      label: 'reindex-bind-group-layout',
    });

    // Create pipelines for all three passes
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
      label: 'reindex-pipeline-layout',
    });

    const statsPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: statsModule, entryPoint: 'main' },
      label: 'reindex-stats-pipeline',
    });

    const reducePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: reduceModule, entryPoint: 'main' },
      label: 'reindex-reduce-pipeline',
    });

    const applyPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: applyModule, entryPoint: 'main' },
      label: 'reindex-apply-pipeline',
    });

    // Create bind group (shared across all passes)
    const computeBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: statsBuffer } },
      ],
      label: 'reindex-bind-group',
    });

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
        getDispatch: () => [1, 1, 1],
      },
      // Pass 3: Apply reindexing
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

    logInfo?.('Reindex: Using three-pass shader (stats + reduce + apply)');

    return {
      shaderMetadata: meta,
      computePipeline: applyPipeline, // Dummy for compatibility
      computeBindGroup,
      computePasses, // Multi-pass configuration drives execution
      paramsBuffer,
      paramsState,
      paramsDirty: false,
      outputBuffer,
      statsBuffer,
      workgroupSize,
      textureWidth: width,
      textureHeight: height,
      enabled: true,
      device,
      resourceSet: {
        destroyAll: () => {
          paramsBuffer.destroy();
          outputBuffer.destroy();
          statsBuffer.destroy();
        },
      },
    };
  }
}

export default ReindexEffect;
