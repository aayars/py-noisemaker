import meta from './meta.json' with { type: 'json' };

const WORKGROUP_SIZE = [8, 8, 1];
const CHANNEL_COUNT = 4;
const FLOAT_SIZE = 4;
const DEFAULT_TEXTURE_USAGE = GPUTextureUsage.STORAGE_BINDING |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

function dispatchDims(width, height, workgroup = WORKGROUP_SIZE) {
  return [
    Math.max(Math.ceil(width / Math.max(workgroup[0], 1)), 1),
    Math.max(Math.ceil(height / Math.max(workgroup[1], 1)), 1),
    1,
  ];
}

function createTexture(device, width, height, usage = DEFAULT_TEXTURE_USAGE) {
  return device.createTexture({
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage,
  });
}

function toBoolean(value, fallback = true) {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  return fallback;
}

class SpatterEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = {
      enabled: true,
      color: true,
      time: 0,
      speed: 1,
    };
    this.resources = null;
  }

  getUIState() {
    return { ...this.userState };
  }

  async updateParams(updates = {}) {
    const changed = [];
    if (Object.prototype.hasOwnProperty.call(updates, 'enabled')) {
      const enabled = toBoolean(updates.enabled, this.userState.enabled);
      if (enabled !== this.userState.enabled) {
        this.userState.enabled = enabled;
        if (this.resources) {
          this.resources.enabled = enabled;
        }
        changed.push('enabled');
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'color')) {
      const color = toBoolean(updates.color, this.userState.color);
      if (color !== this.userState.color) {
        this.userState.color = color;
        if (this.resources?.finalUniform) {
          const { state, offsets } = this.resources.finalUniform;
          state[offsets.colorToggle] = color ? 1 : 0;
          this.resources.finalUniform.dirty = true;
        }
        changed.push('color');
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'time')) {
      const time = Number(updates.time);
      if (Number.isFinite(time) && time !== this.userState.time) {
        this.userState.time = time;
        changed.push('time');
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'speed')) {
      const speed = Number(updates.speed);
      if (Number.isFinite(speed) && speed !== this.userState.speed) {
        this.userState.speed = speed;
        changed.push('speed');
      }
    }

    return { updated: changed };
  }

  destroy() {
    this.invalidateResources();
  }

  invalidateResources() {
    if (!this.resources) {
      return;
    }
    const cleanupFns = this.resources.cleanupFns ?? [];
    for (const fn of cleanupFns.reverse()) {
      try {
        fn?.();
      } catch (error) {
        this.helpers.logWarn?.('Spatter cleanup failed', error);
      }
    }
    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources } = {}) {
    if (!device) {
      throw new Error('SpatterEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('SpatterEffect requires multires output texture.');
    }

    const enabled = this.userState.enabled !== false;
    if (!enabled) {
      if (!this.resources || this.resources.enabled !== false ||
        this.resources.textureWidth !== width ||
        this.resources.textureHeight !== height) {
        this.invalidateResources();
        this.resources = {
          device,
          enabled: false,
          textureWidth: width,
          textureHeight: height,
          computePasses: [],
          workgroupSize: WORKGROUP_SIZE,
          outputTexture: multiresResources.outputTexture,
          blitBindGroup: multiresResources.blitBindGroup,
          cleanupFns: [],
        };
      }
      return this.resources;
    }

    if (this.resources) {
      const matches = this.resources.enabled === true &&
        this.resources.device === device &&
        this.resources.textureWidth === width &&
        this.resources.textureHeight === height;
      if (matches) {
        return this.resources;
      }
      this.invalidateResources();
    }

    this.resources = await this.#createResources({ device, width, height, multiresResources });
    return this.resources;
  }

  beforeDispatch({ device, multiresResources } = {}) {
    if (!device || !this.resources?.enabled) {
      return;
    }

    const seedValue = Number(multiresResources?.frameUniformsState?.[3] ?? 0);
    if (Number.isFinite(seedValue) && this.resources) {
      this.resources.randomSeed = seedValue;
    }

    const { uniformGroups, normalizeStatsBuffer, normalizeStatsInit } = this.resources;
    if (Array.isArray(uniformGroups)) {
      for (const group of uniformGroups) {
        if (!group?.buffer || !group.state) {
          continue;
        }
        let needsWrite = Boolean(group.dirty);
        if (typeof group.update === 'function') {
          try {
            const updated = group.update(group.state, this.userState, {
              seed: this.resources?.randomSeed ?? 0,
            });
            needsWrite = needsWrite || Boolean(updated);
          } catch (error) {
            this.helpers.logWarn?.('Spatter uniform update failed', error);
          }
        }
        if (needsWrite) {
          device.queue.writeBuffer(group.buffer, 0, group.state);
          group.dirty = false;
        }
      }
    }

    if (normalizeStatsBuffer && normalizeStatsInit) {
      device.queue.writeBuffer(normalizeStatsBuffer, 0, normalizeStatsInit);
    }
  }

  async #createResources({ device, width, height, multiresResources }) {
    const {
      getShaderDescriptor,
      getShaderMetadataCached,
      createShaderResourceSet,
      createBindGroupEntriesFromResources,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
      logWarn,
    } = this.helpers;

    const cleanupFns = [];
    const track = (fn) => {
      if (typeof fn === 'function') {
        cleanupFns.push(fn);
      }
    };
    const registerResourceSet = (set) => {
      if (set?.destroyAll) {
        track(() => {
          try {
            set.destroyAll();
          } catch (error) {
            logWarn?.('Failed to destroy shader resource set', error);
          }
        });
      }
    };
    const registerGPUResource = (resource) => {
      if (resource?.destroy) {
        track(() => {
          try {
            resource.destroy();
          } catch (error) {
            logWarn?.('Failed to destroy GPU resource', error);
          }
        });
      }
    };

    const dispatch = dispatchDims(width, height);
    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);

    const textures = {};
    const createNamedTexture = (name) => {
      const texture = createTexture(device, width, height);
      registerGPUResource(texture);
      textures[name] = texture;
      return texture;
    };

    const smearTexture = createNamedTexture('smear');
    const warpTexture = createNamedTexture('warp');
    const spatter1RawTexture = createNamedTexture('spatter1Raw');
    const spatter1BrightTexture = createNamedTexture('spatter1Bright');
    const spatter1FinalTexture = createNamedTexture('spatter1Final');
    const spatter2RawTexture = createNamedTexture('spatter2Raw');
    const spatter2BrightTexture = createNamedTexture('spatter2Bright');
    const spatter2FinalTexture = createNamedTexture('spatter2Final');
    const removalTexture = createNamedTexture('removal');
    const combinedMaskTexture = createNamedTexture('combined');
    const normalizedMaskTexture = createNamedTexture('normalized');
    const finalOutputTexture = createNamedTexture('final');

    const uniformGroups = [];

    const noiseDescriptor = getShaderDescriptor('spatter/noise_seed');
    const noiseMetadata = await getShaderMetadataCached(noiseDescriptor.id);
    const noiseLayout = getOrCreateBindGroupLayout(device, noiseDescriptor.id, 'compute', noiseMetadata);
    const noisePipelineLayout = getOrCreatePipelineLayout(device, noiseDescriptor.id, 'compute', noiseLayout);
    const noisePipeline = await getOrCreateComputePipeline(
      device,
      noiseDescriptor.id,
      noisePipelineLayout,
      noiseDescriptor.entryPoint ?? 'main',
    );

    const createNoiseVariant = (variant, baseSeed, variantSeed, targetTexture) => {
      const resourceSet = createShaderResourceSet(device, noiseDescriptor, noiseMetadata, width, height, {});
      registerResourceSet(resourceSet);

      const paramsState = new Float32Array(8);
      paramsState[0] = width;
      paramsState[1] = height;
      paramsState[2] = CHANNEL_COUNT;
      paramsState[3] = variant;
      paramsState[4] = this.userState.time;
      paramsState[5] = this.userState.speed;
      paramsState[6] = baseSeed;
      paramsState[7] = variantSeed;
      device.queue.writeBuffer(resourceSet.buffers.params, 0, paramsState);

      const bindGroup = device.createBindGroup({
        layout: noiseLayout,
        entries: createBindGroupEntriesFromResources(noiseMetadata.bindings, resourceSet),
      });

      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureLayout,
        entries: [
          { binding: 0, resource: { buffer: resourceSet.buffers.output_buffer } },
          { binding: 1, resource: targetTexture.createView() },
          { binding: 2, resource: { buffer: resourceSet.buffers.params } },
        ],
      });

      const uniformEntry = {
        label: `noise-${variant}`,
        buffer: resourceSet.buffers.params,
        state: paramsState,
        dirty: false,
        update: (state, userState, context = {}) => {
          let dirty = false;
          if (state[4] !== userState.time) {
            state[4] = userState.time;
            dirty = true;
          }
          if (state[5] !== userState.speed) {
            state[5] = userState.speed;
            dirty = true;
          }
          const seedComponent = Number(context.seed ?? 0);
          if (Number.isFinite(seedComponent)) {
            const seedFloat = seedComponent;
            const baseValue = baseSeed + seedFloat * 0.001;
            if (state[6] !== baseValue) {
              state[6] = baseValue;
              dirty = true;
            }
            const variantValue = variantSeed + (seedFloat + variant * 37.0) * 0.001;
            if (state[7] !== variantValue) {
              state[7] = variantValue;
              dirty = true;
            }
          }
          return dirty;
        },
      };
      uniformGroups.push(uniformEntry);

      return { resourceSet, bindGroup, bufferToTextureBindGroup };
    };

    const smearNoise = createNoiseVariant(0, 0, 0, smearTexture);
    const spatter1Noise = createNoiseVariant(1, 10, 5, spatter1RawTexture);
    const spatter2Noise = createNoiseVariant(2, 20, 11, spatter2RawTexture);
    const removalNoise = createNoiseVariant(3, 30, 23, removalTexture);

    const warpDescriptor = getShaderDescriptor('warp');
    const warpMetadata = await getShaderMetadataCached(warpDescriptor.id);
    const warpLayout = getOrCreateBindGroupLayout(device, warpDescriptor.id, 'compute', warpMetadata);
    const warpPipelineLayout = getOrCreatePipelineLayout(device, warpDescriptor.id, 'compute', warpLayout);
    const warpPipeline = await getOrCreateComputePipeline(device, warpDescriptor.id, warpPipelineLayout, warpDescriptor.entryPoint ?? 'main');

    const warpResourceSet = createShaderResourceSet(device, warpDescriptor, warpMetadata, width, height, {
      inputTextures: { input_texture: smearTexture },
    });
    registerResourceSet(warpResourceSet);

    const warpBindGroup = device.createBindGroup({
      layout: warpLayout,
      entries: createBindGroupEntriesFromResources(warpMetadata.bindings, warpResourceSet),
    });

    const warpParamsState = new Float32Array(12);
    warpParamsState[0] = width;
    warpParamsState[1] = height;
    warpParamsState[2] = CHANNEL_COUNT;
    const freqSeed = ((width ^ (height << 2)) % 100) / 200;
    warpParamsState[3] = 2 + freqSeed;
    warpParamsState[4] = ((width + height) % 2) + 1;
    warpParamsState[5] = 1.0 + (((width * 17 + height * 13) % 100) / 100);
    warpParamsState[6] = 3;
    warpParamsState[7] = 0;
    warpParamsState[8] = 1;
    warpParamsState[9] = this.userState.time;
    warpParamsState[10] = this.userState.speed;
    warpParamsState[11] = 0;
    device.queue.writeBuffer(warpResourceSet.buffers.params, 0, warpParamsState);

    const warpUniform = {
      label: 'warp',
      buffer: warpResourceSet.buffers.params,
      state: warpParamsState,
      dirty: false,
      update: (state, userState) => {
        let dirty = false;
        if (state[9] !== userState.time) {
          state[9] = userState.time;
          dirty = true;
        }
        if (state[10] !== userState.speed) {
          state[10] = userState.speed;
          dirty = true;
        }
        return dirty;
      },
    };
    uniformGroups.push(warpUniform);

    const warpBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: warpResourceSet.buffers.output_buffer } },
        { binding: 1, resource: warpTexture.createView() },
        { binding: 2, resource: { buffer: warpResourceSet.buffers.params } },
      ],
    });

    const brightnessDescriptor = getShaderDescriptor('adjust_brightness');
    const brightnessMetadata = await getShaderMetadataCached(brightnessDescriptor.id);
    const brightnessLayout = getOrCreateBindGroupLayout(device, brightnessDescriptor.id, 'compute', brightnessMetadata);
    const brightnessPipelineLayout = getOrCreatePipelineLayout(device, brightnessDescriptor.id, 'compute', brightnessLayout);
    const brightnessPipeline = await getOrCreateComputePipeline(device, brightnessDescriptor.id, brightnessPipelineLayout, brightnessDescriptor.entryPoint ?? 'main');

    const contrastDescriptor = getShaderDescriptor('adjust_contrast');
    const contrastMetadata = await getShaderMetadataCached(contrastDescriptor.id);
    const contrastLayout = getOrCreateBindGroupLayout(device, contrastDescriptor.id, 'compute', contrastMetadata);
    const contrastPipelineLayout = getOrCreatePipelineLayout(device, contrastDescriptor.id, 'compute', contrastLayout);
    const contrastPipeline = await getOrCreateComputePipeline(device, contrastDescriptor.id, contrastPipelineLayout, contrastDescriptor.entryPoint ?? 'main');

    const createBrightnessResources = (inputTexture, amount, targetTexture) => {
      const resourceSet = createShaderResourceSet(device, brightnessDescriptor, brightnessMetadata, width, height, {
        inputTextures: { input_texture: inputTexture },
      });
      registerResourceSet(resourceSet);

      const paramsState = new Float32Array(8);
      paramsState[0] = width;
      paramsState[1] = height;
      paramsState[2] = CHANNEL_COUNT;
      paramsState[3] = amount;
      paramsState[4] = this.userState.time;
      paramsState[5] = this.userState.speed;
      device.queue.writeBuffer(resourceSet.buffers.params, 0, paramsState);

      const bindGroup = device.createBindGroup({
        layout: brightnessLayout,
        entries: createBindGroupEntriesFromResources(brightnessMetadata.bindings, resourceSet),
      });

      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureLayout,
        entries: [
          { binding: 0, resource: { buffer: resourceSet.buffers.output_buffer } },
          { binding: 1, resource: targetTexture.createView() },
          { binding: 2, resource: { buffer: resourceSet.buffers.params } },
        ],
      });

      const uniformEntry = {
        label: 'brightness',
        buffer: resourceSet.buffers.params,
        state: paramsState,
        dirty: false,
        update: (state, userState) => {
          let dirty = false;
          if (state[4] !== userState.time) {
            state[4] = userState.time;
            dirty = true;
          }
          if (state[5] !== userState.speed) {
            state[5] = userState.speed;
            dirty = true;
          }
          return dirty;
        },
      };
      uniformGroups.push(uniformEntry);

      return { resourceSet, bindGroup, bufferToTextureBindGroup };
    };

    const createContrastResources = (inputTexture, amount, targetTexture) => {
      const resourceSet = createShaderResourceSet(device, contrastDescriptor, contrastMetadata, width, height, {
        inputTextures: { input_texture: inputTexture },
      });
      registerResourceSet(resourceSet);

      const paramsState = new Float32Array(8);
      paramsState[0] = width;
      paramsState[1] = height;
      paramsState[2] = CHANNEL_COUNT;
      paramsState[3] = amount;
      paramsState[4] = this.userState.time;
      paramsState[5] = this.userState.speed;
      device.queue.writeBuffer(resourceSet.buffers.params, 0, paramsState);

      const bindGroup = device.createBindGroup({
        layout: contrastLayout,
        entries: createBindGroupEntriesFromResources(contrastMetadata.bindings, resourceSet),
      });

      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureLayout,
        entries: [
          { binding: 0, resource: { buffer: resourceSet.buffers.output_buffer } },
          { binding: 1, resource: targetTexture.createView() },
          { binding: 2, resource: { buffer: resourceSet.buffers.params } },
        ],
      });

      const uniformEntry = {
        label: 'contrast',
        buffer: resourceSet.buffers.params,
        state: paramsState,
        dirty: false,
        update: (state, userState) => {
          let dirty = false;
          if (state[4] !== userState.time) {
            state[4] = userState.time;
            dirty = true;
          }
          if (state[5] !== userState.speed) {
            state[5] = userState.speed;
            dirty = true;
          }
          return dirty;
        },
      };
      uniformGroups.push(uniformEntry);

      return { resourceSet, bindGroup, bufferToTextureBindGroup };
    };

    const spatter1Brightness = createBrightnessResources(spatter1RawTexture, -1.0, spatter1BrightTexture);
    const spatter1Contrast = createContrastResources(spatter1BrightTexture, 4.0, spatter1FinalTexture);
    const spatter2Brightness = createBrightnessResources(spatter2RawTexture, -1.25, spatter2BrightTexture);
    const spatter2Contrast = createContrastResources(spatter2BrightTexture, 4.0, spatter2FinalTexture);

    const combineDescriptor = getShaderDescriptor('spatter/combine');
    const combineMetadata = await getShaderMetadataCached(combineDescriptor.id);
    const combineLayout = getOrCreateBindGroupLayout(device, combineDescriptor.id, 'compute', combineMetadata);
    const combinePipelineLayout = getOrCreatePipelineLayout(device, combineDescriptor.id, 'compute', combineLayout);
    const combinePipeline = await getOrCreateComputePipeline(device, combineDescriptor.id, combinePipelineLayout, combineDescriptor.entryPoint ?? 'main');

    const combineResourceSet = createShaderResourceSet(device, combineDescriptor, combineMetadata, width, height, {
      inputTextures: {
        smear_texture: warpTexture,
        spatter_primary_texture: spatter1FinalTexture,
        spatter_secondary_texture: spatter2FinalTexture,
        removal_texture: removalTexture,
      },
    });
    registerResourceSet(combineResourceSet);

    const combineParamsState = new Float32Array(4);
    combineParamsState[0] = width;
    combineParamsState[1] = height;
    combineParamsState[2] = CHANNEL_COUNT;
    device.queue.writeBuffer(combineResourceSet.buffers.params, 0, combineParamsState);

    const combineBindGroup = device.createBindGroup({
      layout: combineLayout,
      entries: createBindGroupEntriesFromResources(combineMetadata.bindings, combineResourceSet),
    });

    const combineBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: combineResourceSet.buffers.output_buffer } },
        { binding: 1, resource: combinedMaskTexture.createView() },
        { binding: 2, resource: { buffer: combineResourceSet.buffers.params } },
      ],
    });

    uniformGroups.push({
      label: 'combine',
      buffer: combineResourceSet.buffers.params,
      state: combineParamsState,
      dirty: false,
      update: null,
    });

    const normalizeStatsDescriptor = getShaderDescriptor('normalize/stats');
    const normalizeReduceDescriptor = getShaderDescriptor('normalize/reduce');
    const normalizeApplyDescriptor = getShaderDescriptor('normalize/apply');
    const normalizeMetadata = await getShaderMetadataCached('normalize');
    const normalizeLayout = getOrCreateBindGroupLayout(device, 'normalize', 'compute', normalizeMetadata);
    const normalizePipelineLayout = getOrCreatePipelineLayout(device, 'normalize', 'compute', normalizeLayout);

    const normalizeStatsPipeline = await getOrCreateComputePipeline(
      device,
      normalizeStatsDescriptor.id,
      normalizePipelineLayout,
      normalizeStatsDescriptor.entryPoint ?? 'main',
    );
    const normalizeReducePipeline = await getOrCreateComputePipeline(
      device,
      normalizeReduceDescriptor.id,
      normalizePipelineLayout,
      normalizeReduceDescriptor.entryPoint ?? 'main',
    );
    const normalizeApplyPipeline = await getOrCreateComputePipeline(
      device,
      normalizeApplyDescriptor.id,
      normalizePipelineLayout,
      normalizeApplyDescriptor.entryPoint ?? 'main',
    );

    const normParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    registerGPUResource(normParamsBuffer);

    const normParamsState = new Float32Array(8);
    normParamsState[0] = width;
    normParamsState[1] = height;
    normParamsState[2] = CHANNEL_COUNT;
    normParamsState[4] = this.userState.time;
    normParamsState[5] = this.userState.speed;
    device.queue.writeBuffer(normParamsBuffer, 0, normParamsState);

    const normOutputBufferSize = width * height * CHANNEL_COUNT * FLOAT_SIZE;
    const normOutputBuffer = device.createBuffer({
      size: normOutputBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    registerGPUResource(normOutputBuffer);

    const numWorkgroupsX = dispatch[0];
    const numWorkgroupsY = dispatch[1];
    const numWorkgroups = numWorkgroupsX * numWorkgroupsY;
    const normStatsBufferSize = (2 + numWorkgroups * 2) * FLOAT_SIZE;
    const normStatsBuffer = device.createBuffer({
      size: normStatsBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    registerGPUResource(normStatsBuffer);

    const F32_MAX = Number.MAX_VALUE;
    const F32_MIN = -Number.MAX_VALUE;
    const normStatsInit = new Float32Array(2 + numWorkgroups * 2);
    normStatsInit[0] = F32_MAX;
    normStatsInit[1] = F32_MIN;
    for (let i = 0; i < numWorkgroups; i += 1) {
      const base = 2 + i * 2;
      normStatsInit[base] = F32_MAX;
      normStatsInit[base + 1] = F32_MIN;
    }
    device.queue.writeBuffer(normStatsBuffer, 0, normStatsInit);

    const normalizeBindGroup = device.createBindGroup({
      layout: normalizeLayout,
      entries: [
        { binding: 0, resource: combinedMaskTexture.createView() },
        { binding: 1, resource: { buffer: normOutputBuffer } },
        { binding: 2, resource: { buffer: normParamsBuffer } },
        { binding: 3, resource: { buffer: normStatsBuffer } },
      ],
    });

    const normBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: normOutputBuffer } },
        { binding: 1, resource: normalizedMaskTexture.createView() },
        { binding: 2, resource: { buffer: normParamsBuffer } },
      ],
    });

    uniformGroups.push({
      label: 'normalize',
      buffer: normParamsBuffer,
      state: normParamsState,
      dirty: false,
      update: (state, userState) => {
        let dirty = false;
        if (state[4] !== userState.time) {
          state[4] = userState.time;
          dirty = true;
        }
        if (state[5] !== userState.speed) {
          state[5] = userState.speed;
          dirty = true;
        }
        return dirty;
      },
    });

    const finalDescriptor = getShaderDescriptor(meta.id);
    const finalMetadata = await getShaderMetadataCached(meta.id);
    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
    const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');

    const finalResourceSet = createShaderResourceSet(device, finalDescriptor, finalMetadata, width, height, {
      inputTextures: {
        input_texture: multiresResources.outputTexture,
        mask_texture: normalizedMaskTexture,
      },
    });
    registerResourceSet(finalResourceSet);

    const finalParamsState = new Float32Array(12);
    finalParamsState[0] = width;
    finalParamsState[1] = height;
    finalParamsState[2] = CHANNEL_COUNT;
    finalParamsState[4] = this.userState.color ? 1 : 0;
    finalParamsState[5] = 0.875;
    finalParamsState[6] = 0.125;
    finalParamsState[7] = 0.125;
    finalParamsState[8] = this.userState.time;
    finalParamsState[9] = this.userState.speed;
    device.queue.writeBuffer(finalResourceSet.buffers.params, 0, finalParamsState);

    const finalBindGroup = device.createBindGroup({
      layout: finalLayout,
      entries: createBindGroupEntriesFromResources(finalMetadata.bindings, finalResourceSet),
    });

    const finalBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: finalResourceSet.buffers.output_buffer } },
        { binding: 1, resource: finalOutputTexture.createView() },
        { binding: 2, resource: { buffer: finalResourceSet.buffers.params } },
      ],
    });

    const finalOffsets = {
      colorToggle: 4,
      colorR: 5,
      colorG: 6,
      colorB: 7,
      time: 8,
      speed: 9,
    };
    const finalUniform = {
      label: 'final',
      buffer: finalResourceSet.buffers.params,
      state: finalParamsState,
      dirty: false,
      offsets: finalOffsets,
      update: (state, userState) => {
        let dirty = false;
        if (state[finalOffsets.time] !== userState.time) {
          state[finalOffsets.time] = userState.time;
          dirty = true;
        }
        if (state[finalOffsets.speed] !== userState.speed) {
          state[finalOffsets.speed] = userState.speed;
          dirty = true;
        }
        return dirty;
      },
    };
    uniformGroups.push(finalUniform);

    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [{ binding: 0, resource: finalOutputTexture.createView() }],
    });

    const bufferDispatch = dispatch;
    const singleDispatch = [1, 1, 1];

    const computePasses = [
      { pipeline: noisePipeline, bindGroup: smearNoise.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: smearNoise.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: noisePipeline, bindGroup: spatter1Noise.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter1Noise.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: brightnessPipeline, bindGroup: spatter1Brightness.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter1Brightness.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: contrastPipeline, bindGroup: spatter1Contrast.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter1Contrast.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: noisePipeline, bindGroup: spatter2Noise.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter2Noise.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: brightnessPipeline, bindGroup: spatter2Brightness.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter2Brightness.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: contrastPipeline, bindGroup: spatter2Contrast.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: spatter2Contrast.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: noisePipeline, bindGroup: removalNoise.bindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: removalNoise.bufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: warpPipeline, bindGroup: warpBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: warpBufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: combinePipeline, bindGroup: combineBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: combineBufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: normalizeStatsPipeline, bindGroup: normalizeBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: normalizeReducePipeline, bindGroup: normalizeBindGroup, workgroupSize: singleDispatch, dispatch: singleDispatch },
      { pipeline: normalizeApplyPipeline, bindGroup: normalizeBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: normBufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: finalPipeline, bindGroup: finalBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
      { pipeline: bufferToTexturePipeline, bindGroup: finalBufferToTextureBindGroup, workgroupSize: WORKGROUP_SIZE, dispatch: bufferDispatch },
    ];

    return {
      device,
      enabled: true,
      textureWidth: width,
      textureHeight: height,
      computePasses,
      workgroupSize: WORKGROUP_SIZE,
      bufferToTexturePipeline,
  bufferToTextureBindGroup: finalBufferToTextureBindGroup,
      outputBuffer: finalResourceSet.buffers.output_buffer,
      outputTexture: finalOutputTexture,
      blitBindGroup,
      uniformGroups,
      finalUniform,
      normalizeStatsBuffer: normStatsBuffer,
      normalizeStatsInit: normStatsInit,
      cleanupFns,
    };
  }
}

export default SpatterEffect;
export const additionalPasses = {
  'spatter/noise_seed': {
    id: 'spatter/noise_seed',
    label: 'noise_seed.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/spatter/noise_seed.wgsl',
    resources: {
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 32 }
    }
  },
  'spatter/combine': {
    id: 'spatter/combine',
    label: 'combine.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/spatter/combine.wgsl',
    resources: {
      smear_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      spatter_primary_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      spatter_secondary_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      removal_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 16 }
    }
  }
};
