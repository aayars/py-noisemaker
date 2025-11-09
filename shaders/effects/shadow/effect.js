import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const WORKGROUP_SIZE = [8, 8, 1];
const COPY_WORKGROUP_SIZE = [8, 8, 1];

export const additionalPasses = {
  'shadow/value_map': {
    id: 'shadow/value_map',
    label: 'value_map.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/shadow/value_map.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
    },
  },
  'shadow/sobel_distance': {
    id: 'shadow/sobel_distance',
    label: 'sobel_distance.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/shadow/sobel_distance.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
    },
  },
  'shadow/shadow_blend': {
    id: 'shadow/shadow_blend',
    label: 'shadow_blend.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/shadow/shadow_blend.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 32 },
      shade_texture: { kind: 'sampledTexture', format: 'rgba32float' },
    },
  },
};

function getParameterDefault(name, fallback) {
  const param = (meta.parameters ?? []).find((entry) => entry.name === name);
  return param?.default ?? fallback;
}

class ShadowEffect {
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
      alpha: Number(getParameterDefault('alpha', 1.0)) || 1.0,
      time: Number(getParameterDefault('time', 0.0)) || 0.0,
      speed: Number(getParameterDefault('speed', 1.0)) || 1.0,
    };
  }

  invalidateResources() {
    if (!this.resources) {
      return;
    }

    const { logWarn } = this.helpers;
    const { resourceSet } = this.resources;

    if (resourceSet?.destroyAll) {
      try {
        resourceSet.destroyAll();
      } catch (error) {
        logWarn?.('Shadow: failed to destroy GPU resources.', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('ShadowEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('ShadowEffect.ensureResources requires multires output texture.');
    }

    if (this.device && this.device !== device) {
      this.invalidateResources();
    }

    this.device = device;
    this.width = width;
    this.height = height;

    const resources = this.resources;
    if (resources && resources.textureWidth === width && resources.textureHeight === height) {
      return resources;
    }

    this.invalidateResources();
    return this.#createResources({ device, width, height, multiresResources });
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('ShadowEffect.updateParams expects an object.');
    }

    const updated = [];
    const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

    for (const [name, value] of Object.entries(updates)) {
      if (name === 'alpha') {
        const numeric = clamp01(Number(value));
        if (Number.isFinite(numeric) && numeric !== this.userState.alpha) {
          this.userState.alpha = numeric;
          updated.push('alpha');
          if (this.resources?.blendParamsState) {
            this.resources.blendParamsState[3] = numeric;
            this.resources.blendParamsDirty = true;
          }
        }
        continue;
      }

      if (name === 'time' || name === 'speed') {
        const numeric = Number(value);
        if (Number.isFinite(numeric) && numeric !== this.userState[name]) {
          this.userState[name] = numeric;
          updated.push(name);

          if (this.resources) {
            if (this.resources.normalizeParamsState) {
              const timeIndex = 4;
              const speedIndex = 5;
              this.resources.normalizeParamsState[timeIndex] = this.userState.time;
              this.resources.normalizeParamsState[speedIndex] = this.userState.speed;
              this.resources.normalizeParamsDirty = true;
            }

            if (this.resources.convolveParamsState) {
              const timeIndex = 6;
              const speedIndex = 7;
              this.resources.convolveParamsState[timeIndex] = this.userState.time;
              this.resources.convolveParamsState[speedIndex] = this.userState.speed;
              this.resources.convolveParamsDirty = true;
            }

            if (this.resources.blendParamsState) {
              const timeIndex = 4;
              const speedIndex = 5;
              this.resources.blendParamsState[timeIndex] = this.userState.time;
              this.resources.blendParamsState[speedIndex] = this.userState.speed;
              this.resources.blendParamsDirty = true;
            }
          }
        }
      }
    }

    return { updated };
  }

  beforeDispatch({ device }) {
    if (!this.resources || !device) {
      return;
    }

    const {
      normalizeStatsBuffer,
      normalizeStatsInit,
      normalizeParamsDirty,
      normalizeParamsBuffer,
      normalizeParamsState,
      blendParamsDirty,
      blendParamsBuffer,
      blendParamsState,
      convolveParamsDirty,
      convolveParamsBuffer,
      convolveParamsState,
    } = this.resources;

    if (normalizeStatsBuffer && normalizeStatsInit) {
      device.queue.writeBuffer(normalizeStatsBuffer, 0, normalizeStatsInit);
    }

    if (normalizeParamsDirty && normalizeParamsBuffer && normalizeParamsState) {
      device.queue.writeBuffer(normalizeParamsBuffer, 0, normalizeParamsState);
      this.resources.normalizeParamsDirty = false;
    }

    if (blendParamsDirty && blendParamsBuffer && blendParamsState) {
      device.queue.writeBuffer(blendParamsBuffer, 0, blendParamsState);
      this.resources.blendParamsDirty = false;
    }

    if (convolveParamsDirty && convolveParamsBuffer && convolveParamsState) {
      device.queue.writeBuffer(convolveParamsBuffer, 0, convolveParamsState);
      this.resources.convolveParamsDirty = false;
    }
  }

  afterDispatch() {
    // No-op hook for interface parity.
  }

  async #createResources({ device, width, height, multiresResources }) {
    const {
      logInfo,
      logWarn,
      setStatus,
      getBufferToTexturePipeline,
      getShaderDescriptor,
      getShaderMetadataCached,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
    } = this.helpers;

    setStatus?.('Creating shadow resources…');

    try {
      const valueMapDescriptor = getShaderDescriptor('shadow/value_map');
      const sobelDescriptor = getShaderDescriptor('shadow/sobel_distance');
      const blendDescriptor = getShaderDescriptor('shadow/shadow_blend');
      const normalizeStatsDescriptor = getShaderDescriptor('normalize/stats');
      const normalizeReduceDescriptor = getShaderDescriptor('normalize/reduce');
      const normalizeApplyDescriptor = getShaderDescriptor('normalize/apply');
      const convolveDescriptor = getShaderDescriptor('convolve');

      const [
        valueMapMetadata,
        sobelMetadata,
        blendMetadata,
        normalizeMetadata,
        convolveMetadata,
      ] = await Promise.all([
        getShaderMetadataCached(valueMapDescriptor.id),
        getShaderMetadataCached(sobelDescriptor.id),
        getShaderMetadataCached(blendDescriptor.id),
        getShaderMetadataCached('normalize'),
        getShaderMetadataCached(convolveDescriptor.id),
      ]);

      const valueMapBindGroupLayout = getOrCreateBindGroupLayout(
        device,
        valueMapDescriptor.id,
        'compute',
        valueMapMetadata,
      );
      const sobelBindGroupLayout = getOrCreateBindGroupLayout(
        device,
        sobelDescriptor.id,
        'compute',
        sobelMetadata,
      );
      const blendBindGroupLayout = getOrCreateBindGroupLayout(
        device,
        blendDescriptor.id,
        'compute',
        blendMetadata,
      );
      const normalizeBindGroupLayout = getOrCreateBindGroupLayout(
        device,
        'normalize',
        'compute',
        normalizeMetadata,
      );
      const convolveBindGroupLayout = getOrCreateBindGroupLayout(
        device,
        convolveDescriptor.id,
        'compute',
        convolveMetadata,
      );

      const valueMapPipelineLayout = getOrCreatePipelineLayout(
        device,
        valueMapDescriptor.id,
        'compute',
        valueMapBindGroupLayout,
      );
      const sobelPipelineLayout = getOrCreatePipelineLayout(
        device,
        sobelDescriptor.id,
        'compute',
        sobelBindGroupLayout,
      );
      const blendPipelineLayout = getOrCreatePipelineLayout(
        device,
        blendDescriptor.id,
        'compute',
        blendBindGroupLayout,
      );
      const normalizePipelineLayout = getOrCreatePipelineLayout(
        device,
        'normalize',
        'compute',
        normalizeBindGroupLayout,
      );
      const convolvePipelineLayout = getOrCreatePipelineLayout(
        device,
        convolveDescriptor.id,
        'compute',
        convolveBindGroupLayout,
      );

      const valueMapPipeline = await getOrCreateComputePipeline(
        device,
        valueMapDescriptor.id,
        valueMapPipelineLayout,
        valueMapDescriptor.entryPoint ?? 'main',
      );
      const sobelPipeline = await getOrCreateComputePipeline(
        device,
        sobelDescriptor.id,
        sobelPipelineLayout,
        sobelDescriptor.entryPoint ?? 'main',
      );
      const blendPipeline = await getOrCreateComputePipeline(
        device,
        blendDescriptor.id,
        blendPipelineLayout,
        blendDescriptor.entryPoint ?? 'main',
      );
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
      const convolveResetPipeline = await getOrCreateComputePipeline(
        device,
        convolveDescriptor.id,
        convolvePipelineLayout,
        'reset_stats_main',
      );
      const convolveMainPipeline = await getOrCreateComputePipeline(
        device,
        convolveDescriptor.id,
        convolvePipelineLayout,
        'convolve_main',
      );
      const convolveApplyPipeline = await getOrCreateComputePipeline(
        device,
        convolveDescriptor.id,
        convolvePipelineLayout,
        convolveDescriptor.entryPoint ?? 'main',
      );

      const pixelCount = Math.max(width * height, 1);
      const pixelBytes = pixelCount * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const textureUsage = GPUTextureUsage.TEXTURE_BINDING
        | GPUTextureUsage.STORAGE_BINDING
        | GPUTextureUsage.COPY_DST
        | GPUTextureUsage.COPY_SRC;

      const stageTexture = device.createTexture({
        label: 'shadow/stage_texture',
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: textureUsage,
      });
      const outputTexture = device.createTexture({
        label: 'shadow/output_texture',
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: textureUsage,
      });

      const stageBuffer = device.createBuffer({
        label: 'shadow/stage_buffer',
        size: pixelBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const workgroupsX = Math.ceil(width / WORKGROUP_SIZE[0]);
      const workgroupsY = Math.ceil(height / WORKGROUP_SIZE[1]);
      const workgroupCount = workgroupsX * workgroupsY;

      const normalizeStatsInit = new Float32Array(2 + workgroupCount * 2);
      const f32Max = 3.4028235e38;
      const f32Min = -3.4028235e38;
      normalizeStatsInit[0] = f32Max;
      normalizeStatsInit[1] = f32Min;
      for (let i = 0; i < workgroupCount; i += 1) {
        const offset = 2 + i * 2;
        normalizeStatsInit[offset] = f32Max;
        normalizeStatsInit[offset + 1] = f32Min;
      }

      const normalizeStatsBuffer = device.createBuffer({
        label: 'shadow/normalize_stats_buffer',
        size: normalizeStatsInit.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(normalizeStatsBuffer, 0, normalizeStatsInit);

      const convolveStatsBuffer = device.createBuffer({
        label: 'shadow/convolve_stats_buffer',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const normalizeParamsState = new Float32Array(8);
      normalizeParamsState[0] = width;
      normalizeParamsState[1] = height;
      normalizeParamsState[2] = RGBA_CHANNEL_COUNT;
      normalizeParamsState[3] = 0;
      normalizeParamsState[4] = this.userState.time;
      normalizeParamsState[5] = this.userState.speed;
      const normalizeParamsBuffer = device.createBuffer({
        label: 'shadow/normalize_params_buffer',
        size: normalizeParamsState.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(normalizeParamsBuffer, 0, normalizeParamsState);

      const convolveParamsState = new Float32Array(8);
      convolveParamsState[0] = width;
      convolveParamsState[1] = height;
      convolveParamsState[2] = RGBA_CHANNEL_COUNT;
      convolveParamsState[3] = 807;
  convolveParamsState[4] = 1;
      convolveParamsState[5] = 0.5;
      convolveParamsState[6] = this.userState.time;
      convolveParamsState[7] = this.userState.speed;
      const convolveParamsBuffer = device.createBuffer({
        label: 'shadow/convolve_params_buffer',
        size: convolveParamsState.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(convolveParamsBuffer, 0, convolveParamsState);

      const blendParamsState = new Float32Array(8);
      blendParamsState[0] = width;
      blendParamsState[1] = height;
      blendParamsState[2] = RGBA_CHANNEL_COUNT;
      blendParamsState[3] = this.userState.alpha;
      blendParamsState[4] = this.userState.time;
      blendParamsState[5] = this.userState.speed;
      const blendParamsBuffer = device.createBuffer({
        label: 'shadow/blend_params_buffer',
        size: blendParamsState.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(blendParamsBuffer, 0, blendParamsState);

      const copyParamsState = new Float32Array(8);
      copyParamsState[0] = width;
      copyParamsState[1] = height;
      copyParamsState[2] = RGBA_CHANNEL_COUNT;
      const copyParamsBuffer = device.createBuffer({
        label: 'shadow/copy_params_buffer',
        size: copyParamsState.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(copyParamsBuffer, 0, copyParamsState);

      const bufferToTextureResult = await getBufferToTexturePipeline?.(device);
      if (!bufferToTextureResult?.pipeline || !bufferToTextureResult?.bindGroupLayout) {
        throw new Error('Shadow: buffer-to-texture pipeline is unavailable.');
      }
      const bufferToTexturePipeline = bufferToTextureResult.pipeline;
      const bufferToTextureBindGroupLayout = bufferToTextureResult.bindGroupLayout;

      const originalView = multiresResources.outputTexture.createView();
      const stageSampleView = stageTexture.createView();
      const stageStorageView = stageTexture.createView();
      const outputStorageView = outputTexture.createView();
      const outputSampleView = outputTexture.createView();

      const valueMapBindGroup = device.createBindGroup({
        label: 'shadow/value_map_bg',
        layout: valueMapBindGroupLayout,
        entries: [
          { binding: 0, resource: originalView },
          { binding: 1, resource: { buffer: stageBuffer } },
        ],
      });

      const stageCopyBindGroup = device.createBindGroup({
        label: 'shadow/stage_copy_bg',
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: stageBuffer } },
          { binding: 1, resource: stageStorageView },
          { binding: 2, resource: { buffer: copyParamsBuffer } },
        ],
      });

      const sobelBindGroup = device.createBindGroup({
        label: 'shadow/sobel_bg',
        layout: sobelBindGroupLayout,
        entries: [
          { binding: 0, resource: stageSampleView },
          { binding: 1, resource: { buffer: stageBuffer } },
        ],
      });

      const normalizeStatsBindGroup = device.createBindGroup({
        label: 'shadow/normalize_stats_bg',
        layout: normalizeBindGroupLayout,
        entries: [
          { binding: 0, resource: stageSampleView },
          { binding: 1, resource: { buffer: stageBuffer } },
          { binding: 2, resource: { buffer: normalizeParamsBuffer } },
          { binding: 3, resource: { buffer: normalizeStatsBuffer } },
        ],
      });

      const normalizeReduceBindGroup = device.createBindGroup({
        label: 'shadow/normalize_reduce_bg',
        layout: normalizeBindGroupLayout,
        entries: [
          { binding: 0, resource: stageSampleView },
          { binding: 1, resource: { buffer: stageBuffer } },
          { binding: 2, resource: { buffer: normalizeParamsBuffer } },
          { binding: 3, resource: { buffer: normalizeStatsBuffer } },
        ],
      });

      const normalizeApplyBindGroup = device.createBindGroup({
        label: 'shadow/normalize_apply_bg',
        layout: normalizeBindGroupLayout,
        entries: [
          { binding: 0, resource: stageSampleView },
          { binding: 1, resource: { buffer: stageBuffer } },
          { binding: 2, resource: { buffer: normalizeParamsBuffer } },
          { binding: 3, resource: { buffer: normalizeStatsBuffer } },
        ],
      });

      const convolveBindGroup = device.createBindGroup({
        label: 'shadow/convolve_bg',
        layout: convolveBindGroupLayout,
        entries: [
          { binding: 0, resource: stageSampleView },
          { binding: 1, resource: { buffer: stageBuffer } },
          { binding: 2, resource: { buffer: convolveParamsBuffer } },
          { binding: 3, resource: { buffer: convolveStatsBuffer } },
        ],
      });

      const blendBindGroup = device.createBindGroup({
        label: 'shadow/blend_bg',
        layout: blendBindGroupLayout,
        entries: [
          { binding: 0, resource: originalView },
          { binding: 1, resource: { buffer: stageBuffer } },
          { binding: 2, resource: { buffer: blendParamsBuffer } },
          { binding: 3, resource: stageSampleView },
        ],
      });

      const outputCopyBindGroup = device.createBindGroup({
        label: 'shadow/output_copy_bg',
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: stageBuffer } },
          { binding: 1, resource: outputStorageView },
          { binding: 2, resource: { buffer: copyParamsBuffer } },
        ],
      });

      const blitBindGroup = device.createBindGroup({
        label: 'shadow/blit_bg',
        layout: multiresResources.blitBindGroupLayout,
        entries: [{ binding: 0, resource: outputSampleView }],
      });

      const dispatch2D = (workgroup) => ({ width: viewWidth, height: viewHeight }) => {
        const [wgx, wgy, wgz] = workgroup;
        const safeX = Math.max(Math.trunc(wgx ?? 1), 1);
        const safeY = Math.max(Math.trunc(wgy ?? 1), 1);
        const safeZ = Math.max(Math.trunc(wgz ?? 1), 1);
        return [
          Math.ceil(viewWidth / safeX),
          Math.ceil(viewHeight / safeY),
          Math.ceil(1 / safeZ),
        ];
      };

      const computePasses = [
        {
          label: 'shadow/value_map',
          pipeline: valueMapPipeline,
          bindGroup: valueMapBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/value_map_copy',
          pipeline: bufferToTexturePipeline,
          bindGroup: stageCopyBindGroup,
          workgroupSize: COPY_WORKGROUP_SIZE,
          getDispatch: dispatch2D(COPY_WORKGROUP_SIZE),
        },
        {
          label: 'shadow/sobel',
          pipeline: sobelPipeline,
          bindGroup: sobelBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/sobel_copy',
          pipeline: bufferToTexturePipeline,
          bindGroup: stageCopyBindGroup,
          workgroupSize: COPY_WORKGROUP_SIZE,
          getDispatch: dispatch2D(COPY_WORKGROUP_SIZE),
        },
        {
          label: 'shadow/normalize_stats',
          pipeline: normalizeStatsPipeline,
          bindGroup: normalizeStatsBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/normalize_reduce',
          pipeline: normalizeReducePipeline,
          bindGroup: normalizeReduceBindGroup,
          workgroupSize: [1, 1, 1],
          getDispatch: () => [1, 1, 1],
        },
        {
          label: 'shadow/normalize_apply',
          pipeline: normalizeApplyPipeline,
          bindGroup: normalizeApplyBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/normalize_copy',
          pipeline: bufferToTexturePipeline,
          bindGroup: stageCopyBindGroup,
          workgroupSize: COPY_WORKGROUP_SIZE,
          getDispatch: dispatch2D(COPY_WORKGROUP_SIZE),
        },
        {
          label: 'shadow/convolve_reset',
          pipeline: convolveResetPipeline,
          bindGroup: convolveBindGroup,
          workgroupSize: [1, 1, 1],
          getDispatch: () => [1, 1, 1],
        },
        {
          label: 'shadow/convolve_main',
          pipeline: convolveMainPipeline,
          bindGroup: convolveBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/convolve_apply',
          pipeline: convolveApplyPipeline,
          bindGroup: convolveBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
        {
          label: 'shadow/sharpen_copy',
          pipeline: bufferToTexturePipeline,
          bindGroup: stageCopyBindGroup,
          workgroupSize: COPY_WORKGROUP_SIZE,
          getDispatch: dispatch2D(COPY_WORKGROUP_SIZE),
        },
        {
          label: 'shadow/blend',
          pipeline: blendPipeline,
          bindGroup: blendBindGroup,
          workgroupSize: WORKGROUP_SIZE,
          getDispatch: dispatch2D(WORKGROUP_SIZE),
        },
      ];

      logInfo?.('Shadow: Using 13-pass shader (value_map → sobel → normalize → convolve → blend).');

      const destroyList = [
        stageTexture,
        outputTexture,
        stageBuffer,
        normalizeStatsBuffer,
        convolveStatsBuffer,
        normalizeParamsBuffer,
        convolveParamsBuffer,
        blendParamsBuffer,
        copyParamsBuffer,
      ];

      const resources = {
        enabled: true,
        computePipeline: null,
        computeBindGroup: null,
        computePasses,
        workgroupSize: WORKGROUP_SIZE,
        bufferToTexturePipeline,
        bufferToTextureBindGroup: outputCopyBindGroup,
        bufferToTextureWorkgroupSize: COPY_WORKGROUP_SIZE,
        blitBindGroup,
        textureWidth: width,
        textureHeight: height,
        outputBuffer: stageBuffer,
        outputTexture,
        stageBuffer,
        stageTexture,
        normalizeStatsBuffer,
        normalizeStatsInit,
        normalizeParamsBuffer,
        normalizeParamsState,
        normalizeParamsDirty: false,
        convolveParamsBuffer,
        convolveParamsState,
        convolveParamsDirty: false,
        blendParamsBuffer,
        blendParamsState,
        blendParamsDirty: false,
        copyParamsBuffer,
        copyParamsState,
        resourceSet: {
          destroyAll: () => {
            for (const resource of destroyList) {
              if (resource?.destroy) {
                try {
                  resource.destroy();
                } catch (error) {
                  logWarn?.('Shadow: failed to destroy resource.', error);
                }
              }
            }
          },
        },
      };

      this.resources = resources;
      setStatus?.('');
      return resources;
    } catch (error) {
      setStatus?.('');
      logWarn?.('Shadow: failed to create resources.', error);
      throw error;
    }
  }
}

export default ShadowEffect;
