import metadata from './meta.json' with { type: 'json' };
import voronoiMetadata from '../voronoi/meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const BUFFER_TO_TEXTURE_WORKGROUP = [8, 8, 1];

function getParamDefault(metaObj, name, fallback) {
  const entry = (metaObj.parameters ?? []).find((param) => param.name === name);
  if (!entry) {
    return fallback;
  }
  return Number(entry.default ?? fallback);
}

const DISTRIBUTION_VALUES = Object.freeze((metadata.parameters ?? [])
  .find((param) => param.name === 'distrib')?.options?.map((opt) => Number(opt.value)) ?? []);
const DISTRIBUTION_SET = new Set(DISTRIBUTION_VALUES);

const METRIC_VALUES = Object.freeze((metadata.parameters ?? [])
  .find((param) => param.name === 'dist_metric')?.options?.map((opt) => Number(opt.value)) ?? []);
const METRIC_SET = new Set(METRIC_VALUES);

const PARAM_DEFAULTS = Object.freeze({
  enabled: Boolean(getParamDefault(metadata, 'enabled', 1)),
  distrib: Number(getParamDefault(metadata, 'distrib', DISTRIBUTION_VALUES[0] ?? 1000000)),
  freq: Number(getParamDefault(metadata, 'freq', 10)),
  dist_metric: Number(getParamDefault(metadata, 'dist_metric', METRIC_VALUES[0] ?? 1)),
  speed: Number(getParamDefault(metadata, 'speed', 1)),
});

const FINAL_BINDING_OFFSETS = Object.freeze({
  width: 0,
  height: 1,
  channel_count: 2,
  channelCount: 2,
  distrib: 4,
  freq: 5,
  time: 6,
  speed: 7,
  dist_metric: 8,
});

const VORONOI_BINDING_OFFSETS = Object.freeze({
  width: 0,
  height: 1,
  channel_count: 2,
  channelCount: 2,
  diagram_type: 3,
  nth: 4,
  dist_metric: 5,
  sdf_sides: 6,
  alpha: 7,
  with_refract: 8,
  inverse: 9,
  ridges_hint: 12,
  refract_y_from_offset: 13,
  time: 14,
  speed: 15,
  point_freq: 16,
  point_generations: 17,
  point_distrib: 18,
  point_drift: 19,
  point_corners: 20,
  downsample: 21,
  lowpoly_pack: 22,
  lowpolyPack: 22,
});

const NORMALIZE_BINDING_OFFSETS = Object.freeze({
  width: 0,
  height: 1,
  channel_count: 2,
  channelCount: 2,
  time: 4,
  speed: 5,
});

function floatsDiffer(a, b, epsilon = 1e-6) {
  return Math.abs(Number(a) - Number(b)) > epsilon;
}

function clampNumber(value, min, max) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return min;
  }
  if (Number.isFinite(min) && numeric < min) {
    return min;
  }
  if (Number.isFinite(max) && numeric > max) {
    return max;
  }
  return numeric;
}

class LowpolyEffect {
  static id = metadata.id;
  static label = metadata.label;
  static metadata = metadata;

  #timeSeconds = 0;
  #lastTimestamp = null;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = {
      enabled: PARAM_DEFAULTS.enabled,
      distrib: PARAM_DEFAULTS.distrib,
      freq: PARAM_DEFAULTS.freq,
      dist_metric: PARAM_DEFAULTS.dist_metric,
      speed: PARAM_DEFAULTS.speed,
    };
    this.resources = null;
  }

  getUIState() {
    return { ...this.userState };
  }

  destroy() {
    this.invalidateResources();
  }

  invalidateResources() {
    if (!this.resources) {
      return;
    }

    const resources = this.resources;

    try { resources.finalResourceSet?.destroyAll?.(); } catch {}
    try { resources.voronoiRangeResourceSet?.destroyAll?.(); } catch {}
    try { resources.voronoiColorResourceSet?.destroyAll?.(); } catch {}

    for (const texture of [
      resources.voronoiRangeTexture,
      resources.voronoiColorTexture,
      resources.combinedTexture,
      resources.outputTexture,
    ]) {
      try { texture?.destroy?.(); } catch {}
    }
    try { resources.normalizeOutputTexture?.destroy?.(); } catch {}

    for (const buffer of [
      resources.finalParamsBuffer,
      resources.voronoiRangeParamsBuffer,
      resources.voronoiColorParamsBuffer,
      resources.normalizeParamsBuffer,
      resources.normalizeStatsBuffer,
      resources.outputBuffer,
    ]) {
      try { buffer?.destroy?.(); } catch {}
    }

    this.resources = null;
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('LowpolyEffect.updateParams expects an object.');
    }

    const changed = [];

    if (Object.prototype.hasOwnProperty.call(updates, 'enabled')) {
      const enabled = Boolean(updates.enabled);
      if (this.userState.enabled !== enabled) {
        this.userState.enabled = enabled;
        changed.push('enabled');
        if (this.resources) {
          this.resources.enabled = enabled;
        }
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'distrib')) {
      let distrib = clampNumber(updates.distrib, -Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
      if (!DISTRIBUTION_SET.has(distrib)) {
        distrib = PARAM_DEFAULTS.distrib;
      }
      if (floatsDiffer(this.userState.distrib, distrib)) {
        this.userState.distrib = distrib;
        changed.push('distrib');
        this.#updateDistribution(distrib);
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'freq')) {
      const freq = clampNumber(Math.round(Number(updates.freq)), 1, 64);
      if (floatsDiffer(this.userState.freq, freq)) {
        this.userState.freq = freq;
        changed.push('freq');
        this.#updateFrequency(freq);
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'dist_metric')) {
      let metric = clampNumber(updates.dist_metric, -Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
      if (!METRIC_SET.has(metric)) {
        metric = PARAM_DEFAULTS.dist_metric;
      }
      if (floatsDiffer(this.userState.dist_metric, metric)) {
        this.userState.dist_metric = metric;
        changed.push('dist_metric');
        this.#updateMetric(metric);
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'speed')) {
      const speed = clampNumber(updates.speed, 0, 5);
      if (floatsDiffer(this.userState.speed, speed)) {
        this.userState.speed = speed;
        changed.push('speed');
        this.#updateSpeed(speed);
      }
    }

    return { updated: changed };
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('LowpolyEffect requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('LowpolyEffect requires multires output texture.');
    }

    const enabled = this.userState.enabled !== false;
    if (!enabled) {
      if (!this.resources || this.resources.computePipeline) {
        this.invalidateResources();
        this.resources = {
          enabled: false,
          textureWidth: width,
          textureHeight: height,
          computePasses: [],
        };
      } else {
        this.resources.enabled = false;
        this.resources.textureWidth = width;
        this.resources.textureHeight = height;
      }
      return this.resources;
    }

    const reusable = this.resources
      && this.resources.device === device
      && this.resources.textureWidth === width
      && this.resources.textureHeight === height
      && Array.isArray(this.resources.computePasses)
      && this.resources.computePasses.length > 0;

    if (!reusable) {
      this.invalidateResources();
      this.resources = await this.#createResources({ device, width, height, multiresResources });
    }

    this.resources.enabled = true;
    return this.resources;
  }

  beforeDispatch({ device }) {
    const resources = this.resources;
    if (!resources || !resources.enabled) {
      return;
    }

    const timeValue = this.#advanceTime();

    if (resources.finalParamsState && this.#assignIfChanged(resources.finalParamsState, FINAL_BINDING_OFFSETS.time, timeValue)) {
      resources.finalParamsDirty = true;
    }
    if (resources.voronoiRangeParamsState && this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.time, timeValue)) {
      resources.voronoiRangeParamsDirty = true;
    }
    if (resources.voronoiColorParamsState && this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.time, timeValue)) {
      resources.voronoiColorParamsDirty = true;
    }

    if (resources.finalParamsDirty) {
      device.queue.writeBuffer(resources.finalParamsBuffer, 0, resources.finalParamsState);
      resources.finalParamsDirty = false;
    }
    if (resources.voronoiRangeParamsDirty) {
      device.queue.writeBuffer(resources.voronoiRangeParamsBuffer, 0, resources.voronoiRangeParamsState);
      resources.voronoiRangeParamsDirty = false;
    }
    if (resources.voronoiColorParamsDirty) {
      device.queue.writeBuffer(resources.voronoiColorParamsBuffer, 0, resources.voronoiColorParamsState);
      resources.voronoiColorParamsDirty = false;
    }
    if (resources.normalizeParamsDirty) {
      device.queue.writeBuffer(resources.normalizeParamsBuffer, 0, resources.normalizeParamsState);
      resources.normalizeParamsDirty = false;
    }

    if (resources.normalizeStatsBuffer && resources.normalizeStatsResetData) {
      device.queue.writeBuffer(resources.normalizeStatsBuffer, 0, resources.normalizeStatsResetData);
    }
  }

  afterDispatch() {
    // no-op
  }

  #assignIfChanged(array, offset, value) {
    if (!array || typeof offset !== 'number' || offset < 0 || offset >= array.length) {
      return false;
    }
    const current = array[offset];
    if (!floatsDiffer(current, value)) {
      return false;
    }
    array[offset] = value;
    return true;
  }

  #descriptorFromMeta(effectMeta) {
    return {
      id: effectMeta.id,
      label: effectMeta.label || `${effectMeta.id}.wgsl`,
      stage: 'compute',
      entryPoint: (effectMeta.shader && effectMeta.shader.entryPoint) || 'main',
      url: (effectMeta.shader && effectMeta.shader.url) || `/shaders/effects/${effectMeta.id}/${effectMeta.id}.wgsl`,
      resources: effectMeta.resources || {},
    };
  }

  #populateFinalParams(state, width, height) {
    state[FINAL_BINDING_OFFSETS.width] = width;
    state[FINAL_BINDING_OFFSETS.height] = height;
    state[FINAL_BINDING_OFFSETS.channel_count] = RGBA_CHANNEL_COUNT;
    state[FINAL_BINDING_OFFSETS.distrib] = this.userState.distrib;
    state[FINAL_BINDING_OFFSETS.freq] = this.userState.freq;
    state[FINAL_BINDING_OFFSETS.time] = this.#timeSeconds;
    state[FINAL_BINDING_OFFSETS.speed] = this.userState.speed;
    state[FINAL_BINDING_OFFSETS.dist_metric] = this.userState.dist_metric;
  }

  #populateVoronoiParams(state, width, height, { diagramType, nth, packRangeAlpha = false }) {
    const metric = this.userState.dist_metric;
    const useSdf = metric === 201;
    const sdfSides = useSdf ? 6 : 3;
    state[VORONOI_BINDING_OFFSETS.width] = width;
    state[VORONOI_BINDING_OFFSETS.height] = height;
    state[VORONOI_BINDING_OFFSETS.channel_count] = RGBA_CHANNEL_COUNT;
    state[VORONOI_BINDING_OFFSETS.diagram_type] = diagramType;
    state[VORONOI_BINDING_OFFSETS.nth] = nth;
    state[VORONOI_BINDING_OFFSETS.dist_metric] = metric;
    state[VORONOI_BINDING_OFFSETS.sdf_sides] = sdfSides;
    state[VORONOI_BINDING_OFFSETS.alpha] = 1;
    state[VORONOI_BINDING_OFFSETS.with_refract] = 0;
    state[VORONOI_BINDING_OFFSETS.inverse] = 0;
    state[VORONOI_BINDING_OFFSETS.ridges_hint] = 0;
    state[VORONOI_BINDING_OFFSETS.refract_y_from_offset] = 1;
    state[VORONOI_BINDING_OFFSETS.time] = this.#timeSeconds;
    state[VORONOI_BINDING_OFFSETS.speed] = this.userState.speed;
    state[VORONOI_BINDING_OFFSETS.point_freq] = this.userState.freq;
    state[VORONOI_BINDING_OFFSETS.point_generations] = 1;
    state[VORONOI_BINDING_OFFSETS.point_distrib] = this.userState.distrib;
    state[VORONOI_BINDING_OFFSETS.point_drift] = 1.0;
    state[VORONOI_BINDING_OFFSETS.point_corners] = 0;
    state[VORONOI_BINDING_OFFSETS.downsample] = 1;
    if (typeof VORONOI_BINDING_OFFSETS.lowpoly_pack === 'number') {
      state[VORONOI_BINDING_OFFSETS.lowpoly_pack] = packRangeAlpha ? 1 : 0;
    }
  }

  #populateNormalizeParams(state, width, height) {
    state[NORMALIZE_BINDING_OFFSETS.width] = width;
    state[NORMALIZE_BINDING_OFFSETS.height] = height;
    state[NORMALIZE_BINDING_OFFSETS.channel_count] = RGBA_CHANNEL_COUNT;
    state[NORMALIZE_BINDING_OFFSETS.time] = 0;
    state[NORMALIZE_BINDING_OFFSETS.speed] = 1;
  }

  #buildStatsResetData(width, height) {
    const workgroupSize = 8;
    const numWorkgroupsX = Math.ceil(width / workgroupSize);
    const numWorkgroupsY = Math.ceil(height / workgroupSize);
    const numWorkgroups = Math.max(numWorkgroupsX * numWorkgroupsY, 1);
    const statsInit = new Float32Array(2 + numWorkgroups * 2);
    const F32_MAX = 3.4028235e+38;
    const F32_MIN = -3.4028235e+38;
    statsInit[0] = F32_MAX;
    statsInit[1] = F32_MIN;
    for (let i = 0; i < numWorkgroups; i += 1) {
      statsInit[2 + i * 2] = F32_MAX;
      statsInit[2 + i * 2 + 1] = F32_MIN;
    }
    return statsInit;
  }

  #updateDistribution(distrib) {
    const resources = this.resources;
    if (!resources) {
      return;
    }
    if (resources.finalParamsState && this.#assignIfChanged(resources.finalParamsState, FINAL_BINDING_OFFSETS.distrib, distrib)) {
      resources.finalParamsDirty = true;
    }
    if (resources.voronoiRangeParamsState && this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.point_distrib, distrib)) {
      resources.voronoiRangeParamsDirty = true;
    }
    if (resources.voronoiColorParamsState && this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.point_distrib, distrib)) {
      resources.voronoiColorParamsDirty = true;
    }
  }

  #updateFrequency(freq) {
    const resources = this.resources;
    if (!resources) {
      return;
    }
    if (resources.finalParamsState && this.#assignIfChanged(resources.finalParamsState, FINAL_BINDING_OFFSETS.freq, freq)) {
      resources.finalParamsDirty = true;
    }
    if (resources.voronoiRangeParamsState && this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.point_freq, freq)) {
      resources.voronoiRangeParamsDirty = true;
    }
    if (resources.voronoiColorParamsState && this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.point_freq, freq)) {
      resources.voronoiColorParamsDirty = true;
    }
  }

  #updateMetric(metric) {
    const resources = this.resources;
    if (!resources) {
      return;
    }
    const useSdf = metric === 201;
    const sdfSides = useSdf ? 6 : 3;
    if (resources.finalParamsState && this.#assignIfChanged(resources.finalParamsState, FINAL_BINDING_OFFSETS.dist_metric, metric)) {
      resources.finalParamsDirty = true;
    }
    if (resources.voronoiRangeParamsState) {
      let dirty = false;
      dirty = this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.dist_metric, metric) || dirty;
      dirty = this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.sdf_sides, sdfSides) || dirty;
      if (dirty) {
        resources.voronoiRangeParamsDirty = true;
      }
    }
    if (resources.voronoiColorParamsState) {
      let dirty = false;
      dirty = this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.dist_metric, metric) || dirty;
      dirty = this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.sdf_sides, sdfSides) || dirty;
      if (dirty) {
        resources.voronoiColorParamsDirty = true;
      }
    }
  }

  #updateSpeed(speed) {
    const resources = this.resources;
    if (!resources) {
      return;
    }
    if (resources.finalParamsState && this.#assignIfChanged(resources.finalParamsState, FINAL_BINDING_OFFSETS.speed, speed)) {
      resources.finalParamsDirty = true;
    }
    if (resources.voronoiRangeParamsState && this.#assignIfChanged(resources.voronoiRangeParamsState, VORONOI_BINDING_OFFSETS.speed, speed)) {
      resources.voronoiRangeParamsDirty = true;
    }
    if (resources.voronoiColorParamsState && this.#assignIfChanged(resources.voronoiColorParamsState, VORONOI_BINDING_OFFSETS.speed, speed)) {
      resources.voronoiColorParamsDirty = true;
    }
  }

  #advanceTime() {
    const now = typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? performance.now()
      : Date.now();
    if (this.#lastTimestamp === null) {
      this.#lastTimestamp = now;
      return this.#timeSeconds;
    }
    const deltaSeconds = Math.max((now - this.#lastTimestamp) / 1000, 0);
    this.#lastTimestamp = now;
    this.#timeSeconds = (this.#timeSeconds + deltaSeconds) % 100000;
    return this.#timeSeconds;
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
    } = this.helpers;

    const voronoiDescriptor = this.#descriptorFromMeta(voronoiMetadata);
    const voronoiShaderMetadata = await getShaderMetadataCached(voronoiDescriptor.id);
    const voronoiResourceOptions = {
      inputTextures: { input_texture: multiresResources.outputTexture },
    };
    const voronoiRangeResourceSet = createShaderResourceSet(device, voronoiDescriptor, voronoiShaderMetadata, width, height, voronoiResourceOptions);
    const voronoiColorResourceSet = createShaderResourceSet(device, voronoiDescriptor, voronoiShaderMetadata, width, height, voronoiResourceOptions);

    const voronoiLayout = getOrCreateBindGroupLayout(device, voronoiDescriptor.id, 'compute', voronoiShaderMetadata);
    const voronoiPipelineLayout = getOrCreatePipelineLayout(device, voronoiDescriptor.id, 'compute', voronoiLayout);
    const voronoiPipeline = await getOrCreateComputePipeline(device, voronoiDescriptor.id, voronoiPipelineLayout, voronoiDescriptor.entryPoint ?? 'main');

    const voronoiRangeBindGroup = device.createBindGroup({
      layout: voronoiLayout,
      entries: createBindGroupEntriesFromResources(voronoiShaderMetadata.bindings, voronoiRangeResourceSet),
    });
    const voronoiColorBindGroup = device.createBindGroup({
      layout: voronoiLayout,
      entries: createBindGroupEntriesFromResources(voronoiShaderMetadata.bindings, voronoiColorResourceSet),
    });

    const voronoiParamsLength = Number(voronoiDescriptor.resources?.params?.size ?? 96) / Float32Array.BYTES_PER_ELEMENT;
    const voronoiRangeParamsState = new Float32Array(voronoiParamsLength);
    this.#populateVoronoiParams(voronoiRangeParamsState, width, height, { diagramType: 11, nth: 0, packRangeAlpha: false });
    device.queue.writeBuffer(voronoiRangeResourceSet.buffers.params, 0, voronoiRangeParamsState);

    const voronoiColorParamsState = new Float32Array(voronoiParamsLength);
    this.#populateVoronoiParams(voronoiColorParamsState, width, height, { diagramType: 22, nth: 0, packRangeAlpha: false });
    device.queue.writeBuffer(voronoiColorResourceSet.buffers.params, 0, voronoiColorParamsState);

    const voronoiRangeTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const voronoiColorTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const finalDescriptor = this.#descriptorFromMeta(metadata);
    const finalShaderMetadata = await getShaderMetadataCached(finalDescriptor.id);
    const finalResourceSet = createShaderResourceSet(device, finalDescriptor, finalShaderMetadata, width, height, {
      inputTextures: {
        input_texture: multiresResources.outputTexture,
        voronoi_color_texture: voronoiColorTexture,
        voronoi_range_texture: voronoiRangeTexture,
      },
    });

    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalShaderMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
    const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');
    const finalBindGroup = device.createBindGroup({
      layout: finalLayout,
      entries: createBindGroupEntriesFromResources(finalShaderMetadata.bindings, finalResourceSet),
    });

    const finalParamsState = new Float32Array((Number(finalDescriptor.resources?.params?.size ?? 48)) / Float32Array.BYTES_PER_ELEMENT);
    this.#populateFinalParams(finalParamsState, width, height);
    device.queue.writeBuffer(finalResourceSet.buffers.params, 0, finalParamsState);

    const combinedTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);

    const voronoiRangeBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: voronoiRangeResourceSet.buffers.output_buffer } },
        { binding: 1, resource: voronoiRangeTexture.createView() },
        { binding: 2, resource: { buffer: voronoiRangeResourceSet.buffers.params } },
      ],
    });

    const voronoiColorBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: voronoiColorResourceSet.buffers.output_buffer } },
        { binding: 1, resource: voronoiColorTexture.createView() },
        { binding: 2, resource: { buffer: voronoiColorResourceSet.buffers.params } },
      ],
    });

    const combineBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: finalResourceSet.buffers.output_buffer } },
        { binding: 1, resource: combinedTexture.createView() },
        { binding: 2, resource: { buffer: finalResourceSet.buffers.params } },
      ],
    });

    const normalizeParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const normalizeParamsState = new Float32Array(8);
    this.#populateNormalizeParams(normalizeParamsState, width, height);
    device.queue.writeBuffer(normalizeParamsBuffer, 0, normalizeParamsState);

    const normalizeOutputBuffer = device.createBuffer({
      size: width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const statsResetData = this.#buildStatsResetData(width, height);
    const normalizeStatsBuffer = device.createBuffer({
      size: statsResetData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(normalizeStatsBuffer, 0, statsResetData);

    const normalizeMetaId = 'normalize';
    const normalizeShaderMetadata = await getShaderMetadataCached(normalizeMetaId);
    const normalizeLayout = getOrCreateBindGroupLayout(device, normalizeMetaId, 'compute', normalizeShaderMetadata);
    const normalizePipelineLayout = getOrCreatePipelineLayout(device, normalizeMetaId, 'compute', normalizeLayout);

    const statsDescriptor = getShaderDescriptor('normalize/stats');
    const reduceDescriptor = getShaderDescriptor('normalize/reduce');
    const applyDescriptor = getShaderDescriptor('normalize/apply');

    const normalizeStatsPipeline = await getOrCreateComputePipeline(device, 'normalize/stats', normalizePipelineLayout, statsDescriptor.entryPoint ?? 'main');
    const normalizeReducePipeline = await getOrCreateComputePipeline(device, 'normalize/reduce', normalizePipelineLayout, reduceDescriptor.entryPoint ?? 'main');
    const normalizeApplyPipeline = await getOrCreateComputePipeline(device, 'normalize/apply', normalizePipelineLayout, applyDescriptor.entryPoint ?? 'main');

    const normalizeBindGroup = device.createBindGroup({
      layout: normalizeLayout,
      entries: [
        { binding: 0, resource: combinedTexture.createView() },
        { binding: 1, resource: { buffer: normalizeOutputBuffer } },
        { binding: 2, resource: { buffer: normalizeParamsBuffer } },
        { binding: 3, resource: { buffer: normalizeStatsBuffer } },
      ],
    });

    const normalizeOutputTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const normalizeBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: normalizeOutputBuffer } },
        { binding: 1, resource: normalizeOutputTexture.createView() },
        { binding: 2, resource: { buffer: normalizeParamsBuffer } },
      ],
    });

    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [{ binding: 0, resource: normalizeOutputTexture.createView() }],
    });

    const voronoiWorkgroup = voronoiShaderMetadata.workgroupSize ?? [8, 8, 1];
    const finalWorkgroup = finalShaderMetadata.workgroupSize ?? [8, 8, 1];
    const normalizeWorkgroup = normalizeShaderMetadata.workgroupSize ?? [8, 8, 1];

    const voronoiDispatch = [
      Math.ceil(width / Math.max(voronoiWorkgroup[0] ?? 8, 1)),
      Math.ceil(height / Math.max(voronoiWorkgroup[1] ?? 8, 1)),
      Math.max(Math.ceil(1 / Math.max(voronoiWorkgroup[2] ?? 1, 1)), 1),
    ];

    const finalDispatch = [
      Math.ceil(width / Math.max(finalWorkgroup[0] ?? 8, 1)),
      Math.ceil(height / Math.max(finalWorkgroup[1] ?? 8, 1)),
      Math.max(Math.ceil(1 / Math.max(finalWorkgroup[2] ?? 1, 1)), 1),
    ];

    const normalizeDispatch = [
      Math.ceil(width / Math.max(normalizeWorkgroup[0] ?? 8, 1)),
      Math.ceil(height / Math.max(normalizeWorkgroup[1] ?? 8, 1)),
      Math.max(Math.ceil(1 / Math.max(normalizeWorkgroup[2] ?? 1, 1)), 1),
    ];

    const convertDispatch = [Math.ceil(width / 8), Math.ceil(height / 8), 1];

    const computePasses = [
      { id: 'lowpoly:voronoi-range', pipeline: voronoiPipeline, bindGroup: voronoiRangeBindGroup, workgroupSize: voronoiWorkgroup, dispatch: voronoiDispatch },
      { id: 'lowpoly:voronoi-range-convert', pipeline: bufferToTexturePipeline, bindGroup: voronoiRangeBufferToTextureBindGroup, workgroupSize: BUFFER_TO_TEXTURE_WORKGROUP, dispatch: convertDispatch },
      { id: 'lowpoly:voronoi-color', pipeline: voronoiPipeline, bindGroup: voronoiColorBindGroup, workgroupSize: voronoiWorkgroup, dispatch: voronoiDispatch },
      { id: 'lowpoly:voronoi-color-convert', pipeline: bufferToTexturePipeline, bindGroup: voronoiColorBufferToTextureBindGroup, workgroupSize: BUFFER_TO_TEXTURE_WORKGROUP, dispatch: convertDispatch },
      { id: 'lowpoly:combine', pipeline: finalPipeline, bindGroup: finalBindGroup, workgroupSize: finalWorkgroup, dispatch: finalDispatch },
      { id: 'lowpoly:combine-convert', pipeline: bufferToTexturePipeline, bindGroup: combineBufferToTextureBindGroup, workgroupSize: BUFFER_TO_TEXTURE_WORKGROUP, dispatch: convertDispatch },
      { id: 'lowpoly:norm-stats', pipeline: normalizeStatsPipeline, bindGroup: normalizeBindGroup, workgroupSize: normalizeWorkgroup, dispatch: normalizeDispatch },
      { id: 'lowpoly:norm-reduce', pipeline: normalizeReducePipeline, bindGroup: normalizeBindGroup, workgroupSize: [1, 1, 1], dispatch: [1, 1, 1] },
      { id: 'lowpoly:norm-apply', pipeline: normalizeApplyPipeline, bindGroup: normalizeBindGroup, workgroupSize: normalizeWorkgroup, dispatch: normalizeDispatch },
      { id: 'lowpoly:norm-convert', pipeline: bufferToTexturePipeline, bindGroup: normalizeBufferToTextureBindGroup, workgroupSize: BUFFER_TO_TEXTURE_WORKGROUP, dispatch: convertDispatch },
    ];

    return {
      device,
      enabled: true,
      textureWidth: width,
      textureHeight: height,
      computePipeline: null,
      computeBindGroup: null,
      computePasses,
      paramsBuffer: finalResourceSet.buffers.params,
      paramsState: finalParamsState,
      paramsDirty: false,
      finalParamsBuffer: finalResourceSet.buffers.params,
      finalParamsState,
      finalParamsDirty: false,
      voronoiRangeParamsBuffer: voronoiRangeResourceSet.buffers.params,
      voronoiRangeParamsState,
      voronoiRangeParamsDirty: false,
      voronoiColorParamsBuffer: voronoiColorResourceSet.buffers.params,
      voronoiColorParamsState,
      voronoiColorParamsDirty: false,
      normalizeParamsBuffer,
      normalizeParamsState,
      normalizeParamsDirty: false,
      normalizeStatsBuffer,
      normalizeStatsResetData: statsResetData,
      outputBuffer: normalizeOutputBuffer,
      outputTexture: normalizeOutputTexture,
      normalizeOutputTexture,
      bufferToTexturePipeline,
      bufferToTextureBindGroup: normalizeBufferToTextureBindGroup,
      blitBindGroup,
      workgroupSize: finalWorkgroup,
      bindingOffsets: { ...FINAL_BINDING_OFFSETS },
      voronoiBindingOffsets: { ...VORONOI_BINDING_OFFSETS },
      normalizeBindingOffsets: { ...NORMALIZE_BINDING_OFFSETS },
      finalResourceSet,
      voronoiRangeResourceSet,
      voronoiColorResourceSet,
      voronoiRangeTexture,
      voronoiColorTexture,
      combinedTexture,
      voronoiRangeBufferToTextureBindGroup,
      voronoiColorBufferToTextureBindGroup,
      combineBufferToTextureBindGroup,
      normalizeBufferToTextureBindGroup,
      finalBindGroup,
      voronoiRangeBindGroup,
      voronoiColorBindGroup,
      normalizeBindGroup,
      voronoiPipeline,
      finalPipeline,
      normalizePipelines: {
        stats: normalizeStatsPipeline,
        reduce: normalizeReducePipeline,
        apply: normalizeApplyPipeline,
      },
    };
  }
}

export default LowpolyEffect;
