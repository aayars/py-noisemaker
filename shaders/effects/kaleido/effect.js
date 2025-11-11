import metadata from './meta.json' with { type: 'json' };
import voronoiMeta from '../voronoi/meta.json' with { type: 'json' };

const POINT_DISTRIBUTION_ENUMS = [
  1000000, // Random
  1000001, // Square
  1000002, // Waffle
  1000003, // Chess
  1000010, // Hex (H)
  1000011, // Hex (V)
  1000050, // Spiral
  1000100, // Circular
  1000101, // Concentric
  1000102, // Rotating
];

const DEFAULTS = {
  enabled: true,
  sides: 6,
  sdf_sides: 5,
  blend_edges: true,
  point_freq: 1,
  point_generations: 1,
  point_distrib: 0,
  point_drift: 0,
  point_corners: false,
  speed: 1,
};

const FINAL_BINDING_OFFSETS = { width: 0, height: 1, channel_count: 2, sides: 3, sdf_sides: 4, blend_edges: 5, time: 6, speed: 7 };
const VORONOI_BINDING_OFFSETS = {
  width: 0,
  height: 1,
  channel_count: 2,
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
};

function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp(value, minValue, maxValue) {
  let result = toFiniteNumber(value, minValue);
  if (Number.isFinite(minValue)) {
    result = Math.max(result, minValue);
  }
  if (Number.isFinite(maxValue)) {
    result = Math.min(result, maxValue);
  }
  return result;
}

function clampIndex(value, mapping) {
  const length = mapping.length;
  if (length === 0) {
    return 0;
  }
  const numeric = toFiniteNumber(value, 0);
  const rounded = Math.round(numeric);
  return Math.min(Math.max(rounded, 0), length - 1);
}

function mapIndexToEnum(index, mapping) {
  const clamped = clampIndex(index, mapping);
  return mapping[clamped] ?? mapping[0];
}

function mapEnumToIndex(value, mapping) {
  const numeric = toFiniteNumber(value, 0);
  const rounded = Math.round(numeric);
  const existing = mapping.indexOf(rounded);
  if (existing >= 0) {
    return existing;
  }
  return clampIndex(numeric, mapping);
}

class KaleidoEffect {
  static id = metadata.id;
  static label = metadata.label;
  static metadata = metadata;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = { ...DEFAULTS };
    this.resources = null;
    const initialDistribIndex = clampIndex(this.userState.point_distrib, POINT_DISTRIBUTION_ENUMS);
    this.userState.point_distrib = initialDistribIndex;
    this.enumState = {
      point_distrib: mapIndexToEnum(initialDistribIndex, POINT_DISTRIBUTION_ENUMS),
    };
    this.zeroDispatch = [0, 0, 1];
  }

  getUIState() {
    return { ...this.userState };
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      return { updated: [] };
    }

    const changed = [];
    const resources = this.resources;

    const applyFinalUpdate = (offset, value) => {
      if (!resources?.paramsState || typeof offset !== 'number') {
        return;
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return;
      }
      if (resources.paramsState[offset] !== numeric) {
        resources.paramsState[offset] = numeric;
        resources.paramsDirty = true;
      }
    };

    const applyVoronoiUpdate = (offset, value) => {
      if (!resources?.voronoiParamsState || typeof offset !== 'number') {
        return;
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return;
      }
      if (resources.voronoiParamsState[offset] !== numeric) {
        resources.voronoiParamsState[offset] = numeric;
        resources.voronoiParamsDirty = true;
        resources.radiusDirty = true;
      }
    };

    for (const [key, raw] of Object.entries(updates)) {
      switch (key) {
        case 'enabled': {
          const enabled = Boolean(raw);
          if (this.userState.enabled !== enabled) {
            this.userState.enabled = enabled;
            changed.push('enabled');
            if (resources) {
              resources.enabled = enabled;
              if (enabled) {
                resources.radiusDirty = true;
              }
            }
          }
          break;
        }
        case 'sides': {
          const value = clamp(Math.round(raw), 2, 32);
          if (this.userState.sides !== value) {
            this.userState.sides = value;
            changed.push('sides');
            applyFinalUpdate(FINAL_BINDING_OFFSETS.sides, value);
          }
          break;
        }
        case 'sdf_sides': {
          const value = clamp(Math.round(raw), 0, 12);
          if (this.userState.sdf_sides !== value) {
            this.userState.sdf_sides = value;
            changed.push('sdf_sides');
            const sdfValue = value >= 3 ? value : 3;
            const distMetric = value >= 3 ? 201 : 1;
            applyFinalUpdate(FINAL_BINDING_OFFSETS.sdf_sides, sdfValue);
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.sdf_sides, sdfValue);
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.dist_metric, distMetric);
          }
          break;
        }
        case 'blend_edges': {
          const value = Boolean(raw);
          if (this.userState.blend_edges !== value) {
            this.userState.blend_edges = value;
            changed.push('blend_edges');
            applyFinalUpdate(FINAL_BINDING_OFFSETS.blend_edges, value ? 1 : 0);
          }
          break;
        }
        case 'point_freq': {
          const value = clamp(Math.round(raw), 1, 32);
          if (this.userState.point_freq !== value) {
            this.userState.point_freq = value;
            changed.push('point_freq');
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.point_freq, value);
          }
          break;
        }
        case 'point_generations': {
          const value = clamp(Math.round(raw), 1, 5);
          if (this.userState.point_generations !== value) {
            this.userState.point_generations = value;
            changed.push('point_generations');
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.point_generations, value);
          }
          break;
        }
        case 'point_distrib': {
          const index = mapEnumToIndex(raw, POINT_DISTRIBUTION_ENUMS);
          if (this.userState.point_distrib !== index) {
            this.userState.point_distrib = index;
            const enumValue = mapIndexToEnum(index, POINT_DISTRIBUTION_ENUMS);
            this.enumState.point_distrib = enumValue;
            changed.push('point_distrib');
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.point_distrib, enumValue);
          }
          break;
        }
        case 'point_drift': {
          const value = clamp(Number(raw), 0, 1);
          if (this.userState.point_drift !== value) {
            this.userState.point_drift = value;
            changed.push('point_drift');
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.point_drift, value);
          }
          break;
        }
        case 'point_corners': {
          const value = Boolean(raw);
          if (this.userState.point_corners !== value) {
            this.userState.point_corners = value;
            changed.push('point_corners');
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.point_corners, value ? 1 : 0);
          }
          break;
        }
        case 'speed': {
          const value = clamp(Number(raw), 0, 5);
          if (this.userState.speed !== value) {
            this.userState.speed = value;
            changed.push('speed');
            applyFinalUpdate(FINAL_BINDING_OFFSETS.speed, value);
            applyVoronoiUpdate(VORONOI_BINDING_OFFSETS.speed, value);
          }
          break;
        }
        default:
          break;
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

    try {
      this.resources.resourceSet?.destroyAll?.();
    } catch {}

    try {
      this.resources.voronoiResourceSet?.destroyAll?.();
    } catch {}

    try {
      this.resources.outputTexture?.destroy?.();
    } catch {}

    try {
      this.resources.radiusTexture?.destroy?.();
    } catch {}

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('Kaleido requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('Kaleido requires multires output texture.');
    }

    const enabled = this.userState.enabled !== false;
    if (!enabled) {
      if (!this.resources || this.resources.computePipeline) {
        this.invalidateResources();
        this.resources = { enabled: false, textureWidth: width, textureHeight: height, computePasses: [] };
      } else {
        this.resources.enabled = false;
        this.resources.textureWidth = width;
        this.resources.textureHeight = height;
      }
      return this.resources;
    }

    const reuse = this.resources
      && this.resources.device === device
      && this.resources.textureWidth === width
      && this.resources.textureHeight === height
      && Array.isArray(this.resources.computePasses)
      && this.resources.computePasses.length > 0;

    if (!reuse) {
      this.invalidateResources();
      this.resources = await this.#createResources({ device, width, height, multiresResources });
    }

    this.resources.enabled = true;
    return this.resources;
  }

  beforeDispatch() {
    const resources = this.resources;
    if (!resources || !resources.enabled) {
      return;
    }

    const finalState = resources.paramsState;
    const voronoiState = resources.voronoiParamsState;
    const device = resources.device;
    if (!finalState || !voronoiState || !device) {
      return;
    }

    let dirty = false;

    const width = finalState[FINAL_BINDING_OFFSETS.width] ?? resources.textureWidth;
    const height = finalState[FINAL_BINDING_OFFSETS.height] ?? resources.textureHeight;
    if (voronoiState[VORONOI_BINDING_OFFSETS.width] !== width) {
      voronoiState[VORONOI_BINDING_OFFSETS.width] = width;
      dirty = true;
      resources.radiusDirty = true;
    }
    if (voronoiState[VORONOI_BINDING_OFFSETS.height] !== height) {
      voronoiState[VORONOI_BINDING_OFFSETS.height] = height;
      dirty = true;
      resources.radiusDirty = true;
    }

    const timeValue = finalState[FINAL_BINDING_OFFSETS.time] ?? 0;
    if (voronoiState[VORONOI_BINDING_OFFSETS.time] !== timeValue) {
      voronoiState[VORONOI_BINDING_OFFSETS.time] = timeValue;
      dirty = true;
      resources.radiusDirty = true;
    }

    const speedValue = finalState[FINAL_BINDING_OFFSETS.speed] ?? this.userState.speed;
    if (voronoiState[VORONOI_BINDING_OFFSETS.speed] !== speedValue) {
      voronoiState[VORONOI_BINDING_OFFSETS.speed] = speedValue;
      dirty = true;
      resources.radiusDirty = true;
    }

    resources.animateVoronoi = speedValue > 0;

    if (dirty || resources.voronoiParamsDirty) {
      device.queue.writeBuffer(resources.voronoiParamsBuffer, 0, voronoiState);
      resources.voronoiParamsDirty = false;
    }

    if (resources.paramsDirty) {
      device.queue.writeBuffer(resources.paramsBuffer, 0, finalState);
      resources.paramsDirty = false;
    }

    const timeChanged = resources.lastVoronoiTime !== timeValue;
    if (timeChanged) {
      resources.lastVoronoiTime = timeValue;
      if (resources.animateVoronoi) {
        resources.radiusDirty = true;
      }
    }

    const shouldRunVoronoi = Boolean(resources.radiusDirty);
    const computePasses = resources.computePasses;
    if (!Array.isArray(computePasses) || computePasses.length === 0) {
      return;
    }

    const voronoiPass = computePasses[resources.voronoiPassIndex ?? 0];
    const radiusConvertPass = computePasses[resources.radiusConvertPassIndex ?? 1];
    const zeroDispatch = resources.zeroDispatch ?? this.zeroDispatch;

    if (voronoiPass && radiusConvertPass) {
      if (shouldRunVoronoi) {
        voronoiPass.dispatch = voronoiPass.originalDispatch ?? resources.voronoiDispatchBase;
        radiusConvertPass.dispatch = radiusConvertPass.originalDispatch ?? resources.radiusConvertDispatchBase;
        resources.willRunVoronoi = true;
      } else {
        voronoiPass.dispatch = zeroDispatch;
        radiusConvertPass.dispatch = zeroDispatch;
        resources.willRunVoronoi = false;
      }
    }
  }

  afterDispatch() {
    const resources = this.resources;
    if (!resources || !resources.enabled) {
      return;
    }
    if (resources.willRunVoronoi) {
      resources.radiusDirty = false;
      resources.willRunVoronoi = false;
    }
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
    state[FINAL_BINDING_OFFSETS.channel_count] = 4;
    state[FINAL_BINDING_OFFSETS.sides] = this.userState.sides;
    state[FINAL_BINDING_OFFSETS.sdf_sides] = this.userState.sdf_sides >= 3 ? this.userState.sdf_sides : 3;
    state[FINAL_BINDING_OFFSETS.blend_edges] = this.userState.blend_edges ? 1 : 0;
    state[FINAL_BINDING_OFFSETS.time] = 0;
    state[FINAL_BINDING_OFFSETS.speed] = this.userState.speed;
  }

  #populateVoronoiParams(state, width, height) {
    const useSdf = this.userState.sdf_sides >= 3;
    const sdfValue = useSdf ? this.userState.sdf_sides : 3;
    const distribEnum = mapIndexToEnum(this.userState.point_distrib, POINT_DISTRIBUTION_ENUMS);
    this.enumState.point_distrib = distribEnum;
    state[VORONOI_BINDING_OFFSETS.width] = width;
    state[VORONOI_BINDING_OFFSETS.height] = height;
    state[VORONOI_BINDING_OFFSETS.channel_count] = 4;
    state[VORONOI_BINDING_OFFSETS.diagram_type] = 11; // range diagram
    state[VORONOI_BINDING_OFFSETS.nth] = 0;
    state[VORONOI_BINDING_OFFSETS.dist_metric] = useSdf ? 201 : 1;
    state[VORONOI_BINDING_OFFSETS.sdf_sides] = sdfValue;
    state[VORONOI_BINDING_OFFSETS.alpha] = 1;
    state[VORONOI_BINDING_OFFSETS.with_refract] = 0;
    state[VORONOI_BINDING_OFFSETS.inverse] = 0;
    state[VORONOI_BINDING_OFFSETS.ridges_hint] = 0;
    state[VORONOI_BINDING_OFFSETS.refract_y_from_offset] = 1;
    state[VORONOI_BINDING_OFFSETS.time] = 0;
    state[VORONOI_BINDING_OFFSETS.speed] = this.userState.speed;
    state[VORONOI_BINDING_OFFSETS.point_freq] = this.userState.point_freq;
    state[VORONOI_BINDING_OFFSETS.point_generations] = this.userState.point_generations;
    state[VORONOI_BINDING_OFFSETS.point_distrib] = distribEnum;
    state[VORONOI_BINDING_OFFSETS.point_drift] = this.userState.point_drift;
    state[VORONOI_BINDING_OFFSETS.point_corners] = this.userState.point_corners ? 1 : 0;
    state[VORONOI_BINDING_OFFSETS.downsample] = 1;
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

    const voronoiDescriptor = this.#descriptorFromMeta(voronoiMeta);
    const voronoiMetadata = await getShaderMetadataCached(voronoiDescriptor.id);
    const voronoiResourceSet = createShaderResourceSet(device, voronoiDescriptor, voronoiMetadata, width, height, {
      inputTextures: { input_texture: multiresResources.outputTexture },
    });

    const voronoiLayout = getOrCreateBindGroupLayout(device, voronoiDescriptor.id, 'compute', voronoiMetadata);
    const voronoiPipelineLayout = getOrCreatePipelineLayout(device, voronoiDescriptor.id, 'compute', voronoiLayout);
    const voronoiPipeline = await getOrCreateComputePipeline(device, voronoiDescriptor.id, voronoiPipelineLayout, voronoiDescriptor.entryPoint ?? 'main');
    const voronoiBindGroup = device.createBindGroup({
      layout: voronoiLayout,
      entries: createBindGroupEntriesFromResources(voronoiMetadata.bindings, voronoiResourceSet),
    });

    const voronoiParamBytes = Number(voronoiDescriptor.resources?.params?.size ?? 96);
    const voronoiParamsState = new Float32Array(voronoiParamBytes / Float32Array.BYTES_PER_ELEMENT);
    this.#populateVoronoiParams(voronoiParamsState, width, height);
    device.queue.writeBuffer(voronoiResourceSet.buffers.params, 0, voronoiParamsState);

    const radiusTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const finalDescriptor = this.#descriptorFromMeta(metadata);
    const finalMetadata = await getShaderMetadataCached(finalDescriptor.id);
    const finalResourceSet = createShaderResourceSet(device, finalDescriptor, finalMetadata, width, height, {
      inputTextures: {
        input_texture: multiresResources.outputTexture,
        radius_texture: radiusTexture,
      },
    });

    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
    const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');
    const finalBindGroup = device.createBindGroup({
      layout: finalLayout,
      entries: createBindGroupEntriesFromResources(finalMetadata.bindings, finalResourceSet),
    });

    const finalParamsState = new Float32Array(8);
    this.#populateFinalParams(finalParamsState, width, height);
    device.queue.writeBuffer(finalResourceSet.buffers.params, 0, finalParamsState);

    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);

    const voronoiBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: voronoiResourceSet.buffers.output_buffer } },
        { binding: 1, resource: radiusTexture.createView() },
        { binding: 2, resource: { buffer: voronoiResourceSet.buffers.params } },
      ],
    });

    const outputTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const finalBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: finalResourceSet.buffers.output_buffer } },
        { binding: 1, resource: outputTexture.createView() },
        { binding: 2, resource: { buffer: finalResourceSet.buffers.params } },
      ],
    });

    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [{ binding: 0, resource: outputTexture.createView() }],
    });

    const voronoiWorkgroup = voronoiMetadata.workgroupSize ?? [8, 8, 1];
    const voronoiDispatchBase = [
      Math.ceil(width / Math.max(voronoiWorkgroup[0] ?? 8, 1)),
      Math.ceil(height / Math.max(voronoiWorkgroup[1] ?? 8, 1)),
      Math.max(Math.ceil(1 / Math.max(voronoiWorkgroup[2] ?? 1, 1)), 1),
    ];

    const finalWorkgroup = finalMetadata.workgroupSize ?? [8, 8, 1];
    const finalDispatchBase = [
      Math.ceil(width / Math.max(finalWorkgroup[0] ?? 8, 1)),
      Math.ceil(height / Math.max(finalWorkgroup[1] ?? 8, 1)),
      Math.max(Math.ceil(1 / Math.max(finalWorkgroup[2] ?? 1, 1)), 1),
    ];

    const radiusConvertDispatchBase = [Math.ceil(width / 8), Math.ceil(height / 8), 1];
    const finalConvertDispatchBase = radiusConvertDispatchBase.slice();

    const voronoiPass = {
      id: 'kaleido:voronoi',
      pipeline: voronoiPipeline,
      bindGroup: voronoiBindGroup,
      workgroupSize: voronoiWorkgroup,
      dispatch: voronoiDispatchBase,
      originalDispatch: voronoiDispatchBase,
    };

    const radiusConvertPass = {
      id: 'kaleido:radius-convert',
      pipeline: bufferToTexturePipeline,
      bindGroup: voronoiBufferToTextureBindGroup,
      workgroupSize: [8, 8, 1],
      dispatch: radiusConvertDispatchBase,
      originalDispatch: radiusConvertDispatchBase,
    };

    const finalPass = {
      id: 'kaleido:final',
      pipeline: finalPipeline,
      bindGroup: finalBindGroup,
      workgroupSize: finalWorkgroup,
      dispatch: finalDispatchBase,
      originalDispatch: finalDispatchBase,
    };

    const finalConvertPass = {
      id: 'kaleido:final-convert',
      pipeline: bufferToTexturePipeline,
      bindGroup: finalBufferToTextureBindGroup,
      workgroupSize: [8, 8, 1],
      dispatch: finalConvertDispatchBase,
      originalDispatch: finalConvertDispatchBase,
    };

    const computePasses = [voronoiPass, radiusConvertPass, finalPass, finalConvertPass];

    return {
      device,
      enabled: true,
      textureWidth: width,
      textureHeight: height,
      computePipeline: finalPipeline,
      computeBindGroup: finalBindGroup,
      computePasses,
      paramsBuffer: finalResourceSet.buffers.params,
      paramsState: finalParamsState,
      paramsDirty: false,
      outputBuffer: finalResourceSet.buffers.output_buffer,
      outputTexture,
      bufferToTexturePipeline,
      bufferToTextureBindGroup: finalBufferToTextureBindGroup,
      workgroupSize: finalWorkgroup,
      blitBindGroup,
      bindingOffsets: { ...FINAL_BINDING_OFFSETS },
      voronoiParamsBuffer: voronoiResourceSet.buffers.params,
      voronoiParamsState,
      voronoiParamsDirty: false,
      voronoiBindingOffsets: { ...VORONOI_BINDING_OFFSETS },
      radiusTexture,
      resourceSet: finalResourceSet,
      voronoiResourceSet,
      voronoiDispatchBase,
      radiusConvertDispatchBase,
      finalDispatchBase,
      finalConvertDispatchBase,
      voronoiPassIndex: 0,
      radiusConvertPassIndex: 1,
      finalPassIndex: 2,
      finalConvertPassIndex: 3,
      zeroDispatch: this.zeroDispatch,
      radiusDirty: true,
      willRunVoronoi: false,
      lastVoronoiTime: null,
      animateVoronoi: this.userState.speed > 0,
    };
  }
}

export default KaleidoEffect;
