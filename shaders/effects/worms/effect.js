import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const FLOATS_PER_AGENT = 9; // [x, y, rot, stride, r, g, b, seed, age]
const PARAM_FLOAT_COUNT = 20; // 5 vec4 slots as defined in uniforms.json
const MAX_AGENT_COUNT = 131072;
const DENSITY_SCALE = 0.2;
const STRIDE_SCALE = 0.1;
const TAU = Math.PI * 2;
const RIGHT_ANGLE = Math.PI * 0.5;

const PARAM_INDEX = Object.freeze({
  width: 0,
  height: 1,
  channelCount: 2,
  behavior: 4,
  density: 5,
  stride: 6,
  strideDeviation: 8,
  alpha: 9,
  kink: 10,
  quantize: 12,
  time: 13,
  intensity: 15,
  inputIntensity: 16,
  lifetime: 17,
});

const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

function toNumber(value, fallback) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : fallback;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toBoolean(value, fallback) {
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  if (typeof value === 'string') {
    return value.toLowerCase() === 'true';
  }
  return fallback;
}

function clamp(value, min, max) {
  let result = value;
  if (Number.isFinite(min)) {
    result = Math.max(min, result);
  }
  if (Number.isFinite(max)) {
    result = Math.min(max, result);
  }
  return result;
}

function clampInt(value, min, max) {
  return clamp(Math.round(value), min, max);
}

function wrapAngle(value) {
  const mod = value % TAU;
  return mod < 0 ? mod + TAU : mod;
}

function computeAgentCount(width, height, density) {
  const maxDim = Math.max(width, height);
  if (maxDim <= 0 || density <= 0) {
    return 0;
  }
  const desired = Math.floor(maxDim * density);
  return Math.max(1, Math.min(desired, MAX_AGENT_COUNT));
}

function normalizedStride(stride, width, height) {
  const scale = Math.max(width, height) / 1024;
  const base = stride * (scale <= 0 ? 1 : scale);
  return Math.max(0.1, base);
}

function createRotationFactory(behavior, count) {
  const mode = Number.isFinite(behavior) ? Math.round(behavior) : 1;
  const total = Math.max(1, count);
  const baseHeading = Math.random() * TAU;
  const quarterSize = Math.max(1, Math.floor(total / 4));

  return (index) => {
    switch (mode) {
      case 0:
        return 0;
      case 1:
        return baseHeading;
      case 2:
        return wrapAngle(baseHeading + Math.floor(Math.random() * 4) * RIGHT_ANGLE);
      case 3:
        return baseHeading + (Math.random() - 0.5) * 0.25;
      case 4:
        return Math.random() * TAU;
      case 5: {
        const band = Math.floor(index / quarterSize);
        if (band <= 0) {
          return baseHeading;
        }
        if (band === 1) {
          return wrapAngle(baseHeading + Math.floor(Math.random() * 4) * RIGHT_ANGLE);
        }
        if (band === 2) {
          return wrapAngle(baseHeading + (Math.random() - 0.5) * 0.25);
        }
        return Math.random() * TAU;
      }
      case 10:
        return Math.random();
      default:
        return Math.random() * TAU;
    }
  };
}

function seedAgents(width, height, density, stride, behavior) {
  const count = computeAgentCount(width, height, density);
  if (count <= 0) {
    return { data: new Float32Array(0), count: 0 };
  }

  const data = new Float32Array(count * FLOATS_PER_AGENT);
  const strideDeviation = 0.05;
  const normalized = normalizedStride(stride, width, height);
  const rotationForIndex = createRotationFactory(behavior, count);

  for (let i = 0; i < count; i += 1) {
    const base = i * FLOATS_PER_AGENT;
    data[base + 0] = Math.random() * width;
    data[base + 1] = Math.random() * height;
    data[base + 2] = rotationForIndex(i);
    const strideVariation = normalized * (1 + (Math.random() - 0.5) * 2 * strideDeviation);
    data[base + 3] = Math.max(0.1, strideVariation);
    data[base + 4] = 0.5 + Math.random() * 0.5;
    data[base + 5] = 0.5 + Math.random() * 0.5;
    data[base + 6] = 0.5 + Math.random() * 0.5;
    data[base + 7] = Math.random() * 1000000;
    data[base + 8] = -1.0;
  }

  return { data, count };
}

const PARAMETER_DEFAULTS = Object.freeze({
  behavior: getDefaultParamValue('behavior', 1),
  density: getDefaultParamValue('density', 20),
  stride: getDefaultParamValue('stride', 10),
  kink: getDefaultParamValue('kink', 1.0),
  strideDeviation: getDefaultParamValue('strideDeviation', 0.05),
  quantize: getDefaultParamValue('quantize', false),
  intensity: getDefaultParamValue('intensity', 90),
  inputIntensity: getDefaultParamValue('inputIntensity', 100),
  lifetime: getDefaultParamValue('lifetime', 30),
  enabled: getDefaultParamValue('enabled', true),
});

class WormsEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  #timeSeconds = 0;
  #lastTimestamp = null;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.device = null;
    this.width = 0;
    this.height = 0;
    this.resources = null;
    this.userState = {
      behavior: PARAMETER_DEFAULTS.behavior,
      density: PARAMETER_DEFAULTS.density,
      stride: PARAMETER_DEFAULTS.stride,
      kink: PARAMETER_DEFAULTS.kink,
      strideDeviation: PARAMETER_DEFAULTS.strideDeviation,
      quantize: Boolean(PARAMETER_DEFAULTS.quantize),
      intensity: PARAMETER_DEFAULTS.intensity,
      inputIntensity: PARAMETER_DEFAULTS.inputIntensity,
      lifetime: PARAMETER_DEFAULTS.lifetime,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null;
    this.lastDensity = null;
    this.lastStride = null;
    this.lastStrideDeviation = null;
    this.lastBehavior = null;
    this.feedbackTexture = null;
  }

  #densitySlider() {
    return clampInt(toNumber(this.userState.density, PARAMETER_DEFAULTS.density), 1, 100);
  }

  #getActualDensity() {
    return this.#densitySlider() * DENSITY_SCALE;
  }

  #strideSlider() {
    return clampInt(toNumber(this.userState.stride, PARAMETER_DEFAULTS.stride), 1, 100);
  }

  #getActualStride() {
    return Math.max(0.1, this.#strideSlider() * STRIDE_SCALE);
  }

  #getStrideDeviation() {
    return toNumber(this.userState.strideDeviation, PARAMETER_DEFAULTS.strideDeviation);
  }

  #getIntensityScalar() {
    const percent = clamp(toNumber(this.userState.intensity, PARAMETER_DEFAULTS.intensity), 0, 100);
    return percent * 0.01;
  }

  #getInputIntensityScalar() {
    const percent = clamp(toNumber(this.userState.inputIntensity, PARAMETER_DEFAULTS.inputIntensity), 0, 100);
    return percent * 0.01;
  }

  #getLifetimeSeconds() {
    const seconds = clamp(toNumber(this.userState.lifetime, PARAMETER_DEFAULTS.lifetime), 0, 60);
    return seconds;
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

  #populateParamsState(state) {
    if (!state) {
      return;
    }

    const width = Math.max(1, Math.floor(this.width));
    const height = Math.max(1, Math.floor(this.height));
    const actualDensity = this.#getActualDensity();
    const actualStride = this.#getActualStride();
    const strideDeviation = this.#getStrideDeviation();
    const intensity = this.#getIntensityScalar();
    const inputIntensity = this.#getInputIntensityScalar();
    const lifetime = this.#getLifetimeSeconds();
    const behavior = clampInt(toNumber(this.userState.behavior, PARAMETER_DEFAULTS.behavior), 0, 10);
    const quantizeFlag = this.userState.quantize ? 1 : 0;

    state[PARAM_INDEX.width] = width;
    state[PARAM_INDEX.height] = height;
    state[PARAM_INDEX.channelCount] = RGBA_CHANNEL_COUNT;
    state[3] = 0;

    state[PARAM_INDEX.behavior] = behavior;
    state[PARAM_INDEX.density] = actualDensity;
    state[PARAM_INDEX.stride] = actualStride;
    state[7] = 0;

    state[PARAM_INDEX.strideDeviation] = strideDeviation;
    state[PARAM_INDEX.alpha] = 1.0;
    state[PARAM_INDEX.kink] = Math.max(0, toNumber(this.userState.kink, PARAMETER_DEFAULTS.kink));
    state[11] = 0;

    state[PARAM_INDEX.quantize] = quantizeFlag;
    state[PARAM_INDEX.time] = this.#timeSeconds;
    state[14] = 0;
    state[PARAM_INDEX.intensity] = intensity;

    state[PARAM_INDEX.inputIntensity] = inputIntensity;
    state[PARAM_INDEX.lifetime] = lifetime;
    state[18] = 0;
    state[19] = 0;
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
        logWarn?.('Failed to destroy worms resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy worms output texture during invalidation:', error);
      }
    }

    if (resources.feedbackTexture?.destroy) {
      try {
        resources.feedbackTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy worms feedback texture during invalidation:', error);
      }
    }

    this.#destroyAgentBuffers();
    this.resources = null;
    this.lastDensity = null;
    this.lastStride = null;
    this.lastStrideDeviation = null;
    this.lastBehavior = null;
    this.feedbackTexture = null;
  }

  #destroyAgentBuffers() {
    const { logWarn } = this.helpers;
    if (this.agentBuffers) {
      [this.agentBuffers.a, this.agentBuffers.b].forEach((buffer) => {
        if (buffer?.destroy) {
          try {
            buffer.destroy();
          } catch (error) {
            logWarn?.('Failed to destroy worms agent buffer:', error);
          }
        }
      });
      this.agentBuffers = null;
    }
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('WormsEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('WormsEffect.ensureResources requires multires output texture.');
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
      throw new TypeError('WormsEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    const numericParams = [
      'behavior',
      'density',
      'stride',
      'kink',
      'strideDeviation',
      'intensity',
      'inputIntensity',
      'lifetime',
    ];

    let requiresReset = false;

    numericParams.forEach((name) => {
      if (!hasOwn(updates, name)) {
        return;
      }
      const numeric = Number(updates[name]);
      if (!Number.isFinite(numeric)) {
        logWarn?.(`updateWormsParams: ${name} must be a finite number.`);
        return;
      }
      const oldValue = this.userState[name];
      this.userState[name] = numeric;
      updated.push(name);

      const densityChanged = name === 'density' && oldValue !== numeric;
      const strideChanged = name === 'stride' && oldValue !== numeric;
      const strideDeviationChanged = name === 'strideDeviation' && oldValue !== numeric;
      const behaviorChanged = name === 'behavior' && oldValue !== numeric;

      if (densityChanged || strideChanged || strideDeviationChanged || behaviorChanged) {
        requiresReset = true;
      }
    });

    const booleanParams = ['enabled', 'quantize'];
    booleanParams.forEach((name) => {
      if (!hasOwn(updates, name)) {
        return;
      }
      const value = Boolean(updates[name]);
      this.userState[name] = value;
      if (name === 'enabled' && this.resources) {
        this.resources.enabled = value;
      }
      if (name === 'quantize' && this.resources?.paramsState) {
        this.#populateParamsState(this.resources.paramsState);
        this.resources.paramsDirty = true;
      }
      updated.push(name);
    });

    if (hasOwn(updates, 'resetState') && updates.resetState) {
      requiresReset = true;
    }

    if (requiresReset) {
      this.#resetSimulation();
    } else if (this.resources?.paramsState) {
      this.#populateParamsState(this.resources.paramsState);
      this.resources.paramsDirty = true;
    }

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
    this.#timeSeconds = 0;
    this.#lastTimestamp = null;
  }

  #initializeAgentBuffers(device, width, height, behavior) {
    const density = this.#getActualDensity();
    const stride = this.#getActualStride();

    const agents = seedAgents(width, height, density, stride, behavior);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    let bufferA = null;
    let bufferB = null;

    if (agents.count > 0 && agents.data.byteLength > 0) {
      bufferA = device.createBuffer({ size: agents.data.byteLength, usage });
      bufferB = device.createBuffer({ size: agents.data.byteLength, usage });
      device.queue.writeBuffer(bufferA, 0, agents.data);
      device.queue.writeBuffer(bufferB, 0, agents.data);
    }

    return { a: bufferA, b: bufferB, current: 'a', agentCount: agents.count };
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
    } = this.helpers;

    setStatus?.('Creating worms resources…');

    try {
      const descriptor = getShaderDescriptor('worms');
      const shaderMetadata = await getShaderMetadataCached('worms');

      const initDescriptor = getShaderDescriptor('worms/init_from_prev');
      const moveDescriptor = getShaderDescriptor('worms/agent_move');
      const finalDescriptor = getShaderDescriptor('worms/final_blend');
  const bufferToTextureDescriptor = getShaderDescriptor('worms/buffer_to_texture');
  const composeDescriptor = getShaderDescriptor('worms/compose_to_texture');

      const actualDensity = this.#getActualDensity();
      const actualStride = this.#getActualStride();
      const strideDeviation = this.#getStrideDeviation();
      const behavior = clampInt(toNumber(this.userState.behavior, PARAMETER_DEFAULTS.behavior), 0, 10);

      const densityChanged = this.lastDensity !== actualDensity;
      const strideChanged = this.lastStride !== actualStride;
      const strideDeviationChanged = this.lastStrideDeviation !== strideDeviation;
      const behaviorChanged = this.lastBehavior !== behavior;

      if (!this.agentBuffers || densityChanged || strideChanged || strideDeviationChanged || behaviorChanged) {
        if (this.agentBuffers) {
          this.#destroyAgentBuffers();
        }
        this.agentBuffers = this.#initializeAgentBuffers(device, width, height, behavior);
      }
      this.lastDensity = actualDensity;
      this.lastStride = actualStride;
      this.lastStrideDeviation = strideDeviation;
      this.lastBehavior = behavior;

      const paramsSize = PARAM_FLOAT_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const outputBuffer = device.createBuffer({
        size: Math.max(1, outputBufferSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      const textureSize = { width, height, depthOrArrayLayers: 1 };
      const outputTexture = device.createTexture({
        size: textureSize,
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      });
      const outputTextureView = outputTexture.createView();

      const feedbackTexture = device.createTexture({
        size: textureSize,
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING
          | GPUTextureUsage.STORAGE_BINDING
          | GPUTextureUsage.COPY_SRC
          | GPUTextureUsage.COPY_DST,
      });
      const feedbackView = feedbackTexture.createView();
      this.feedbackTexture = feedbackTexture;

      const paramsState = new Float32Array(PARAM_FLOAT_COUNT);
      this.#populateParamsState(paramsState);
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);

      let initPipeline;
      let movePipeline;
      let finalPipeline;
      let useMultiPass = true;

      try {
        initPipeline = await getOrCreateComputePipeline(device, initDescriptor.id, computePipelineLayout, initDescriptor.entryPoint ?? 'main');
        movePipeline = await getOrCreateComputePipeline(device, moveDescriptor.id, computePipelineLayout, moveDescriptor.entryPoint ?? 'main');
        finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, computePipelineLayout, finalDescriptor.entryPoint ?? 'main');
        logInfo?.('Worms: Using multi-pass compute pipeline.');
      } catch (error) {
        logWarn?.('Worms: Multi-pass pipeline creation failed, falling back to single-pass implementation:', error);
        useMultiPass = false;
      }

      const computePipeline = useMultiPass
        ? movePipeline
        : await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

      const frontBuffer = this.agentBuffers?.a;
      const backBuffer = this.agentBuffers?.b;
      if (!frontBuffer || !backBuffer) {
        throw new Error('WormsEffect: agent buffers not initialized.');
      }

      const inputView = multiresResources.outputTexture.createView();
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: inputView },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackView },
          { binding: 4, resource: { buffer: frontBuffer } },
          { binding: 5, resource: { buffer: backBuffer } },
        ],
      });

      const bufferToTextureMetadata = await getShaderMetadataCached(bufferToTextureDescriptor.id);
      const bufferToTextureLayout = getOrCreateBindGroupLayout(device, bufferToTextureDescriptor.id, 'compute', bufferToTextureMetadata);
      const bufferToTexturePipelineLayout = getOrCreatePipelineLayout(device, bufferToTextureDescriptor.id, 'compute', bufferToTextureLayout);
      const bufferToTexturePipeline = await getOrCreateComputePipeline(
        device,
        bufferToTextureDescriptor.id,
        bufferToTexturePipelineLayout,
        bufferToTextureDescriptor.entryPoint ?? 'main',
      );

      const composeMetadata = await getShaderMetadataCached(composeDescriptor.id);
      const composeLayout = getOrCreateBindGroupLayout(device, composeDescriptor.id, 'compute', composeMetadata);
      const composePipelineLayout = getOrCreatePipelineLayout(device, composeDescriptor.id, 'compute', composeLayout);
      const composePipeline = await getOrCreateComputePipeline(
        device,
        composeDescriptor.id,
        composePipelineLayout,
        composeDescriptor.entryPoint ?? 'main',
      );

      const trailBufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: feedbackView },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const blitBindGroup = device.createBindGroup({
        layout: multiresResources.blitBindGroupLayout,
        entries: [{ binding: 0, resource: outputTextureView }],
      });

  const pixelWorkgroupSize = [8, 8, 1];
  const agentWorkgroupSize = [64, 1, 1];
  const bufferToTextureWorkgroupSize = [8, 8, 1];

      let computePasses = null;
      if (useMultiPass) {
        computePasses = [
          {
            pipeline: initPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: pixelWorkgroupSize,
            getDispatch: () => [
              Math.ceil(width / pixelWorkgroupSize[0]),
              Math.ceil(height / pixelWorkgroupSize[1]),
              1,
            ],
          },
          {
            pipeline: movePipeline,
            bindGroup: computeBindGroup,
            workgroupSize: agentWorkgroupSize,
            getDispatch: () => [
              Math.max(1, Math.ceil(Math.max(this.agentBuffers?.agentCount ?? 0, 1) / agentWorkgroupSize[0])),
              1,
              1,
            ],
          },
          {
            pipeline: finalPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: pixelWorkgroupSize,
            getDispatch: () => [
              Math.ceil(width / pixelWorkgroupSize[0]),
              Math.ceil(height / pixelWorkgroupSize[1]),
              1,
            ],
          },
          {
            pipeline: bufferToTexturePipeline,
            bindGroup: trailBufferToTextureBindGroup,
            workgroupSize: bufferToTextureWorkgroupSize,
            getDispatch: () => [
              Math.ceil(width / bufferToTextureWorkgroupSize[0]),
              Math.ceil(height / bufferToTextureWorkgroupSize[1]),
              1,
            ],
          },
        ];
      }

      const shouldCopyOutputToPrev = useMultiPass ? false : true;

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
        outputTextureView,
        feedbackTexture,
        feedbackTextureView: feedbackView,
  bufferToTexturePipeline: composePipeline,
  bufferToTextureBindGroup: null,
  bufferToTextureWorkgroupSize,
  trailBufferToTextureBindGroup,
  composeBindGroupLayout: composeLayout,
        blitBindGroup,
        workgroupSize: pixelWorkgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
  device,
  computeBindGroupLayout,
  shouldCopyOutputToPrev,
      };

      setStatus?.('Worms resources ready.');
      return this.resources;
    } catch (error) {
      this.helpers.logWarn?.('Failed to create worms resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources }) {
    if (!this.agentBuffers || !this.resources || !this.resources.enabled) {
      return;
    }

    const resources = this.resources;
    const paramsState = resources.paramsState;
    const timeValue = this.#advanceTime();
    if (paramsState && paramsState[PARAM_INDEX.time] !== timeValue) {
      paramsState[PARAM_INDEX.time] = timeValue;
      resources.paramsDirty = true;
    }

    if (resources.paramsDirty) {
      device.queue.writeBuffer(resources.paramsBuffer, 0, paramsState);
      resources.paramsDirty = false;
    }

    const currentIsA = this.agentBuffers.current === 'a';
    const inputBuffer = currentIsA ? this.agentBuffers.a : this.agentBuffers.b;
    const outputBuffer = currentIsA ? this.agentBuffers.b : this.agentBuffers.a;

    const inputView = multiresResources.outputTexture.createView();
    const feedbackView = resources.feedbackTextureView ?? resources.feedbackTexture.createView();
    resources.feedbackTextureView = feedbackView;

    const previousBindGroup = resources.computeBindGroup;
    const newBindGroup = device.createBindGroup({
      layout: resources.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: inputView },
        { binding: 1, resource: { buffer: resources.outputBuffer } },
        { binding: 2, resource: { buffer: resources.paramsBuffer } },
        { binding: 3, resource: feedbackView },
        { binding: 4, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
      ],
    });

    resources.computeBindGroup = newBindGroup;
    if (resources.computePasses && Array.isArray(resources.computePasses)) {
      resources.computePasses.forEach((pass) => {
        if (pass.bindGroup === previousBindGroup) {
          pass.bindGroup = newBindGroup;
        }
      });
    }

    const composeBindGroup = device.createBindGroup({
      layout: resources.composeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: resources.outputBuffer } },
        { binding: 1, resource: resources.outputTextureView ?? resources.outputTexture.createView() },
        { binding: 2, resource: { buffer: resources.paramsBuffer } },
        { binding: 3, resource: inputView },
      ],
    });

    resources.bufferToTextureBindGroup = composeBindGroup;
  }

  afterDispatch() {
    if (!this.agentBuffers) {
      return;
    }
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }

  #resetSimulation() {
    this.invalidateResources();
    this.#timeSeconds = 0;
    this.#lastTimestamp = null;
  }
}

export default WormsEffect;

export const additionalPasses = {
  'worms/init_from_prev': {
    id: 'worms/init_from_prev',
    label: 'init_from_prev.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/init_from_prev.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 80 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba16float', persistent: true },
    },
  },
  'worms/agent_move': {
    id: 'worms/agent_move',
    label: 'agent_move.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/agent_move.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 80 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba16float', persistent: true },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom', persistent: true },
      agent_state_out: { kind: 'storageBuffer', size: 'custom', persistent: true },
    },
  },
  'worms/final_blend': {
    id: 'worms/final_blend',
    label: 'final_blend.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/final_blend.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 80 },
    },
  },
  'worms/buffer_to_texture': {
    id: 'worms/buffer_to_texture',
    label: 'buffer_to_texture.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/buffer_to_texture.wgsl',
    resources: {
      output_buffer: { kind: 'readOnlyStorageBuffer', size: 'pixel-f32x4' },
      output_texture: { kind: 'storageTexture', format: 'rgba16float' },
      params: { kind: 'uniformBuffer', size: 80 },
    },
  },
  'worms/compose_to_texture': {
    id: 'worms/compose_to_texture',
    label: 'compose_to_texture.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/compose_to_texture.wgsl',
    resources: {
      output_buffer: { kind: 'readOnlyStorageBuffer', size: 'pixel-f32x4' },
      output_texture: { kind: 'storageTexture', format: 'rgba16float' },
      params: { kind: 'uniformBuffer', size: 80 },
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
    },
  },
};
