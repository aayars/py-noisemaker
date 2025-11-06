import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const FLOATS_PER_AGENT = 8; // [x, y, rot, stride, r, g, b, seed]
const PARAM_FLOAT_COUNT = 20;
const MAX_AGENT_COUNT = 131072;
const DENSITY_SCALE = 0.2;
const STRIDE_SCALE = 0.1;
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

function seedAgents(width, height, density, stride, strideDeviation) {
  const count = computeAgentCount(width, height, density);
  if (count <= 0) {
    return { data: new Float32Array(0), count: 0 };
  }
  const deviation = Number.isFinite(strideDeviation) ? strideDeviation : 0.05;
  const data = new Float32Array(count * FLOATS_PER_AGENT);
  const normalized = normalizedStride(stride, width, height);
  for (let i = 0; i < count; i += 1) {
    const base = i * FLOATS_PER_AGENT;
    data[base + 0] = Math.random() * width;
    data[base + 1] = Math.random() * height;
    data[base + 2] = Math.random() * Math.PI * 2;
    const strideVariation = normalized * (1 + (Math.random() - 0.5) * 2 * deviation);
    data[base + 3] = Math.max(0.1, strideVariation);
    data[base + 4] = 0.5 + Math.random() * 0.5;
    data[base + 5] = 0.5 + Math.random() * 0.5;
    data[base + 6] = 0.5 + Math.random() * 0.5;
    data[base + 7] = Math.random() * 1000000;
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
  drunkenness: getDefaultParamValue('drunkenness', 0),
  intensity: getDefaultParamValue('intensity', 90),
  inputIntensity: getDefaultParamValue('inputIntensity', 100),
  enabled: getDefaultParamValue('enabled', true),
});

class WormsEffect {
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
      behavior: PARAMETER_DEFAULTS.behavior,
      density: PARAMETER_DEFAULTS.density,
      stride: PARAMETER_DEFAULTS.stride,
      kink: PARAMETER_DEFAULTS.kink,
      strideDeviation: PARAMETER_DEFAULTS.strideDeviation,
      quantize: Boolean(PARAMETER_DEFAULTS.quantize),
      drunkenness: PARAMETER_DEFAULTS.drunkenness,
      intensity: PARAMETER_DEFAULTS.intensity,
      inputIntensity: PARAMETER_DEFAULTS.inputIntensity,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null;
    this.lastDensity = null;
    this.lastStride = null;
    this.lastStrideDeviation = null;
    this.feedbackTexture = null; // Persistent feedback texture (preserved across density changes)
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

  #populateParamsState(paramsState) {
    if (!paramsState) {
      return;
    }

    const width = Math.max(1, Math.floor(this.width));
    const height = Math.max(1, Math.floor(this.height));
    const actualDensity = this.#getActualDensity();
    const actualStride = this.#getActualStride();
    const strideDeviation = this.#getStrideDeviation();
    const intensity = this.#getIntensityScalar();
    const inputIntensity = this.#getInputIntensityScalar();
    const behavior = clamp(toNumber(this.userState.behavior, PARAMETER_DEFAULTS.behavior), 1, 10);
    const kink = Math.max(0, toNumber(this.userState.kink, PARAMETER_DEFAULTS.kink));
    const drunkenness = toNumber(this.userState.drunkenness, PARAMETER_DEFAULTS.drunkenness);
    const quantizeFlag = this.userState.quantize ? 1 : 0;

    paramsState[0] = width;
    paramsState[1] = height;
    paramsState[2] = RGBA_CHANNEL_COUNT;
    paramsState[3] = 0;

    paramsState[4] = behavior;
    paramsState[5] = actualDensity;
    paramsState[6] = actualStride;
    paramsState[7] = 0;

    paramsState[8] = strideDeviation;
    paramsState[9] = 1.0;
    paramsState[10] = kink;
    paramsState[11] = drunkenness;

    paramsState[12] = quantizeFlag;
    paramsState[13] = 0;
    paramsState[14] = 0;
    paramsState[15] = intensity;

    paramsState[16] = inputIntensity;
    paramsState[17] = 0;
    paramsState[18] = 0;
    paramsState[19] = 0;
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
      'drunkenness',
      'intensity',
      'inputIntensity',
    ];
    numericParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const numeric = Number(updates[name]);
        if (Number.isFinite(numeric)) {
          const oldValue = this.userState[name];
          this.userState[name] = numeric;

          const densityChanged = name === 'density' && oldValue !== numeric;
          const strideChanged = name === 'stride' && oldValue !== numeric;
          const strideDeviationChanged = name === 'strideDeviation' && oldValue !== numeric;

          if (densityChanged || strideChanged || strideDeviationChanged) {
            this.lastDensity = null;
            this.lastStride = null;
            this.lastStrideDeviation = null;
            this.invalidateResources();
          }

          if (this.resources?.paramsState) {
            this.#populateParamsState(this.resources.paramsState);
            this.resources.paramsDirty = true;
          }

          updated.push(name);
        } else {
          logWarn?.(`updateWormsParams: ${name} must be a finite number.`);
        }
      }
    });

    const booleanParams = ['enabled', 'quantize'];
    booleanParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const value = Boolean(updates[name]);
        this.userState[name] = value;
        if (name === 'enabled' && this.resources) {
          this.resources.enabled = value;
        } else if (name === 'quantize' && this.resources?.paramsState) {
          this.#populateParamsState(this.resources.paramsState);
          this.resources.paramsDirty = true;
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

  #initializeAgentBuffers(device, width, height) {
    const density = this.#getActualDensity();
    const stride = this.#getActualStride();
    const strideDeviation = this.#getStrideDeviation();

    const agents = seedAgents(width, height, density, stride, strideDeviation);
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
      getBufferToTexturePipeline,
    } = this.helpers;

    setStatus?.('Creating worms resources…');

    try {
      const descriptor = getShaderDescriptor('worms');
      const shaderMetadata = await getShaderMetadataCached('worms');
      
      // Get descriptors for multi-pass shaders
      const initFromPrevDescriptor = getShaderDescriptor('worms/init_from_prev');
      const agentMoveDescriptor = getShaderDescriptor('worms/agent_move');
      const finalBlendDescriptor = getShaderDescriptor('worms/final_blend');

      const actualDensity = this.#getActualDensity();
      const actualStride = this.#getActualStride();
      const strideDeviation = this.#getStrideDeviation();

      // Create or recreate agent buffers if key parameters changed
      if (
        !this.agentBuffers ||
        this.lastDensity !== actualDensity ||
        this.lastStride !== actualStride ||
        this.lastStrideDeviation !== strideDeviation
      ) {
        if (this.agentBuffers) {
          this.#destroyAgentBuffers();
        }
        this.agentBuffers = this.#initializeAgentBuffers(device, width, height);
      }
      this.lastDensity = actualDensity;
      this.lastStride = actualStride;
      this.lastStrideDeviation = strideDeviation;

      const feedbackTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });
      const feedbackSampleView = feedbackTexture.createView();
      const feedbackStorageView = feedbackTexture.createView();

      const paramsSize = PARAM_FLOAT_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const paramsState = new Float32Array(PARAM_FLOAT_COUNT);
      this.#populateParamsState(paramsState);
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
      
      let initFromPrevPipeline, agentMovePipeline, finalBlendPipeline;
      let useMultiPass = true;
      
      try {
        // Pass 0: Init from prev_texture
        initFromPrevPipeline = await getOrCreateComputePipeline(
          device,
          'worms/init_from_prev',
          computePipelineLayout,
          initFromPrevDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 1: Agent movement
        agentMovePipeline = await getOrCreateComputePipeline(
          device, 
          'worms/agent_move',
          computePipelineLayout, 
          agentMoveDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 2: Final blend
        finalBlendPipeline = await getOrCreateComputePipeline(
          device,
          'worms/final_blend',
          computePipelineLayout,
          finalBlendDescriptor.entryPoint ?? 'main'
        );
        
        logInfo?.('Worms: Using optimized multi-pass shaders');
      } catch (error) {
        logWarn?.('Worms: Multi-pass pipeline creation failed, falling back to single-pass:', error);
        useMultiPass = false;
      }
      
      // Fallback: use original single-pass shader (if multi-pass fails)
      const computePipeline = useMultiPass 
        ? agentMovePipeline 
        : await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

      this.agentBuffers.current = 'a';

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      const outputTextureView = outputTexture.createView();

      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackSampleView },
          { binding: 4, resource: { buffer: this.agentBuffers.a } },
          { binding: 5, resource: { buffer: this.agentBuffers.b } },
        ],
      });

      const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);
      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: outputTextureView },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const feedbackBufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: feedbackStorageView },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const blitBindGroup = device.createBindGroup({
        layout: multiresResources.blitBindGroupLayout,
        entries: [{ binding: 0, resource: outputTextureView }],
      });

      // Configure multi-pass execution
      const agentCount = this.agentBuffers.agentCount;
      const agentWorkgroupSize = [64, 1, 1];
      const blendWorkgroupSize = [8, 8, 1];
      
      let computePasses = null;
      
      if (useMultiPass) {
        computePasses = [
          // Pass 0: Copy prev_texture to output_buffer (pixel-parallel)
          {
            pipeline: initFromPrevPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: blendWorkgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / blendWorkgroupSize[0]),
              Math.ceil(height / blendWorkgroupSize[1]),
              1
            ],
          },
          // Pass 1: Move agents and deposit trails (agent-parallel)
          {
            pipeline: agentMovePipeline,
            bindGroup: computeBindGroup,
            workgroupSize: agentWorkgroupSize,
            getDispatch: () => [
              Math.max(1, Math.ceil(Math.max(agentCount, 1) / agentWorkgroupSize[0])),
              1,
              1
            ],
          },
          // Pass 2: Write trail buffer to feedback texture (pixel-parallel)
          {
            pipeline: bufferToTexturePipeline,
            bindGroup: feedbackBufferToTextureBindGroup,
            workgroupSize: blendWorkgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / blendWorkgroupSize[0]),
              Math.ceil(height / blendWorkgroupSize[1]),
              1
            ],
          },
          // Pass 3: Final blend with input (pixel-parallel)
          {
            pipeline: finalBlendPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: blendWorkgroupSize,
            getDispatch: ({ width, height }) => [
              Math.ceil(width / blendWorkgroupSize[0]),
              Math.ceil(height / blendWorkgroupSize[1]),
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
        computePasses, // Multi-pass configuration
        paramsBuffer,
        paramsState,
        outputBuffer,
        outputTexture,
        feedbackTexture,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        feedbackBufferToTextureBindGroup,
        blitBindGroup,
        workgroupSize: blendWorkgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        device,
        shouldCopyOutputToPrev: useMultiPass ? false : true,
        computeBindGroupLayout,
      };

      setStatus?.('Worms resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create worms resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources }) {
    if (!this.agentBuffers || !this.resources) return;

    // Update params buffer if dirty
    if (this.resources.paramsDirty) {
      device.queue.writeBuffer(this.resources.paramsBuffer, 0, this.resources.paramsState);
      this.resources.paramsDirty = false;
    }

    const currentIsA = this.agentBuffers.current === 'a';
    const inputBuffer = currentIsA ? this.agentBuffers.a : this.agentBuffers.b;
    const outputBuffer = currentIsA ? this.agentBuffers.b : this.agentBuffers.a;

    // Recreate bind group with swapped buffers
    const previousBindGroup = this.resources.computeBindGroup;

    const newBindGroup = device.createBindGroup({
      layout: this.resources.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.resources.outputBuffer } },
        { binding: 2, resource: { buffer: this.resources.paramsBuffer } },
        { binding: 3, resource: this.resources.feedbackTexture.createView() },
        { binding: 4, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
      ],
    });
    
    this.resources.computeBindGroup = newBindGroup;
    
    // Update bind groups in all compute passes
    if (this.resources.computePasses && Array.isArray(this.resources.computePasses)) {
      this.resources.computePasses.forEach((pass) => {
        if (pass && pass.bindGroup === previousBindGroup) {
          pass.bindGroup = newBindGroup;
        }
      });
    }
  }

  afterDispatch() {
    if (!this.agentBuffers) return;
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }
}

export default WormsEffect;

// Multi-pass shader descriptors
export const additionalPasses = {
  'worms/init_from_prev': {
    id: 'worms/init_from_prev',
    label: 'init_from_prev.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/worms/init_from_prev.wgsl',
    resources: {
      params: { kind: 'uniformBuffer', size: 80 },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' }
    }
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
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom' },
      agent_state_out: { kind: 'storageBuffer', size: 'custom' }
    }
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
  params: { kind: 'uniformBuffer', size: 80 }
    }
  }
};

