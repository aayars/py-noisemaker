import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const FLOATS_PER_AGENT = 9;
const PARAM_FLOAT_COUNT = 16;
const MAX_AGENT_COUNT = 131072;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

function clampNumber(value, min, max) {
  let result = value;
  if (Number.isFinite(min)) {
    result = Math.max(min, result);
  }
  if (Number.isFinite(max)) {
    result = Math.min(max, result);
  }
  return result;
}

function toFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toBoolean(value, fallback = false) {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  if (typeof value === 'string') {
    const lower = value.trim().toLowerCase();
    if (lower === 'true') {
      return true;
    }
    if (lower === 'false') {
      return false;
    }
  }
  return value === undefined ? fallback : Boolean(value);
}

const PARAMETER_DEFAULTS = Object.freeze({
  density: getDefaultParamValue('density', 5),
  stride: getDefaultParamValue('stride', 1.0),
  quantize: getDefaultParamValue('quantize', false),
  intensity: getDefaultParamValue('intensity', 90),
  inverse: getDefaultParamValue('inverse', false),
  xy_blend: getDefaultParamValue('xy_blend', false),
  worm_lifetime: getDefaultParamValue('worm_lifetime', 30),
  inputIntensity: getDefaultParamValue('inputIntensity', 100),
  enabled: getDefaultParamValue('enabled', true),
});

class ErosionWormsEffect {
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
      density: PARAMETER_DEFAULTS.density,
      stride: PARAMETER_DEFAULTS.stride,
      quantize: Boolean(PARAMETER_DEFAULTS.quantize),
      intensity: PARAMETER_DEFAULTS.intensity,
      inverse: Boolean(PARAMETER_DEFAULTS.inverse),
      xy_blend: Boolean(PARAMETER_DEFAULTS.xy_blend),
      worm_lifetime: PARAMETER_DEFAULTS.worm_lifetime,
      inputIntensity: PARAMETER_DEFAULTS.inputIntensity,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null;
    this.lastDensity = null;
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
        logWarn?.('Failed to destroy erosion worms resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy erosion worms output texture during invalidation:', error);
      }
    }

    if (resources.feedbackTexture?.destroy) {
      try {
        resources.feedbackTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy erosion worms feedback texture during invalidation:', error);
      }
    }

    this.#destroyAgentBuffers();

    this.resources = null;
  }

  #destroyAgentBuffers() {
    const { logWarn } = this.helpers;
    if (!this.agentBuffers) {
      return;
    }

    [this.agentBuffers.a, this.agentBuffers.b].forEach((buffer) => {
      if (buffer?.destroy) {
        try {
          buffer.destroy();
        } catch (error) {
          logWarn?.('Failed to destroy erosion worms agent buffer:', error);
        }
      }
    });

    this.agentBuffers = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('ErosionWormsEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('ErosionWormsEffect.ensureResources requires multires output texture.');
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

  updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('ErosionWormsEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    const updateParamState = (key, value) => {
      const resources = this.resources;
      if (!resources?.paramsState || !resources.bindingOffsets) {
        return;
      }
      const offset = resources.bindingOffsets[key];
      if (typeof offset === 'number') {
        resources.paramsState[offset] = value;
        resources.paramsDirty = true;
      }
    };

    if (hasOwn(updates, 'density')) {
      const numeric = toFiniteNumber(updates.density);
      if (numeric === null) {
        logWarn?.('updateErosionWormsParams: density must be a finite number.');
      } else {
        const clamped = clampNumber(numeric, 1, 100);
        if (this.userState.density !== clamped) {
          this.userState.density = clamped;
          this.lastDensity = null;
          this.invalidateResources();
        }
        updated.push('density');
      }
    }

    if (hasOwn(updates, 'stride')) {
      const numeric = toFiniteNumber(updates.stride);
      if (numeric === null) {
        logWarn?.('updateErosionWormsParams: stride must be a finite number.');
      } else {
        const clamped = clampNumber(numeric, 0.1, 10.0);
        this.userState.stride = clamped;
        updateParamState('stride', clamped);
        updated.push('stride');
      }
    }

    if (hasOwn(updates, 'quantize')) {
      const value = toBoolean(updates.quantize, this.userState.quantize);
      this.userState.quantize = value;
      updateParamState('quantize', value ? 1.0 : 0.0);
      updated.push('quantize');
    }

    if (hasOwn(updates, 'intensity')) {
      const numeric = toFiniteNumber(updates.intensity);
      if (numeric === null) {
        logWarn?.('updateErosionWormsParams: intensity must be a finite number.');
      } else {
        const percent = clampNumber(numeric, 0, 100);
        this.userState.intensity = percent;
        updateParamState('intensity', percent * 0.01);
        updated.push('intensity');
      }
    }

    if (hasOwn(updates, 'inverse')) {
      const value = toBoolean(updates.inverse, this.userState.inverse);
      this.userState.inverse = value;
      updateParamState('inverse', value ? 1.0 : 0.0);
      updated.push('inverse');
    }

    if (hasOwn(updates, 'xy_blend')) {
      const value = toBoolean(updates.xy_blend, this.userState.xy_blend);
      this.userState.xy_blend = value;
      updateParamState('xy_blend', value ? 1.0 : 0.0);
      updated.push('xy_blend');
    }

    if (hasOwn(updates, 'worm_lifetime')) {
      const numeric = toFiniteNumber(updates.worm_lifetime);
      if (numeric === null) {
        logWarn?.('updateErosionWormsParams: worm_lifetime must be a finite number.');
      } else {
        const clamped = clampNumber(numeric, 0, 60);
        this.userState.worm_lifetime = clamped;
        updateParamState('worm_lifetime', clamped);
        updated.push('worm_lifetime');
      }
    }

    if (hasOwn(updates, 'inputIntensity')) {
      const numeric = toFiniteNumber(updates.inputIntensity);
      if (numeric === null) {
        logWarn?.('updateErosionWormsParams: inputIntensity must be a finite number.');
      } else {
        const percent = clampNumber(numeric, 0, 100);
        this.userState.inputIntensity = percent;
        updateParamState('inputIntensity', percent * 0.01);
        updated.push('inputIntensity');
      }
    }

    if (hasOwn(updates, 'enabled')) {
      const value = toBoolean(updates.enabled, this.userState.enabled);
      this.userState.enabled = value;
      if (this.resources) {
        this.resources.enabled = value;
      }
      updated.push('enabled');
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
  }

  #initializeAgentBuffers(device, width, height) {
    const density = clampNumber(this.userState.density, 1, 100);
    const maxDim = Math.max(width, height);
    const desiredCount = Math.floor(maxDim * density);
    const agentCount = Math.max(1, Math.min(desiredCount, MAX_AGENT_COUNT));
    const bufferSize = agentCount * FLOATS_PER_AGENT * Float32Array.BYTES_PER_ELEMENT;

    const initialData = new Float32Array(agentCount * FLOATS_PER_AGENT);
    for (let i = 0; i < agentCount; i += 1) {
      const base = i * FLOATS_PER_AGENT;
      initialData[base + 0] = Math.random() * width;
      initialData[base + 1] = Math.random() * height;
      const angle = Math.random() * Math.PI * 2;
      initialData[base + 2] = Math.cos(angle);
      initialData[base + 3] = Math.sin(angle);
      initialData[base + 4] = 0.0;
      initialData[base + 5] = 0.0;
      initialData[base + 6] = 0.0;
      initialData[base + 7] = clampNumber(0.7 + Math.random() * 0.3, 0, 1);
      initialData[base + 8] = -1.0;
    }

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const bufferA = device.createBuffer({ size: bufferSize, usage });
    const bufferB = device.createBuffer({ size: bufferSize, usage });
    device.queue.writeBuffer(bufferA, 0, initialData);
    device.queue.writeBuffer(bufferB, 0, initialData);

    return { a: bufferA, b: bufferB, current: 'a', agentCount };
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

    setStatus?.('Creating erosion worms resources…');

    try {
      const descriptor = getShaderDescriptor('erosion_worms');
      const shaderMetadata = await getShaderMetadataCached('erosion_worms');
      const initDescriptor = getShaderDescriptor('erosion_worms/init_from_prev');
      const moveDescriptor = getShaderDescriptor('erosion_worms/agent_move');
      const finalDescriptor = getShaderDescriptor('erosion_worms/final_blend');

      if (!this.agentBuffers || this.lastDensity !== this.userState.density) {
        this.#destroyAgentBuffers();
        this.agentBuffers = this.#initializeAgentBuffers(device, width, height);
        this.lastDensity = this.userState.density;
      }

      if (!this.agentBuffers) {
        throw new Error('ErosionWormsEffect: failed to initialize agent buffers.');
      }

      const paramsBuffer = device.createBuffer({
        size: PARAM_FLOAT_COUNT * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const outputBufferSize = Math.max(1, width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT);
      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      });
      const outputTextureView = outputTexture.createView();

      const feedbackTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      });
      const feedbackTextureView = feedbackTexture.createView();

      const paramsState = new Float32Array(PARAM_FLOAT_COUNT);
      const bindingOffsets = {
        width: 0,
        height: 1,
        channel_count: 2,
        channelCount: 2,
        density: 4,
        stride: 5,
        quantize: 6,
        intensity: 8,
        inverse: 9,
        xy_blend: 10,
        time: 11,
        worm_lifetime: 12,
        inputIntensity: 13,
      };

      paramsState[0] = width;
      paramsState[1] = height;
      paramsState[2] = RGBA_CHANNEL_COUNT;
      paramsState[3] = 0.0;
      paramsState[4] = this.userState.density;
      paramsState[5] = this.userState.stride;
      paramsState[6] = this.userState.quantize ? 1.0 : 0.0;
      paramsState[7] = 0.0;
      paramsState[8] = this.userState.intensity * 0.01;
      paramsState[9] = this.userState.inverse ? 1.0 : 0.0;
      paramsState[10] = this.userState.xy_blend ? 1.0 : 0.0;
      paramsState[11] = 0.0;
      paramsState[12] = this.userState.worm_lifetime;
      paramsState[13] = this.userState.inputIntensity * 0.01;
      paramsState[14] = 0.0;
      paramsState[15] = 0.0;
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);

      let initPipeline;
      let movePipeline;
      let finalPipeline;
      let useMultiPass = true;

      try {
        initPipeline = await getOrCreateComputePipeline(
          device,
          initDescriptor.id,
          computePipelineLayout,
          initDescriptor.entryPoint ?? 'main',
        );
        movePipeline = await getOrCreateComputePipeline(
          device,
          moveDescriptor.id,
          computePipelineLayout,
          moveDescriptor.entryPoint ?? 'main',
        );
        finalPipeline = await getOrCreateComputePipeline(
          device,
          finalDescriptor.id,
          computePipelineLayout,
          finalDescriptor.entryPoint ?? 'main',
        );
        logInfo?.('Erosion worms: using multi-pass compute pipeline.');
      } catch (error) {
        logWarn?.('Erosion worms: multi-pass pipeline creation failed, using fallback path.', error);
        useMultiPass = false;
      }

      const computePipeline = useMultiPass
        ? movePipeline
        : await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

      const inputTextureView = multiresResources.outputTexture.createView();
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: inputTextureView },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTextureView },
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
      const feedbackCopyBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: feedbackTextureView },
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
      let feedbackCopyPass = null;
      if (useMultiPass) {
        const agentDispatch = Math.max(1, Math.ceil(this.agentBuffers.agentCount / agentWorkgroupSize[0]));
        feedbackCopyPass = {
          pipeline: bufferToTexturePipeline,
          bindGroup: feedbackCopyBindGroup,
          workgroupSize: bufferToTextureWorkgroupSize,
          getDispatch: ({ width: w, height: h }) => [
            Math.ceil(w / bufferToTextureWorkgroupSize[0]),
            Math.ceil(h / bufferToTextureWorkgroupSize[1]),
            1,
          ],
        };

        computePasses = [
          {
            pipeline: initPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: pixelWorkgroupSize,
            getDispatch: ({ width: w, height: h }) => [
              Math.ceil(w / pixelWorkgroupSize[0]),
              Math.ceil(h / pixelWorkgroupSize[1]),
              1,
            ],
          },
          {
            pipeline: movePipeline,
            bindGroup: computeBindGroup,
            workgroupSize: agentWorkgroupSize,
            getDispatch: () => [agentDispatch, 1, 1],
          },
          feedbackCopyPass,
          {
            pipeline: finalPipeline,
            bindGroup: computeBindGroup,
            workgroupSize: pixelWorkgroupSize,
            getDispatch: ({ width: w, height: h }) => [
              Math.ceil(w / pixelWorkgroupSize[0]),
              Math.ceil(h / pixelWorkgroupSize[1]),
              1,
            ],
          },
        ];
      }

      this.agentBuffers.current = 'a';

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
        feedbackTextureView,
        feedbackCopyBindGroup,
        feedbackCopyPass,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        bufferToTextureBindGroupLayout,
        bufferToTextureWorkgroupSize,
        blitBindGroup,
        workgroupSize: pixelWorkgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        device,
        bindingOffsets,
        shouldCopyOutputToPrev: useMultiPass ? false : true,
        computeBindGroupLayout,
      };

      setStatus?.('Erosion worms resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create erosion worms resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources }) {
    if (!this.agentBuffers || !this.resources || !this.resources.enabled) {
      return;
    }

    const resources = this.resources;
    if (resources.paramsDirty && resources.paramsState) {
      device.queue.writeBuffer(resources.paramsBuffer, 0, resources.paramsState);
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
    if (Array.isArray(resources.computePasses)) {
      resources.computePasses.forEach((pass) => {
        if (pass && pass.bindGroup === previousBindGroup) {
          pass.bindGroup = newBindGroup;
        }
      });
    }
  }

  afterDispatch() {
    if (!this.agentBuffers) {
      return;
    }
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }
}

export default ErosionWormsEffect;

export const additionalPasses = {
  'erosion_worms/init_from_prev': {
    id: 'erosion_worms/init_from_prev',
    label: 'init_from_prev.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/erosion_worms/init_from_prev.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float', persistent: true },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom', persistent: true },
      agent_state_out: { kind: 'storageBuffer', size: 'custom', persistent: true },
    },
  },
  'erosion_worms/agent_move': {
    id: 'erosion_worms/agent_move',
    label: 'agent_move.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/erosion_worms/agent_move.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float', persistent: true },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom', persistent: true },
      agent_state_out: { kind: 'storageBuffer', size: 'custom', persistent: true },
    },
  },
  'erosion_worms/final_blend': {
    id: 'erosion_worms/final_blend',
    label: 'final_blend.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/erosion_worms/final_blend.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 64 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float', persistent: true },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom', persistent: true },
      agent_state_out: { kind: 'storageBuffer', size: 'custom', persistent: true },
    },
  },
};
