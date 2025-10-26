import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const AGENT_FLOATS_PER_WORM = 8; // [x, y, rot, stride, r, g, b, seed]
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  behavior: getDefaultParamValue('behavior', 1),
  density: getDefaultParamValue('density', 4.0),
  duration: getDefaultParamValue('duration', 4.0),
  stride: getDefaultParamValue('stride', 1.0),
  alpha: getDefaultParamValue('alpha', 0.5),
  kink: getDefaultParamValue('kink', 1.0),
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
      duration: PARAMETER_DEFAULTS.duration,
      stride: PARAMETER_DEFAULTS.stride,
      alpha: PARAMETER_DEFAULTS.alpha,
      kink: PARAMETER_DEFAULTS.kink,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null;
    this.lastDensity = null;
    this.feedbackTexture = null; // Persistent feedback texture (preserved across density changes)
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

    const numericParams = ['behavior', 'density', 'duration', 'stride', 'alpha', 'kink'];
    numericParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const numeric = Number(updates[name]);
        if (Number.isFinite(numeric)) {
          const oldValue = this.userState[name];
          this.userState[name] = numeric;
          
          // If density changed, invalidate resources to force recreation with new agent count
          if (name === 'density' && oldValue !== numeric) {
            this.lastDensity = null;
            this.invalidateResources();
          } else if (this.resources?.paramsState && this.resources?.bindingOffsets?.[name] !== undefined) {
            this.resources.paramsState[this.resources.bindingOffsets[name]] = numeric;
            this.resources.paramsDirty = true;
          }
          
          updated.push(name);
        } else {
          logWarn?.(`updateWormsParams: ${name} must be a finite number.`);
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

  #initializeAgentBuffers(device, width, height) {
    const agentCount = Math.max(Math.floor(Math.max(width, height) * this.userState.density), 1);
    const bufferSize = agentCount * AGENT_FLOATS_PER_WORM * Float32Array.BYTES_PER_ELEMENT;
    
    const initialData = new Float32Array(agentCount * AGENT_FLOATS_PER_WORM);
    for (let i = 0; i < agentCount; i++) {
      const base = i * AGENT_FLOATS_PER_WORM;
      
      // Initialize position randomly within frame
      initialData[base + 0] = Math.random() * width;   // x
      initialData[base + 1] = Math.random() * height;  // y
      
      // Initialize rotation randomly
      initialData[base + 2] = Math.random() * Math.PI * 2; // rot
      
      // Initialize stride with deviation
      const strideDeviation = 0.05; // Default from Python
      const stride = this.userState.stride * (1 + (Math.random() - 0.5) * 2 * strideDeviation);
      const normalizedStride = stride * (Math.max(width, height) / 1024.0);
      initialData[base + 3] = Math.max(0.1, normalizedStride); // stride
      
      // Sample starting color (random for now, could sample from texture)
      initialData[base + 4] = 0.5 + Math.random() * 0.5; // r
      initialData[base + 5] = 0.5 + Math.random() * 0.5; // g
      initialData[base + 6] = 0.5 + Math.random() * 0.5; // b
      
      // Initialize seed for RNG
      initialData[base + 7] = Math.random() * 1000000; // seed
    }

    const bufferA = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const bufferB = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

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

    setStatus?.('Creating worms resources…');

    try {
      const descriptor = getShaderDescriptor('worms');
      const shaderMetadata = await getShaderMetadataCached('worms');
      
      // Get descriptors for multi-pass shaders
      const initFromPrevDescriptor = getShaderDescriptor('worms/init_from_prev');
      const agentMoveDescriptor = getShaderDescriptor('worms/agent_move');
      const finalBlendDescriptor = getShaderDescriptor('worms/final_blend');

      // Create or recreate agent buffers if density changed
      if (!this.agentBuffers || this.lastDensity !== this.userState.density) {
        if (this.agentBuffers) {
          this.#destroyAgentBuffers();
        }
        this.agentBuffers = this.#initializeAgentBuffers(device, width, height);
        this.lastDensity = this.userState.density;
      }

      const feedbackTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });

      const paramsSize = 64;
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
        width: 0, height: 1, channel_count: 2, channelCount: 2,
        behavior: 4, density: 5, duration: 6, stride: 7,
        stride_deviation: 8, alpha: 9, kink: 10, drunkenness: 11,
        quantize: 12, time: 13, speed: 14,
      };

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channel_count] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.behavior] = this.userState.behavior;
      paramsState[bindingOffsets.density] = this.userState.density;
      paramsState[bindingOffsets.duration] = this.userState.duration;
      paramsState[bindingOffsets.stride] = this.userState.stride;
      paramsState[bindingOffsets.stride_deviation] = 0.05; // Default
      paramsState[bindingOffsets.alpha] = this.userState.alpha;
      paramsState[bindingOffsets.kink] = this.userState.kink;
      paramsState[bindingOffsets.drunkenness] = 0.0; // Default
      paramsState[bindingOffsets.quantize] = 0.0; // Default false
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = 1.0; // Default

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

      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTexture.createView() },
          { binding: 4, resource: { buffer: this.agentBuffers.a } },
          { binding: 5, resource: { buffer: this.agentBuffers.b } },
        ],
      });

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
              Math.ceil(agentCount / agentWorkgroupSize[0]),
              1,
              1
            ],
          },
          // Pass 2: Final blend with input (pixel-parallel)
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
        blitBindGroup,
        workgroupSize: blendWorkgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        device,
        bindingOffsets,
        shouldCopyOutputToPrev: true,
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
      this.resources.computePasses.forEach(pass => {
        pass.bindGroup = newBindGroup;
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
      params: { kind: 'uniformBuffer', size: 64 },
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
      params: { kind: 'uniformBuffer', size: 64 }
    }
  }
};

