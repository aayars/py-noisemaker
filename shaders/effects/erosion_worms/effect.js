import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const AGENT_FLOATS_PER_WORM = 9; // [x, y, x_dir, y_dir, r, g, b, inertia, age]
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  density: getDefaultParamValue('density', 5),
  contraction: getDefaultParamValue('contraction', 1.0),
  quantize: getDefaultParamValue('quantize', false),
  alpha: getDefaultParamValue('alpha', 0.25),
  inverse: getDefaultParamValue('inverse', false),
  xy_blend: getDefaultParamValue('xy_blend', false),
  speed: getDefaultParamValue('speed', 2.0),
  worm_lifetime: getDefaultParamValue('worm_lifetime', 30),
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
      contraction: PARAMETER_DEFAULTS.contraction,
      quantize: Boolean(PARAMETER_DEFAULTS.quantize),
      alpha: PARAMETER_DEFAULTS.alpha,
      inverse: Boolean(PARAMETER_DEFAULTS.inverse),
      xy_blend: Boolean(PARAMETER_DEFAULTS.xy_blend),
      speed: PARAMETER_DEFAULTS.speed,
      worm_lifetime: PARAMETER_DEFAULTS.worm_lifetime,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null; // Persistent agent state (ping-pong)
    this.lastDensity = null; // Track density changes for buffer recreation
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
    if (this.agentBuffers) {
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

  async updateParams(updates = {}) {
    console.log('[EROSION] updateParams called:', updates);
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('ErosionWormsEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    const numericParams = ['density', 'contraction', 'alpha', 'speed', 'worm_lifetime'];
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
            // For non-density params, update in-place
            console.log(`[EROSION] Updating ${name} to ${numeric}, setting paramsDirty=true`);
            this.resources.paramsState[this.resources.bindingOffsets[name]] = numeric;
            this.resources.paramsDirty = true;
          }
          
          updated.push(name);
        } else {
          logWarn?.(`updateErosionWormsParams: ${name} must be a finite number.`);
        }
      }
    });

    const booleanParams = ['quantize', 'inverse', 'xy_blend', 'enabled'];
    booleanParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const value = Boolean(updates[name]);
        this.userState[name] = value;
        if (name !== 'enabled' && this.resources?.paramsState && this.resources?.bindingOffsets?.[name] !== undefined) {
          this.resources.paramsState[this.resources.bindingOffsets[name]] = value ? 1.0 : 0.0;
          this.resources.paramsDirty = true;
        } else if (name === 'enabled' && this.resources) {
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
    const agentCount = Math.max(Math.floor(Math.sqrt(width * height) * (this.userState.density / 50)), 1);
    const bufferSize = agentCount * AGENT_FLOATS_PER_WORM * Float32Array.BYTES_PER_ELEMENT;
    
    const initialData = new Float32Array(agentCount * AGENT_FLOATS_PER_WORM);
    for (let i = 0; i < agentCount; i++) {
      const base = i * AGENT_FLOATS_PER_WORM;
      
      // Initialize position randomly
      initialData[base + 0] = Math.random() * width;  // x
      initialData[base + 1] = Math.random() * height; // y
      
      // Initialize direction as normalized random vector (matching Python)
      // Python: x_dir, y_dir = normal(); length = sqrt(x^2 + y^2); normalize
      const dir_x = (Math.random() - 0.5) * 2; // Random normal-ish
      const dir_y = (Math.random() - 0.5) * 2;
      const dir_len = Math.sqrt(dir_x * dir_x + dir_y * dir_y) || 1;
      initialData[base + 2] = dir_x / dir_len; // x_dir (normalized)
      initialData[base + 3] = dir_y / dir_len; // y_dir (normalized)
      
      // Sample starting color from input texture position
      // This will be the persistent "starting_colors" from Python
      initialData[base + 4] = 0.5 + Math.random() * 0.5; // r
      initialData[base + 5] = 0.5 + Math.random() * 0.5; // g
      initialData[base + 6] = 0.5 + Math.random() * 0.5; // b
      
      // Python: inertia = normal(mean=0.75, stddev=0.25)
      // Simple approximation: 0.75 +/- random * 0.5 clamped to [0,1]
      initialData[base + 7] = Math.max(0, Math.min(1, 0.75 + (Math.random() - 0.5) * 0.5)); // inertia
      
      // Initialize age to 0 - respawn timing is handled by (age + index) % lifetime
      initialData[base + 8] = 0.0; // age
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

    setStatus?.('Creating erosion worms resources…');

    try {
      const descriptor = getShaderDescriptor('erosion_worms');
      const shaderMetadata = await getShaderMetadataCached('erosion_worms');
      
      // Get descriptors for multi-pass shaders (they share the same bind group layout)
      const initFromPrevDescriptor = getShaderDescriptor('erosion_worms/init_from_prev');
      const agentMoveDescriptor = getShaderDescriptor('erosion_worms/agent_move');
      const finalBlendDescriptor = getShaderDescriptor('erosion_worms/final_blend');

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
        width: 0, height: 1, channel_count: 2, channelCount: 2, // alias for main.js
        density: 4, contraction: 5, quantize: 6,
        alpha: 8, inverse: 9, xy_blend: 10, time: 11, speed: 12, worm_lifetime: 13,
      };

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channel_count] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.density] = this.userState.density;
      paramsState[bindingOffsets.contraction] = this.userState.contraction;
      paramsState[bindingOffsets.quantize] = this.userState.quantize ? 1.0 : 0.0;
      paramsState[bindingOffsets.alpha] = this.userState.alpha;
      paramsState[bindingOffsets.inverse] = this.userState.inverse ? 1.0 : 0.0;
      paramsState[bindingOffsets.xy_blend] = this.userState.xy_blend ? 1.0 : 0.0;
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = this.userState.speed;
      paramsState[bindingOffsets.worm_lifetime] = this.userState.worm_lifetime;

      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      // Create pipelines for multi-pass execution
      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
      
      let initFromPrevPipeline, agentMovePipeline, finalBlendPipeline;
      let useMultiPass = true;
      
      try {
        // Pass 0: Init from prev_texture (uses init_from_prev.wgsl)
        initFromPrevPipeline = await getOrCreateComputePipeline(
          device,
          'erosion_worms/init_from_prev',
          computePipelineLayout,
          initFromPrevDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 1: Agent movement (uses agent_move.wgsl)
        agentMovePipeline = await getOrCreateComputePipeline(
          device, 
          'erosion_worms/agent_move',
          computePipelineLayout, 
          agentMoveDescriptor.entryPoint ?? 'main'
        );
        
        // Pass 2: Final blend (uses final_blend.wgsl)
        finalBlendPipeline = await getOrCreateComputePipeline(
          device,
          'erosion_worms/final_blend',
          computePipelineLayout,
          finalBlendDescriptor.entryPoint ?? 'main'
        );
        
        logInfo?.('Erosion worms: Using optimized multi-pass shaders');
      } catch (error) {
        logWarn?.('Erosion worms: Multi-pass pipeline creation failed, falling back to single-pass:', error);
        useMultiPass = false;
      }
      
      // Fallback: use original single-pass shader
      const computePipeline = useMultiPass 
        ? agentMovePipeline 
        : await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');

      // Agent buffer management - both buffers stored, will swap each frame
      this.agentBuffers.current = 'a';

      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTexture.createView() },
          { binding: 4, resource: { buffer: this.agentBuffers.a } }, // agent_state_in
          { binding: 5, resource: { buffer: this.agentBuffers.b } }, // agent_state_out
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
        computeBindGroupLayout, // Store for recreating bind group
      };

      setStatus?.('Erosion worms resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create erosion worms resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources }) {
    // Swap agent buffers before each dispatch (ping-pong)
    if (!this.agentBuffers || !this.resources) return;

    // Update params buffer if dirty
    if (this.resources.paramsDirty) {
      console.log('[EROSION] Writing params, worm_lifetime=', this.resources.paramsState[this.resources.bindingOffsets.worm_lifetime]);
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
    // Swap the current marker after dispatch completes
    if (!this.agentBuffers) return;
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }
}

export default ErosionWormsEffect;

// Multi-pass shader descriptors
export const additionalPasses = {
  'erosion_worms/init_from_prev': {
    id: 'erosion_worms/init_from_prev',
    label: 'init_from_prev.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/erosion_worms/init_from_prev.wgsl',
    resources: {
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' }
    }
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
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom' },
      agent_state_out: { kind: 'storageBuffer', size: 'custom' }
    }
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
      params: { kind: 'uniformBuffer', size: 64 }
    }
  }
};
