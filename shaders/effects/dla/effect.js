import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;
const AGENT_FLOATS_PER_WALKER = 8; // [x, y, rot, stride, r, g, b, seed]
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

function getDefaultParamValue(name, fallback) {
  const param = (meta.parameters ?? []).find((p) => p.name === name);
  return param?.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  padding: getDefaultParamValue('padding', 2.0),
  seed_density: getDefaultParamValue('seed_density', 0.01),
  density: getDefaultParamValue('density', 0.05),
  alpha: getDefaultParamValue('alpha', 1.0),
  speed: getDefaultParamValue('speed', 2.0),
  enabled: getDefaultParamValue('enabled', true),
});

class DLAEffect {
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
      padding: PARAMETER_DEFAULTS.padding,
      seed_density: PARAMETER_DEFAULTS.seed_density,
      density: PARAMETER_DEFAULTS.density,
      alpha: PARAMETER_DEFAULTS.alpha,
      speed: PARAMETER_DEFAULTS.speed,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
    this.agentBuffers = null; // Persistent agent state (ping-pong)
    this.resourceCreateCount = 0; // DEBUG: Track resource recreation
    this.seedsInitialized = false; // Track whether seeds have been initialized
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
        logWarn?.('Failed to destroy DLA resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy DLA output texture during invalidation:', error);
      }
    }

    if (resources.feedbackTexture?.destroy) {
      try {
        resources.feedbackTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy DLA feedback texture during invalidation:', error);
      }
    }

    if (resources.outputBuffer?.destroy) {
      try {
        resources.outputBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy DLA output buffer during invalidation:', error);
      }
    }

    if (resources.gliderBuffer?.destroy) {
      try {
        resources.gliderBuffer.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy DLA glider buffer during invalidation:', error);
      }
    }

    if (this.agentBuffers) {
      [this.agentBuffers.a, this.agentBuffers.b].forEach((buffer) => {
        if (buffer?.destroy) {
          try {
            buffer.destroy();
          } catch (error) {
            logWarn?.('Failed to destroy DLA agent buffer during invalidation:', error);
          }
        }
      });
      this.agentBuffers = null;
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('DLAEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('DLAEffect.ensureResources requires multires output texture.');
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
      const hasPipelines = resources.computePasses || resources.computePipeline;
      if (hasPipelines && sizeMatches) {
        return resources;
      }
      this.invalidateResources();
    }

    return this.#createResources({ device, width, height, multiresResources });
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('DLAEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    const numericParams = ['padding', 'seed_density', 'density', 'alpha', 'speed'];
    numericParams.forEach((name) => {
      if (hasOwn(updates, name)) {
        const numeric = Number(updates[name]);
        if (Number.isFinite(numeric)) {
          this.userState[name] = numeric;
          if (this.resources?.paramsState && this.resources?.bindingOffsets?.[name] !== undefined) {
            this.resources.paramsState[this.resources.bindingOffsets[name]] = numeric;
            this.resources.paramsDirty = true;
          }
          updated.push(name);
          
          // Changing seed_density requires resetting seeds (but keeping resources)
          if (name === 'seed_density') {
            this.seedsInitialized = false;
            // Clear feedback texture by resetting frame count - next frame will reinitialize seeds
            this._frameCount = 0;
            console.log('DLA: seed_density changed, will reinitialize seeds');
          }
          // Changing density requires recreating agent buffers
          else if (name === 'density') {
            // Just mark for recreation - beforeDispatch will handle it
            this.agentBuffers = null;
          }
        } else {
          logWarn?.(`updateDLAParams: ${name} must be a finite number.`);
        }
      }
    });

    if (hasOwn(updates, 'enabled')) {
      this.userState.enabled = Boolean(updates.enabled);
      if (this.resources) {
        this.resources.enabled = this.userState.enabled;
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
    const agentCount = Math.max(Math.floor(Math.sqrt(width * height) * this.userState.density), 1);
    const bufferSize = agentCount * AGENT_FLOATS_PER_WALKER * Float32Array.BYTES_PER_ELEMENT;
    
    console.log(`DLA: Initializing ${agentCount} agents for ${width}x${height} (density=${this.userState.density})`);
    
    // Seed RNG with time to get different results each initialization
    const timeSeed = Date.now();
    let rngState = timeSeed;
    const seededRandom = () => {
      rngState = (rngState * 1664525 + 1013904223) | 0;
      return ((rngState >>> 0) / 4294967296);
    };
    
    const initialData = new Float32Array(agentCount * AGENT_FLOATS_PER_WALKER);
    
    for (let i = 0; i < agentCount; i++) {
      const base = i * AGENT_FLOATS_PER_WALKER;
      
      initialData[base + 0] = seededRandom() * width;  // x
      initialData[base + 1] = seededRandom() * height; // y
      initialData[base + 2] = 0;           // rot
      initialData[base + 3] = 1;           // stride
      initialData[base + 4] = 0.0;
      initialData[base + 5] = 1.0;
      initialData[base + 6] = 0.0;
      initialData[base + 7] = seededRandom();
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
      createBindGroupEntriesFromResources,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    setStatus?.('Creating DLA resources…');

    try {
      // Get buffer-to-texture pipeline (used multiple times)
      const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);

      // Get shader descriptors for all 4 passes (init_seeds, init_from_prev, agent_walk, final_blend)
      const seedsDescriptor = getShaderDescriptor('dla/init_seeds');
      const initDescriptor = getShaderDescriptor('dla/init_from_prev');
      const agentDescriptor = getShaderDescriptor('dla/agent_walk');
      const blendDescriptor = getShaderDescriptor('dla/final_blend');
      
      const seedsMetadata = await getShaderMetadataCached('dla/init_seeds');
      const initMetadata = await getShaderMetadataCached('dla/init_from_prev');
      const agentMetadata = await getShaderMetadataCached('dla/agent_walk');
      const blendMetadata = await getShaderMetadataCached('dla/final_blend');

      if (!this.agentBuffers) {
        this.agentBuffers = this.#initializeAgentBuffers(device, width, height);
      }

      const feedbackTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });

      logInfo?.(`DLA: Feedback texture created for ${width}x${height}`);

      const paramsSize = 48;
      const paramsBuffer = device.createBuffer({
        size: paramsSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
      console.log(`DLA: Creating outputBuffer for ${width}x${height}, size: ${outputBufferSize} bytes`);
      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      console.log('DLA: outputBuffer created:', outputBuffer);

      const gliderBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      console.log('DLA: gliderBuffer created:', gliderBuffer);

      this.resourceCreateCount++;
      console.log(`DLA: Creating resources #${this.resourceCreateCount}`);

      const clearedData = new Float32Array(width * height * RGBA_CHANNEL_COUNT);
      device.queue.writeBuffer(outputBuffer, 0, clearedData);
      device.queue.writeBuffer(gliderBuffer, 0, clearedData);

      const paramsLength = paramsSize / Float32Array.BYTES_PER_ELEMENT;
      const paramsState = new Float32Array(paramsLength);
      const bindingOffsets = {
        width: 0, height: 1, channelCount: 2, padding: 3,
        seed_density: 4, density: 5, alpha: 6, time: 7, speed: 8,
      };

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channelCount] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.padding] = this.userState.padding;
      paramsState[bindingOffsets.seed_density] = this.userState.seed_density;
      paramsState[bindingOffsets.density] = this.userState.density;
      paramsState[bindingOffsets.alpha] = this.userState.alpha;
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = this.userState.speed;

      console.log(`DLA: params buffer dimensions: ${width}x${height}`);
      console.log(`DLA: multires texture dimensions: ${multiresResources.outputTexture.width}x${multiresResources.outputTexture.height}`);
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      // Create pipelines for all 4 passes
      const seedsBindGroupLayout = getOrCreateBindGroupLayout(device, seedsDescriptor.id, 'compute', seedsMetadata);
      const seedsPipelineLayout = getOrCreatePipelineLayout(device, seedsDescriptor.id, 'compute', seedsBindGroupLayout);
      const seedsPipeline = await getOrCreateComputePipeline(device, seedsDescriptor.id, seedsPipelineLayout, seedsDescriptor.entryPoint ?? 'main');

      const initBindGroupLayout = getOrCreateBindGroupLayout(device, initDescriptor.id, 'compute', initMetadata);
      const initPipelineLayout = getOrCreatePipelineLayout(device, initDescriptor.id, 'compute', initBindGroupLayout);
      const initPipeline = await getOrCreateComputePipeline(device, initDescriptor.id, initPipelineLayout, initDescriptor.entryPoint ?? 'main');

      const agentBindGroupLayout = getOrCreateBindGroupLayout(device, agentDescriptor.id, 'compute', agentMetadata);
      const agentPipelineLayout = getOrCreatePipelineLayout(device, agentDescriptor.id, 'compute', agentBindGroupLayout);
      const agentPipeline = await getOrCreateComputePipeline(device, agentDescriptor.id, agentPipelineLayout, agentDescriptor.entryPoint ?? 'main');

      const blendBindGroupLayout = getOrCreateBindGroupLayout(device, blendDescriptor.id, 'compute', blendMetadata);
      const blendPipelineLayout = getOrCreatePipelineLayout(device, blendDescriptor.id, 'compute', blendBindGroupLayout);
      const blendPipeline = await getOrCreateComputePipeline(device, blendDescriptor.id, blendPipelineLayout, blendDescriptor.entryPoint ?? 'main');

      // Agent buffer management - both buffers stored, will swap each frame
      this.agentBuffers.current = 'a';

      // Create bind group for seed initialization (run once)
      const seedsBindGroup = device.createBindGroup({
        layout: seedsBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: { buffer: paramsBuffer } },
        ],
      });

      // Run seed initialization pass ONLY if seeds haven't been initialized yet
      if (!this.seedsInitialized) {
        console.log('DLA: Running seed initialization (FIRST TIME ONLY)');
        const seedEncoder = device.createCommandEncoder();
        const seedPass = seedEncoder.beginComputePass();
        seedPass.setPipeline(seedsPipeline);
        seedPass.setBindGroup(0, seedsBindGroup);
        const seedDispatchX = Math.ceil(width / 8);
        const seedDispatchY = Math.ceil(height / 8);
        seedPass.dispatchWorkgroups(seedDispatchX, seedDispatchY, 1);
        seedPass.end();
        
        // Copy seeds from outputBuffer to feedbackTexture
        const seedCopyBindGroup = device.createBindGroup({
          layout: bufferToTextureBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: outputBuffer } },
            { binding: 1, resource: feedbackTexture.createView() },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });
        const seedCopyPass = seedEncoder.beginComputePass();
        seedCopyPass.setPipeline(bufferToTexturePipeline);
        seedCopyPass.setBindGroup(0, seedCopyBindGroup);
        const copyDispatchX = Math.ceil(width / 8);
        const copyDispatchY = Math.ceil(height / 8);
        seedCopyPass.dispatchWorkgroups(copyDispatchX, copyDispatchY, 1);
        seedCopyPass.end();
        
        device.queue.submit([seedEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        this.seedsInitialized = true;
        console.log(`DLA: Seed initialization complete (seed_density=${this.userState.seed_density}, expected ~${Math.floor(width * height * this.userState.seed_density)} seeds)`);
        logInfo?.(`DLA: Seed initialization complete`);
      } else {
        console.log('DLA: Skipping seed initialization (already initialized)');
      }

      // Create bind groups for all 3 per-frame passes (will be updated in beforeDispatch)
      const initBindGroup = device.createBindGroup({
        layout: initBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTexture.createView() },
          { binding: 4, resource: { buffer: gliderBuffer } },
        ],
      });

      const agentBindGroup = device.createBindGroup({
        layout: agentBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTexture.createView() },
          { binding: 4, resource: { buffer: this.agentBuffers.a } }, // agent_state_in
          { binding: 5, resource: { buffer: this.agentBuffers.b } }, // agent_state_out
          { binding: 6, resource: { buffer: gliderBuffer } },
        ],
      });

      const blendBindGroup = device.createBindGroup({
        layout: blendBindGroupLayout,
        entries: [
          { binding: 0, resource: multiresResources.outputTexture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
          { binding: 3, resource: feedbackTexture.createView() },
          { binding: 4, resource: { buffer: gliderBuffer } },
        ],
      });

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: outputTexture.createView() },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });
      
      console.log('DLA: buffer-to-texture bind group created with outputBuffer:', outputBuffer);

      const blitBindGroup = device.createBindGroup({
        layout: multiresResources.blitBindGroupLayout,
        entries: [{ binding: 0, resource: outputTexture.createView() }],
      });

      console.log('DLA: Created outputTexture:', outputTexture);
      console.log('DLA: Created blitBindGroup with outputTexture, enabled:', this.userState.enabled);
      console.log('DLA: Multires outputTexture:', multiresResources.outputTexture);
      console.log('DLA: Are they the same?', outputTexture === multiresResources.outputTexture);

      const agentCount = this.agentBuffers.agentCount;
      const pixelWorkgroupSize = [8, 8, 1];
      const agentWorkgroupSize = [64, 1, 1];

      // Create multi-pass configuration
      const computePasses = [
        // Pass 0: Copy prev_texture to output_buffer (pixel-parallel)
        {
          pipeline: initPipeline,
          bindGroup: initBindGroup,
          workgroupSize: pixelWorkgroupSize,
          getDispatch: ({ width, height }) => [
            Math.max(1, Math.ceil(width / pixelWorkgroupSize[0])),
            Math.max(1, Math.ceil(height / pixelWorkgroupSize[1])),
            1
          ],
        },
        // Pass 1: Agent random walk and sticking (agent-parallel)
        {
          pipeline: agentPipeline,
          bindGroup: agentBindGroup,
          workgroupSize: agentWorkgroupSize,
          getDispatch: () => [
            Math.max(1, Math.ceil(agentCount / agentWorkgroupSize[0])),
            1,
            1
          ],
        },
        // Pass 2: Final blend with input (pixel-parallel)
        {
          pipeline: blendPipeline,
          bindGroup: blendBindGroup,
          workgroupSize: pixelWorkgroupSize,
          getDispatch: ({ width, height }) => [
            Math.max(1, Math.ceil(width / pixelWorkgroupSize[0])),
            Math.max(1, Math.ceil(height / pixelWorkgroupSize[1])),
            1
          ],
        },
      ];
      console.log('DLA: computePasses length', computePasses.length);

      this.resources = {
        // Store all 3 passes
        initDescriptor,
        agentDescriptor,
        blendDescriptor,
        initMetadata,
        agentMetadata,
        blendMetadata,
        initPipeline,
        agentPipeline,
        blendPipeline,
        initBindGroup,
        agentBindGroup,
        blendBindGroup,
        initBindGroupLayout,
        agentBindGroupLayout,
        blendBindGroupLayout,
        // Store seeds initialization resources for reuse
        seedsPipeline,
        seedsBindGroup,
        bufferToTexturePipeline,
        bufferToTextureBindGroupLayout,
        computePipeline: null, // Not used in multi-pass
        computeBindGroup: null, // Not used in multi-pass
        computePasses, // Multi-pass configuration
        paramsBuffer,
        paramsState,
        outputBuffer,
        gliderBuffer,
        outputTexture,
        feedbackTexture,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        blitBindGroup,
        workgroupSize: pixelWorkgroupSize,
        agentWorkgroupSize: agentWorkgroupSize,
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        width, // Store dimensions for seed reinit
        height,
        paramsDirty: false,
        device,
        bindingOffsets,
        shouldCopyOutputToPrev: true,
      };

      setStatus?.('DLA resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create DLA resources:', error);
      throw error;
    }
  }

  beforeDispatch({ device, multiresResources, encoder }) {
    if (!this.resources) return;
    
    // Reinitialize seeds if seed_density changed
    if (!this.seedsInitialized && this.resources.seedsPipeline && this.resources.seedsBindGroup) {
      console.log('DLA: Reinitializing seeds due to seed_density change');
      
      const { width, height, outputBuffer, feedbackTexture, paramsBuffer } = this.resources;
      
      // Run seed initialization pass
      const seedEncoder = device.createCommandEncoder();
      const seedPass = seedEncoder.beginComputePass();
      seedPass.setPipeline(this.resources.seedsPipeline);
      seedPass.setBindGroup(0, this.resources.seedsBindGroup);
      const seedDispatchX = Math.ceil(width / 8);
      const seedDispatchY = Math.ceil(height / 8);
      seedPass.dispatchWorkgroups(seedDispatchX, seedDispatchY, 1);
      seedPass.end();
      
      // Copy seeds from outputBuffer to feedbackTexture
      const seedCopyBindGroup = device.createBindGroup({
        layout: this.resources.bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: feedbackTexture.createView() },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });
      const seedCopyPass = seedEncoder.beginComputePass();
      seedCopyPass.setPipeline(this.resources.bufferToTexturePipeline);
      seedCopyPass.setBindGroup(0, seedCopyBindGroup);
      const copyDispatchX = Math.ceil(width / 8);
      const copyDispatchY = Math.ceil(height / 8);
      seedCopyPass.dispatchWorkgroups(copyDispatchX, copyDispatchY, 1);
      seedCopyPass.end();
      
      device.queue.submit([seedEncoder.finish()]);
      
      this.seedsInitialized = true;
      this._frameCount = 0; // Reset frame count so we don't copy old output over new seeds
      console.log(`DLA: Seed reinitialization complete (seed_density=${this.userState.seed_density})`);
    }
    
    // Recreate agent buffers if they were cleared (e.g., density changed)
    if (!this.agentBuffers) {
      this.agentBuffers = this.#initializeAgentBuffers(device, this.width, this.height);
      console.log('DLA: Recreated agent buffers due to parameter change');
      
      // Update the agent pass dispatch function to use new agent count
      if (this.resources.computePasses && this.resources.computePasses[1]) {
        const agentCount = this.agentBuffers.agentCount;
        const agentWorkgroupSize = [64, 1, 1];
        this.resources.computePasses[1].getDispatch = () => [
          Math.max(1, Math.ceil(agentCount / agentWorkgroupSize[0])),
          1,
          1
        ];
        console.log('DLA: Updated agent dispatch for', agentCount, 'agents');
      }
    }

    // Initialize frame counter
    if (!this._frameCount) {
      this._frameCount = 0;
      console.log('DLA beforeDispatch: FIRST CALL - skipping texture copy to preserve seeds');
    }

    // Skip texture copy on first frame to preserve initial seeds in feedbackTexture
    const copyWidth = Math.max(Math.trunc(this.width ?? 0), 0);
    const copyHeight = Math.max(Math.trunc(this.height ?? 0), 0);

    if (this._frameCount > 0 && copyWidth > 0 && copyHeight > 0) {
        encoder.copyTextureToTexture(
            { texture: this.resources.outputTexture },
            { texture: this.resources.feedbackTexture },
            { width: copyWidth, height: copyHeight, depthOrArrayLayers: 1 },
        );
    }
    
    this._frameCount++;

    const currentIsA = this.agentBuffers.current === 'a';
    const inputBuffer = currentIsA ? this.agentBuffers.a : this.agentBuffers.b;
    const outputBuffer = currentIsA ? this.agentBuffers.b : this.agentBuffers.a;

    // Recreate bind groups for all 3 passes with swapped agent buffers
    this.resources.initBindGroup = device.createBindGroup({
      layout: this.resources.initBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.resources.outputBuffer } },
        { binding: 2, resource: { buffer: this.resources.paramsBuffer } },
        { binding: 3, resource: this.resources.feedbackTexture.createView() },
        { binding: 4, resource: { buffer: this.resources.gliderBuffer } },
      ],
    });

    this.resources.agentBindGroup = device.createBindGroup({
      layout: this.resources.agentBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.resources.outputBuffer } },
        { binding: 2, resource: { buffer: this.resources.paramsBuffer } },
        { binding: 3, resource: this.resources.feedbackTexture.createView() },
        { binding: 4, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
        { binding: 6, resource: { buffer: this.resources.gliderBuffer } },
      ],
    });

    this.resources.blendBindGroup = device.createBindGroup({
      layout: this.resources.blendBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.resources.outputBuffer } },
        { binding: 2, resource: { buffer: this.resources.paramsBuffer } },
        { binding: 3, resource: this.resources.feedbackTexture.createView() },
        { binding: 4, resource: { buffer: this.resources.gliderBuffer } },
      ],
    });

    // Update bind groups in all compute passes
    if (this.resources.computePasses && Array.isArray(this.resources.computePasses)) {
      this.resources.computePasses[0].bindGroup = this.resources.initBindGroup;
      this.resources.computePasses[1].bindGroup = this.resources.agentBindGroup;
      // Pass 2 (final_blend) is currently disabled
      if (this.resources.computePasses[2]) {
        this.resources.computePasses[2].bindGroup = this.resources.blendBindGroup;
      }
    }
  }

  afterDispatch() {
    // Swap the current marker after dispatch completes
    if (!this.agentBuffers) return;
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }
}

export default DLAEffect;

// Multi-pass shader descriptors
export const additionalPasses = {
  'dla/init_seeds': {
    id: 'dla/init_seeds',
    label: 'init_seeds.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/dla/init_seeds.wgsl',
    resources: {
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 48 }
    }
  },
  'dla/init_from_prev': {
    id: 'dla/init_from_prev',
    label: 'init_from_prev.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/dla/init_from_prev.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 48 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      glider_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' }
    }
  },
  'dla/agent_walk': {
    id: 'dla/agent_walk',
    label: 'agent_walk.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/dla/agent_walk.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 48 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      agent_state_in: { kind: 'readOnlyStorageBuffer', size: 'custom' },
      agent_state_out: { kind: 'storageBuffer', size: 'custom' },
      glider_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' }
    }
  },
  'dla/final_blend': {
    id: 'dla/final_blend',
    label: 'final_blend.wgsl',
    stage: 'compute',
    entryPoint: 'main',
    url: '/shaders/effects/dla/final_blend.wgsl',
    resources: {
      input_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      output_buffer: { kind: 'storageBuffer', size: 'pixel-f32x4' },
      params: { kind: 'uniformBuffer', size: 48 },
      prev_texture: { kind: 'sampledTexture', format: 'rgba32float' },
      glider_buffer: { kind: 'readOnlyStorageBuffer', size: 'pixel-f32x4' }
    }
  }
};
