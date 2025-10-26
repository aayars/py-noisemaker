import meta from './meta.json' with { type: 'json' };

const RETRY_BASE_DELAY_MS = 2000;
const RETRY_MAX_DELAY_MS = 30000;
const RGBA_CHANNEL_COUNT = 4;
const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);
const PARAMETER_DEFINITIONS = new Map((meta.parameters ?? []).map((param) => [param.name, param]));
const PARAMETER_BINDINGS = new Map(Object.entries(meta.parameterBindings ?? {}));

function toFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function getDefaultParamValue(name, fallback) {
  const param = PARAMETER_DEFINITIONS.get(name);
  if (!param) {
    return fallback;
  }
  return param.default ?? fallback;
}

const PARAMETER_DEFAULTS = Object.freeze({
  speed: getDefaultParamValue('speed', 1.0),
  enabled: getDefaultParamValue('enabled', true),
});

class FrameEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.device = null;
    this.width = 0;
    this.height = 0;
    this.resources = null;
    this.retryDelayMs = RETRY_BASE_DELAY_MS;
    this.userState = {
      speed: PARAMETER_DEFAULTS.speed,
      enabled: Boolean(PARAMETER_DEFAULTS.enabled),
    };
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
        logWarn?.('Failed to destroy frame resources during invalidation:', error);
      }
    }

    if (resources.outputTexture?.destroy) {
      try {
        resources.outputTexture.destroy();
      } catch (error) {
        logWarn?.('Failed to destroy frame output texture during invalidation:', error);
      }
    }

    this.resources = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('FrameEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('FrameEffect.ensureResources requires multires output texture.');
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
      const hasPipeline = Boolean(resources.computePipeline);
      if (!hasPipeline) {
        const now = Date.now();
        const retryAt = resources.retryAt ?? 0;
        if (now < retryAt && sizeMatches) {
          return resources;
        }
        this.invalidateResources();
      } else if (sizeMatches) {
        return resources;
      } else {
        this.invalidateResources();
      }
    }

    return this.#createResources({ device, width, height, multiresResources });
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('FrameEffect.updateParams expects an object.');
    }

    const updated = [];
    const { logWarn } = this.helpers;

    Object.keys(updates).forEach((name) => {
      if (!PARAMETER_DEFINITIONS.has(name)) {
        logWarn?.(`updateFrameParams: Unknown parameter '${name}'.`);
      }
    });

    if (hasOwn(updates, 'speed')) {
      const numeric = toFiniteNumber(updates.speed);
      if (numeric === null) {
        logWarn?.('updateFrameParams: speed must be a finite number.');
      } else {
        const { min, max } = this.#getNumericBounds('speed');
        const clamped = Math.min(Math.max(numeric, min), max);
        this.userState.speed = clamped;
        if (this.resources?.paramsState) {
          const offset = this.resources?.bindingOffsets?.speed ?? this.#getBindingOffset('speed');
          this.resources.paramsState[offset] = clamped;
          this.resources.paramsDirty = true;
        }
        updated.push('speed');
      }
    }

    if (hasOwn(updates, 'enabled')) {
      this.#getParameterDefinition('enabled');
      const enabled = Boolean(updates.enabled);
      this.userState.enabled = enabled;
      if (this.resources) {
        this.resources.enabled = enabled;
        if (enabled && !this.resources.computePipeline) {
          this.resources.retryAt = Date.now();
          this.invalidateResources();
        }
      }
      updated.push('enabled');
    }

    return { updated };
  }

  getUIState() {
    return {
      speed: this.userState.speed,
      enabled: this.userState.enabled,
    };
  }

  destroy() {
    this.invalidateResources();
    this.device = null;
    this.width = 0;
    this.height = 0;
    this.retryDelayMs = RETRY_BASE_DELAY_MS;
  }

  #getParameterDefinition(name) {
    const definition = PARAMETER_DEFINITIONS.get(name);
    if (!definition) {
      throw new Error(`Frame metadata missing parameter definition for '${name}'.`);
    }
    return definition;
  }

  #getNumericBounds(name) {
    const definition = this.#getParameterDefinition(name);
    const min = Number(definition.min);
    const max = Number(definition.max);
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      throw new Error(`Frame parameter '${name}' must define finite min and max bounds.`);
    }
    if (min > max) {
      throw new Error(`Frame parameter '${name}' metadata has min greater than max.`);
    }
    return { min, max };
  }

  #getBindingOffset(name) {
    const binding = PARAMETER_BINDINGS.get(name);
    if (!binding) {
      throw new Error(`Frame metadata missing parameter binding for '${name}'.`);
    }
    if (binding.buffer !== 'params') {
      throw new Error(`Frame parameter '${name}' binding must target the 'params' buffer.`);
    }
    const offset = Number(binding.offset);
    if (!Number.isFinite(offset)) {
      throw new Error(`Frame parameter '${name}' binding offset must be a finite number.`);
    }
    return offset;
  }

  #getParamsFloatLength() {
    const sizeBytes = Number(meta.resources?.params?.size);
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      throw new Error('Frame metadata must provide a positive params buffer size.');
    }
    if (sizeBytes % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error('Frame params buffer size must be divisible by 4 bytes.');
    }
    return sizeBytes / Float32Array.BYTES_PER_ELEMENT;
  }

  async #createResources({ device, width, height, multiresResources }) {
    const {
      logInfo,
      logWarn,
      setStatus,
      getShaderDescriptor,
      getShaderMetadataCached,
      warnOnNonContiguousBindings,
      createShaderResourceSet,
      createBindGroupEntriesFromResources,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    setStatus?.('Creating frame resources…');

    try {
      logInfo?.('Attempting to load frame shader descriptor...');
      const descriptor = getShaderDescriptor('frame');
      logInfo?.(`Frame descriptor loaded: ${descriptor.label}`);
      const shaderMetadata = await getShaderMetadataCached('frame');
      logInfo?.(`Frame metadata parsed, bindings count: ${shaderMetadata.bindings.length}`);
      warnOnNonContiguousBindings?.(shaderMetadata.bindings, descriptor.id);

      logInfo?.(`Creating frame resource set with input texture: ${Boolean(multiresResources.outputTexture)}`);
      const resourceSet = createShaderResourceSet(device, descriptor, shaderMetadata, width, height, {
        inputTextures: {
          inputTexture: multiresResources.outputTexture,
        },
      });
      logInfo?.(`Frame resource set created. Buffers: ${Object.keys(resourceSet.buffers).join(', ')}, Textures: ${Object.keys(resourceSet.textures).join(', ')}`);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
      const computePipeline = await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: createBindGroupEntriesFromResources(shaderMetadata.bindings, resourceSet),
      });

      const paramsBuffer = resourceSet.buffers.params;
      const outputBuffer = resourceSet.buffers.outputBuffer;

      if (!paramsBuffer || !outputBuffer) {
        const missing = !paramsBuffer ? 'params buffer' : 'output buffer';
        return this.#createDisabledResources({ width, height }, missing ? `Missing ${missing} in resource set.` : undefined);
      }

      const paramsLength = this.#getParamsFloatLength();
      const paramsState = new Float32Array(paramsLength);

      const bindingOffsets = {
        width: this.#getBindingOffset('width'),
        height: this.#getBindingOffset('height'),
        channelCount: this.#getBindingOffset('channel_count'),
        time: this.#getBindingOffset('time'),
        speed: this.#getBindingOffset('speed'),
      };

      Object.entries(bindingOffsets).forEach(([name, offset]) => {
        if (offset < 0 || offset >= paramsLength) {
          throw new Error(`Frame parameter '${name}' binding offset ${offset} exceeds params buffer length ${paramsLength}.`);
        }
      });

      paramsState[bindingOffsets.width] = width;
      paramsState[bindingOffsets.height] = height;
      paramsState[bindingOffsets.channelCount] = RGBA_CHANNEL_COUNT;
      paramsState[bindingOffsets.time] = 0;
      paramsState[bindingOffsets.speed] = this.userState.speed;

      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });

      const storageView = outputTexture.createView();
      const sampleView = outputTexture.createView();

      const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);
      const bufferToTextureBindGroup = device.createBindGroup({
        layout: bufferToTextureBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: storageView },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const blitBindGroupLayout = multiresResources.blitBindGroupLayout;
      const blitBindGroup = device.createBindGroup({
        layout: blitBindGroupLayout,
        entries: [
          { binding: 0, resource: sampleView },
        ],
      });

      this.resources = {
        descriptor,
        shaderMetadata,
        resourceSet,
        computePipeline,
        computeBindGroup,
        paramsBuffer,
        paramsState,
        outputBuffer,
        outputTexture,
        storageView,
        sampleView,
        bufferToTexturePipeline,
        bufferToTextureBindGroup,
        bufferToTextureWorkgroupSize: [8, 8, 1],
        blitBindGroup,
        workgroupSize: shaderMetadata.workgroupSize ?? [8, 8, 1],
        enabled: this.userState.enabled,
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        retryAt: undefined,
        lastFailure: undefined,
        lastFailureReason: undefined,
        device,
        bindingOffsets,
      };

      this.retryDelayMs = RETRY_BASE_DELAY_MS;
      setStatus?.('Frame resources ready.');
      return this.resources;
    } catch (error) {
      logWarn?.('Failed to create frame resources:', error);
      return this.#createDisabledResources({ width, height }, error?.message ?? error);
    }
  }

  #createDisabledResources({ width, height }, reason) {
    const { logWarn, setStatus } = this.helpers;
    if (reason) {
      logWarn?.(`Frame effect disabled: ${reason}`);
    } else {
      logWarn?.('Frame effect disabled.');
    }

    setStatus?.('Frame effect unavailable. Using generator output directly.');

    const now = Date.now();
    const retryAt = now + this.retryDelayMs;
    this.retryDelayMs = Math.min(this.retryDelayMs * 2, RETRY_MAX_DELAY_MS);

    const stub = {
      descriptor: null,
      shaderMetadata: null,
      resourceSet: {
        destroyAll() {
          // No-op for disabled state.
        },
      },
      computePipeline: null,
      computeBindGroup: null,
      paramsBuffer: null,
      paramsState: null,
      outputBuffer: null,
      outputTexture: null,
      storageView: null,
      sampleView: null,
      bufferToTexturePipeline: null,
      bufferToTextureBindGroup: null,
      bufferToTextureWorkgroupSize: [8, 8, 1],
      blitBindGroup: null,
      workgroupSize: [8, 8, 1],
      enabled: false,
      textureWidth: width,
      textureHeight: height,
      paramsDirty: false,
      retryAt,
      lastFailure: now,
      lastFailureReason: reason ?? 'Unknown reason',
      device: this.device,
    };

    this.resources = stub;
    return stub;
  }
}

export default FrameEffect;
