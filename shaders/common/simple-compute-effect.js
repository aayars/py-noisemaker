const RGBA_CHANNEL_COUNT = 4;

function toCamelCase(value) {
  return String(value ?? '')
    .split('_')
    .filter(Boolean)
    .map((part, index) => (index === 0 ? part.toLowerCase() : part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()))
    .join('');
}

function resolveDefaultValue(param) {
  if (!param) {
    return undefined;
  }
  if (param.type === 'boolean') {
    return Boolean(param.default);
  }
  if (typeof param.default !== 'undefined') {
    return Number(param.default);
  }
  if (param.type === 'int') {
    return 0;
  }
  if (param.type === 'float' || param.type === 'number') {
    return 0;
  }
  return undefined;
}

function clampNumeric(value, minValue, maxValue) {
  if (!Number.isFinite(value)) {
    return Number.isFinite(minValue) ? minValue : 0;
  }
  let clamped = value;
  if (Number.isFinite(minValue)) {
    clamped = Math.max(clamped, minValue);
  }
  if (Number.isFinite(maxValue)) {
    clamped = Math.min(clamped, maxValue);
  }
  return clamped;
}

function coerceValue(definition, rawValue) {
  if (!definition) {
    return rawValue;
  }

  switch (definition.type) {
    case 'boolean':
      return Boolean(rawValue);
    case 'int': {
      const numeric = clampNumeric(Math.round(Number(rawValue)), Number(definition.min), Number(definition.max));
      return Number.isFinite(numeric) ? numeric : 0;
    }
    case 'float':
    case 'number':
    default: {
      const numeric = Number(rawValue);
      const coerced = clampNumeric(numeric, Number(definition.min), Number(definition.max));
      return Number.isFinite(coerced) ? coerced : 0;
    }
  }
}

function floatsEqual(a, b) {
  return Math.abs(a - b) <= 1e-6;
}

class SimpleComputeEffect {
  constructor({ helpers } = {}) {
    const metadata = this.constructor.metadata;
    if (!metadata) {
      throw new Error('SimpleComputeEffect subclasses must define static metadata.');
    }

    this.helpers = helpers ?? {};
    this.metadata = metadata;
    this.effectId = metadata.id ?? this.constructor.id;
    if (!this.effectId) {
      throw new Error('Effect metadata is missing an id.');
    }

    const parameterList = Array.isArray(metadata.parameters) ? metadata.parameters : [];
    this.parameterDefinitions = new Map(parameterList.map((param) => [param.name, param]));
    this.parameterBindings = new Map(Object.entries(metadata.parameterBindings ?? {}));

    this.enabledParamName = this.#resolveEnabledParameterName();
    this.userState = {};
    parameterList.forEach((param) => {
      const defaultValue = resolveDefaultValue(param);
      if (typeof defaultValue !== 'undefined') {
        this.userState[param.name] = defaultValue;
      }
    });
    if (!this.parameterDefinitions.has(this.enabledParamName)) {
      this.userState[this.enabledParamName] = true;
    }

    this.resources = null;
    this.paramOffsets = new Map();
  }

  static get label() {
    return this.metadata?.label ?? this.metadata?.id ?? 'Effect';
  }

  get metadataLabel() {
    return this.metadata?.label ?? this.effectId;
  }

  getUIState() {
    const state = {};
    this.parameterDefinitions.forEach((_, name) => {
      state[name] = this.userState[name];
    });
    return state;
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
    } catch (error) {
      this.helpers.logWarn?.(`Failed to destroy resources for '${this.effectId}':`, error);
    }

    try {
      this.resources.outputTexture?.destroy?.();
    } catch (error) {
      this.helpers.logWarn?.(`Failed to destroy output texture for '${this.effectId}':`, error);
    }

    this.resources = null;
    this.paramOffsets.clear();
  }

  async ensureResources(context = {}) {
    const { device, width, height, multiresResources } = context;
    if (!device) {
      throw new Error(`${this.effectId} requires a GPUDevice.`);
    }
    if (!multiresResources?.outputTexture) {
      throw new Error(`${this.effectId} requires multires output texture.`);
    }

    const enabled = this.userState[this.enabledParamName] !== false;
    if (!enabled) {
      if (!this.resources || this.resources.computePipeline) {
        this.resources = this.#createDisabledResources(width, height, 'Effect disabled by user.');
      } else {
        this.resources.enabled = false;
        this.resources.textureWidth = width;
        this.resources.textureHeight = height;
      }
      return this.resources;
    }

    if (this.resources) {
      const sizeMatches = this.resources.textureWidth === width && this.resources.textureHeight === height;
      const sameDevice = this.resources.device === device;
      const validPipelines = Boolean(this.resources.computePipeline) && Boolean(this.resources.computeBindGroup);

      if (sameDevice && sizeMatches && validPipelines) {
        this.resources.enabled = true;
        return this.resources;
      }

      this.invalidateResources();
    }

    this.resources = await this.#createResources({ device, width, height, multiresResources });
    return this.resources;
  }

  async updateParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('SimpleComputeEffect.updateParams expects an object.');
    }

    const changed = [];

    for (const [name, value] of Object.entries(updates)) {
      if (name === this.enabledParamName) {
        const enabled = Boolean(value);
        if (Boolean(this.userState[name]) !== enabled) {
          this.userState[name] = enabled;
          changed.push(name);
          if (this.resources) {
            this.resources.enabled = enabled;
            if (enabled) {
              if (!this.resources.computePipeline) {
                this.invalidateResources();
              }
            } else if (this.resources.computePipeline || this.resources.computeBindGroup) {
              const { textureWidth, textureHeight } = this.resources;
              this.invalidateResources();
              this.resources = this.#createDisabledResources(textureWidth, textureHeight, 'Effect disabled by user.');
            }
          }
        }
        continue;
      }

      const definition = this.parameterDefinitions.get(name);
      if (!definition) {
        this.helpers.logWarn?.(`${this.effectId}: Unknown parameter '${name}'.`);
        continue;
      }

      const coerced = coerceValue(definition, value);
      const previous = this.userState[name];
      const differs = definition.type === 'boolean'
        ? Boolean(previous) !== Boolean(coerced)
        : !floatsEqual(Number(previous ?? 0), Number(coerced));

      if (!differs) {
        continue;
      }

      this.userState[name] = coerced;
      changed.push(name);

      const offset = this.paramOffsets.get(name);
      if (typeof offset === 'number' && this.resources?.paramsState) {
        this.resources.paramsState[offset] = Number(coerced);
        this.resources.paramsDirty = true;
      }
    }

    return { updated: changed };
  }

  getResourceCreationOptions({ multiresResources }) {
    return {
      inputTextures: {
        input_texture: multiresResources?.outputTexture ?? null,
      },
    };
  }

  async onResourcesCreated(resources, _context) {
    return resources;
  }

  #resolveEnabledParameterName() {
    return 'enabled';
  }

  #createDisabledResources(width, height, reason) {
    const numericWidth = Number(width);
    const numericHeight = Number(height);
    const safeWidth = Number.isFinite(numericWidth) ? numericWidth : 0;
    const safeHeight = Number.isFinite(numericHeight) ? numericHeight : 0;
    if (reason) {
      this.helpers.logWarn?.(`${this.effectId} disabled: ${reason}`);
    }

    return {
      descriptor: null,
      shaderMetadata: null,
      resourceSet: {
        destroyAll() {},
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
      textureWidth: safeWidth,
      textureHeight: safeHeight,
      paramsDirty: false,
      device: null,
      bindingOffsets: {},
    };
  }

  #resolveParamsFloatLength() {
    const sizeBytes = Number(this.metadata?.resources?.params?.size);
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      throw new Error(`${this.effectId} metadata must specify a positive params buffer size.`);
    }
    if (sizeBytes % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${this.effectId} params buffer size must be divisible by 4 bytes.`);
    }
    return sizeBytes / Float32Array.BYTES_PER_ELEMENT;
  }

  #resolveBindingOffset(name) {
    const binding = this.parameterBindings.get(name);
    if (!binding) {
      return null;
    }
    if (binding.kind === 'toggle') {
      return null;
    }
    if (binding.buffer && binding.buffer !== 'params') {
      throw new Error(`${this.effectId} parameter '${name}' must target the params buffer.`);
    }
    if (typeof binding.offset !== 'undefined') {
      const offset = Number(binding.offset);
      if (!Number.isFinite(offset)) {
        throw new Error(`${this.effectId} parameter '${name}' binding offset must be a finite number.`);
      }
      return offset;
    }
    return null;
  }

  #initialValueForBinding(name, context) {
    switch (name) {
      case 'width':
        return context.width;
      case 'height':
        return context.height;
      case 'channel_count':
        return RGBA_CHANNEL_COUNT;
      case 'time':
        return 0;
      default: {
        const definition = this.parameterDefinitions.get(name);
        if (!definition) {
          return 0;
        }
        const current = this.userState[name];
        const value = typeof current === 'undefined' ? resolveDefaultValue(definition) : current;
        return Number(coerceValue(definition, value));
      }
    }
  }

  #createBindingOffsetsMap() {
    const map = {};
    this.paramOffsets.forEach((offset, name) => {
      if (typeof offset !== 'number') {
        return;
      }
      map[name] = offset;
      map[toCamelCase(name)] = offset;
    });
    return map;
  }

  #populateParamsState(paramsState, context) {
    this.paramOffsets.clear();

    this.parameterBindings.forEach((binding, name) => {
      const offset = this.#resolveBindingOffset(name);
      if (offset === null || typeof offset === 'undefined') {
        return;
      }
      if (offset < 0 || offset >= paramsState.length) {
        throw new Error(`${this.effectId} binding offset for '${name}' exceeds params buffer length.`);
      }
      const value = this.#initialValueForBinding(name, context);
      paramsState[offset] = Number(value ?? 0);
      this.paramOffsets.set(name, offset);
    });

    if (!this.paramOffsets.has('width') && paramsState.length > 0) {
      paramsState[0] = context.width;
      this.paramOffsets.set('width', 0);
    }
    if (!this.paramOffsets.has('height') && paramsState.length > 1) {
      paramsState[1] = context.height;
      this.paramOffsets.set('height', 1);
    }
    if (!this.paramOffsets.has('channel_count') && paramsState.length > 2) {
      paramsState[2] = RGBA_CHANNEL_COUNT;
      this.paramOffsets.set('channel_count', 2);
    }
  }

  async #createResources(context) {
    const { device, width, height, multiresResources } = context;
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

    setStatus?.(`Preparing ${this.metadataLabel} resources…`);

    try {
      const descriptor = getShaderDescriptor(this.effectId);
      const shaderMetadata = await getShaderMetadataCached(this.effectId);
      warnOnNonContiguousBindings?.(shaderMetadata.bindings, descriptor.id);

      const resourceOptions = this.getResourceCreationOptions(context);
      const resourceSet = createShaderResourceSet(device, descriptor, shaderMetadata, width, height, resourceOptions);

      const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
      const computePipeline = await getOrCreateComputePipeline(device, descriptor.id, computePipelineLayout, descriptor.entryPoint ?? 'main');
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: createBindGroupEntriesFromResources(shaderMetadata.bindings, resourceSet),
      });

      const paramsBuffer = resourceSet.buffers.params;
      const outputBuffer = resourceSet.buffers.output_buffer;
      if (!paramsBuffer || !outputBuffer) {
        logWarn?.(`${this.effectId}: Missing required GPU buffers.`);
        return this.#createDisabledResources(width, height, 'Missing params or output buffer.');
      }

      const paramsLength = this.#resolveParamsFloatLength();
      const paramsState = new Float32Array(paramsLength);
      this.#populateParamsState(paramsState, context);
      device.queue.writeBuffer(paramsBuffer, 0, paramsState);

      const outputTexture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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

      const bindingOffsets = this.#createBindingOffsetsMap();

      const resources = {
        descriptor,
        shaderMetadata,
        resourceSet,
        computeBindGroupLayout,
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
        enabled: true,  // FORCE ENABLED
        textureWidth: width,
        textureHeight: height,
        paramsDirty: false,
        device,
        bindingOffsets,
      };

      const augmented = await this.onResourcesCreated(resources, context);
      const finalResources = augmented ?? resources;
      logInfo?.(`${this.metadataLabel} resources ready.`);
      return finalResources;
    } catch (error) {
      logWarn?.(`Failed to create resources for '${this.effectId}':`, error);
      return this.#createDisabledResources(width, height, error?.message ?? 'Resource creation failed.');
    }
  }
}

export default SimpleComputeEffect;
