const WEBGPU_ENABLE_HINT = 'WebGPU not available. Enable chrome://flags/#enable-unsafe-webgpu and restart the browser.';

function createWebGPURuntime(config = {}) {
  const {
    canvas,
    getShaderDescriptor,
    parseShaderMetadata,
    logInfo = () => {},
    logWarn = () => {},
    logError = () => {},
    setStatus = () => {},
    fatal = (message) => {
      throw new Error(typeof message === 'string' ? message : String(message));
    },
    onCachesCleared,
    onDeviceInvalidated,
  } = config;

  if (!canvas) {
    throw new Error('createWebGPURuntime requires a canvas element.');
  }
  if (typeof getShaderDescriptor !== 'function') {
    throw new Error('createWebGPURuntime requires getShaderDescriptor().');
  }
  if (typeof parseShaderMetadata !== 'function') {
    throw new Error('createWebGPURuntime requires parseShaderMetadata().');
  }

  const shaderSourceCache = new Map();
  const shaderMetadataCache = new Map();
  const shaderModuleCache = new Map();
  const bindGroupLayoutCache = new Map();
  const pipelineLayoutCache = new Map();
  const computePipelineCache = new Map();
  const blitShaderModuleCache = new Map();
  const bufferToTexturePipelineCache = new WeakMap();
  const pipelineCache = new Map();
  const warnedShaders = new Set();

  let cachedDevice = null;
  let cachedWebGPUState = null;
  let webgpuStateInitPromise = null;
  let singletonWebGPUInit = null;

  function clearGPUObjectCaches() {
    logInfo('Clearing all GPU object caches');
    shaderModuleCache.clear();
    bindGroupLayoutCache.clear();
    pipelineLayoutCache.clear();
    computePipelineCache.clear();
    blitShaderModuleCache.clear();
    pipelineCache.clear();
    onCachesCleared?.();
  }

  function alignTo(value, multiple) {
    if (multiple <= 0) {
      return value;
    }
    return Math.ceil(value / multiple) * multiple;
  }

  function resolveUniformBufferSize(size) {
    if (typeof size === 'number' && Number.isFinite(size) && size > 0) {
      return alignTo(Math.max(size, 256), 256);
    }
    return 256;
  }

  function resolveStorageBufferSize(sizeDescriptor, width, height) {
    if (typeof sizeDescriptor === 'number' && Number.isFinite(sizeDescriptor) && sizeDescriptor > 0) {
      return alignTo(sizeDescriptor, 16);
    }
    if (sizeDescriptor === 'pixel-f32x4') {
      const pixels = Math.max(width * height, 1);
      return alignTo(pixels * 4 * 4, 16);
    }
    if (sizeDescriptor === 'pixel-sort-f32x4') {
      const maxDim = Math.max(Math.max(width, height), 1);
      const want = Math.min(Math.max(maxDim * 2, maxDim, 1), 4096);
      const pixels = Math.max(want * want, 1);
      return alignTo(pixels * 4 * 4, 16);
    }
    if (sizeDescriptor === 'dynamic-histogram') {
      const bins = Math.max(Math.max(width, height), 1);
      return alignTo(bins * Uint32Array.BYTES_PER_ELEMENT, 16);
    }
    return 16;
  }

  function ensureCachesMatchDevice(device) {
    if (cachedDevice !== device) {
      logWarn(`Device changed! Old: ${cachedDevice ? 'exists' : 'null'}, New: ${device ? 'exists' : 'null'}, Same object: ${cachedDevice === device}`);
      clearGPUObjectCaches();
      cachedDevice = device;
    }
  }

  async function loadShaderSource(shaderId) {
    if (shaderSourceCache.has(shaderId)) {
      return shaderSourceCache.get(shaderId);
    }

    let descriptor;
    try {
      descriptor = getShaderDescriptor(shaderId);
    } catch (error) {
      fatal(error?.message ?? error);
    }

    let response;
    try {
      const url = `${descriptor.url}?t=${Date.now()}`;
      response = await fetch(url);
    } catch (error) {
      fatal(`Failed to fetch ${descriptor.label ?? shaderId}: ${error?.message ?? error}`);
    }

    if (!response?.ok) {
      fatal(`Failed to fetch ${descriptor.label ?? shaderId}: ${response?.status ?? 'Request failed'}`);
    }

    const source = await response.text();
    shaderSourceCache.set(shaderId, source);
    return source;
  }

  async function getShaderMetadataCached(shaderId) {
    if (shaderMetadataCache.has(shaderId)) {
      return shaderMetadataCache.get(shaderId);
    }
    const source = await loadShaderSource(shaderId);
    const metadata = parseShaderMetadata(source);
    shaderMetadataCache.set(shaderId, metadata);
    return metadata;
  }

  async function compileShaderModuleWithValidation(device, code, { label } = {}) {
    const descriptor = label ? { code, label } : { code };
    let shaderModule;
    try {
      shaderModule = device.createShaderModule(descriptor);
    } catch (error) {
      fatal(`Failed to create ${label ?? 'unnamed'} shader module: ${error?.message ?? error}`);
    }

    if (shaderModule?.getCompilationInfo) {
      try {
        const info = await shaderModule.getCompilationInfo();
        const messages = info?.messages ?? [];
        const errors = messages.filter((message) => message.type === 'error');
        const warnings = messages.filter((message) => message.type === 'warning');

        if (warnings.length > 0) {
          warnings.forEach((warning) => {
            const loc = typeof warning.lineNum === 'number' ? `Line ${warning.lineNum}: ` : '';
            logWarn(`[Shader Warning${label ? `: ${label}` : ''}] ${loc}${warning.message}`);
          });
        }

        if (errors.length > 0) {
          const details = errors
            .map((message) => {
              const lineInfo = typeof message.lineNum === 'number' ? `Line ${message.lineNum}: ` : '';
              return `${lineInfo}${message.message}`;
            })
            .join('\n');
          fatal(`${label ?? 'Shader'} compilation failed:\n${details}`);
        }
      } catch (error) {
        fatal(`Failed to validate ${label ?? 'shader'} compilation: ${error?.message ?? error}`);
      }
    }

    return shaderModule;
  }

  async function getOrCreateShaderModule(device, shaderId) {
    ensureCachesMatchDevice(device);
    if (shaderModuleCache.has(shaderId)) {
      return shaderModuleCache.get(shaderId);
    }
    const descriptor = getShaderDescriptor(shaderId);
    const source = await loadShaderSource(shaderId);
    const module = await compileShaderModuleWithValidation(device, source, { label: descriptor.label });
    shaderModuleCache.set(shaderId, module);
    return module;
  }

  function createBindGroupLayoutEntriesFromMetadata(bindings, stage) {
    const visibility = stage === 'render' ? GPUShaderStage.FRAGMENT : GPUShaderStage.COMPUTE;
    return bindings
      .filter((binding) => binding.group === 0)
      .map((binding) => {
        if (binding.resource === 'uniformBuffer') {
          return {
            binding: binding.binding,
            visibility,
            buffer: { type: 'uniform' },
          };
        }

        if (binding.resource === 'storageBuffer') {
          return {
            binding: binding.binding,
            visibility,
            buffer: { type: 'storage' },
          };
        }

        if (binding.resource === 'readOnlyStorageBuffer') {
          return {
            binding: binding.binding,
            visibility,
            buffer: { type: 'read-only-storage' },
          };
        }

        if (binding.resource === 'storageTexture') {
          return {
            binding: binding.binding,
            visibility,
            storageTexture: {
              access: binding.storageTextureAccess ?? 'write-only',
              format: binding.storageTextureFormat ?? 'rgba32float',
            },
          };
        }

        if (binding.resource === 'sampledTexture') {
          return {
            binding: binding.binding,
            visibility,
            texture: { sampleType: 'unfilterable-float' },
          };
        }

        if (binding.resource === 'sampler') {
          return {
            binding: binding.binding,
            visibility,
            sampler: {},
          };
        }

        fatal(`Unsupported binding resource type for ${binding.name} (binding ${binding.binding}).`);
        return null;
      })
      .filter(Boolean)
      .sort((a, b) => a.binding - b.binding);
  }

  function createShaderResourceSet(device, descriptor, metadata, width, height, options = {}) {
    const buffers = {};
    const textures = {};
    const samplers = {};
    const destroyables = [];
    let destroyed = false;
    const templates = descriptor.resources ?? {};
    const groupZeroBindings = metadata.bindings.filter((binding) => binding.group === 0);
    const providedTextures = options.inputTextures ?? {};
    const providedSamplers = options.samplers ?? {};

    for (const binding of groupZeroBindings) {
      const template = templates[binding.name];

      if (binding.resource === 'uniformBuffer') {
        if (!template?.size) {
          logWarn(`No resource template size for uniform buffer '${binding.name}' in shader '${descriptor.id}'. Defaulting to 256 bytes.`);
        }
        const size = resolveUniformBufferSize(template?.size);
        const buffer = device.createBuffer({
          size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        buffers[binding.name] = buffer;
        destroyables.push(buffer);
        continue;
      }

      if (binding.resource === 'storageBuffer' || binding.resource === 'readOnlyStorageBuffer') {
        const size = resolveStorageBufferSize(template?.size, width, height);
        const buffer = device.createBuffer({
          size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        buffers[binding.name] = buffer;
        destroyables.push(buffer);
        continue;
      }

      if (binding.resource === 'storageTexture') {
        const format = template?.format ?? binding.storageTextureFormat ?? 'rgba32float';
        const texture = device.createTexture({
          size: { width, height, depthOrArrayLayers: 1 },
          format,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        textures[binding.name] = texture;
        destroyables.push(texture);
        continue;
      }

      if (binding.resource === 'sampledTexture') {
        const provided = providedTextures[binding.name];
        if (provided) {
          textures[binding.name] = provided;
        } else {
          const format = template?.format ?? 'rgba32float';
          const texture = device.createTexture({
            size: { width, height, depthOrArrayLayers: 1 },
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
          textures[binding.name] = texture;
          destroyables.push(texture);
        }
        continue;
      }

      if (binding.resource === 'sampler') {
        const provided = providedSamplers[binding.name];
        if (provided) {
          samplers[binding.name] = provided;
        } else {
          const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
          samplers[binding.name] = sampler;
        }
        continue;
      }
    }

    return {
      buffers,
      textures,
      samplers,
      destroyAll() {
        if (destroyed) {
          return;
        }
        destroyed = true;
        for (const resource of destroyables) {
          if (resource?.destroy) {
            try {
              resource.destroy();
            } catch (error) {
              logWarn('Failed to destroy GPU resource during cleanup:', error);
            }
          }
        }
      },
    };
  }

  function createBindGroupEntriesFromResources(bindings, resourceSet) {
    return bindings
      .filter((binding) => binding.group === 0)
      .map((binding) => {
        if (binding.resource === 'uniformBuffer' || binding.resource === 'storageBuffer' || binding.resource === 'readOnlyStorageBuffer') {
          const buffer = resourceSet.buffers[binding.name];
          if (!buffer) {
            fatal(`Missing GPU buffer for binding ${binding.name}.`);
          }
          return { binding: binding.binding, resource: { buffer } };
        }

        if (binding.resource === 'storageTexture' || binding.resource === 'sampledTexture') {
          const texture = resourceSet.textures[binding.name];
          if (!texture) {
            fatal(`Missing GPU texture for binding ${binding.name}.`);
          }
          return { binding: binding.binding, resource: texture.createView() };
        }

        if (binding.resource === 'sampler') {
          const sampler = resourceSet.samplers?.[binding.name];
          if (!sampler) {
            fatal(`Missing sampler for binding ${binding.name}.`);
          }
          return { binding: binding.binding, resource: sampler };
        }

        fatal(`Unsupported bind group entry resource for ${binding.name}.`);
        return null;
      })
      .filter(Boolean)
      .sort((a, b) => a.binding - b.binding);
  }

  function warnOnNonContiguousBindings(bindings, shaderId) {
    if (warnedShaders.has(shaderId)) {
      return;
    }
    const groupZero = bindings.filter((b) => b.group === 0).map((b) => b.binding).sort((a, b) => a - b);
    for (let i = 1; i < groupZero.length; i += 1) {
      if (groupZero[i] !== groupZero[i - 1] + 1) {
        logWarn(`Non-contiguous binding indices detected for shader '${shaderId}': [${groupZero.join(', ')}]`);
        warnedShaders.add(shaderId);
        break;
      }
    }
  }

  function getOrCreateBindGroupLayout(device, shaderId, stage, metadata) {
    ensureCachesMatchDevice(device);
    const cacheKey = `${shaderId}|${stage}`;
    if (bindGroupLayoutCache.has(cacheKey)) {
      return bindGroupLayoutCache.get(cacheKey);
    }
    const layout = device.createBindGroupLayout({
      entries: createBindGroupLayoutEntriesFromMetadata(metadata.bindings, stage),
    });
    bindGroupLayoutCache.set(cacheKey, layout);
    return layout;
  }

  function getOrCreatePipelineLayout(device, shaderId, stage, bindGroupLayout) {
    ensureCachesMatchDevice(device);
    const cacheKey = `${shaderId}|${stage}`;
    if (pipelineLayoutCache.has(cacheKey)) {
      return pipelineLayoutCache.get(cacheKey);
    }
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    pipelineLayoutCache.set(cacheKey, layout);
    return layout;
  }

  async function getOrCreateComputePipeline(device, shaderId, pipelineLayout, entryPoint) {
    ensureCachesMatchDevice(device);
    const normalizedEntryPoint = entryPoint ?? 'main';
    const cacheKey = `compute|${shaderId}|${normalizedEntryPoint}`;
    if (computePipelineCache.has(cacheKey)) {
      return computePipelineCache.get(cacheKey);
    }
    const module = await getOrCreateShaderModule(device, shaderId);
    device.pushErrorScope('validation');
    let pipeline;
    try {
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: normalizedEntryPoint },
      });
    } catch (error) {
      await device.popErrorScope();
      fatal(`Failed to create compute pipeline: ${error?.message ?? error}`);
    }
    const err = await device.popErrorScope();
    if (err) {
      fatal(`Compute pipeline validation failed: ${err?.message ?? err}`);
    }
    computePipelineCache.set(cacheKey, pipeline);
    return pipeline;
  }

  const BUFFER_TO_TEXTURE_SHADER = `struct AberrationParams {
  size : vec4<f32>,
  anim : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> input_buffer : array<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : AberrationParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(params.size.x, 0.0));
  let height : u32 = u32(max(params.size.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let index : u32 = (gid.y * width + gid.x) * 4u;
  let color : vec4<f32> = vec4<f32>(
    input_buffer[index + 0u],
    input_buffer[index + 1u],
    input_buffer[index + 2u],
    input_buffer[index + 3u]
  );

  textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}`;

  async function getBufferToTexturePipeline(device) {
    let entry = bufferToTexturePipelineCache.get(device);
    if (entry) {
      return entry;
    }

    const module = await compileShaderModuleWithValidation(device, BUFFER_TO_TEXTURE_SHADER, { label: 'buffer-to-texture shader' });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    device.pushErrorScope('validation');
    let pipeline;
    try {
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: 'main' },
      });
    } catch (error) {
      await device.popErrorScope();
      fatal(`Failed to create buffer-to-texture pipeline: ${error?.message ?? error}`);
    }

    const pipelineError = await device.popErrorScope();
    if (pipelineError) {
      fatal(`Buffer-to-texture pipeline validation failed: ${pipelineError?.message ?? pipelineError}`);
    }

    entry = { pipeline, bindGroupLayout };
    bufferToTexturePipelineCache.set(device, entry);
    return entry;
  }

  const FLAT_COLOR_SHADER = `@vertex
fn vertex_main(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4<f32> {
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}`;

  async function renderFlatColor(options = {}) {
    const { alphaMode, clearColor } = options ?? {};

    setStatus('Preparing flat color render…');

    const { device, webgpuContext } = await getWebGPUState({ alphaMode });

    const cacheKey = `${webgpuContext.format}|${webgpuContext.alphaMode ?? 'opaque'}`;
    let pipeline = pipelineCache.get(cacheKey);

    if (!pipeline) {
      const shaderModule = await compileShaderModuleWithValidation(device, FLAT_COLOR_SHADER, { label: 'flat-color shader' });
      try {
        pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module: shaderModule, entryPoint: 'vertex_main' },
          fragment: {
            module: shaderModule,
            entryPoint: 'fragment_main',
            targets: [{ format: webgpuContext.format }],
          },
          primitive: { topology: 'triangle-list' },
        });
      } catch (error) {
        fatal(`Failed to create render pipeline: ${error?.message ?? error}`);
      }

      pipelineCache.set(cacheKey, pipeline);
    }

    let encoder;
    try {
      encoder = device.createCommandEncoder();
    } catch (error) {
      fatal(`Failed to create command encoder: ${error?.message ?? error}`);
    }

    const textureView = webgpuContext.getCurrentTextureView();
    const attachment = {
      view: textureView,
      loadOp: 'clear',
      clearValue: clearColor ?? { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    };

    let pass;
    try {
      pass = encoder.beginRenderPass({ colorAttachments: [attachment] });
    } catch (error) {
      fatal(`Failed to begin render pass: ${error?.message ?? error}`);
    }

    pass.setPipeline(pipeline);
    pass.draw(3, 1, 0, 0);
    pass.end();

    let commandBuffer;
    try {
      commandBuffer = encoder.finish();
    } catch (error) {
      fatal(`Failed to finalize GPU commands: ${error?.message ?? error}`);
    }

    try {
      device.queue.submit([commandBuffer]);
    } catch (error) {
      fatal(`Failed to submit GPU work: ${error?.message ?? error}`);
    }

    setStatus('Rendered flat color.');

    return {
      format: webgpuContext.format,
      alphaMode: webgpuContext.alphaMode,
    };
  }

  class WebGPUContext {
    constructor({ adapter, device, canvasElement, onDeviceLost, onContextLost }) {
      if (!canvasElement) {
        fatal('WebGPUContext requires a canvas element.');
      }
      if (!device) {
        fatal('WebGPUContext requires a GPU device.');
      }

      this.adapter = adapter ?? null;
      this.device = device;
      this.queue = device.queue;
      this.canvas = canvasElement;
      this.context = null;
      this.format = null;
      this.alphaMode = 'opaque';
      this._onContextLost = onContextLost;
      this._onDeviceLost = onDeviceLost;
      this._contextLostListener = null;

      if (device?.lost && typeof device.lost.then === 'function') {
        device.lost
          .then((info) => {
            if (typeof this._onDeviceLost === 'function') {
              this._onDeviceLost(info);
            } else {
              const message = info?.message ?? 'Device lost for unknown reasons.';
              fatal(`WebGPU device lost: ${message}`);
            }
          })
          .catch((error) => {
            fatal(`WebGPU device lost: ${error?.message ?? error}`);
          });
      }
    }

    configureCanvas(options = {}) {
      const alphaMode = options.alphaMode ?? 'opaque';

      if (!this.canvas) {
        fatal('WebGPUContext has no canvas to configure.');
      }

      if (!navigator.gpu?.getPreferredCanvasFormat) {
        fatal('navigator.gpu.getPreferredCanvasFormat() is unavailable.');
      }

      const context = this.canvas.getContext('webgpu');
      if (!context) {
        fatal('Unable to acquire WebGPU canvas context.');
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device: this.device, format, alphaMode });

      this.context = context;
      this.format = format;
      this.alphaMode = alphaMode;

      if (!this._contextLostListener) {
        this._contextLostListener = (event) => {
          event?.preventDefault?.();
          logError('WebGPU canvas context lost.', event);
          if (typeof this._onContextLost === 'function') {
            this._onContextLost(event);
          } else {
            setStatus('WebGPU context lost. Refresh the page to recover.');
          }
        };
        this.canvas.addEventListener('contextlost', this._contextLostListener, { once: true });
      }
    }

    getCurrentTextureView() {
      if (!this.context) {
        fatal('Canvas is not configured. Call configureCanvas() first.');
      }
      const texture = this.context.getCurrentTexture();
      if (!texture?.createView) {
        fatal('Current texture is unavailable.');
      }
      return texture.createView();
    }
  }

  async function ensureWebGPU() {
    if (singletonWebGPUInit) {
      logInfo('ensureWebGPU: Returning cached singleton');
      return singletonWebGPUInit;
    }

    setStatus('Checking WebGPU support…');

    if (!navigator.gpu) {
      fatal(WEBGPU_ENABLE_HINT);
    }

    let adapter;
    try {
      adapter = await navigator.gpu.requestAdapter();
    } catch (error) {
      fatal(`Failed to request GPU adapter: ${error.message ?? error}`);
    }

    if (!adapter) {
      fatal('Unable to acquire GPU adapter. Ensure WebGPU is enabled for your browser profile.');
    }

    let device;
    try {
      device = await adapter.requestDevice();
      logInfo('Created new WebGPU device');
    } catch (error) {
      fatal(`Failed to request GPU device: ${error.message ?? error}`);
    }

    if (!device) {
      fatal('GPU adapter returned no device.');
    }

    setStatus('WebGPU ready.');
    singletonWebGPUInit = { adapter, device };
    return singletonWebGPUInit;
  }

  function getPipelineCacheKey(format, alphaMode) {
    return `${format}|${alphaMode ?? 'opaque'}`;
  }

  async function getWebGPUState(options = {}) {
    const desiredAlphaMode = options.alphaMode ?? cachedWebGPUState?.webgpuContext?.alphaMode ?? 'premultiplied';

    if (!cachedWebGPUState) {
      if (webgpuStateInitPromise) {
        logInfo('getWebGPUState: Waiting for in-progress initialization');
        await webgpuStateInitPromise;
        return cachedWebGPUState;
      }

      logInfo('getWebGPUState: No cached state, creating new device');
      webgpuStateInitPromise = (async () => {
        const { adapter, device } = await ensureWebGPU();
        const webgpuContext = new WebGPUContext({
          adapter,
          device,
          canvasElement: canvas,
          onDeviceLost: (info) => {
            logWarn('WebGPU device lost, clearing caches:', info);
            cachedWebGPUState = null;
            cachedDevice = null;
            webgpuStateInitPromise = null;
            pipelineCache.clear();
            clearGPUObjectCaches();
            onDeviceInvalidated?.();
            fatal(`WebGPU device lost: ${info?.message ?? 'Unknown reason'}. Please refresh.`);
          },
        });
        webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
        cachedWebGPUState = { adapter, device, webgpuContext };
        cachedDevice = device;
        logInfo('getWebGPUState: Cached new state');
      })();

      await webgpuStateInitPromise;
      webgpuStateInitPromise = null;
      return cachedWebGPUState;
    }

    const { webgpuContext } = cachedWebGPUState;

    if (!webgpuContext.context) {
      webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
    } else if (desiredAlphaMode !== webgpuContext.alphaMode) {
      webgpuContext.configureCanvas({ alphaMode: desiredAlphaMode });
    }

    return cachedWebGPUState;
  }

  return {
    WEBGPU_ENABLE_HINT,
    ensureWebGPU,
    getWebGPUState,
    clearGPUObjectCaches,
    ensureCachesMatchDevice,
    loadShaderSource,
    getShaderMetadataCached,
    getOrCreateShaderModule,
    createBindGroupLayoutEntriesFromMetadata,
    createShaderResourceSet,
    createBindGroupEntriesFromResources,
    warnOnNonContiguousBindings,
    getOrCreateBindGroupLayout,
    getOrCreatePipelineLayout,
    getOrCreateComputePipeline,
    getBufferToTexturePipeline,
    compileShaderModuleWithValidation,
    renderFlatColor,
    pipelineCache,
    blitShaderModuleCache,
  };
}

export { createWebGPURuntime, WEBGPU_ENABLE_HINT };
