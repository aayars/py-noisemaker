import meta from './meta.json' with { type: 'json' };
import bloomMeta from '../bloom/meta.json' with { type: 'json' };

const DOWNSAMPLE_DIVISOR = 100;
const OFFSET_SCALE = -0.05;

function clampAlpha(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0.0;
  }
  return Math.min(Math.max(numeric, 0.0), 1.0);
}

function descriptorFromMeta(effectMeta) {
  return {
    id: effectMeta.id,
    label: effectMeta.label || `${effectMeta.id}.wgsl`,
    stage: 'compute',
    entryPoint: (effectMeta.shader && effectMeta.shader.entryPoint) || 'main',
    url: (effectMeta.shader && effectMeta.shader.url) || `/shaders/effects/${effectMeta.id}/${effectMeta.id}.wgsl`,
    resources: effectMeta.resources || {},
  };
}

class VaselineEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    const alphaParam = Array.isArray(meta.parameters)
      ? meta.parameters.find((param) => param.name === 'alpha')
      : null;
    const defaultAlpha = alphaParam?.default;
    this.userState = {
      enabled: true,
      alpha: clampAlpha(defaultAlpha ?? 0.5),
    };
    this.resources = null;
    this.finalParamsState = null;
  }

  getUIState() {
    return { ...this.userState };
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
      this.helpers?.logWarn?.('Vaseline: failed to destroy GPU resources.', error);
    }

    try {
      this.resources.bloomOutputTexture?.destroy?.();
    } catch (error) {
      this.helpers?.logWarn?.('Vaseline: failed to destroy bloom texture.', error);
    }

    try {
      this.resources.outputTexture?.destroy?.();
    } catch (error) {
      this.helpers?.logWarn?.('Vaseline: failed to destroy output texture.', error);
    }

    this.resources = null;
    this.finalParamsState = null;
  }

  async updateParams(updates = {}) {
    const changed = [];
    if (Object.prototype.hasOwnProperty.call(updates, 'enabled')) {
      const enabled = Boolean(updates.enabled);
      if (this.userState.enabled !== enabled) {
        this.userState.enabled = enabled;
        changed.push('enabled');
        if (this.resources) {
          this.resources.enabled = enabled;
        }
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'alpha')) {
      const alpha = clampAlpha(updates.alpha);
      if (this.userState.alpha !== alpha) {
        this.userState.alpha = alpha;
        changed.push('alpha');
        const offset = this.resources?.bindingOffsets?.alpha;
        if (Number.isInteger(offset) && this.finalParamsState) {
          this.finalParamsState[offset] = alpha;
          if (this.resources) {
            this.resources.paramsDirty = true;
          }
        }
      }
    }

    return { updated: changed };
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('VaselineEffect requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('VaselineEffect requires multires output texture.');
    }

    const enabled = this.userState.enabled !== false;
    if (!enabled) {
      if (!this.resources || this.resources.computePipeline) {
        this.invalidateResources();
        this.resources = {
          enabled: false,
          textureWidth: width,
          textureHeight: height,
          computePasses: [],
        };
      } else {
        this.resources.enabled = false;
        this.resources.textureWidth = width;
        this.resources.textureHeight = height;
      }
      return this.resources;
    }

    if (this.resources) {
      const sameDevice = this.resources.device === device;
      const sizeMatches = this.resources.textureWidth === width && this.resources.textureHeight === height;
      if (sameDevice && sizeMatches) {
        this.resources.enabled = true;
        return this.resources;
      }
      this.invalidateResources();
    }

    this.resources = await this.#createResources({ device, width, height, multiresResources });
    return this.resources;
  }

  async #createResources({ device, width, height, multiresResources }) {
    const {
      logWarn,
      getShaderDescriptor,
      getShaderMetadataCached,
      createShaderResourceSet,
      createBindGroupEntriesFromResources,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    const safeWidth = Math.max(Math.trunc(Number(width) || 0), 1);
    const safeHeight = Math.max(Math.trunc(Number(height) || 0), 1);

    // === Bloom pass setup ===
    const bloomDescriptor = descriptorFromMeta(bloomMeta);
    const bloomMetadata = await getShaderMetadataCached(bloomDescriptor.id);
    const bloomResourceSet = createShaderResourceSet(device, bloomDescriptor, bloomMetadata, safeWidth, safeHeight, {
      inputTextures: { input_texture: multiresResources.outputTexture },
    });
    const bloomLayout = getOrCreateBindGroupLayout(device, bloomDescriptor.id, 'compute', bloomMetadata);
    const bloomPipelineLayout = getOrCreatePipelineLayout(device, bloomDescriptor.id, 'compute', bloomLayout);
    const bloomDownsamplePipeline = await getOrCreateComputePipeline(device, bloomDescriptor.id, bloomPipelineLayout, 'downsample_main');
    const bloomPipeline = await getOrCreateComputePipeline(device, bloomDescriptor.id, bloomPipelineLayout, bloomDescriptor.entryPoint ?? 'upsample_main');
    const bloomBindGroup = device.createBindGroup({
      layout: bloomLayout,
      entries: createBindGroupEntriesFromResources(bloomMetadata.bindings, bloomResourceSet),
    });

    const bloomParamsBuffer = bloomResourceSet.buffers.params;
    if (!bloomParamsBuffer) {
      throw new Error('Vaseline: bloom params buffer missing.');
    }
    const bloomParamsState = new Float32Array(12);
    const bloomOffsets = {
      width: 0,
      height: 1,
      channelCount: 2,
      alpha: 3,
      time: 4,
      speed: 5,
      downWidth: 6,
      downHeight: 7,
      invDownWidth: 8,
      invDownHeight: 9,
      offsetX: 10,
      offsetY: 11,
    };
    const downsampleWidth = Math.max(Math.trunc(safeWidth / DOWNSAMPLE_DIVISOR), 1);
    const downsampleHeight = Math.max(Math.trunc(safeHeight / DOWNSAMPLE_DIVISOR), 1);
    const invDownsampleWidth = downsampleWidth > 0 ? 1 / downsampleWidth : 1;
    const invDownsampleHeight = downsampleHeight > 0 ? 1 / downsampleHeight : 1;
    const offsetX = Math.trunc(safeWidth * OFFSET_SCALE);
    const offsetY = Math.trunc(safeHeight * OFFSET_SCALE);

    bloomParamsState[bloomOffsets.width] = safeWidth;
    bloomParamsState[bloomOffsets.height] = safeHeight;
    bloomParamsState[bloomOffsets.channelCount] = 4;
    bloomParamsState[bloomOffsets.alpha] = 1.0; // Always request full bloom for vaseline
    bloomParamsState[bloomOffsets.time] = 0.0;
    bloomParamsState[bloomOffsets.speed] = 1.0;
    bloomParamsState[bloomOffsets.downWidth] = downsampleWidth;
    bloomParamsState[bloomOffsets.downHeight] = downsampleHeight;
    bloomParamsState[bloomOffsets.invDownWidth] = invDownsampleWidth;
    bloomParamsState[bloomOffsets.invDownHeight] = invDownsampleHeight;
    bloomParamsState[bloomOffsets.offsetX] = offsetX;
    bloomParamsState[bloomOffsets.offsetY] = offsetY;
    device.queue.writeBuffer(bloomParamsBuffer, 0, bloomParamsState);

    const downsampleDispatch = [
      Math.max(Math.ceil(downsampleWidth / 8), 1),
      Math.max(Math.ceil(downsampleHeight / 8), 1),
      1,
    ];

    // === Buffer->Texture for bloom (reuse for final mask blend) ===
    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);
    const bloomOutputTexture = device.createTexture({
      size: { width: safeWidth, height: safeHeight, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const bloomBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: bloomResourceSet.buffers.output_buffer } },
        { binding: 1, resource: bloomOutputTexture.createView() },
        { binding: 2, resource: { buffer: bloomParamsBuffer } },
      ],
    });

    // === Final vaseline combine pass ===
    const finalDescriptor = descriptorFromMeta(meta);
    const finalMetadata = await getShaderMetadataCached(finalDescriptor.id);
    const finalResourceSet = createShaderResourceSet(device, finalDescriptor, finalMetadata, safeWidth, safeHeight, {
      inputTextures: {
        input_texture: multiresResources.outputTexture,
        bloom_texture: bloomOutputTexture,
      },
    });
    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
    const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');
    const finalBindGroup = device.createBindGroup({
      layout: finalLayout,
      entries: createBindGroupEntriesFromResources(finalMetadata.bindings, finalResourceSet),
    });

    const finalParamsBuffer = finalResourceSet.buffers.params;
    if (!finalParamsBuffer) {
      throw new Error('Vaseline: final params buffer missing.');
    }
    const finalParamsState = new Float32Array(8);
    const finalBindingOffsets = {
      width: 0,
      height: 1,
      channel_count: 2,
      channelCount: 2,
      alpha: 4,
      time: 5,
      speed: 6,
    };
    finalParamsState[finalBindingOffsets.width] = safeWidth;
    finalParamsState[finalBindingOffsets.height] = safeHeight;
    finalParamsState[finalBindingOffsets.channel_count] = 4;
    finalParamsState[3] = 0.0; // padding slot between channel_count and alpha
    finalParamsState[finalBindingOffsets.alpha] = this.userState.alpha;
    finalParamsState[finalBindingOffsets.time] = 0.0;
    finalParamsState[finalBindingOffsets.speed] = 1.0;
    finalParamsState[7] = 0.0; // trailing pad
    device.queue.writeBuffer(finalParamsBuffer, 0, finalParamsState);
    this.finalParamsState = finalParamsState;

    const finalOutputBuffer = finalResourceSet.buffers.output_buffer;
    const finalOutputTexture = device.createTexture({
      size: { width: safeWidth, height: safeHeight, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const finalBufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureLayout,
      entries: [
        { binding: 0, resource: { buffer: finalOutputBuffer } },
        { binding: 1, resource: finalOutputTexture.createView() },
        { binding: 2, resource: { buffer: finalParamsBuffer } },
      ],
    });

    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: finalOutputTexture.createView() },
      ],
    });

    const workgroupSize = [8, 8, 1];
    const dispatchDims = [
      Math.max(Math.ceil(safeWidth / workgroupSize[0]), 1),
      Math.max(Math.ceil(safeHeight / workgroupSize[1]), 1),
      1,
    ];

    const passes = [
      { pipeline: bloomDownsamplePipeline, bindGroup: bloomBindGroup, workgroupSize, dispatch: downsampleDispatch },
      { pipeline: bloomPipeline, bindGroup: bloomBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: bufferToTexturePipeline, bindGroup: bloomBufferToTextureBindGroup, workgroupSize: [8, 8, 1], dispatch: [Math.ceil(safeWidth / 8), Math.ceil(safeHeight / 8), 1] },
      { pipeline: finalPipeline, bindGroup: finalBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: bufferToTexturePipeline, bindGroup: finalBufferToTextureBindGroup, workgroupSize: [8, 8, 1], dispatch: [Math.ceil(safeWidth / 8), Math.ceil(safeHeight / 8), 1] },
    ];

    return {
      device,
      enabled: true,
      textureWidth: safeWidth,
      textureHeight: safeHeight,
      computePipeline: finalPipeline,
      computeBindGroup: finalBindGroup,
      computePasses: passes,
      paramsBuffer: finalParamsBuffer,
      paramsState: finalParamsState,
      outputBuffer: finalOutputBuffer,
      outputTexture: finalOutputTexture,
      bloomOutputTexture,
      bufferToTexturePipeline,
      bufferToTextureBindGroup: finalBufferToTextureBindGroup,
      bufferToTextureWorkgroupSize: [8, 8, 1],
      blitBindGroup,
      workgroupSize,
      paramsDirty: false,
      bindingOffsets: finalBindingOffsets,
      resourceSet: {
        destroyAll() {
          try { bloomResourceSet.destroyAll(); } catch (error) { logWarn?.('Vaseline: failed to destroy bloom resources.', error); }
          try { finalResourceSet.destroyAll(); } catch (error) { logWarn?.('Vaseline: failed to destroy final resources.', error); }
        },
      },
    };
  }
}

export default VaselineEffect;
export const additionalPasses = {};
