import meta from './meta.json' with { type: 'json' };
// Directly import convolve meta to avoid dependency on external manifest registration order.
// This eliminates the stale runtime error: "Sobel effect requires convolve effect to be registered." seen when
// the shader registry module was cached without the convolve entry.
import convolveMeta from '../convolve/meta.json' with { type: 'json' };

// Sobel multi-pass effect: Reuses the existing Convolve shader pipelines
// (kernels 808=Sobel X, 809=Sobel Y) to populate two intermediate buffers,
// then runs a combine pass (sobel.wgsl) to compute edge magnitude.

class SobelEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = { enabled: true, dist_metric: 1, alpha: 1.0 };
  this.resources = null;
  this.sobelParamsState = null; // Float32Array backing the combine pass uniform buffer
  }

  getUIState() {
    return { ...this.userState };
  }

  async updateParams(updates = {}) {
    const changed = [];
    if (typeof updates.enabled !== 'undefined') {
      const enabled = Boolean(updates.enabled);
      if (this.userState.enabled !== enabled) {
        this.userState.enabled = enabled;
        changed.push('enabled');
        if (this.resources) this.resources.enabled = enabled;
      }
    }
    const device = this.resources?.device;
    if (typeof updates.dist_metric !== 'undefined') {
      const metric = Number(updates.dist_metric);
      if (Number.isFinite(metric) && this.userState.dist_metric !== metric) {
        this.userState.dist_metric = metric;
        changed.push('dist_metric');
        const offset = this.resources?.bindingOffsets?.dist_metric;
        if (Number.isInteger(offset) && this.sobelParamsState && device) {
          this.sobelParamsState[offset] = metric;
          // write only the single float (offset * 4 bytes)
          device.queue.writeBuffer(this.resources.paramsBuffer, offset * 4, new Float32Array([metric]));
        }
      }
    }
    if (typeof updates.alpha !== 'undefined') {
      const alpha = Math.min(Math.max(Number(updates.alpha), 0), 1);
      if (this.userState.alpha !== alpha) {
        this.userState.alpha = alpha;
        changed.push('alpha');
        const offset = this.resources?.bindingOffsets?.alpha;
        if (Number.isInteger(offset) && this.sobelParamsState && device) {
          this.sobelParamsState[offset] = alpha;
          device.queue.writeBuffer(this.resources.paramsBuffer, offset * 4, new Float32Array([alpha]));
        }
      }
    }
    return { updated: changed };
  }

  destroy() { this.invalidateResources(); }

  invalidateResources() {
    if (!this.resources) return;
    try { this.resources.resourceSet?.destroyAll?.(); } catch {}
    try { this.resources.outputTexture?.destroy?.(); } catch {}
    this.resources = null;
  this.sobelParamsState = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) throw new Error('SobelEffect requires a GPUDevice.');
    if (!multiresResources?.outputTexture) throw new Error('SobelEffect requires multires output texture.');

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
      }
      return this.resources;
    }

    if (this.resources) {
      const sizeMatches = this.resources.textureWidth === width && this.resources.textureHeight === height;
      if (sizeMatches && this.resources.device === device) {
        return this.resources;
      }
      this.invalidateResources();
    }

    this.resources = await this.#createResources({ device, width, height, multiresResources });
    return this.resources;
  }

  async #createResources({ device, width, height, multiresResources }) {
    const {
      getShaderDescriptor,
      getShaderMetadataCached,
      createShaderResourceSet,
      createBindGroupEntriesFromResources,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    // Final combine shader descriptor
    const sobelDescriptor = getShaderDescriptor('sobel');
    const sobelMetadata = await getShaderMetadataCached('sobel');
    const sobelBindGroupLayout = getOrCreateBindGroupLayout(device, sobelDescriptor.id, 'compute', sobelMetadata);
    const sobelPipelineLayout = getOrCreatePipelineLayout(device, sobelDescriptor.id, 'compute', sobelBindGroupLayout);
    const sobelPipeline = await getOrCreateComputePipeline(device, sobelDescriptor.id, sobelPipelineLayout, sobelDescriptor.entryPoint ?? 'main');

    // Convolve descriptor/metadata reused for Sobel X/Y passes
    // Build the convolve descriptor directly from its meta (manifest fallback-less)
    const convolveDescriptor = {
      id: convolveMeta.id,
      label: convolveMeta.label || `${convolveMeta.id}.wgsl`,
      stage: 'compute',
      entryPoint: (convolveMeta.shader && convolveMeta.shader.entryPoint) || 'main',
      url: (convolveMeta.shader && convolveMeta.shader.url) || `/shaders/effects/${convolveMeta.id}/${convolveMeta.id}.wgsl`,
      resources: convolveMeta.resources || {},
    };
    const convolveMetadata = await getShaderMetadataCached(convolveDescriptor.id);
    const convolveBindGroupLayout = getOrCreateBindGroupLayout(device, 'convolve', 'compute', convolveMetadata);
    const convolvePipelineLayout = getOrCreatePipelineLayout(device, 'convolve', 'compute', convolveBindGroupLayout);

    const convolveResetPipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'reset_stats_main');
    const convolveMainPipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'convolve_main');
    const convolveFinalizePipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'main');

    // Create two resource sets for sobel X/Y passes (each mimics convolve bindings)
    const commonOptions = { inputTextures: { input_texture: multiresResources.outputTexture } };
    const resourceSetX = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, commonOptions);
    const resourceSetY = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, commonOptions);

    const paramsBufferX = resourceSetX.buffers.params;
    const paramsBufferY = resourceSetY.buffers.params;
    const outputBufferX = resourceSetX.buffers.output_buffer;
    const outputBufferY = resourceSetY.buffers.output_buffer;
    const statsBufferX = resourceSetX.buffers.stats_buffer;
    const statsBufferY = resourceSetY.buffers.stats_buffer;

    const paramsLength = 32 / Float32Array.BYTES_PER_ELEMENT;
    const paramsStateX = new Float32Array(paramsLength);
    const paramsStateY = new Float32Array(paramsLength);
    // Populate static params: width, height, channel_count, kernel, with_normalize, alpha, time
    paramsStateX[0] = width; // width
    paramsStateX[1] = height; // height
    paramsStateX[2] = 4; // channel_count
    paramsStateX[3] = 808; // kernel sobel x
    paramsStateX[4] = 1.0; // with_normalize true (required for proper Sobel edge detection)
    paramsStateX[5] = 1.0; // alpha
    paramsStateX[6] = 0.0; // time
    paramsStateX[7] = 1.0; // speed

    paramsStateY.set(paramsStateX);
    paramsStateY[3] = 809; // kernel sobel y

    device.queue.writeBuffer(paramsBufferX, 0, paramsStateX);
    device.queue.writeBuffer(paramsBufferY, 0, paramsStateY);

    const bindGroupX = device.createBindGroup({
      layout: convolveBindGroupLayout,
      entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, resourceSetX),
    });
    const bindGroupY = device.createBindGroup({
      layout: convolveBindGroupLayout,
      entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, resourceSetY),
    });

    // Final combine params buffer/state
    // Create a dedicated uniform buffer for the final combine pass so we do NOT conflict with
    // the kernel id field (offset 3) used by the convolve passes. Reusing the convolve params
    // buffer previously caused the combine pass to read the kernel id (808/809) instead of the
    // intended distance metric. This fixes that by giving the combine stage its own params.
    const sobelParamsBuffer = device.createBuffer({
      size: 32, // 8 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  const sobelParamsState = new Float32Array(8);
  // Use binding offset keys expected by the runtime (camelCase for channelCount)
  const bindingOffsets = { width: 0, height: 1, channelCount: 2, dist_metric: 3, /* pad:4 */ alpha: 5, time: 6, speed: 7 };
  sobelParamsState[bindingOffsets.width] = width;
  sobelParamsState[bindingOffsets.height] = height;
  sobelParamsState[bindingOffsets.channelCount] = 4;
    sobelParamsState[bindingOffsets.dist_metric] = this.userState.dist_metric;
    sobelParamsState[bindingOffsets.alpha] = this.userState.alpha;
    sobelParamsState[bindingOffsets.time] = 0.0;
    sobelParamsState[bindingOffsets.speed] = 1.0;
    device.queue.writeBuffer(sobelParamsBuffer, 0, sobelParamsState);
    this.sobelParamsState = sobelParamsState;

    // Output texture & conversion pipeline
    const outputTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const storageView = outputTexture.createView();
    const sampleView = outputTexture.createView();

    // Dedicated final output buffer for the Sobel combine pass to avoid read/write hazards.
    const pixelCount = Math.max(width * height, 1);
    const finalOutputBufferSize = Math.max(pixelCount * 4 * 4, 16);
    const finalOutputBuffer = device.createBuffer({
      size: finalOutputBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);
    const bufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: finalOutputBuffer } },
        { binding: 1, resource: storageView },
        { binding: 2, resource: { buffer: sobelParamsBuffer } },
      ],
    });
    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [{ binding: 0, resource: sampleView }],
    });

    // Bind group for final combine pass (maps sobel_x/y buffers + output + params)
    const sobelCombineBindGroup = device.createBindGroup({
      layout: sobelBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outputBufferX } },
        { binding: 1, resource: { buffer: outputBufferY } },
        { binding: 2, resource: { buffer: finalOutputBuffer } }, // final output target (dedicated)
        { binding: 3, resource: { buffer: sobelParamsBuffer } },
      ],
    });

    // Compute pass configurations
    const workgroupSize = [8, 8, 1];
    const dispatchDims = [
      Math.ceil(width / workgroupSize[0]),
      Math.ceil(height / workgroupSize[1]),
      1,
    ];

    const passes = [
      // Sobel X: reset -> convolve -> finalize
      { pipeline: convolveResetPipeline, bindGroup: bindGroupX, workgroupSize: [1,1,1], dispatch: [1,1,1] },
      { pipeline: convolveMainPipeline, bindGroup: bindGroupX, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: convolveFinalizePipeline, bindGroup: bindGroupX, workgroupSize, dispatch: dispatchDims.slice() },
      // Sobel Y
      { pipeline: convolveResetPipeline, bindGroup: bindGroupY, workgroupSize: [1,1,1], dispatch: [1,1,1] },
      { pipeline: convolveMainPipeline, bindGroup: bindGroupY, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: convolveFinalizePipeline, bindGroup: bindGroupY, workgroupSize, dispatch: dispatchDims.slice() },
      // Final combine
      { pipeline: sobelPipeline, bindGroup: sobelCombineBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
    ];

    return {
      device,
      enabled: true,
      textureWidth: width,
      textureHeight: height,
      computePipeline: sobelPipeline, // dummy primary pipeline
      computeBindGroup: sobelCombineBindGroup,
      computePasses: passes,
      paramsBuffer: sobelParamsBuffer,
      paramsState: sobelParamsState,
      outputBuffer: finalOutputBuffer,
      outputTexture,
      storageView,
      sampleView,
      bufferToTexturePipeline,
      bufferToTextureBindGroup,
      blitBindGroup,
      workgroupSize,
      paramsDirty: false,
      resourceSet: { destroyAll() { resourceSetX.destroyAll(); resourceSetY.destroyAll(); try { finalOutputBuffer.destroy?.(); } catch {} } },
      bindingOffsets,
    };
  }
}

export default SobelEffect;

// No additional sub-shaders to register in the manifest for sobel
export const additionalPasses = {};
