import meta from './meta.json' with { type: 'json' };
import convolveMeta from '../convolve/meta.json' with { type: 'json' };
import sobelMeta from '../sobel/meta.json' with { type: 'json' };

// OutlineEffect: minimal multi-pass orchestrator reusing existing low-level shaders.
// Pass chain:
//   1. Convolve blur (kernel 800) -> buffer->texture (blur texture)
//   2. Convolve Sobel X (808) passes
//   3. Convolve Sobel Y (809) passes
//   4. Sobel combine (existing sobel shader) -> buffer->texture (edges texture)
//   5. Final outline blend (outline.wgsl) multiplying edges (optionally inverted) with base
//   6. Buffer->texture (final output) for blit

class OutlineEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = { enabled: true, sobel_metric: 1, invert: false };
    this.resources = null;
    this.finalParamsState = null;
    this.sobelParamsState = null;
  }

  getUIState() { return { ...this.userState }; }

  async updateParams(updates = {}) {
    const changed = [];
    if (typeof updates.enabled !== 'undefined') {
      const enabled = Boolean(updates.enabled);
      if (this.userState.enabled !== enabled) {
        this.userState.enabled = enabled; changed.push('enabled'); if (this.resources) this.resources.enabled = enabled;
      }
    }
    const device = this.resources?.device;
    if (typeof updates.sobel_metric !== 'undefined') {
      const metric = Number(updates.sobel_metric);
      if (Number.isFinite(metric) && this.userState.sobel_metric !== metric) {
        this.userState.sobel_metric = metric; changed.push('sobel_metric');
        // Update sobel combine params
        const sobelOff = this.resources?.sobelBindingOffsets?.dist_metric; if (Number.isInteger(sobelOff) && this.sobelParamsState && device) {
          this.sobelParamsState[sobelOff] = metric; device.queue.writeBuffer(this.resources.sobelParamsBuffer, sobelOff * 4, new Float32Array([metric]));
        }
        // Update final pass params
        const finalOff = this.resources?.finalBindingOffsets?.sobel_metric; if (Number.isInteger(finalOff) && this.finalParamsState) {
          this.finalParamsState[finalOff] = metric; this.resources.finalParamsDirty = true;
        }
      }
    }
    if (typeof updates.invert !== 'undefined') {
      const invert = Boolean(updates.invert);
      if (this.userState.invert !== invert) {
        this.userState.invert = invert; changed.push('invert');
        const invOff = this.resources?.finalBindingOffsets?.invert_flag; if (Number.isInteger(invOff) && this.finalParamsState) {
          this.finalParamsState[invOff] = invert ? 1.0 : 0.0; this.resources.finalParamsDirty = true;
        }
      }
    }
    return { updated: changed };
  }

  destroy() { this.invalidateResources(); }
  invalidateResources() { if (!this.resources) return; try { this.resources.resourceSet?.destroyAll?.(); } catch {}; try { this.resources.outputTexture?.destroy?.(); } catch {}; this.resources = null; this.finalParamsState = null; this.sobelParamsState = null; }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) throw new Error('OutlineEffect requires a GPUDevice.');
    if (!multiresResources?.outputTexture) throw new Error('OutlineEffect requires multires output texture.');
    const enabled = this.userState.enabled !== false;
    if (!enabled) {
      if (!this.resources || this.resources.computePipeline) { this.invalidateResources(); this.resources = { enabled: false, textureWidth: width, textureHeight: height, computePasses: [] }; }
      return this.resources;
    }
    if (this.resources) {
      const sizeMatches = this.resources.textureWidth === width && this.resources.textureHeight === height;
      if (sizeMatches && this.resources.device === device) { return this.resources; }
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

    // Convolve descriptor (blur + sobel passes)
    const convolveDescriptor = {
      id: convolveMeta.id,
      label: convolveMeta.label || `${convolveMeta.id}.wgsl`,
      stage: 'compute',
      entryPoint: (convolveMeta.shader && convolveMeta.shader.entryPoint) || 'main',
      url: (convolveMeta.shader && convolveMeta.shader.url) || `/shaders/effects/${convolveMeta.id}/${convolveMeta.id}.wgsl`,
      resources: convolveMeta.resources || {},
    };
    const convolveMetadata = await getShaderMetadataCached(convolveDescriptor.id);
    const convolveLayout = getOrCreateBindGroupLayout(device, convolveDescriptor.id, 'compute', convolveMetadata);
    const convolvePipelineLayout = getOrCreatePipelineLayout(device, convolveDescriptor.id, 'compute', convolveLayout);
  const convolveResetPipeline = await getOrCreateComputePipeline(device, convolveDescriptor.id, convolvePipelineLayout, 'reset_stats_main');
  const convolveMainPipeline = await getOrCreateComputePipeline(device, convolveDescriptor.id, convolvePipelineLayout, 'convolve_main');
  const convolveFinalizePipeline = await getOrCreateComputePipeline(device, convolveDescriptor.id, convolvePipelineLayout, convolveDescriptor.entryPoint ?? 'main');
  // We will reuse a second finalize pass later to blur edges (kernel 800) after Sobel combine.

    // Sobel X/Y resource sets (kernels 808 / 809) using base multires texture directly
    const sobelSetX = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, { inputTextures: { input_texture: multiresResources.outputTexture } });
    const sobelSetY = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, { inputTextures: { input_texture: multiresResources.outputTexture } });
    const paramsStateX = new Float32Array(8); const paramsStateY = new Float32Array(8);
    paramsStateX[0] = width; paramsStateX[1] = height; paramsStateX[2] = 4; paramsStateX[3] = 808; paramsStateX[4] = 1.0; paramsStateX[5] = 1.0; paramsStateX[6] = 0.0; paramsStateX[7] = 1.0;
    paramsStateY.set(paramsStateX); paramsStateY[3] = 809;
    device.queue.writeBuffer(sobelSetX.buffers.params, 0, paramsStateX);
    device.queue.writeBuffer(sobelSetY.buffers.params, 0, paramsStateY);
    const sobelBindGroupX = device.createBindGroup({ layout: convolveLayout, entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, sobelSetX) });
    const sobelBindGroupY = device.createBindGroup({ layout: convolveLayout, entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, sobelSetY) });

    // Sobel combine descriptor
    const sobelDescriptor = {
      id: sobelMeta.id,
      label: sobelMeta.label || `${sobelMeta.id}.wgsl`,
      stage: 'compute',
      entryPoint: (sobelMeta.shader && sobelMeta.shader.entryPoint) || 'main',
      url: (sobelMeta.shader && sobelMeta.shader.url) || `/shaders/effects/${sobelMeta.id}/${sobelMeta.id}.wgsl`,
      resources: sobelMeta.resources || {},
    };
    const sobelMetadata = await getShaderMetadataCached(sobelDescriptor.id);
    const sobelLayout = getOrCreateBindGroupLayout(device, sobelDescriptor.id, 'compute', sobelMetadata);
    const sobelPipelineLayout = getOrCreatePipelineLayout(device, sobelDescriptor.id, 'compute', sobelLayout);
    const sobelPipeline = await getOrCreateComputePipeline(device, sobelDescriptor.id, sobelPipelineLayout, sobelDescriptor.entryPoint ?? 'main');
    const sobelParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sobelParamsState = new Float32Array(8);
    const sobelBindingOffsets = { width: 0, height: 1, channelCount: 2, dist_metric: 3, _pad0: 4, alpha: 5, time: 6, speed: 7 };
    sobelParamsState[sobelBindingOffsets.width] = width;
    sobelParamsState[sobelBindingOffsets.height] = height;
    sobelParamsState[sobelBindingOffsets.channelCount] = 4;
    sobelParamsState[sobelBindingOffsets.dist_metric] = this.userState.sobel_metric;
    sobelParamsState[sobelBindingOffsets.alpha] = 1.0;
    sobelParamsState[sobelBindingOffsets.time] = 0.0;
    sobelParamsState[sobelBindingOffsets.speed] = 1.0;
    device.queue.writeBuffer(sobelParamsBuffer, 0, sobelParamsState);
    this.sobelParamsState = sobelParamsState;
    const sobelPixelCount = Math.max(width * height, 1);
    const sobelFinalBufferSize = sobelPixelCount * 4 * 4;
    const sobelFinalBuffer = device.createBuffer({ size: sobelFinalBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const sobelCombineBindGroup = device.createBindGroup({ layout: sobelLayout, entries: [
      { binding: 0, resource: { buffer: sobelSetX.buffers.output_buffer } },
      { binding: 1, resource: { buffer: sobelSetY.buffers.output_buffer } },
      { binding: 2, resource: { buffer: sobelFinalBuffer } },
      { binding: 3, resource: { buffer: sobelParamsBuffer } },
    ] });

    // Convert buffers to textures (sobel final)
    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);
    const sobelOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    const sobelBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureLayout, entries: [
      { binding: 0, resource: { buffer: sobelFinalBuffer } },
      { binding: 1, resource: sobelOutputTexture.createView() },
      { binding: 2, resource: { buffer: sobelParamsBuffer } },
    ] });

    // Normalize passes (stats -> reduce -> apply) on sobelOutputTexture to expand dynamic range
    const normalizeStatsDescriptor = getShaderDescriptor('normalize/stats');
    const normalizeReduceDescriptor = getShaderDescriptor('normalize/reduce');
    const normalizeApplyDescriptor = getShaderDescriptor('normalize/apply');
    const normalizeMetadata = await getShaderMetadataCached('normalize');
    const normalizeLayout = getOrCreateBindGroupLayout(device, 'normalize', 'compute', normalizeMetadata);
    const normalizePipelineLayout = getOrCreatePipelineLayout(device, 'normalize', 'compute', normalizeLayout);
    const normalizeStatsPipeline = await getOrCreateComputePipeline(device, 'normalize/stats', normalizePipelineLayout, normalizeStatsDescriptor.entryPoint ?? 'main');
    const normalizeReducePipeline = await getOrCreateComputePipeline(device, 'normalize/reduce', normalizePipelineLayout, normalizeReduceDescriptor.entryPoint ?? 'main');
    const normalizeApplyPipeline = await getOrCreateComputePipeline(device, 'normalize/apply', normalizePipelineLayout, normalizeApplyDescriptor.entryPoint ?? 'main');
    const normParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const normParamsState = new Float32Array(8);
    normParamsState[0] = width; normParamsState[1] = height; normParamsState[2] = 4; normParamsState[4] = 0.0; normParamsState[5] = 1.0;
    device.queue.writeBuffer(normParamsBuffer, 0, normParamsState);
    const normOutputBufferSize = width * height * 4 * 4;
    const normOutputBuffer = device.createBuffer({ size: normOutputBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const numWorkgroupsX = Math.ceil(width / 8);
    const numWorkgroupsY = Math.ceil(height / 8);
    const numWorkgroups = numWorkgroupsX * numWorkgroupsY;
    const normStatsBufferSize = (2 + numWorkgroups * 2) * 4;
    const normStatsBuffer = device.createBuffer({ size: normStatsBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const statsInit = new Float32Array(2 + numWorkgroups * 2);
    const F32_MAX = 3.4028235e+38; const F32_MIN = -3.4028235e+38;
    statsInit[0] = F32_MAX; statsInit[1] = F32_MIN;
    for (let i = 0; i < numWorkgroups; i++) { statsInit[2 + i * 2] = F32_MAX; statsInit[2 + i * 2 + 1] = F32_MIN; }
    device.queue.writeBuffer(normStatsBuffer, 0, statsInit);
    // Skipping separate normalize-to-texture conversion; use sobelOutputTexture directly for final combine

    // Final outline combine pass descriptor
    const finalDescriptor = getShaderDescriptor(meta.id);
    const finalMetadata = await getShaderMetadataCached(meta.id);
    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
    const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');
    const finalParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const finalParamsState = new Float32Array(8);
  // binding offsets (mirrors meta.parameterBindings keys). Provide both snake_case and camelCase aliases for runtime convenience.
  const finalBindingOffsets = { width: 0, height: 1, channel_count: 2, channelCount: 2, invert_flag: 3, sobel_metric: 4, time: 6, speed: 7 };
    finalParamsState[finalBindingOffsets.width] = width;
    finalParamsState[finalBindingOffsets.height] = height;
    finalParamsState[finalBindingOffsets.channel_count] = 4;
  finalParamsState[finalBindingOffsets.invert_flag] = this.userState.invert ? 1.0 : 0.0;
  finalParamsState[finalBindingOffsets.sobel_metric] = this.userState.sobel_metric;
    finalParamsState[finalBindingOffsets.time] = 0.0;
    finalParamsState[finalBindingOffsets.speed] = 1.0;
    device.queue.writeBuffer(finalParamsBuffer, 0, finalParamsState);
    this.finalParamsState = finalParamsState;
    const finalOutputBufferSize = width * height * 4 * 4;
    const finalOutputBuffer = device.createBuffer({ size: finalOutputBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const finalBindGroup = device.createBindGroup({ layout: finalLayout, entries: [
      { binding: 0, resource: multiresResources.outputTexture.createView() },
      { binding: 1, resource: { buffer: finalOutputBuffer } },
      { binding: 2, resource: { buffer: finalParamsBuffer } },
      { binding: 3, resource: sobelOutputTexture.createView() },
    ] });
  const finalOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const finalBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureLayout, entries: [
      { binding: 0, resource: { buffer: finalOutputBuffer } },
      { binding: 1, resource: finalOutputTexture.createView() },
      { binding: 2, resource: { buffer: finalParamsBuffer } },
    ] });
    const blitBindGroup = device.createBindGroup({ layout: multiresResources.blitBindGroupLayout, entries: [ { binding: 0, resource: finalOutputTexture.createView() } ] });

    // Pass schedule
    const workgroupSize = [8, 8, 1];
    const dispatchDims = [ Math.ceil(width / workgroupSize[0]), Math.ceil(height / workgroupSize[1]), 1 ];
    const passes = [
      // Sobel X
      { pipeline: convolveResetPipeline, bindGroup: sobelBindGroupX, workgroupSize: [1,1,1], dispatch: [1,1,1] },
      { pipeline: convolveMainPipeline, bindGroup: sobelBindGroupX, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: convolveFinalizePipeline, bindGroup: sobelBindGroupX, workgroupSize, dispatch: dispatchDims.slice() },
      // Sobel Y
      { pipeline: convolveResetPipeline, bindGroup: sobelBindGroupY, workgroupSize: [1,1,1], dispatch: [1,1,1] },
      { pipeline: convolveMainPipeline, bindGroup: sobelBindGroupY, workgroupSize, dispatch: dispatchDims.slice() },
      { pipeline: convolveFinalizePipeline, bindGroup: sobelBindGroupY, workgroupSize, dispatch: dispatchDims.slice() },
  // Sobel combine
  { pipeline: sobelPipeline, bindGroup: sobelCombineBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
  // Buffer->Texture (Sobel)
  { pipeline: bufferToTexturePipeline, bindGroup: sobelBufferToTextureBindGroup, workgroupSize: [8,8,1], dispatch: [ Math.ceil(width/8), Math.ceil(height/8), 1 ] },
      // Final combine (outline multiply)
      { pipeline: finalPipeline, bindGroup: finalBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
      // Buffer->Texture (Final)
      { pipeline: bufferToTexturePipeline, bindGroup: finalBufferToTextureBindGroup, workgroupSize: [8,8,1], dispatch: [ Math.ceil(width/8), Math.ceil(height/8), 1 ] },
    ];

    return {
      device,
      enabled: true,
      textureWidth: width,
      textureHeight: height,
      computePipeline: finalPipeline,
      computeBindGroup: finalBindGroup,
      computePasses: passes,
      paramsBuffer: finalParamsBuffer,
      paramsState: finalParamsState,
      outputBuffer: finalOutputBuffer,
      outputTexture: finalOutputTexture,
      bufferToTexturePipeline,
      bufferToTextureBindGroup: finalBufferToTextureBindGroup,
      blitBindGroup,
      workgroupSize,
  paramsDirty: false,
      bindingOffsets: finalBindingOffsets,
      sobelParamsBuffer,
      sobelParamsState,
      sobelBindingOffsets,
  resourceSet: { destroyAll() { try { sobelSetX.destroyAll(); sobelSetY.destroyAll(); } catch {}; } },
    };
  }
}

export default OutlineEffect;
export const additionalPasses = {};
