import meta from './meta.json' with { type: 'json' };
import posterizeMeta from '../posterize/meta.json' with { type: 'json' };
import bloomMeta from '../bloom/meta.json' with { type: 'json' };
import normalizeMeta from '../normalize/meta.json' with { type: 'json' };
import convolveMeta from '../convolve/meta.json' with { type: 'json' };

// GlowingEdgesEffect orchestrates reuse of existing low-level passes:
// posterize -> sobel (via convolve kernels 808/809 + sobel combine) -> bloom -> normalize -> final blend
// Only the final blend has bespoke shader code (glowing_edges.wgsl). All prior steps re-use existing pipelines.

class GlowingEdgesEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.userState = { enabled: true, sobel_metric: 1, alpha: 1.0 };
    this.resources = null;
    this.finalParamsState = null; // uniform buffer state for final combine pass
  }

  getUIState() { return { ...this.userState }; }

  async updateParams(updates = {}) {
    const changed = [];
    if (typeof updates.enabled !== 'undefined') {
      const enabled = Boolean(updates.enabled);
      if (this.userState.enabled !== enabled) { this.userState.enabled = enabled; changed.push('enabled'); if (this.resources) this.resources.enabled = enabled; }
    }
    if (typeof updates.sobel_metric !== 'undefined') {
      const metric = Number(updates.sobel_metric);
      if (Number.isFinite(metric) && this.userState.sobel_metric !== metric) {
        this.userState.sobel_metric = metric; changed.push('sobel_metric');
        const off = this.resources?.bindingOffsets?.sobel_metric; if (Number.isInteger(off) && this.finalParamsState) { this.finalParamsState[off] = metric; this.resources.paramsDirty = true; }
        // Also update sobel params buffer if present
        const sobelOffsets = this.resources?.sobelBindingOffsets; const sobelParamsState = this.resources?.sobelParamsState; const sobelParamsBuffer = this.resources?.sobelParamsBuffer; const device = this.resources?.device;
        if (Number.isInteger(sobelOffsets?.dist_metric) && sobelParamsState && device && sobelParamsBuffer) {
          sobelParamsState[sobelOffsets.dist_metric] = metric;
          device.queue.writeBuffer(sobelParamsBuffer, sobelOffsets.dist_metric * 4, new Float32Array([metric]));
        }
      }
    }
    if (typeof updates.alpha !== 'undefined') {
      const alpha = Math.min(Math.max(Number(updates.alpha), 0), 1);
      if (this.userState.alpha !== alpha) { this.userState.alpha = alpha; changed.push('alpha'); const off = this.resources?.bindingOffsets?.alpha; if (Number.isInteger(off) && this.finalParamsState) { this.finalParamsState[off] = alpha; this.resources.paramsDirty = true; } }
    }
    return { updated: changed };
  }

  destroy() { this.invalidateResources(); }
  invalidateResources() { if (!this.resources) return; try { this.resources.resourceSet?.destroyAll?.(); } catch {}; try { this.resources.outputTexture?.destroy?.(); } catch {}; this.resources = null; this.finalParamsState = null; }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) throw new Error('GlowingEdgesEffect requires a GPUDevice.');
    if (!multiresResources?.outputTexture) throw new Error('GlowingEdgesEffect requires multires output texture.');
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

    // === Posterize pass ===
    const posterizeDescriptor = this.#descriptorFromMeta(posterizeMeta);
    const posterizeMetadata = await getShaderMetadataCached(posterizeDescriptor.id);
    const posterizeResourceSet = createShaderResourceSet(device, posterizeDescriptor, posterizeMetadata, width, height, {
      inputTextures: { input_texture: multiresResources.outputTexture },
    });
    const posterizeLayout = getOrCreateBindGroupLayout(device, posterizeDescriptor.id, 'compute', posterizeMetadata);
    const posterizePipelineLayout = getOrCreatePipelineLayout(device, posterizeDescriptor.id, 'compute', posterizeLayout);
    const posterizePipeline = await getOrCreateComputePipeline(device, posterizeDescriptor.id, posterizePipelineLayout, posterizeDescriptor.entryPoint);
    const posterizeBindGroup = device.createBindGroup({ layout: posterizeLayout, entries: createBindGroupEntriesFromResources(posterizeMetadata.bindings, posterizeResourceSet) });

    // === Sobel (convolve passes 808/809 -> sobel combine) ===
    const convolveDescriptor = this.#descriptorFromMeta(convolveMeta);
    const convolveMetadata = await getShaderMetadataCached(convolveDescriptor.id);
    const convolveLayout = getOrCreateBindGroupLayout(device, 'convolve', 'compute', convolveMetadata);
    const convolvePipelineLayout = getOrCreatePipelineLayout(device, 'convolve', 'compute', convolveLayout);
    const convolveResetPipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'reset_stats_main');
    const convolveMainPipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'convolve_main');
    const convolveFinalizePipeline = await getOrCreateComputePipeline(device, 'convolve', convolvePipelineLayout, 'main');

    // Two resource sets for Sobel X/Y using posterize output texture as input
    const sobelInputTexture = posterizeResourceSet.textures.input_texture;
    const sobelCommonOptions = { inputTextures: { input_texture: sobelInputTexture } };
    const sobelSetX = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, sobelCommonOptions);
    const sobelSetY = createShaderResourceSet(device, convolveDescriptor, convolveMetadata, width, height, sobelCommonOptions);
    // Configure params (kernel ids 808/809, with_normalize=true)
    const paramsLength = 32 / Float32Array.BYTES_PER_ELEMENT;
    const paramsStateX = new Float32Array(paramsLength); const paramsStateY = new Float32Array(paramsLength);
    paramsStateX[0] = width; paramsStateX[1] = height; paramsStateX[2] = 4; paramsStateX[3] = 808; paramsStateX[4] = 1.0; paramsStateX[5] = 1.0; paramsStateX[6] = 0.0; paramsStateX[7] = 1.0;
    paramsStateY.set(paramsStateX); paramsStateY[3] = 809;
    device.queue.writeBuffer(sobelSetX.buffers.params, 0, paramsStateX);
    device.queue.writeBuffer(sobelSetY.buffers.params, 0, paramsStateY);
    const sobelBindGroupX = device.createBindGroup({ layout: convolveLayout, entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, sobelSetX) });
    const sobelBindGroupY = device.createBindGroup({ layout: convolveLayout, entries: createBindGroupEntriesFromResources(convolveMetadata.bindings, sobelSetY) });

    // Sobel combine (reuse sobel descriptor/metadata)
    const sobelDescriptor = getShaderDescriptor('sobel');
    const sobelMetadata = await getShaderMetadataCached('sobel');
    const sobelLayout = getOrCreateBindGroupLayout(device, sobelDescriptor.id, 'compute', sobelMetadata);
    const sobelPipelineLayout = getOrCreatePipelineLayout(device, sobelDescriptor.id, 'compute', sobelLayout);
    const sobelPipeline = await getOrCreateComputePipeline(device, sobelDescriptor.id, sobelPipelineLayout, sobelDescriptor.entryPoint ?? 'main');

    const sobelParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sobelParamsState = new Float32Array(8);
    const sobelBindingOffsets = { width: 0, height: 1, channelCount: 2, dist_metric: 3, alpha: 5, time: 6, speed: 7 };
    sobelParamsState[sobelBindingOffsets.width] = width;
    sobelParamsState[sobelBindingOffsets.height] = height;
    sobelParamsState[sobelBindingOffsets.channelCount] = 4;
    sobelParamsState[sobelBindingOffsets.dist_metric] = this.userState.sobel_metric;
    sobelParamsState[sobelBindingOffsets.alpha] = 1.0; // full magnitude
    sobelParamsState[sobelBindingOffsets.time] = 0.0;
    sobelParamsState[sobelBindingOffsets.speed] = 1.0;
    device.queue.writeBuffer(sobelParamsBuffer, 0, sobelParamsState);

    const sobelFinalOutputPixelCount = Math.max(width * height, 1);
    const sobelFinalBufferSize = sobelFinalOutputPixelCount * 4 * 4;
    const sobelFinalBuffer = device.createBuffer({ size: sobelFinalBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const sobelCombineBindGroup = device.createBindGroup({ layout: sobelLayout, entries: [
      { binding: 0, resource: { buffer: sobelSetX.buffers.output_buffer } },
      { binding: 1, resource: { buffer: sobelSetY.buffers.output_buffer } },
      { binding: 2, resource: { buffer: sobelFinalBuffer } },
      { binding: 3, resource: { buffer: sobelParamsBuffer } },
    ] });

    // Output texture for sobel final buffer (for subsequent bloom pass)
    const sobelOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = await getBufferToTexturePipeline(device);
    const sobelBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureBindGroupLayout, entries: [
      { binding: 0, resource: { buffer: sobelFinalBuffer } },
      { binding: 1, resource: sobelOutputTexture.createView() },
      { binding: 2, resource: { buffer: sobelParamsBuffer } },
    ] });

    // === Bloom pass (input = sobelOutputTexture) ===
    const bloomDescriptor = this.#descriptorFromMeta(bloomMeta);
    const bloomMetadata = await getShaderMetadataCached(bloomDescriptor.id);
    const bloomResourceSet = createShaderResourceSet(device, bloomDescriptor, bloomMetadata, width, height, { inputTextures: { input_texture: sobelOutputTexture } });
    const bloomLayout = getOrCreateBindGroupLayout(device, bloomDescriptor.id, 'compute', bloomMetadata);
    const bloomPipelineLayout = getOrCreatePipelineLayout(device, bloomDescriptor.id, 'compute', bloomLayout);
    const bloomDownsamplePipeline = await getOrCreateComputePipeline(device, bloomDescriptor.id, bloomPipelineLayout, 'downsample_main');
    const bloomPipeline = await getOrCreateComputePipeline(device, bloomDescriptor.id, bloomPipelineLayout, bloomDescriptor.entryPoint ?? 'upsample_main');
    const bloomBindGroup = device.createBindGroup({ layout: bloomLayout, entries: createBindGroupEntriesFromResources(bloomMetadata.bindings, bloomResourceSet) });
    // Write bloom params (alpha fixed 0.5 per Python ref)
  const bloomParamsState = new Float32Array(12);
  const DOWNSAMPLE_DIVISOR = 100;
  const OFFSET_SCALE = -0.05;
  const downsampleWidth = Math.max(Math.trunc(width / DOWNSAMPLE_DIVISOR), 1);
  const downsampleHeight = Math.max(Math.trunc(height / DOWNSAMPLE_DIVISOR), 1);
  const invDownsampleWidth = downsampleWidth > 0 ? 1 / downsampleWidth : 1;
  const invDownsampleHeight = downsampleHeight > 0 ? 1 / downsampleHeight : 1;
  const offsetX = Math.trunc(width * OFFSET_SCALE);
  const offsetY = Math.trunc(height * OFFSET_SCALE);
  bloomParamsState[0] = width; // width
  bloomParamsState[1] = height; // height
  bloomParamsState[2] = 4; // channel_count
  bloomParamsState[3] = 0.5; // alpha per Python ref
  bloomParamsState[4] = 0.0; // time
  bloomParamsState[5] = 1.0; // speed
  bloomParamsState[6] = downsampleWidth;
  bloomParamsState[7] = downsampleHeight;
  bloomParamsState[8] = invDownsampleWidth;
  bloomParamsState[9] = invDownsampleHeight;
  bloomParamsState[10] = offsetX;
  bloomParamsState[11] = offsetY;
  device.queue.writeBuffer(bloomResourceSet.buffers.params, 0, bloomParamsState);

    // After bloom we get bloomResourceSet.buffers.output_buffer -> convert to texture for normalize pass
    const bloomOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const bloomBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureBindGroupLayout, entries: [
      { binding: 0, resource: { buffer: bloomResourceSet.buffers.output_buffer } },
      { binding: 1, resource: bloomOutputTexture.createView() },
      { binding: 2, resource: { buffer: bloomResourceSet.buffers.params } },
    ] });

    // === Normalize pass (input = bloomOutputTexture) ===
    // Use full 3-pass normalize: stats -> reduce -> apply
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
    normParamsState[0] = width; normParamsState[1] = height; normParamsState[2] = 4; normParamsState[4] = 0; normParamsState[5] = 1;
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
    const normalizeBindGroup = device.createBindGroup({ layout: normalizeLayout, entries: [
      { binding: 0, resource: bloomOutputTexture.createView() },
      { binding: 1, resource: { buffer: normOutputBuffer } },
      { binding: 2, resource: { buffer: normParamsBuffer } },
      { binding: 3, resource: { buffer: normStatsBuffer } },
    ] });
    const normOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const normBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureBindGroupLayout, entries: [
      { binding: 0, resource: { buffer: normOutputBuffer } },
      { binding: 1, resource: normOutputTexture.createView() },
      { binding: 2, resource: { buffer: normParamsBuffer } },
    ] });

    // === Final combine pass (input: original multires texture + normalized edges texture) ===
    const finalDescriptor = getShaderDescriptor(meta.id);
    const finalMetadata = await getShaderMetadataCached(meta.id);
    const finalLayout = getOrCreateBindGroupLayout(device, finalDescriptor.id, 'compute', finalMetadata);
    const finalPipelineLayout = getOrCreatePipelineLayout(device, finalDescriptor.id, 'compute', finalLayout);
  const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, finalPipelineLayout, finalDescriptor.entryPoint ?? 'main');
    const finalParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const finalParamsState = new Float32Array(8);
  const bindingOffsets = { width: 0, height: 1, channel_count: 2, channelCount: 2, alpha: 3, sobel_metric: 4, time: 6, speed: 7 };
    finalParamsState[bindingOffsets.width] = width;
    finalParamsState[bindingOffsets.height] = height;
    finalParamsState[bindingOffsets.channel_count] = 4;
    finalParamsState[bindingOffsets.alpha] = this.userState.alpha;
    finalParamsState[bindingOffsets.sobel_metric] = this.userState.sobel_metric;
    finalParamsState[bindingOffsets.time] = 0.0;
    finalParamsState[bindingOffsets.speed] = 1.0;
    device.queue.writeBuffer(finalParamsBuffer, 0, finalParamsState);
    this.finalParamsState = finalParamsState;

    const finalOutputBufferSize = width * height * 4 * 4;
    const finalOutputBuffer = device.createBuffer({ size: finalOutputBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const finalBindGroup = device.createBindGroup({ layout: finalLayout, entries: [
      { binding: 0, resource: multiresResources.outputTexture.createView() },
      { binding: 1, resource: { buffer: finalOutputBuffer } },
      { binding: 2, resource: { buffer: finalParamsBuffer } },
      { binding: 3, resource: normOutputTexture.createView() },
    ] });

    // Output texture for final result
    const finalOutputTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const finalBufferToTextureBindGroup = device.createBindGroup({ layout: bufferToTextureBindGroupLayout, entries: [
      { binding: 0, resource: { buffer: finalOutputBuffer } },
      { binding: 1, resource: finalOutputTexture.createView() },
      { binding: 2, resource: { buffer: finalParamsBuffer } },
    ] });
    const blitBindGroup = device.createBindGroup({ layout: multiresResources.blitBindGroupLayout, entries: [ { binding: 0, resource: finalOutputTexture.createView() } ] });

    // === Compute pass schedule ===
    const workgroupSize = [8, 8, 1];
    const dispatchDims = [ Math.ceil(width / workgroupSize[0]), Math.ceil(height / workgroupSize[1]), 1 ];
  const downsampleDispatch = [ Math.ceil(Math.max(1, Math.trunc(width / 100)) / 8), Math.ceil(Math.max(1, Math.trunc(height / 100)) / 8), 1 ];
  const passes = [
      // Posterize
      { pipeline: posterizePipeline, bindGroup: posterizeBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
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
  // Bloom downsample + upsample
  { pipeline: bloomDownsamplePipeline, bindGroup: bloomBindGroup, workgroupSize, dispatch: downsampleDispatch },
      { pipeline: bloomPipeline, bindGroup: bloomBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
      // Buffer->Texture (Bloom)
      { pipeline: bufferToTexturePipeline, bindGroup: bloomBufferToTextureBindGroup, workgroupSize: [8,8,1], dispatch: [ Math.ceil(width/8), Math.ceil(height/8), 1 ] },
  // Normalize stats -> reduce -> apply
  { pipeline: normalizeStatsPipeline, bindGroup: normalizeBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
  { pipeline: normalizeReducePipeline, bindGroup: normalizeBindGroup, workgroupSize: [1,1,1], dispatch: [1,1,1] },
  { pipeline: normalizeApplyPipeline, bindGroup: normalizeBindGroup, workgroupSize, dispatch: dispatchDims.slice() },
      // Buffer->Texture (Normalize)
      { pipeline: bufferToTexturePipeline, bindGroup: normBufferToTextureBindGroup, workgroupSize: [8,8,1], dispatch: [ Math.ceil(width/8), Math.ceil(height/8), 1 ] },
      // Final combine
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
      resourceSet: { destroyAll() { try { posterizeResourceSet.destroyAll(); } catch {}; try { sobelSetX.destroyAll(); sobelSetY.destroyAll(); } catch {}; try { bloomResourceSet.destroyAll(); } catch {}; } },
      bindingOffsets,
      sobelParamsBuffer,
      sobelParamsState,
      sobelBindingOffsets,
    };
  }

  #descriptorFromMeta(effectMeta) {
    return {
      id: effectMeta.id,
      label: effectMeta.label || `${effectMeta.id}.wgsl`,
      stage: 'compute',
      entryPoint: (effectMeta.shader && effectMeta.shader.entryPoint) || 'main',
      url: (effectMeta.shader && effectMeta.shader.url) || `/shaders/effects/${effectMeta.id}/${effectMeta.id}.wgsl`,
      resources: effectMeta.resources || {},
    };
  }
}

export default GlowingEdgesEffect;
export const additionalPasses = {};
