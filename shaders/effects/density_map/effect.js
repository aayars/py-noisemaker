import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const DEFAULT_PIXEL_WORKGROUP = [8, 8, 1];
const HISTOGRAM_RESET_WORKGROUP = [64, 1, 1];
const HISTOGRAM_FINALIZE_WORKGROUP = [64, 1, 1];

class DensityMapEffect extends SimpleComputeEffect {
  static metadata = metadata;

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const finalResources = baseResources ?? resources;
    const { device, descriptor, shaderMetadata } = finalResources ?? {};
    if (!device || !descriptor || !shaderMetadata) {
      return finalResources;
    }

    const width = Math.max(Math.trunc(context.width ?? finalResources.textureWidth ?? 0), 1);
    const height = Math.max(Math.trunc(context.height ?? finalResources.textureHeight ?? 0), 1);
    const binCount = Math.max(width, height, 1);

    try {
      const resourceSet = finalResources.resourceSet;
      const textures = resourceSet?.textures ?? {};
      const buffers = resourceSet?.buffers ?? {};
      const inputTexture = textures.input_texture;
      const outputBuffer = finalResources.outputBuffer ?? buffers.output_buffer ?? null;
      const paramsBuffer = finalResources.paramsBuffer ?? buffers.params ?? null;

      const histogramSize = binCount * Uint32Array.BYTES_PER_ELEMENT;
      const previousHistogram = buffers.histogram_buffer;
      if (previousHistogram?.destroy) {
        try {
          previousHistogram.destroy();
        } catch (error) {
          this.helpers?.logWarn?.('Density Map: failed to destroy prior histogram buffer:', error);
        }
      }

      finalResources.histogramBuffer = device.createBuffer({
        label: 'Density Map Histogram Buffer',
        size: histogramSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      if (buffers) {
        resourceSet.buffers.histogram_buffer = finalResources.histogramBuffer;
      }

      const statsSize = Uint32Array.BYTES_PER_ELEMENT * 4;
      const previousStats = buffers.stats_buffer;
      if (previousStats?.destroy) {
        try {
          previousStats.destroy();
        } catch (error) {
          this.helpers?.logWarn?.('Density Map: failed to destroy prior stats buffer:', error);
        }
      }

      finalResources.statsBuffer = device.createBuffer({
        label: 'Density Map Stats Buffer',
        size: statsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      if (buffers) {
        resourceSet.buffers.stats_buffer = finalResources.statsBuffer;
      }

      const bindGroupLayout = finalResources.computeBindGroupLayout;
      if (bindGroupLayout && inputTexture && outputBuffer && paramsBuffer && finalResources.histogramBuffer && finalResources.statsBuffer) {
        finalResources.computeBindGroup = device.createBindGroup({
          label: 'Density Map Compute Bind Group',
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: inputTexture.createView() },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: finalResources.histogramBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
            { binding: 4, resource: { buffer: finalResources.statsBuffer } },
          ],
        });
      }

      const layout = this.helpers.getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const pipelineLayout = this.helpers.getOrCreatePipelineLayout(device, descriptor.id, 'compute', layout);

      finalResources.resetPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'reset_histogram_main',
      );

      finalResources.reducePipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'reduce_minmax_main',
      );

      finalResources.histogramPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'histogram_main',
      );

      finalResources.finalizePipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'finalize_histogram_main',
      );

      // Each entry point has explicit workgroup size in shader:
      // reset_histogram_main: @workgroup_size(64, 1, 1)
      // reduce_minmax_main: @workgroup_size(8, 8, 1)
      // histogram_main: @workgroup_size(8, 8, 1)
      // finalize_histogram_main: @workgroup_size(64, 1, 1)
      // main (remap): @workgroup_size(8, 8, 1)
      finalResources.resetWorkgroupSize = [64, 1, 1];
      finalResources.reduceWorkgroupSize = [8, 8, 1];
      finalResources.histogramWorkgroupSize = [8, 8, 1];
      finalResources.finalizeWorkgroupSize = [64, 1, 1];
      finalResources.remapWorkgroupSize = [8, 8, 1];
      finalResources.workgroupSize = [8, 8, 1];
      finalResources.histogramBinCount = binCount;
    } catch (error) {
      this.helpers?.logWarn?.('Density Map: failed to prepare GPU resources.', error);
      finalResources.resetPipeline = null;
      finalResources.reducePipeline = null;
      finalResources.histogramPipeline = null;
      finalResources.finalizePipeline = null;
      finalResources.resetWorkgroupSize = [64, 1, 1];
      finalResources.reduceWorkgroupSize = [8, 8, 1];
      finalResources.histogramWorkgroupSize = [8, 8, 1];
      finalResources.finalizeWorkgroupSize = [64, 1, 1];
      finalResources.remapWorkgroupSize = [8, 8, 1];
    }

    return finalResources;
  }

  async ensureResources(context = {}) {
    const resources = await super.ensureResources(context);
    if (!resources) {
      return resources;
    }

    if (resources.enabled && resources.computeBindGroup) {
      this.#configureComputePasses(resources, context);
    } else {
      resources.computePasses = [];
    }

    return resources;
  }

  #configureComputePasses(resources, context) {
    const width = Math.max(Math.trunc(context.width ?? resources.textureWidth ?? 0), 1);
    const height = Math.max(Math.trunc(context.height ?? resources.textureHeight ?? 0), 1);
    const binCount = Math.max(width, height, 1);

    const bindGroup = resources.computeBindGroup;
    if (!bindGroup) {
      resources.computePasses = [];
      return;
    }

    const resetPipeline = resources.resetPipeline;
    const reducePipeline = resources.reducePipeline;
    const histogramPipeline = resources.histogramPipeline;
    const finalizePipeline = resources.finalizePipeline;
    const remapPipeline = resources.computePipeline;

    const resetWorkgroup = Array.isArray(resources.resetWorkgroupSize)
      ? resources.resetWorkgroupSize
      : HISTOGRAM_RESET_WORKGROUP;
    const reduceWorkgroup = Array.isArray(resources.reduceWorkgroupSize)
      ? resources.reduceWorkgroupSize
      : DEFAULT_PIXEL_WORKGROUP;
    const histogramWorkgroup = Array.isArray(resources.histogramWorkgroupSize)
      ? resources.histogramWorkgroupSize
      : reduceWorkgroup;
    const finalizeWorkgroup = Array.isArray(resources.finalizeWorkgroupSize)
      ? resources.finalizeWorkgroupSize
      : HISTOGRAM_FINALIZE_WORKGROUP;
    const remapWorkgroup = Array.isArray(resources.remapWorkgroupSize)
      ? resources.remapWorkgroupSize
      : DEFAULT_PIXEL_WORKGROUP;

    const resetDispatch = [
      Math.max(Math.ceil(binCount / Math.max(resetWorkgroup[0] ?? 64, 1)), 1),
      1,
      1,
    ];

    const pixelDispatch = [
      Math.max(Math.ceil(width / Math.max(reduceWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(reduceWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(reduceWorkgroup[2] ?? 1, 1)), 1),
    ];

    const histogramDispatch = [
      Math.max(Math.ceil(width / Math.max(histogramWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(histogramWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(histogramWorkgroup[2] ?? 1, 1)), 1),
    ];

    const finalizeDispatch = [
      Math.max(Math.ceil(binCount / Math.max(finalizeWorkgroup[0] ?? 64, 1)), 1),
      1,
      1,
    ];

    const remapDispatch = [
      Math.max(Math.ceil(width / Math.max(remapWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(remapWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(remapWorkgroup[2] ?? 1, 1)), 1),
    ];

    console.log('[density_map] width:', width, 'height:', height, 'binCount:', binCount);
    console.log('[density_map] remapWorkgroup:', remapWorkgroup);
    console.log('[density_map] remapDispatch:', remapDispatch);

    const computePasses = [];
    if (resetPipeline) {
      computePasses.push({
        pipeline: resetPipeline,
        bindGroup,
        workgroupSize: resetWorkgroup,
        dispatch: resetDispatch,
      });
    }
    if (reducePipeline) {
      computePasses.push({
        pipeline: reducePipeline,
        bindGroup,
        workgroupSize: reduceWorkgroup,
        dispatch: pixelDispatch,
      });
    }
    if (histogramPipeline) {
      computePasses.push({
        pipeline: histogramPipeline,
        bindGroup,
        workgroupSize: histogramWorkgroup,
        dispatch: histogramDispatch,
      });
    }
    if (finalizePipeline) {
      computePasses.push({
        pipeline: finalizePipeline,
        bindGroup,
        workgroupSize: finalizeWorkgroup,
        dispatch: finalizeDispatch,
      });
    }
    if (remapPipeline) {
      computePasses.push({
        pipeline: remapPipeline,
        bindGroup,
        workgroupSize: remapWorkgroup,
        dispatch: remapDispatch,
      });
    }

    resources.computePasses = computePasses;
    resources.workgroupSize = remapWorkgroup;
    resources.histogramBinCount = binCount;
  }

  destroy() {
    if (this.resources?.histogramBuffer) {
      this.resources.histogramBuffer.destroy();
      this.resources.histogramBuffer = null;
    }
    if (this.resources?.statsBuffer) {
      this.resources.statsBuffer.destroy();
      this.resources.statsBuffer = null;
    }
    super.destroy();
  }
}

export default DensityMapEffect;
