import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const PREPARE_WORKGROUP = [8, 8, 1];
const SORT_WORKGROUP = [1, 1, 1];
const FINALIZE_WORKGROUP = [8, 8, 1];
const FLOAT_EPSILON = 1e-6;
const MAX_ROW_PIXELS = 4096;

function toNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

class PixelSortEffect extends SimpleComputeEffect {
  static metadata = metadata;

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const { device, descriptor, shaderMetadata } = baseResources ?? {};

    if (!device || !descriptor) {
      return baseResources;
    }

    try {
      const layout = this.helpers.getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const pipelineLayout = this.helpers.getOrCreatePipelineLayout(device, descriptor.id, 'compute', layout);

      baseResources.preparePipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'prepare',
      );

      baseResources.sortPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'sort_rows',
      );

      baseResources.prepareWorkgroupSize = PREPARE_WORKGROUP.slice();
      baseResources.sortWorkgroupSize = SORT_WORKGROUP.slice();
      baseResources.finalizeWorkgroupSize = FINALIZE_WORKGROUP.slice();
    } catch (error) {
      this.helpers.logWarn?.('Pixel Sort: failed to prepare auxiliary pipelines.', error);
      baseResources.preparePipeline = null;
      baseResources.sortPipeline = null;
      baseResources.prepareWorkgroupSize = PREPARE_WORKGROUP.slice();
      baseResources.sortWorkgroupSize = SORT_WORKGROUP.slice();
      baseResources.finalizeWorkgroupSize = FINALIZE_WORKGROUP.slice();
    }

    return baseResources;
  }

  async ensureResources(context = {}) {
    const resources = await super.ensureResources(context);
    if (!resources || !resources.enabled || !resources.paramsState) {
      return resources;
    }

    const width = Math.max(Math.trunc(toNumber(context.width ?? resources.textureWidth, 0)), 1);
    const height = Math.max(Math.trunc(toNumber(context.height ?? resources.textureHeight, 0)), 1);
    const want = this.#computeTargetSize(width, height);

    this.#writeParam(resources, resources.bindingOffsets?.want_size ?? resources.bindingOffsets?.wantSize, want);

    this.#configureComputePasses(resources, { width, height, want });
    return resources;
  }

  #computeTargetSize(width, height) {
    const maxDim = Math.max(width, height, 1);
    const doubled = maxDim * 2;
    const clamped = Math.min(Math.max(Math.round(doubled), maxDim), MAX_ROW_PIXELS);
    return clamped;
  }

  #configureComputePasses(resources, dimensions) {
    const { width, height, want } = dimensions;
    const bindGroup = resources.computeBindGroup;
    if (!bindGroup) {
      resources.computePasses = [];
      return;
    }

    const passes = [];
    const preparePipeline = resources.preparePipeline;
    const sortPipeline = resources.sortPipeline;
    const finalizePipeline = resources.computePipeline;

    if (preparePipeline) {
      const workgroup = Array.isArray(resources.prepareWorkgroupSize) ? resources.prepareWorkgroupSize : PREPARE_WORKGROUP;
      passes.push({
        pipeline: preparePipeline,
        bindGroup,
        workgroupSize: workgroup,
        getDispatch: () => [
          Math.max(Math.ceil(want / Math.max(workgroup[0] ?? 8, 1)), 1),
          Math.max(Math.ceil(want / Math.max(workgroup[1] ?? 8, 1)), 1),
          Math.max(Math.ceil(1 / Math.max(workgroup[2] ?? 1, 1)), 1),
        ],
      });
    }

    if (sortPipeline && want > 0) {
      const workgroup = Array.isArray(resources.sortWorkgroupSize) ? resources.sortWorkgroupSize : SORT_WORKGROUP;
      passes.push({
        pipeline: sortPipeline,
        bindGroup,
        workgroupSize: workgroup,
        dispatch: [1, Math.max(want, 1), 1],
      });
    }

    if (finalizePipeline) {
      const workgroup = Array.isArray(resources.finalizeWorkgroupSize)
        ? resources.finalizeWorkgroupSize
        : Array.isArray(resources.workgroupSize)
          ? resources.workgroupSize
          : FINALIZE_WORKGROUP;
      passes.push({
        pipeline: finalizePipeline,
        bindGroup,
        workgroupSize: workgroup,
        getDispatch: () => [
          Math.max(Math.ceil(width / Math.max(workgroup[0] ?? 8, 1)), 1),
          Math.max(Math.ceil(height / Math.max(workgroup[1] ?? 8, 1)), 1),
          Math.max(Math.ceil(1 / Math.max(workgroup[2] ?? 1, 1)), 1),
        ],
      });
      resources.workgroupSize = workgroup;
    }

    resources.computePasses = passes;
  }

  #writeParam(resources, offset, value) {
    if (!Number.isInteger(offset) || offset < 0) {
      return;
    }

    const params = resources.paramsState;
    if (!params || offset >= params.length) {
      return;
    }

    const numeric = toNumber(value, params[offset] ?? 0);
    if (Math.abs(params[offset] - numeric) <= FLOAT_EPSILON) {
      return;
    }

    params[offset] = numeric;
    resources.paramsDirty = true;
  }
}

export default PixelSortEffect;

