import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const DEFAULT_WORKGROUP = [8, 8, 1];
const FLOAT_EPSILON = 1e-6;

function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

class BlurEffect extends SimpleComputeEffect {
  static metadata = metadata;

  constructor(options = {}) {
    super(options);
  }

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const { device, descriptor, shaderMetadata } = baseResources ?? {};

    if (!device || !descriptor) {
      return baseResources;
    }

    try {
      const layout = this.helpers.getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const pipelineLayout = this.helpers.getOrCreatePipelineLayout(device, descriptor.id, 'compute', layout);
      const downsamplePipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'downsample_main',
      );

      baseResources.downsamplePipeline = downsamplePipeline;
      baseResources.downsampleWorkgroupSize = DEFAULT_WORKGROUP.slice();
    } catch (error) {
      this.helpers.logWarn?.('Blur: failed to prepare downsample pipeline.', error);
      baseResources.downsamplePipeline = null;
      baseResources.downsampleWorkgroupSize = DEFAULT_WORKGROUP.slice();
    }

    return baseResources;
  }

  async ensureResources(context = {}) {
    const resources = await super.ensureResources(context);
    if (!resources || !resources.enabled || !resources.paramsState) {
      return resources;
    }

    this.#updateDerivedParameters(resources, context);
    return resources;
  }

  #updateDerivedParameters(resources, context) {
    const width = Math.max(Math.trunc(toFiniteNumber(context.width ?? resources.textureWidth, 0)), 1);
    const height = Math.max(Math.trunc(toFiniteNumber(context.height ?? resources.textureHeight, 0)), 1);
    const amount = Math.max(toFiniteNumber(this.userState.amount, 1), 1);

    const downsampleWidth = Math.max(Math.trunc(width / amount), 1);
    const downsampleHeight = Math.max(Math.trunc(height / amount), 1);
    const invDownsampleWidth = downsampleWidth > 0 ? 1 / downsampleWidth : 1;
    const invDownsampleHeight = downsampleHeight > 0 ? 1 / downsampleHeight : 1;

    const offsets = resources.bindingOffsets ?? {};
    this.#writeParam(resources, offsets.downsample_width ?? offsets.downsampleWidth, downsampleWidth);
    this.#writeParam(resources, offsets.downsample_height ?? offsets.downsampleHeight, downsampleHeight);
    this.#writeParam(resources, offsets.inv_downsample_width ?? offsets.invDownsampleWidth, invDownsampleWidth);
    this.#writeParam(resources, offsets.inv_downsample_height ?? offsets.invDownsampleHeight, invDownsampleHeight);

    const downsampleWorkgroup = Array.isArray(resources.downsampleWorkgroupSize)
      ? resources.downsampleWorkgroupSize
      : DEFAULT_WORKGROUP;
    const upsampleWorkgroup = Array.isArray(resources.workgroupSize)
      ? resources.workgroupSize
      : DEFAULT_WORKGROUP;

    const downsampleDispatch = [
      Math.max(Math.ceil(downsampleWidth / Math.max(downsampleWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(downsampleHeight / Math.max(downsampleWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(downsampleWorkgroup[2] ?? 1, 1)), 1),
    ];

    const upsampleDispatch = [
      Math.max(Math.ceil(width / Math.max(upsampleWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(upsampleWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(upsampleWorkgroup[2] ?? 1, 1)), 1),
    ];

    const computePasses = [];
    if (resources.downsamplePipeline && resources.computeBindGroup) {
      computePasses.push({
        pipeline: resources.downsamplePipeline,
        bindGroup: resources.computeBindGroup,
        workgroupSize: downsampleWorkgroup,
        dispatch: downsampleDispatch,
      });
    }

    if (resources.computePipeline && resources.computeBindGroup) {
      computePasses.push({
        pipeline: resources.computePipeline,
        bindGroup: resources.computeBindGroup,
        workgroupSize: upsampleWorkgroup,
        dispatch: upsampleDispatch,
      });
    }

    resources.computePasses = computePasses;
    resources.workgroupSize = upsampleWorkgroup;

  }

  #writeParam(resources, offset, value) {
    if (!Number.isInteger(offset) || offset < 0) {
      return;
    }

    const params = resources.paramsState;
    if (!params || offset >= params.length) {
      return;
    }

    const numeric = toFiniteNumber(value, params[offset] ?? 0);
    if (Math.abs(params[offset] - numeric) <= FLOAT_EPSILON) {
      return;
    }

    params[offset] = numeric;
    resources.paramsDirty = true;
  }
}

export default BlurEffect;
