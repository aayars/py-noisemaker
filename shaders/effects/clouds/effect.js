import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const DEFAULT_WORKGROUP = [8, 8, 1];
const FLOAT_EPSILON = 1e-6;
const DEFAULT_PRE_SCALE = 0.25;
const SHADE_OFFSET_MAX = 15;

function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function hashToUnit(seedA, seedB, seedC) {
  const mixed = seedA * 374761393 + seedB * 668265263 + seedC * 69069;
  const sinValue = Math.sin(mixed);
  return sinValue - Math.floor(sinValue);
}

function randomIntInRange(minValue, maxValue, seedA, seedB, seedC) {
  const unit = hashToUnit(seedA, seedB, seedC);
  const min = Math.min(minValue, maxValue);
  const max = Math.max(minValue, maxValue);
  const span = max - min + 1;
  const clampedUnit = Math.min(Math.max(unit, 0), 1 - Number.EPSILON);
  return min + Math.floor(clampedUnit * span);
}

class CloudsEffect extends SimpleComputeEffect {
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

        try {
          const shadePipeline = await this.helpers.getOrCreateComputePipeline(
            device,
            descriptor.id,
            pipelineLayout,
            'shade_main',
          );
          baseResources.shadePipeline = shadePipeline;
          baseResources.shadeWorkgroupSize = DEFAULT_WORKGROUP.slice();
          // create normalize pipeline for min/max normalization pass
          const normalizePipeline = await this.helpers.getOrCreateComputePipeline(
            device,
            descriptor.id,
            pipelineLayout,
            'normalize_main',
          );
          baseResources.normalizePipeline = normalizePipeline;
        } catch (shadeError) {
        this.helpers.logWarn?.('Clouds: failed to prepare shade pipeline.', shadeError);
        baseResources.shadePipeline = null;
      baseResources.shadeWorkgroupSize = DEFAULT_WORKGROUP.slice();
    }
  } catch (error) {
    this.helpers.logWarn?.('Clouds: failed to prepare downsample pipeline.', error);
    baseResources.downsamplePipeline = null;
    baseResources.downsampleWorkgroupSize = DEFAULT_WORKGROUP.slice();
    baseResources.shadePipeline = null;
    baseResources.shadeWorkgroupSize = DEFAULT_WORKGROUP.slice();
  }

  return baseResources;
}  async ensureResources(context = {}) {
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

  const downsampleWidth = Math.max(Math.floor(width * DEFAULT_PRE_SCALE), 1);
  const downsampleHeight = Math.max(Math.floor(height * DEFAULT_PRE_SCALE), 1);
    const invDownsampleWidth = downsampleWidth > 0 ? 1 / downsampleWidth : 1;
    const invDownsampleHeight = downsampleHeight > 0 ? 1 / downsampleHeight : 1;

    const offsets = resources.bindingOffsets ?? {};
    this.#writeParam(resources, offsets.downsample_width ?? offsets.downsampleWidth, downsampleWidth);
    this.#writeParam(resources, offsets.downsample_height ?? offsets.downsampleHeight, downsampleHeight);
    this.#writeParam(resources, offsets.inv_downsample_width ?? offsets.invDownsampleWidth, invDownsampleWidth);
    this.#writeParam(resources, offsets.inv_downsample_height ?? offsets.invDownsampleHeight, invDownsampleHeight);
    this.#writeParam(resources, offsets.pre_scale ?? offsets.preScale, DEFAULT_PRE_SCALE);

    const [shadeOffsetX, shadeOffsetY] = this.#computeShadeOffsets(width, height);
    this.#writeParam(resources, offsets.shade_offset_x ?? offsets.shadeOffsetX, shadeOffsetX);
    this.#writeParam(resources, offsets.shade_offset_y ?? offsets.shadeOffsetY, shadeOffsetY);

    const downsampleWorkgroup = Array.isArray(resources.downsampleWorkgroupSize)
      ? resources.downsampleWorkgroupSize
      : DEFAULT_WORKGROUP;
    const shadeWorkgroup = Array.isArray(resources.shadeWorkgroupSize)
      ? resources.shadeWorkgroupSize
      : downsampleWorkgroup;
    const upsampleWorkgroup = Array.isArray(resources.workgroupSize)
      ? resources.workgroupSize
      : DEFAULT_WORKGROUP;

    const downsampleDispatch = [
      Math.max(Math.ceil(downsampleWidth / Math.max(downsampleWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(downsampleHeight / Math.max(downsampleWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(downsampleWorkgroup[2] ?? 1, 1)), 1),
    ];

    const shadeDispatch = [
      Math.max(Math.ceil(downsampleWidth / Math.max(shadeWorkgroup[0] ?? downsampleWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(downsampleHeight / Math.max(shadeWorkgroup[1] ?? downsampleWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(shadeWorkgroup[2] ?? downsampleWorkgroup[2] ?? 1, 1)), 1),
    ];

    const upsampleDispatch = [
      Math.max(Math.ceil(width / Math.max(upsampleWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(upsampleWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(upsampleWorkgroup[2] ?? 1, 1)), 1),
    ];

    const computePasses = [];
    const hasBindGroup = Boolean(resources.computeBindGroup);
    const hasDownsample = hasBindGroup && Boolean(resources.downsamplePipeline);
    const hasShade = hasBindGroup && Boolean(resources.shadePipeline);
    const hasUpsample = hasBindGroup && Boolean(resources.computePipeline);

    if (hasDownsample) {
      computePasses.push({
        pipeline: resources.downsamplePipeline,
        bindGroup: resources.computeBindGroup,
        workgroupSize: downsampleWorkgroup,
        dispatch: downsampleDispatch,
      });
    }

    // Normalize control values to compute min/max before shading
    if (resources.normalizePipeline && hasBindGroup) {
      const normalizeDispatch = [1, 1, 1];
      computePasses.push({
        pipeline: resources.normalizePipeline,
        bindGroup: resources.computeBindGroup,
        workgroupSize: [1, 1, 1],
        dispatch: normalizeDispatch,
      });
    }

    if (hasShade) {
      computePasses.push({
        pipeline: resources.shadePipeline,
        bindGroup: resources.computeBindGroup,
        workgroupSize: shadeWorkgroup,
        dispatch: shadeDispatch,
      });
    }

    if (hasDownsample && hasUpsample) {
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

  #computeShadeOffsets(width, height) {
    const seedA = Math.trunc(width) || 1;
    const seedB = Math.trunc(height) || 1;
    const offsetX = randomIntInRange(-SHADE_OFFSET_MAX, SHADE_OFFSET_MAX, seedA, seedB, 0x1f123bb);
    const offsetY = randomIntInRange(-SHADE_OFFSET_MAX, SHADE_OFFSET_MAX, seedB, seedA ^ 0x9e3779b, 0x632be59);
    return [offsetX, offsetY];
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
    if (Math.abs((params[offset] ?? 0) - numeric) <= FLOAT_EPSILON) {
      return;
    }

    params[offset] = numeric;
    resources.paramsDirty = true;
  }
}

export default CloudsEffect;
