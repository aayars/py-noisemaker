import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const FLOAT_SIZE = 4;
const CHANNEL_COUNT = 4;
const DEFAULT_WORKGROUP = [8, 8, 1];

function hsvToRgb(h, s, v) {
  const hue = ((h % 1) + 1) % 1;
  const scaled = hue * 6;
  const sector = Math.floor(scaled);
  const fraction = scaled - sector;
  const p = v * (1 - s);
  const q = v * (1 - fraction * s);
  const t = v * (1 - (1 - fraction) * s);

  switch (sector % 6) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    case 5:
    default:
      return [v, p, q];
  }
}

function createGradientData(width, height) {
  const safeWidth = Math.max(width, 1);
  const safeHeight = Math.max(height, 1);
  const bytesPerRow = Math.ceil((safeWidth * CHANNEL_COUNT * FLOAT_SIZE) / 256) * 256;
  const floatsPerRow = bytesPerRow / FLOAT_SIZE;
  const data = new Float32Array(floatsPerRow * safeHeight);

  for (let y = 0; y < safeHeight; y += 1) {
    const v = safeHeight > 1 ? y / (safeHeight - 1) : 0;
    const saturation = 0.55 + 0.4 * v;
    const value = 0.35 + 0.6 * v;
    const rowOffset = y * floatsPerRow;
    for (let x = 0; x < safeWidth; x += 1) {
      const u = safeWidth > 1 ? x / (safeWidth - 1) : 0;
      const [r, g, b] = hsvToRgb(u, saturation, value);
      const idx = rowOffset + x * CHANNEL_COUNT;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 1.0;
    }
  }

  return { data, bytesPerRow };
}

class ColorMapEffect extends SimpleComputeEffect {
  static metadata = metadata;

  constructor(options = {}) {
    super(options);
    this._clutSignature = null;
  }

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const { device, descriptor, shaderMetadata } = baseResources ?? {};
    if (!device || !descriptor || !shaderMetadata) {
      return baseResources;
    }

    try {
      const layout = this.helpers.getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const pipelineLayout = this.helpers.getOrCreatePipelineLayout(device, descriptor.id, 'compute', layout);

      const resetPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'reset_stats_main',
      );

      const minmaxPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'minmax_main',
      );

      baseResources.resetPipeline = resetPipeline;
      baseResources.resetWorkgroupSize = [1, 1, 1];
      baseResources.minmaxPipeline = minmaxPipeline;
      baseResources.minmaxWorkgroupSize = shaderMetadata.workgroupSize ?? DEFAULT_WORKGROUP.slice();
    } catch (error) {
      this.helpers.logWarn?.('Color Map: failed to prepare auxiliary pipelines.', error);
      baseResources.resetPipeline = null;
      baseResources.minmaxPipeline = null;
      baseResources.resetWorkgroupSize = [1, 1, 1];
      baseResources.minmaxWorkgroupSize = DEFAULT_WORKGROUP.slice();
    }

    return baseResources;
  }

  async ensureResources(context = {}) {
    const resources = await super.ensureResources(context);
    if (!resources || !resources.resourceSet) {
      return resources;
    }

    if (resources.enabled && resources.computeBindGroup) {
      this.#configureComputePasses(resources, context);
    }

    this.#ensureClutTexture(resources, context);

    return resources;
  }

  invalidateResources() {
    super.invalidateResources();
    this._clutSignature = null;
  }

  #configureComputePasses(resources, context) {
    const width = Math.max(Math.trunc(context.width ?? resources.textureWidth ?? 0), 1);
    const height = Math.max(Math.trunc(context.height ?? resources.textureHeight ?? 0), 1);

    const bindGroup = resources.computeBindGroup;
    if (!bindGroup) {
      resources.computePasses = [];
      return;
    }

    const resetPipeline = resources.resetPipeline;
    const minmaxPipeline = resources.minmaxPipeline;
    const colorPipeline = resources.computePipeline;

    const minmaxWorkgroup = Array.isArray(resources.minmaxWorkgroupSize)
      ? resources.minmaxWorkgroupSize
      : DEFAULT_WORKGROUP;
    const colorWorkgroup = Array.isArray(resources.workgroupSize)
      ? resources.workgroupSize
      : DEFAULT_WORKGROUP;

    const minmaxDispatch = [
      Math.max(Math.ceil(width / Math.max(minmaxWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(minmaxWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(minmaxWorkgroup[2] ?? 1, 1)), 1),
    ];

    const colorDispatch = [
      Math.max(Math.ceil(width / Math.max(colorWorkgroup[0] ?? 8, 1)), 1),
      Math.max(Math.ceil(height / Math.max(colorWorkgroup[1] ?? 8, 1)), 1),
      Math.max(Math.ceil(1 / Math.max(colorWorkgroup[2] ?? 1, 1)), 1),
    ];

    const computePasses = [];
    if (resetPipeline) {
      computePasses.push({
        pipeline: resetPipeline,
        bindGroup,
        workgroupSize: resources.resetWorkgroupSize ?? [1, 1, 1],
        dispatch: [1, 1, 1],
      });
    }

    if (minmaxPipeline) {
      computePasses.push({
        pipeline: minmaxPipeline,
        bindGroup,
        workgroupSize: minmaxWorkgroup,
        dispatch: minmaxDispatch,
      });
    }

    if (colorPipeline) {
      computePasses.push({
        pipeline: colorPipeline,
        bindGroup,
        workgroupSize: colorWorkgroup,
        dispatch: colorDispatch,
      });
    }

    resources.computePasses = computePasses;
    resources.workgroupSize = colorWorkgroup;
  }

  #ensureClutTexture(resources, context) {
    const texture = resources.resourceSet.textures?.clut_texture;
    if (!texture || !resources.device) {
      return;
    }

    const width = Math.max(context.width ?? resources.textureWidth ?? 0, 1);
    const height = Math.max(context.height ?? resources.textureHeight ?? 0, 1);
    const signature = `${width}x${height}`;

    if (resources._clutInitialized && this._clutSignature === signature) {
      return;
    }

    const { data, bytesPerRow } = createGradientData(width, height);
    resources.device.queue.writeTexture(
      { texture },
      data,
      {
        bytesPerRow,
        rowsPerImage: height,
      },
      {
        width,
        height,
        depthOrArrayLayers: 1,
      },
    );

    resources._clutInitialized = true;
    this._clutSignature = signature;
  }
}

export default ColorMapEffect;
