import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const DEFAULT_WORKGROUP = [8, 8, 1];

class ConvolveEffect extends SimpleComputeEffect {
  static metadata = metadata;

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const finalResources = baseResources ?? resources;
    const { device, descriptor, shaderMetadata } = finalResources ?? {};
    if (!device || !descriptor || !shaderMetadata) {
      return finalResources;
    }

    try {
      const layout = this.helpers.getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
      const pipelineLayout = this.helpers.getOrCreatePipelineLayout(device, descriptor.id, 'compute', layout);

      finalResources.resetPipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'reset_stats_main',
      );

      finalResources.convolvePipeline = await this.helpers.getOrCreateComputePipeline(
        device,
        descriptor.id,
        pipelineLayout,
        'convolve_main',
      );

      finalResources.resetWorkgroupSize = [1, 1, 1];
      finalResources.convolveWorkgroupSize = DEFAULT_WORKGROUP.slice();
      finalResources.normalizeWorkgroupSize = DEFAULT_WORKGROUP.slice();
      finalResources.workgroupSize = DEFAULT_WORKGROUP.slice();
    } catch (error) {
      this.helpers.logWarn?.('Convolve: failed to prepare auxiliary pipelines.', error);
      finalResources.resetPipeline = null;
      finalResources.convolvePipeline = null;
      finalResources.resetWorkgroupSize = [1, 1, 1];
      finalResources.convolveWorkgroupSize = DEFAULT_WORKGROUP.slice();
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

    const bindGroup = resources.computeBindGroup;
    if (!bindGroup) {
      resources.computePasses = [];
      return;
    }

    const resetPipeline = resources.resetPipeline;
    const convolvePipeline = resources.convolvePipeline;
    const normalizePipeline = resources.computePipeline;

    const resetPass = resetPipeline
      ? {
          pipeline: resetPipeline,
          bindGroup,
          workgroupSize: resources.resetWorkgroupSize ?? [1, 1, 1],
          dispatch: [1, 1, 1],
        }
      : null;

    const convolveWorkgroup = Array.isArray(resources.convolveWorkgroupSize)
      ? resources.convolveWorkgroupSize
      : DEFAULT_WORKGROUP;

    const normalizeWorkgroup = Array.isArray(resources.normalizeWorkgroupSize)
      ? resources.normalizeWorkgroupSize
      : Array.isArray(resources.workgroupSize)
        ? resources.workgroupSize
        : DEFAULT_WORKGROUP;

    const convolvePass = convolvePipeline
      ? {
          pipeline: convolvePipeline,
          bindGroup,
          workgroupSize: convolveWorkgroup,
          dispatch: [
            Math.max(Math.ceil(width / Math.max(convolveWorkgroup[0] ?? 8, 1)), 1),
            Math.max(Math.ceil(height / Math.max(convolveWorkgroup[1] ?? 8, 1)), 1),
            Math.max(Math.ceil(1 / Math.max(convolveWorkgroup[2] ?? 1, 1)), 1),
          ],
        }
      : null;

    const normalizePass = normalizePipeline
      ? {
          pipeline: normalizePipeline,
          bindGroup,
          workgroupSize: normalizeWorkgroup,
          dispatch: [
            Math.max(Math.ceil(width / Math.max(normalizeWorkgroup[0] ?? 8, 1)), 1),
            Math.max(Math.ceil(height / Math.max(normalizeWorkgroup[1] ?? 8, 1)), 1),
            Math.max(Math.ceil(1 / Math.max(normalizeWorkgroup[2] ?? 1, 1)), 1),
          ],
        }
      : null;

    const computePasses = [];
    if (resetPass) {
      computePasses.push(resetPass);
    }
    if (convolvePass) {
      computePasses.push(convolvePass);
    }
    if (normalizePass) {
      computePasses.push(normalizePass);
    }

    resources.computePasses = computePasses;
    resources.workgroupSize = normalizeWorkgroup;
  }
}

export default ConvolveEffect;
