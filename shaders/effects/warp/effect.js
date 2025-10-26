import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import meta from './meta.json' with { type: 'json' };

export default class WarpEffect extends SimpleComputeEffect {
  static metadata = meta;

  async onResourcesCreated(resources, context) {
    const paramsState = resources?.paramsState;
    const bindingOffsets = resources?.bindingOffsets ?? {};
    const device = context?.device;
    const paramsBuffer = resources?.paramsBuffer;

    if (paramsState) {
      let dirty = false;

      const splineOffset = bindingOffsets.spline_order;
      if (typeof splineOffset === 'number' && paramsState[splineOffset] === 0) {
        paramsState[splineOffset] = 3; // Bicubic by default.
        dirty = true;
      }

      const signedOffset = bindingOffsets.signed_range;
      if (typeof signedOffset === 'number' && paramsState[signedOffset] === 0) {
        paramsState[signedOffset] = 1;
        dirty = true;
      }

      const speedOffset = bindingOffsets.speed;
      if (typeof speedOffset === 'number' && paramsState[speedOffset] === 0) {
        paramsState[speedOffset] = 1;
        dirty = true;
      }

      if (dirty && device && paramsBuffer) {
        device.queue.writeBuffer(paramsBuffer, 0, paramsState);
        resources.paramsDirty = false;
      }
    }

    return resources;
  }
}
