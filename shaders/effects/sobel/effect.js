import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import meta from './meta.json' with { type: 'json' };

export default class SobelEffect extends SimpleComputeEffect {
  static metadata = meta;
}
