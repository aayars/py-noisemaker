import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

class KaleidoEffect extends SimpleComputeEffect {
  static metadata = metadata;

  constructor(options = {}) {
    super(options);
  }
}

export default KaleidoEffect;
