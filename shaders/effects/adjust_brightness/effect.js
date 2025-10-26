import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

class AdjustBrightnessEffect extends SimpleComputeEffect {
	static metadata = metadata;
}

export default AdjustBrightnessEffect;
