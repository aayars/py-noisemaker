import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

class ScanlineErrorEffect extends SimpleComputeEffect {
	static metadata = metadata;
}

export default ScanlineErrorEffect;
