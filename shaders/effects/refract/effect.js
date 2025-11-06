import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

class RefractEffect extends SimpleComputeEffect {
	static metadata = metadata;

	getResourceCreationOptions(context = {}) {
		const base = super.getResourceCreationOptions(context) ?? {};
		const inputTextures = { ...(base.inputTextures ?? {}) };
		const sourceTexture = context?.multiresResources?.outputTexture;
		if (sourceTexture) {
			inputTextures.reference_x_texture = sourceTexture;
			inputTextures.reference_y_texture = sourceTexture;
		}

		return {
			...base,
			inputTextures,
		};
	}
}

export default RefractEffect;
