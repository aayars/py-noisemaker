/**
 * Stray Hair effect - Single-frame strand generation
 * 
 * Unlike the multi-frame animated worms effect, stray_hair generates complete strands
 * in a single frame, matching the Python implementation which calls worms() once.
 * 
 * Python reference:
 *   mask = values(4, ...)  # Flow field at freq=4
 *   mask = worms(mask, behavior=unruly, density=0.0025-0.00375, duration=8-16, 
 *                kink=5-50, stride=0.5, stride_deviation=0.25, alpha=1)
 *   brightness = values(32, ...)  # High-freq brightness
 *   return blend(tensor, brightness * 0.333, mask * 0.666)
 * 
 * Implementation approach:
 *   1. Generate flow field (value noise @ freq=4)
 *   2. Generate complete strands in one pass (agent simulation loop in shader)
 *   3. Blend with input using brightness modulation
 */

import meta from './meta.json' with { type: 'json' };
import SimpleComputeEffect from '../../common/simple-compute-effect.js';

export default class StrayHairEffect extends SimpleComputeEffect {
  static id = meta.id;
  static label = meta.label;
  static metadata = meta;

  constructor(options) {
    super(options);
  }
}
