import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import meta from './meta.json' with { type: 'json' };

const DIAGRAM_TYPE_VALUES = [0, 11, 12, 21, 22, 31, 41, 42];
const POINT_DISTRIBUTION_VALUES = [
  1000000, // Random
  1000001, // Square
  1000002, // Waffle
  1000003, // Chess
  1000010, // Hex (H)
  1000011, // Hex (V)
  1000050, // Spiral
  1000100, // Circular
  1000101, // Concentric
  1000102, // Rotating
];

function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clampIndex(value, mapping) {
  const length = mapping.length;
  if (length === 0) {
    return 0;
  }
  const numeric = toFiniteNumber(value, 0);
  const rounded = Math.round(numeric);
  return Math.min(Math.max(rounded, 0), length - 1);
}

function mapIndexToEnum(index, mapping) {
  const clamped = clampIndex(index, mapping);
  return mapping[clamped] ?? mapping[0];
}

function mapEnumToIndex(value, mapping) {
  const numeric = toFiniteNumber(value, 0);
  const rounded = Math.round(numeric);
  const existing = mapping.indexOf(rounded);
  if (existing >= 0) {
    return existing;
  }
  return clampIndex(numeric, mapping);
}

export default class VoronoiEffect extends SimpleComputeEffect {
  static metadata = meta;

  constructor(options = {}) {
    super(options);

    const initialDiagramIndex = mapEnumToIndex(this.userState.diagram_type, DIAGRAM_TYPE_VALUES);
    const initialDistribIndex = mapEnumToIndex(this.userState.point_distrib, POINT_DISTRIBUTION_VALUES);

    this.userState.diagram_type = initialDiagramIndex;
    this.userState.point_distrib = initialDistribIndex;

    this.enumState = {
      diagram_type: mapIndexToEnum(initialDiagramIndex, DIAGRAM_TYPE_VALUES),
      point_distrib: mapIndexToEnum(initialDistribIndex, POINT_DISTRIBUTION_VALUES),
    };
  }

  async updateParams(updates = {}) {
    const coerced = { ...updates };

    const hasDiagramUpdate = Object.prototype.hasOwnProperty.call(coerced, 'diagram_type');
    const hasDistribUpdate = Object.prototype.hasOwnProperty.call(coerced, 'point_distrib');

    if (hasDiagramUpdate) {
      coerced.diagram_type = mapEnumToIndex(coerced.diagram_type, DIAGRAM_TYPE_VALUES);
    }

    if (hasDistribUpdate) {
      coerced.point_distrib = mapEnumToIndex(coerced.point_distrib, POINT_DISTRIBUTION_VALUES);
    }

    const result = await super.updateParams(coerced);

    if (hasDiagramUpdate) {
      const index = mapEnumToIndex(this.userState.diagram_type, DIAGRAM_TYPE_VALUES);
      this.userState.diagram_type = index;
      const enumValue = mapIndexToEnum(index, DIAGRAM_TYPE_VALUES);
      this.enumState.diagram_type = enumValue;
      this.#writeEnumToParams('diagram_type', enumValue);
    }

    if (hasDistribUpdate) {
      const index = mapEnumToIndex(this.userState.point_distrib, POINT_DISTRIBUTION_VALUES);
      this.userState.point_distrib = index;
      const enumValue = mapIndexToEnum(index, POINT_DISTRIBUTION_VALUES);
      this.enumState.point_distrib = enumValue;
      this.#writeEnumToParams('point_distrib', enumValue);
    }

    return result;
  }

  async onResourcesCreated(resources, context) {
    const base = await super.onResourcesCreated(resources, context);
    const target = base ?? resources;
    
    // Override the index values with the actual enum constants
    const diagram_offset = this.paramOffsets.get('diagram_type');
    const distrib_offset = this.paramOffsets.get('point_distrib');
    
    if (typeof diagram_offset === 'number' && target.paramsState) {
      target.paramsState[diagram_offset] = Number(this.enumState.diagram_type);
    }
    
    if (typeof distrib_offset === 'number' && target.paramsState) {
      target.paramsState[distrib_offset] = Number(this.enumState.point_distrib);
    }
    
    // Force write the corrected params to GPU
    if (target.paramsBuffer && target.paramsState && context.device) {
      context.device.queue.writeBuffer(target.paramsBuffer, 0, target.paramsState);
    }
    
    return target;
  }

  #writeEnumToParams(name, enumValue, targetResources = this.resources) {
    const offset = this.paramOffsets.get(name);
    const resources = targetResources;
    if (typeof offset !== 'number' || !resources?.paramsState) {
      return;
    }
    const paramsState = resources.paramsState;
    const numericValue = Number(enumValue ?? 0);
    if (paramsState[offset] !== numericValue) {
      paramsState[offset] = numericValue;
      resources.paramsDirty = true;
    }
  }
}
