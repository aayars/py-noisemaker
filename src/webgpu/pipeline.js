import * as SHADERS from './shaders.js';

const DEFAULT_WORKGROUP_SIZE = [8, 8, 1];

/**
 * Represents a single std140 uniform field within a stage specific buffer.
 * Offsets and sizes are computed so callers can mirror CPU side packing
 * routines without guessing alignment requirements.
 *
 * @typedef {Object} UniformField
 * @property {string} name Human readable parameter name.
 * @property {'f32'|'i32'|'u32'} scalarType Scalar type stored in the buffer.
 * @property {number} components Number of scalar components (1 for scalars,
 *   2-4 for vector like entries).
 * @property {number} offset Byte offset from the beginning of the uniform
 *   struct following std140 alignment rules.
 * @property {number} size Total byte size of the field, including padding
 *   inserted to satisfy alignment constraints.
 * @property {boolean} [bool] True when the value represents a boolean. Booleans
 *   are encoded as `u32(0|1)` so JS callers can mirror Python's packing logic.
 * @property {Array<number>|number} defaultValue Default scalar/vector value used
 *   when the preset does not override the parameter. Arrays are plain numbers so
 *   that they can be cloned into `Float32Array` instances without mutation.
 */

/**
 * Metadata describing the layout of a stage specific uniform buffer.
 *
 * @typedef {Object} UniformLayout
 * @property {number} size Total byte size of the uniform struct. The size is
 *   always padded up to the next 16-byte boundary so staging buffers can be
 *   double buffered without alignment concerns.
 * @property {UniformField[]} fields Ordered list of fields contained in the
 *   struct. Offsets respect std140 packing rules.
 */

/**
 * Description of a compute stage derived from a preset. Descriptors are ordered
 * exactly as they will execute (generator → octave effects → post effects →
 * final effects) so later tasks can bind resources without recomputing the
 * topology.
 *
 * @typedef {Object} StageDescriptor
 * @property {number} index Sequential index inside the program.
 * @property {'generator'|'effect'} stageType Stage classification.
 * @property {string} signature Stable topology key `(category:name)` used to
 *   detect ordering changes.
 * @property {string} category Logical preset bucket (`generator`, `octave`,
 *   `post`, `final`).
 * @property {string} label Human readable label for debug logs.
 * @property {string|null} shaderId Export name from {@link SHADERS} identifying
 *   the WGSL template backing this stage. `null` when no GPU variant is known.
 * @property {string|null} shaderSource WGSL code after applying specialization
 *   constants. `null` when unavailable.
 * @property {object|null} pipeline Cached `GPUComputePipeline` instance. The
 *   pipeline is optional because unit tests run without WebGPU access.
 * @property {UniformLayout|null} uniformLayout std140 layout for the stage
 *   specific uniform buffer or `null` when the stage does not expose uniforms.
 * @property {Record<string, number|boolean|Array<number>>} uniformDefaults
 *   Default scalar parameters mirrored from the CPU implementation.
 * @property {Array<{name: string, resourceType: string, value: unknown}>}
 *   resourceParams Parameters that require auxiliary textures/samplers or other
 *   non-uniform bindings. These are recorded for future tasks to translate into
 *   bind group entries.
 * @property {{
 *   hasUniform: boolean,
 *   hasFrameUniform: boolean,
 *   readsTexture: boolean,
 *   writesTexture: boolean,
 *   auxiliary: Array<{ name: string, resourceType: string, optional: boolean }>,
 * }} bindings Summary of bind group requirements.
 * @property {{
 *   workgroupSize: [number, number, number],
 *   constants: Record<string, unknown>,
 * }} specialization Specialization constants that were applied to the WGSL
 *   template. Resolution and channel counts are recorded in `constants` so the
 *   command encoder can adjust dispatch sizes later.
 * @property {boolean} gpuSupported True when the stage can execute on the GPU
 *   (shader template discovered and all uniforms represent std140 compatible
 *   scalars/vectors). Stages with `false` remain in the descriptor list so the
 *   CPU pipeline can fall back selectively.
 * @property {Array<{ level: 'info'|'warn'|'error', message: string }>} issues
 *   Validation notes captured while building the descriptor.
 */

/**
 * Compile a preset into an ordered list of GPU ready stage descriptors wrapped
 * in a {@link PresetProgram}. The returned program owns double buffered uniform
 * views so animation code can flip between them without reallocating.
 *
 * @param {import('../composer.js').Preset} preset Preset instance to compile.
 * @param {import('../context.js').Context} ctx Active rendering context.
 * @param {object} [sharedResources] Optional helpers for shader templating or
 *   pipeline caching.
 * @returns {PresetProgram}
 */
export function compilePreset(preset, ctx, sharedResources = {}) {
  if (!preset) {
    throw new Error('compilePreset requires a Preset instance.');
  }

  const stageSources = collectPresetStages(preset);
  const descriptors = stageSources.map((stage, index) =>
    finalizeStageDescriptor(stage, index, ctx, sharedResources)
  );
  return new PresetProgram(ctx, descriptors, sharedResources);
}

/**
 * Immutable program containing ordered stage descriptors and double buffered
 * uniform views. The class exposes helpers so callers can diff preset topology
 * and retrieve per-stage uniform buffers without reinterpreting metadata.
 */
export class PresetProgram {
  constructor(ctx, stages, sharedResources = {}) {
    this.ctx = ctx;
    this.stages = stages;
    this.sharedResources = sharedResources;
    this.topologySignature = stages.map((stage) => stage.signature).join('|');
    this.uniformBuffers = stages.map((stage) => buildUniformBufferPair(stage));
  }

  /**
   * Number of GPU stages described by the program.
   * @returns {number}
   */
  get stageCount() {
    return this.stages.length;
  }

  /**
   * Retrieve the descriptor for the given stage index.
   * @param {number} index
   * @returns {StageDescriptor}
   */
  getStageDescriptor(index) {
    return this.stages[index];
  }

  /**
   * Obtain a double buffered uniform view for the requested stage.
   *
   * @param {number} index Stage index.
   * @param {number} [bufferIndex=0] Selects either buffer in the double buffered
   *   pair. The index is modulo the available views so callers can pass frame
   *   counters directly.
   * @returns {{ buffer: ArrayBuffer, view: DataView, layout: UniformLayout }|null}
   */
  getUniformBufferView(index, bufferIndex = 0) {
    const holder = this.uniformBuffers[index];
    if (!holder) return null;
    const views = holder.views;
    if (!views.length) return null;
    const selected = views[((bufferIndex % views.length) + views.length) % views.length];
    return {
      buffer: holder.buffers[selected.index],
      view: selected.view,
      layout: this.stages[index].uniformLayout,
    };
  }

  /**
   * Check whether the program topology matches a new preset instance. Only the
   * generator/effect ordering is compared; parameter changes do not require
   * recompilation.
   *
   * @param {import('../composer.js').Preset} preset
   * @returns {boolean}
   */
  matchesPreset(preset) {
    if (!preset) return false;
    const signature = buildTopologySignatureFromPreset(preset);
    return signature === this.topologySignature;
  }
}

function buildUniformBufferPair(stage) {
  const layout = stage.uniformLayout;
  if (!layout || !layout.size) {
    return { buffers: [], views: [] };
  }
  const buffers = [new ArrayBuffer(layout.size), new ArrayBuffer(layout.size)];
  const views = buffers.map((buffer, index) => ({ view: new DataView(buffer), index }));
  return { buffers, views };
}

function collectPresetStages(preset) {
  const stages = [];
  const generatorParams = cloneStageParams(preset.generator || {});
  stages.push({
    stageType: 'generator',
    category: 'generator',
    name: generatorParams.generator || 'multires',
    label: 'generator',
    params: generatorParams,
    source: preset.generator || {},
  });

  const buckets = [
    ['octave_effects', 'octave'],
    ['post_effects', 'post'],
    ['final_effects', 'final'],
  ];
  for (const [key, category] of buckets) {
    const effects = Array.isArray(preset[key]) ? preset[key] : [];
    for (const effect of effects) {
      stages.push(buildEffectStage(effect, category));
    }
  }

  return stages;
}

function buildEffectStage(effect, category) {
  if (typeof effect === 'function' && effect.__effectName) {
    const params = cloneStageParams(effect.__params || {});
    return {
      stageType: 'effect',
      category,
      name: effect.__effectName,
      label: effect.__effectName,
      params,
      source: effect,
    };
  }
  if (effect && typeof effect === 'object') {
    return {
      stageType: 'effect',
      category,
      name: effect.name || 'nested',
      label: effect.name || 'nested',
      params: cloneStageParams(effect.params || {}),
      source: effect,
      unsupported: true,
    };
  }
  return {
    stageType: 'effect',
    category,
    name: 'anonymous',
    label: 'anonymous',
    params: {},
    source: effect,
    unsupported: true,
  };
}

function finalizeStageDescriptor(stage, index, ctx, sharedResources) {
  const descriptor = {
    index,
    stageType: stage.stageType,
    category: stage.category,
    label: stage.label,
    signature: `${stage.category}:${stage.name}`,
    shaderId: resolveShaderId(stage),
    shaderSource: null,
    pipeline: null,
    uniformLayout: null,
    uniformDefaults: {},
    resourceParams: [],
    bindings: {
      hasUniform: false,
      hasFrameUniform: true,
      readsTexture: stage.stageType !== 'generator',
      writesTexture: true,
      auxiliary: [],
    },
    specialization: {
      workgroupSize: Array.isArray(sharedResources.workgroupSize)
        ? [...sharedResources.workgroupSize]
        : [...DEFAULT_WORKGROUP_SIZE],
      constants: {},
    },
    gpuSupported: !stage.unsupported,
    issues: [],
  };

  const paramAnalysis = analyseStageParams(stage.params || {});
  descriptor.uniformLayout = paramAnalysis.layout;
  descriptor.uniformDefaults = paramAnalysis.defaults;
  descriptor.resourceParams = paramAnalysis.resources;
  descriptor.bindings.hasUniform = Boolean(descriptor.uniformLayout);
  descriptor.bindings.auxiliary = paramAnalysis.resources.map((r) => ({
    name: r.name,
    resourceType: r.resourceType,
    optional: r.value === null || r.value === undefined,
  }));
  descriptor.issues.push(...paramAnalysis.issues);

  if (!descriptor.shaderId) {
    descriptor.gpuSupported = false;
    descriptor.issues.push({
      level: 'error',
      message: `No shader template registered for stage "${stage.label}"`,
    });
  }

  if (!descriptor.uniformLayout && paramAnalysis.requiresUniforms) {
    descriptor.gpuSupported = false;
  }

  if (stage.unsupported) {
    descriptor.gpuSupported = false;
    descriptor.issues.push({
      level: 'error',
      message: `Stage "${stage.label}" uses unsupported dynamic behaviour`,
    });
  }

  descriptor.shaderSource = resolveShaderSource(descriptor, sharedResources);
  descriptor.pipeline = acquirePipeline(descriptor, ctx, sharedResources);

  if (!descriptor.shaderSource) {
    descriptor.gpuSupported = false;
  }

  return descriptor;
}

function resolveShaderId(stage) {
  if (!stage || stage.unsupported) return null;
  if (stage.stageType === 'generator') {
    return 'VALUE_WGSL';
  }
  const name = stage.name || 'anonymous';
  const base = name
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/__+/g, '_')
    .replace(/^_|_$/g, '')
    .toUpperCase();
  const candidate = `${base}_WGSL`;
  if (Object.prototype.hasOwnProperty.call(SHADERS, candidate)) {
    return candidate;
  }
  return null;
}

function resolveShaderSource(descriptor, sharedResources) {
  if (!descriptor.shaderId) return null;
  const template = SHADERS[descriptor.shaderId];
  if (!template) return null;
  if (sharedResources && typeof sharedResources.specializeShaderSource === 'function') {
    return sharedResources.specializeShaderSource(template, descriptor);
  }
  return template;
}

function acquirePipeline(descriptor, ctx, sharedResources) {
  if (!descriptor.shaderSource) return null;
  if (sharedResources && typeof sharedResources.getComputePipeline === 'function') {
    try {
      return sharedResources.getComputePipeline(descriptor, ctx);
    } catch (err) {
      descriptor.issues.push({
        level: 'warn',
        message: `Pipeline cache rejected stage "${descriptor.label}": ${err}`,
      });
      return null;
    }
  }
  return null;
}

function cloneStageParams(params) {
  const out = {};
  if (!params || typeof params !== 'object') {
    return out;
  }
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) {
      out[key] = value.slice();
    } else if (ArrayBuffer.isView(value)) {
      out[key] = Array.from(value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function analyseStageParams(params) {
  const defaults = {};
  const resources = [];
  const issues = [];
  const uniformFields = [];
  let requiresUniforms = false;

  for (const [name, value] of Object.entries(params)) {
    const classification = classifyParam(value);
    if (classification.type === 'uniform') {
      requiresUniforms = true;
      uniformFields.push({
        name,
        scalarType: classification.scalarType,
        components: classification.components,
        bool: classification.bool,
        defaultValue: classification.value,
      });
      defaults[name] = classification.original;
    } else if (classification.type === 'resource') {
      resources.push({ name, resourceType: classification.resourceType, value });
    } else {
      issues.push({ level: 'error', message: classification.reason });
    }
  }

  const layout = uniformFields.length ? buildStd140Layout(uniformFields, issues) : null;

  return { layout, defaults, resources, issues, requiresUniforms };
}

function classifyParam(value) {
  if (value === null || value === undefined) {
    return { type: 'resource', resourceType: 'null' };
  }
  if (typeof value === 'number') {
    const scalarType = Number.isInteger(value) ? 'i32' : 'f32';
    return {
      type: 'uniform',
      scalarType,
      components: 1,
      bool: false,
      value,
      original: value,
    };
  }
  if (typeof value === 'boolean') {
    return {
      type: 'uniform',
      scalarType: 'u32',
      components: 1,
      bool: true,
      value: value ? 1 : 0,
      original: value,
    };
  }
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    const arr = Array.isArray(value) ? value : Array.from(value);
    if (!arr.length) {
      return { type: 'resource', resourceType: 'empty-array', value: arr };
    }
    if (!arr.every((v) => typeof v === 'number')) {
      return {
        type: 'unsupported',
        reason: 'Non-numeric array parameters are not representable in uniforms.',
      };
    }
    if (arr.length > 4) {
      return {
        type: 'unsupported',
        reason: `Array parameter with ${arr.length} entries exceeds vec4 capacity.`,
      };
    }
    const scalarType = arr.every((v) => Number.isInteger(v)) ? 'i32' : 'f32';
    return {
      type: 'uniform',
      scalarType,
      components: arr.length,
      bool: false,
      value: arr.slice(),
      original: arr.slice(),
    };
  }
  if (typeof value === 'string') {
    return { type: 'resource', resourceType: 'string' };
  }
  if (typeof value === 'function') {
    return {
      type: 'unsupported',
      reason: 'Function valued parameters cannot be expressed as uniforms.',
    };
  }
  return { type: 'resource', resourceType: typeof value };
}

function buildStd140Layout(fields, issues) {
  let offset = 0;
  const layoutFields = [];
  for (const field of fields) {
    const { components, scalarType, bool, defaultValue, name } = field;
    const alignSize = std140AlignSize(components);
    if (!alignSize) {
      issues.push({ level: 'error', message: `Unsupported component count for "${name}"` });
      continue;
    }
    offset = alignTo(offset, alignSize.align);
    layoutFields.push({
      name,
      scalarType,
      components,
      offset,
      size: alignSize.size,
      bool,
      defaultValue,
    });
    offset += alignSize.size;
  }
  const finalSize = alignTo(offset, 16);
  return { size: finalSize, fields: layoutFields };
}

function std140AlignSize(components) {
  switch (components) {
    case 1:
      return { align: 4, size: 4 };
    case 2:
      return { align: 8, size: 8 };
    case 3:
    case 4:
      return { align: 16, size: 16 };
    default:
      return null;
  }
}

function alignTo(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : value + (alignment - remainder);
}

function buildTopologySignatureFromPreset(preset) {
  const stages = collectPresetStages(preset);
  return stages.map((stage) => `${stage.category}:${stage.name}`).join('|');
}

