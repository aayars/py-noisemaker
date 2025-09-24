import {
  ColorSpace,
  InterpolationType,
  OctaveBlending,
  ValueDistribution,
  ValueMask,
  isNativeSize,
  isValueMaskProcedural,
} from '../constants.js';
import { maskValues } from '../masks.js';
import { resample } from '../value.js';
import { PALETTES } from '../palettes.js';
import * as SHADERS from './shaders.js';

const { getShaderFilename, inheritShaderFilename } = SHADERS;
import { OpenSimplex } from '../simplex.js';

const DEFAULT_WORKGROUP_SIZE = [8, 8, 1];
const COMPUTE_STAGE_VISIBILITY =
  typeof GPUShaderStage !== 'undefined' ? GPUShaderStage.COMPUTE : 0;
const STORAGE_TEXTURE_FORMAT = 'rgba32float';
const PRESENT_PARAMS_FLOATS = 16;
const PRESENT_PARAMS_SIZE = PRESENT_PARAMS_FLOATS * 4;
const MASK_STORAGE_PAD = 16;
const UINT32_BYTES = 4;
const PERMUTATION_TABLE_SIZE = 256;
const PERMUTATION_ENTRY_UINTS = 4 + PERMUTATION_TABLE_SIZE * 2;
const PERMUTATION_HEADER_UINTS = 4;
const CHANNEL_SEED_DELTA = 65535;
const F32_MAX = 3.402823466e38;

const FLOAT_ORDER_TMP = new ArrayBuffer(4);
const FLOAT_ORDER_VIEW = new DataView(FLOAT_ORDER_TMP);
const FLOAT_SIGN_BIT = 0x80000000;

function floatToOrderedUintBits(value) {
  let val = Number(value);
  if (!Number.isFinite(val)) {
    if (val > 0) {
      val = F32_MAX;
    } else if (val < 0) {
      val = -F32_MAX;
    } else {
      val = 0;
    }
  }
  FLOAT_ORDER_VIEW.setFloat32(0, val, true);
  let bits = FLOAT_ORDER_VIEW.getUint32(0, true);
  if ((bits & FLOAT_SIGN_BIT) !== 0) {
    return (~bits) >>> 0;
  }
  return (bits ^ FLOAT_SIGN_BIT) >>> 0;
}

const PERMUTATION_TABLE_CACHE = new Map();

export class StageValidationError extends Error {
  constructor(descriptor, cause, bindings, context = {}) {
    const stageLabel = descriptor?.label ?? `stage ${descriptor?.index ?? '?'}`;
    const baseMessage = cause?.message || cause || 'Unknown validation error';
    super(`Stage "${stageLabel}" validation error: ${baseMessage}`);
    this.name = 'StageValidationError';
    this.stage = {
      index: descriptor?.index ?? null,
      label: descriptor?.label ?? null,
      category: descriptor?.category ?? null,
      shaderId: descriptor?.shaderId ?? null,
    };
    this.bindings = Array.isArray(bindings) ? bindings : [];
    this.cause = cause;
    this.context = context;
  }
}

function describeBindingResource(resource) {
  if (!resource) return 'null';
  if (resource.buffer) return 'buffer';
  if (resource.texture) return 'texture';
  if (resource.sampler) return 'sampler';
  if (resource.resource) {
    return describeBindingResource(resource.resource);
  }
  const type = resource.constructor?.name;
  if (type && type !== 'Object') {
    return type;
  }
  if (typeof resource === 'object') {
    return 'object';
  }
  return typeof resource;
}

function summarizeBindings(entries) {
  if (!Array.isArray(entries)) return [];
  return entries.map((entry) => ({
    binding: entry?.binding ?? null,
    resourceType: describeBindingResource(entry?.resource),
  }));
}

async function safePopErrorScope(device) {
  if (!device?.popErrorScope) return null;
  try {
    return await device.popErrorScope();
  } catch (err) {
    return err;
  }
}

function logStageValidationError(descriptor, error, bindings, extras = {}) {
  if (typeof console === 'undefined' || typeof console.error !== 'function') {
    return;
  }
  const payload = {
    stage: {
      index: descriptor?.index ?? null,
      label: descriptor?.label ?? null,
      category: descriptor?.category ?? null,
      shaderId: descriptor?.shaderId ?? null,
    },
    bindings,
    ...extras,
  };
  console.error('WebGPU validation error', error, payload);
}

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
 * @property {string|null} bindingSignature Stable key describing the bind-group
 *   requirements for caching.
 * @property {GPUBindGroupLayout|null} bindGroupLayout Cached bind-group layout
 *   when a WebGPU device is available.
 * @property {GPUPipelineLayout|null} pipelineLayout Cached pipeline layout when
 *   a WebGPU device is available.
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
    this.uniformBuffers = stages.map((stage) => buildUniformBufferPair(stage, ctx));
    let gpuStageCount = 0;
    let hasUnsupportedStages = false;
    let generatorStageIndex = -1;
    for (let i = 0; i < stages.length; i += 1) {
      const descriptor = stages[i];
      if (!descriptor) continue;
      const gpuSupported = descriptor.gpuSupported !== false;
      if (descriptor.stageType === 'generator' && generatorStageIndex === -1) {
        generatorStageIndex = i;
      }
      if (gpuSupported) {
        gpuStageCount += 1;
      } else {
        hasUnsupportedStages = true;
      }
    }
    this._gpuStageCount = gpuStageCount;
    this._hasUnsupportedStages = hasUnsupportedStages;
    this._generatorStageIndex = generatorStageIndex;
    this._generatorStageSupported =
      generatorStageIndex >= 0 && stages[generatorStageIndex]?.gpuSupported !== false;
    this._allStagesSupported = !hasUnsupportedStages && gpuStageCount > 0;
  }

  _ensureNormalizationState(descriptor, ctx, size = 16) {
    if (!descriptor || !ctx || !ctx.device) {
      return null;
    }
    const byteSize = Math.max(16, Math.floor(Number(size) || 0));
    let state = descriptor._normalizationState;
    if (!state || state.size !== byteSize) {
      const usage =
        typeof GPUBufferUsage !== 'undefined'
          ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          : (1 << 7) | (1 << 3);
      const zeroArray = new Uint32Array(byteSize / 4);
      const buffer = ctx.createGPUBuffer(zeroArray, usage);
      const arrayBuffer = new ArrayBuffer(byteSize);
      state = {
        buffer,
        arrayBuffer,
        uint32: new Uint32Array(arrayBuffer),
        size: byteSize,
      };
      descriptor._normalizationState = state;
    }
    if (state && state.uint32) {
      state.uint32.fill(0);
      if (state.uint32.length >= 2) {
        state.uint32[0] = floatToOrderedUintBits(F32_MAX);
        state.uint32[1] = floatToOrderedUintBits(-F32_MAX);
      }
    }
    return state;
  }

  _ensureSinNormalizationState(descriptor, ctx, size = 16) {
    if (!descriptor || !ctx || !ctx.device) {
      return null;
    }
    const byteSize = Math.max(16, Math.floor(Number(size) || 0));
    let state = descriptor._sinNormalizationState;
    if (!state || state.size !== byteSize) {
      const usage =
        typeof GPUBufferUsage !== 'undefined'
          ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          : (1 << 7) | (1 << 3);
      const zeroArray = new Uint32Array(byteSize / 4);
      const buffer = ctx.createGPUBuffer(zeroArray, usage);
      const arrayBuffer = new ArrayBuffer(byteSize);
      state = {
        buffer,
        arrayBuffer,
        uint32: new Uint32Array(arrayBuffer),
        size: byteSize,
      };
      descriptor._sinNormalizationState = state;
    }
    if (state && state.uint32) {
      state.uint32.fill(0);
    }
    return state;
  }

  async _dispatchMultiresNormalizationPass({
    descriptor,
    context,
    pingPong,
    uniformView,
    frameUniformInfo,
    normalizationState,
    width,
    height,
    frameIndex,
    stageIndex,
    encoder,
  }) {
    if (!descriptor || !context || context.isCPU) {
      return false;
    }
    const device = context.device;
    if (!device || !pingPong?.writeFbo || !pingPong?.readFbo || !encoder) {
      return false;
    }
    if (!uniformView?.gpuBuffer || !frameUniformInfo?.buffer) {
      return false;
    }
    if (!normalizationState?.buffer) {
      return false;
    }

    let pipeline = descriptor._normalizationPipeline;
    if (pipeline && typeof pipeline.then === 'function') {
      try {
        pipeline = await pipeline;
      } catch (err) {
        console.warn('Failed to resolve multires normalization pipeline', err);
        descriptor._normalizationPipeline = null;
        pipeline = null;
      }
    }
    if (!pipeline) {
      const shaderSource =
        descriptor.normalizationShaderSource || SHADERS.MULTIRES_NORMALIZE_WGSL;
      if (!shaderSource) {
        return false;
      }
      const shaderFilename =
        descriptor.normalizationShaderFilename || getShaderFilename(shaderSource) || null;
      if (!descriptor.normalizationShaderFilename && shaderFilename) {
        descriptor.normalizationShaderFilename = shaderFilename;
      }
      const pipelineRequest = {
        code: shaderSource,
        pipelineLayout: descriptor.normalizationPipelineLayout || null,
        label: descriptor.label ? `stage:${descriptor.label}:normalize` : undefined,
        shaderFilename,
      };
      try {
        pipeline = await context.createComputePipeline(pipelineRequest);
        descriptor._normalizationPipeline = pipeline;
      } catch (err) {
        console.warn('Failed to create multires normalization pipeline', err);
        throw err;
      }
    }
    if (!pipeline) {
      return false;
    }

    let layout = null;
    if (pipeline?.getBindGroupLayout) {
      try {
        layout = pipeline.getBindGroupLayout(0);
        if (layout) {
          descriptor.normalizationBindGroupLayout = layout;
        }
      } catch (err) {
        console.warn('Failed to query normalization bind group layout', err);
        layout = null;
      }
    }
    if (!layout) {
      layout = descriptor.normalizationBindGroupLayout || null;
    }
    if (!layout) {
      return false;
    }

    const entries = [
      { binding: 0, resource: { buffer: uniformView.gpuBuffer } },
      { binding: 1, resource: { buffer: frameUniformInfo.buffer } },
      { binding: 2, resource: pingPong.writeFbo },
      { binding: 3, resource: pingPong.readFbo },
      { binding: 4, resource: { buffer: normalizationState.buffer } },
    ];
    const normalizationBindingSummary = summarizeBindings(entries);

    let bindGroup = null;
    try {
      bindGroup = device.createBindGroup({ layout, entries });
    } catch (err) {
      logStageValidationError(descriptor, err, normalizationBindingSummary, {
        width,
        height,
        frameIndex,
        stageIndex,
        normalizationPass: true,
      });
      throw err;
    }

    const baseWorkgroup = Array.isArray(descriptor.normalizationWorkgroupSize)
      ? descriptor.normalizationWorkgroupSize
      : Array.isArray(descriptor.specialization?.workgroupSize)
      ? descriptor.specialization.workgroupSize
      : DEFAULT_WORKGROUP_SIZE;
    const wgX = Math.max(1, Math.floor(baseWorkgroup[0] || DEFAULT_WORKGROUP_SIZE[0]));
    const wgY = Math.max(1, Math.floor(baseWorkgroup[1] || DEFAULT_WORKGROUP_SIZE[1]));
    const dispatchX = Math.max(1, Math.ceil(width / wgX));
    const dispatchY = Math.max(1, Math.ceil(height / wgY));

    let scopeActive = false;
    if (device?.pushErrorScope && device?.popErrorScope) {
      try {
        device.pushErrorScope('validation');
        scopeActive = true;
      } catch (_) {
        scopeActive = false;
      }
    }

    let pass = null;
    try {
      pass = context.beginComputePass(encoder);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dispatchX, dispatchY, 1);
    } catch (err) {
      if (pass) {
        try {
          pass.end();
        } catch (_) {
          /* ignore */
        }
        pass = null;
      }
      let validationErr = null;
      if (scopeActive) {
        validationErr = await safePopErrorScope(device);
        scopeActive = false;
      }
      if (validationErr) {
        logStageValidationError(descriptor, validationErr, normalizationBindingSummary, {
          width,
          height,
          frameIndex,
          stageIndex,
          normalizationPass: true,
          cause: err,
        });
        throw new StageValidationError(descriptor, validationErr, normalizationBindingSummary, {
          width,
          height,
          frameIndex,
          stageIndex,
          normalizationPass: true,
          cause: err,
        });
      }
      if (err instanceof StageValidationError) {
        throw err;
      }
      logStageValidationError(descriptor, err, normalizationBindingSummary, {
        width,
        height,
        frameIndex,
        stageIndex,
        normalizationPass: true,
      });
      throw new StageValidationError(descriptor, err, normalizationBindingSummary, {
        width,
        height,
        frameIndex,
        stageIndex,
        normalizationPass: true,
      });
    } finally {
      if (pass) {
        try {
          pass.end();
        } catch (_) {
          /* ignore */
        }
      }
    }

    if (scopeActive) {
      const validationErr = await safePopErrorScope(device);
      scopeActive = false;
      if (validationErr) {
        logStageValidationError(descriptor, validationErr, normalizationBindingSummary, {
          width,
          height,
          frameIndex,
          stageIndex,
          normalizationPass: true,
        });
        throw new StageValidationError(descriptor, validationErr, normalizationBindingSummary, {
          width,
          height,
          frameIndex,
          stageIndex,
          normalizationPass: true,
        });
      }
    }

    return true;
  }

  _ensureMaskBuffer(descriptor, ctx) {
    if (!descriptor) return null;
    const holder = ensureMaskDataHolder(descriptor);
    const array = holder?.array instanceof Float32Array ? holder.array : new Float32Array([1, 1, 1, 1]);
    const arrayByteLength = array.byteLength || MASK_STORAGE_PAD;
    if (!ctx || !ctx.device) {
      descriptor._maskGPUBuffer = null;
      descriptor._maskGPUBufferSize = 0;
      return { buffer: null, array };
    }
    const usage = (typeof GPUBufferUsage !== 'undefined'
      ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      : (1 << 5) | (1 << 3));
    const requiredSize = Math.max(arrayByteLength, MASK_STORAGE_PAD);
    let buffer = descriptor._maskGPUBuffer;
    const currentSize =
      descriptor._maskGPUBufferSize ?? buffer?.size ?? buffer?._size ?? 0;
    if (!buffer || currentSize < requiredSize) {
      if (buffer && typeof buffer.destroy === 'function') {
        try {
          buffer.destroy();
        } catch (err) {
          void err;
        }
      }
      buffer = ctx.device.createBuffer({
        size: requiredSize,
        usage,
      });
      // Some implementations (notably Safari's Metal backend) omit the ``size``
      // property on returned GPUBuffer objects. Track the allocation manually so
      // future writes know the true capacity and can grow the buffer if needed.
      buffer._size = requiredSize;
      descriptor._maskGPUBuffer = buffer;
      descriptor._maskGPUBufferSize = requiredSize;
      descriptor._maskDataDirty = true;
    }
    if (
      buffer &&
      (!Number.isFinite(descriptor._maskGPUBufferSize) || descriptor._maskGPUBufferSize <= 0)
    ) {
      const knownSize = buffer.size ?? buffer._size ?? requiredSize;
      descriptor._maskGPUBufferSize = Number.isFinite(knownSize) && knownSize > 0 ? knownSize : requiredSize;
    }
    if (descriptor._maskDataDirty && ctx.queue) {
      ctx.queue.writeBuffer(buffer, 0, array);
      descriptor._maskDataDirty = false;
    }
    return { buffer };
  }

  _ensurePermutationTableBuffer(descriptor, ctx, options = {}) {
    if (!descriptor) return null;
    const plan = descriptor._multiresPermutationPlan;
    if (!plan) {
      descriptor._permutationGPUBuffer = null;
      descriptor._permutationGPUBufferSize = 0;
      return { buffer: null };
    }
    const width = Math.max(1, Math.floor(Number(options.width) || 1));
    const height = Math.max(1, Math.floor(Number(options.height) || 1));
    const seedValue = Number(options.seed) || 0;
    const frameSeed = Number.isFinite(seedValue) ? Math.trunc(seedValue) >>> 0 : 0;
    const seeds = computePermutationSeeds(plan, width, height, frameSeed);
    if (!seeds.length) {
      seeds.push(0);
    }
    const previousSeeds = descriptor._permutationSeedList || [];
    let seedsChanged = seeds.length !== previousSeeds.length;
    if (!seedsChanged) {
      for (let i = 0; i < seeds.length; i += 1) {
        if (seeds[i] !== previousSeeds[i]) {
          seedsChanged = true;
          break;
        }
      }
    }
    if (seedsChanged) {
      descriptor._permutationSeedList = seeds.slice();
      descriptor._permutationDataDirty = true;
    }

    const entryCount = seeds.length;
    const totalU32 = PERMUTATION_HEADER_UINTS + entryCount * PERMUTATION_ENTRY_UINTS;
    let array = descriptor._permutationArray;
    if (!array || array.length !== totalU32) {
      array = new Uint32Array(totalU32);
      descriptor._permutationArray = array;
      descriptor._permutationDataDirty = true;
    }
    if (descriptor._permutationDataDirty && array) {
      array.fill(0);
      array[0] = entryCount >>> 0;
      for (let i = 0; i < entryCount; i += 1) {
        const seed = seeds[i] >>> 0;
        const entryBase = PERMUTATION_HEADER_UINTS + i * PERMUTATION_ENTRY_UINTS;
        array[entryBase] = seed;
        const tables = getPermutationTables(seed);
        array.set(tables.perm, entryBase + 4);
        array.set(tables.grad, entryBase + 4 + PERMUTATION_TABLE_SIZE);
      }
    }

    if (!ctx || !ctx.device) {
      return { buffer: null };
    }

    const usage =
      typeof GPUBufferUsage !== 'undefined'
        ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        : (1 << 5) | (1 << 3);
    const byteLength = array ? array.length * UINT32_BYTES : PERMUTATION_HEADER_UINTS * UINT32_BYTES;
    const requiredSize = Math.max(byteLength, MASK_STORAGE_PAD);
    let buffer = descriptor._permutationGPUBuffer;
    const currentSize = descriptor._permutationGPUBufferSize ?? buffer?.size ?? buffer?._size ?? 0;
    if (!buffer || currentSize < requiredSize) {
      if (buffer && typeof buffer.destroy === 'function') {
        try {
          buffer.destroy();
        } catch (err) {
          void err;
        }
      }
      buffer = ctx.device.createBuffer({
        size: requiredSize,
        usage,
      });
      buffer._size = requiredSize;
      descriptor._permutationGPUBuffer = buffer;
      descriptor._permutationGPUBufferSize = requiredSize;
      descriptor._permutationDataDirty = true;
    }

    if (
      buffer &&
      (!Number.isFinite(descriptor._permutationGPUBufferSize) || descriptor._permutationGPUBufferSize <= 0)
    ) {
      const knownSize = buffer.size ?? buffer._size ?? requiredSize;
      descriptor._permutationGPUBufferSize = Number.isFinite(knownSize) && knownSize > 0 ? knownSize : requiredSize;
    }

    if (descriptor._permutationDataDirty && array && ctx.queue) {
      ctx.queue.writeBuffer(buffer, 0, array);
      descriptor._permutationDataDirty = false;
    }

    return { buffer };
  }

  /**
   * Number of GPU stages described by the program.
   * @returns {number}
   */
  get stageCount() {
    return this.stages.length;
  }

  /**
   * Number of GPU-capable stages contained in the program.
   * @returns {number}
   */
  get gpuStageCount() {
    return this._gpuStageCount;
  }

  /**
   * True when any stage falls back to CPU.
   * @returns {boolean}
   */
  get hasUnsupportedStages() {
    return this._hasUnsupportedStages;
  }

  /**
   * True when the generator stage is GPU-capable.
   * @returns {boolean}
   */
  get generatorStageSupported() {
    return this._generatorStageSupported;
  }

  /**
   * True when every stage has a GPU implementation.
   * @returns {boolean}
   */
  get allStagesSupported() {
    return this._allStagesSupported;
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
   * @returns {{
   *   arrayBuffer: ArrayBuffer,
   *   view: DataView,
   *   layout: UniformLayout,
   *   gpuBuffer: GPUBuffer|null,
   *   flush: () => void,
   *   index: number,
   * }|null}
   */
  getUniformBufferView(index, bufferIndex = 0) {
    const holder = this.uniformBuffers[index];
    if (!holder) return null;
    const views = holder.views || [];
    if (!views.length) return null;
    const selected = views[((bufferIndex % views.length) + views.length) % views.length];
    return {
      arrayBuffer: selected.arrayBuffer,
      view: selected.view,
      layout: this.stages[index].uniformLayout,
      gpuBuffer: selected.gpuBuffer || null,
      flush: selected.flush || (() => {}),
      index: selected.index,
    };
  }

  async execute(ctx, options = {}) {
    const context = ctx || this.ctx;
    if (!context || !context.device || !context.queue) {
      throw new Error('PresetProgram.execute requires an initialized WebGPU context.');
    }
    if (!Array.isArray(this.stages) || !this.stages.length) {
      throw new Error('PresetProgram.execute requires at least one GPU stage.');
    }
    if (!this._gpuStageCount) {
      throw new Error('PresetProgram.execute requires at least one GPU-capable stage.');
    }

    const {
      encoder: providedEncoder = null,
      width: optWidth,
      height: optHeight,
      time = 0,
      frameIndex = 0,
      seed = 0,
      present = false,
      target: targetOverride,
      presentationTarget,
      readback = false,
    } = options || {};

    const width = Math.max(1, Math.floor(Number(optWidth) || 0));
    const height = Math.max(1, Math.floor(Number(optHeight) || 0));
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      throw new Error('PresetProgram.execute requires positive width and height.');
    }

    const autoSubmit = !providedEncoder;
    const device = context.device;
    const queue = context.queue;
    const pingPong = context.pingPong(width, height);
    const frameUniformInfo = context.ensureFrameUniforms(width, height, seed, time, frameIndex);
    const presentationTargetOption =
      presentationTarget !== undefined ? presentationTarget : targetOverride;
    const gpuMapModeRead = typeof GPUMapMode !== 'undefined' ? GPUMapMode.READ : 1;

    const encode = async (encoder) => {
      const state = {
        finalTexture: pingPong.readTex,
        finalView: pingPong.readFbo,
        readbackRequest: null,
        profileInfo: null,
      };

      let timestampInfo = null;
      if (typeof context.prepareTimestampQueryResources === 'function') {
        const resources = context.prepareTimestampQueryResources(2);
        if (
          resources?.querySet &&
          resources?.resolveBuffer &&
          typeof encoder.writeTimestamp === 'function' &&
          typeof encoder.resolveQuerySet === 'function'
        ) {
          timestampInfo = {
            querySet: resources.querySet,
            resolveBuffer: resources.resolveBuffer,
            count: resources.count || 2,
            usedTimestampQuery: true,
          };
          encoder.writeTimestamp(resources.querySet, 0);
        }
      }

      for (let index = 0; index < this.stages.length; index += 1) {
        const descriptor = this.stages[index];
        if (!descriptor || descriptor.gpuSupported === false) {
          continue;
        }

        let pipeline = descriptor.pipeline;
        if (pipeline && typeof pipeline.then === 'function') {
          try {
            pipeline = await pipeline;
          } catch (err) {
            console.warn('Failed to resolve pipeline for stage', descriptor?.label, err);
            pipeline = null;
          }
        }
        if (!pipeline && descriptor.shaderSource && typeof context.createComputePipeline === 'function') {
          try {
            const pipelineRequest = {
              code: descriptor.shaderSource,
              pipelineLayout: descriptor.pipelineLayout || null,
              label: descriptor.label ? `stage:${descriptor.label}` : undefined,
              shaderFilename: descriptor.shaderFilename || getShaderFilename(descriptor.shaderSource) || null,
            };
            pipeline = await context.createComputePipeline(pipelineRequest);
            descriptor.pipeline = pipeline;
            if (pipeline?.getBindGroupLayout) {
              try {
                const stageLayout = pipeline.getBindGroupLayout(0);
                if (stageLayout) {
                  descriptor.bindGroupLayout = stageLayout;
                }
              } catch (layoutErr) {
                console.warn('Failed to query bind group layout for stage', descriptor?.label, layoutErr);
              }
            }
          } catch (err) {
            console.warn('Failed to create pipeline for stage', descriptor?.label, err);
            pipeline = null;
          }
        }
        if (!pipeline) {
          continue;
        }

        const uniformView = this.getUniformBufferView(index, frameIndex);
        if (uniformView && typeof uniformView.flush === 'function') {
          uniformView.flush();
        }

        const entries = [];
        if (descriptor.bindings?.hasUniform && uniformView?.gpuBuffer) {
          entries.push({ binding: 0, resource: { buffer: uniformView.gpuBuffer } });
        }
        if (descriptor.bindings?.hasFrameUniform && frameUniformInfo?.buffer) {
          entries.push({ binding: 1, resource: { buffer: frameUniformInfo.buffer } });
        }
        if (descriptor.bindings?.readsTexture && pingPong.readFbo) {
          entries.push({ binding: 2, resource: pingPong.readFbo });
        }
        if (descriptor.bindings?.writesTexture && pingPong.writeFbo) {
          entries.push({ binding: 3, resource: pingPong.writeFbo });
        }

        let normalizationState = null;
        const resources = Array.isArray(descriptor.resourceParams) ? descriptor.resourceParams : [];
        for (let auxIndex = 0; auxIndex < resources.length; auxIndex += 1) {
          const binding = 4 + auxIndex;
          const resourceSpec = resources[auxIndex];
          let entry = null;
          const helperArgs = {
            ctx: context,
            program: this,
            descriptor,
            resource: resourceSpec,
            binding,
            stageIndex: index,
            pingPong,
            width,
            height,
            frameIndex,
          };
          if (
            !entry &&
            resourceSpec?.resourceType === 'storage-buffer' &&
            resourceSpec?.name === 'normalizationState'
          ) {
            normalizationState = this._ensureNormalizationState(
              descriptor,
              context,
              resourceSpec.size,
            );
            if (normalizationState?.buffer && queue && typeof queue.writeBuffer === 'function') {
              queue.writeBuffer(normalizationState.buffer, 0, normalizationState.arrayBuffer);
            }
            if (normalizationState?.buffer) {
              entry = { resource: { buffer: normalizationState.buffer } };
            }
          }
          if (
            !entry &&
            resourceSpec?.resourceType === 'storage-buffer' &&
            resourceSpec?.name === 'sinNormalizationState'
          ) {
            const sinState = this._ensureSinNormalizationState(
              descriptor,
              context,
              resourceSpec.size,
            );
            if (sinState?.buffer && queue && typeof queue.writeBuffer === 'function') {
              queue.writeBuffer(sinState.buffer, 0, sinState.arrayBuffer);
            }
            if (sinState?.buffer) {
              entry = { resource: { buffer: sinState.buffer } };
            }
          }
          if (
            !entry &&
            resourceSpec?.resourceType === 'storage-buffer' &&
            resourceSpec?.name === 'maskData'
          ) {
            const maskState = this._ensureMaskBuffer(descriptor, context);
            if (maskState?.buffer) {
              entry = { resource: { buffer: maskState.buffer } };
            }
          }
          if (
            !entry &&
            resourceSpec?.resourceType === 'storage-buffer' &&
            resourceSpec?.name === 'permutationTables'
          ) {
            const permutationState = this._ensurePermutationTableBuffer(descriptor, context, {
              seed,
              width,
              height,
            });
            if (permutationState?.buffer) {
              entry = { resource: { buffer: permutationState.buffer } };
            }
          }
          if (this.sharedResources) {
            const helpers = this.sharedResources;
            if (!entry && typeof helpers.resolveResourceBinding === 'function') {
              entry = helpers.resolveResourceBinding(helperArgs);
            }
            if (!entry && typeof helpers.getAuxiliaryBindGroupEntry === 'function') {
              entry = helpers.getAuxiliaryBindGroupEntry(helperArgs);
            }
            if (!entry && typeof helpers.buildStageBindGroupEntry === 'function') {
              entry = helpers.buildStageBindGroupEntry(helperArgs);
            }
          }
          if (!entry && resourceSpec?.value) {
            const value = resourceSpec.value;
            if (value && typeof value === 'object') {
              if (typeof value.createView === 'function') {
                entry = { resource: value.createView() };
              } else if ('resource' in value) {
                entry = { resource: value.resource };
              }
            }
            if (!entry && value) {
              entry = { resource: value };
            }
          }
          if (entry && entry.resource) {
            entries.push({ binding, resource: entry.resource });
          }
        }

        const bindingSummary = summarizeBindings(entries);
        let scopeActive = false;
        if (device?.pushErrorScope && device?.popErrorScope) {
          try {
            device.pushErrorScope('validation');
            scopeActive = true;
          } catch (_) {
            scopeActive = false;
          }
        }

        let pass = null;
        try {
          let layout = null;
          if (pipeline?.getBindGroupLayout) {
            try {
              layout = pipeline.getBindGroupLayout(0);
              if (layout) {
                descriptor.bindGroupLayout = layout;
              }
            } catch (layoutErr) {
              console.warn('Failed to fetch pipeline bind group layout for stage', descriptor?.label, layoutErr);
              layout = null;
            }
          }
          if (!layout) {
            layout = descriptor.bindGroupLayout || null;
          }
          if (!layout) {
            if (scopeActive) {
              const validationErr = await safePopErrorScope(device);
              scopeActive = false;
              if (validationErr) {
                logStageValidationError(descriptor, validationErr, bindingSummary, {
                  width,
                  height,
                  frameIndex,
                  stageIndex: index,
                });
                throw new StageValidationError(descriptor, validationErr, bindingSummary, {
                  width,
                  height,
                  frameIndex,
                  stageIndex: index,
                });
              }
            }
            continue;
          }
          const bindGroup = device.createBindGroup({ layout, entries });
          pass = context.beginComputePass(encoder);
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          const workgroupSize = Array.isArray(descriptor.specialization?.workgroupSize)
            ? descriptor.specialization.workgroupSize
            : DEFAULT_WORKGROUP_SIZE;
          const wgX = Math.max(1, Math.floor(workgroupSize[0] || DEFAULT_WORKGROUP_SIZE[0]));
          const wgY = Math.max(1, Math.floor(workgroupSize[1] || DEFAULT_WORKGROUP_SIZE[1]));
          const dispatchX = Math.max(1, Math.ceil(width / wgX));
          const dispatchY = Math.max(1, Math.ceil(height / wgY));
          pass.dispatchWorkgroups(dispatchX, dispatchY, 1);
        } catch (err) {
          if (pass) {
            try {
              pass.end();
            } catch (_) {
              /* ignore */
            }
            pass = null;
          }
          let validationErr = null;
          if (scopeActive) {
            validationErr = await safePopErrorScope(device);
            scopeActive = false;
          }
          if (validationErr) {
            logStageValidationError(descriptor, validationErr, bindingSummary, {
              width,
              height,
              frameIndex,
              stageIndex: index,
              cause: err,
            });
            throw new StageValidationError(descriptor, validationErr, bindingSummary, {
              width,
              height,
              frameIndex,
              stageIndex: index,
              cause: err,
            });
          }
          if (err instanceof StageValidationError) {
            throw err;
          }
          logStageValidationError(descriptor, err, bindingSummary, {
            width,
            height,
            frameIndex,
            stageIndex: index,
          });
          throw new StageValidationError(descriptor, err, bindingSummary, {
            width,
            height,
            frameIndex,
            stageIndex: index,
          });
        } finally {
          if (pass) {
            try {
              pass.end();
            } catch (_) {
              /* ignore */
            }
          }
        }

        if (scopeActive) {
          const validationErr = await safePopErrorScope(device);
          scopeActive = false;
          if (validationErr) {
            logStageValidationError(descriptor, validationErr, bindingSummary, {
              width,
              height,
              frameIndex,
              stageIndex: index,
            });
            throw new StageValidationError(descriptor, validationErr, bindingSummary, {
              width,
              height,
              frameIndex,
              stageIndex: index,
            });
          }
        }

        let normalizationHandled = false;
        if (
          descriptor.requiresNormalizationResolve &&
          normalizationState?.buffer &&
          !context.isCPU
        ) {
          normalizationHandled = await this._dispatchMultiresNormalizationPass({
            descriptor,
            context,
            pingPong,
            uniformView,
            frameUniformInfo,
            normalizationState,
            width,
            height,
            frameIndex,
            stageIndex: index,
            encoder,
          });
        }

        if (normalizationHandled) {
          state.finalTexture = pingPong.readTex;
          state.finalView = pingPong.readFbo;
        } else {
          pingPong.swap();
          state.finalTexture = pingPong.readTex;
          state.finalView = pingPong.readFbo;
        }
      }

      if (state.finalTexture) {
        try {
          state.finalTexture._noisemakerPresentationNormalized = true;
          state.finalTexture._noisemakerChannels = 4;
          state.finalTexture._noisemakerShape = [height, width, 4];
        } catch (err) {
          void err;
        }
      }

      if (present && state.finalView && context.presentationFormat) {
        if (!context._renderPipeline) {
          const vs = `
            struct VSOut {
              @builtin(position) pos: vec4<f32>,
            };
            @vertex
            fn main(@builtin(vertex_index) idx : u32) -> VSOut {
              var pos = array<vec2<f32>,6>(
                vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
                vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0)
              );
              var out: VSOut;
              out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
              return out;
            }
          `;
          const fs = `
            struct PresentParams {
              minVals: vec4<f32>,
              invRange: vec4<f32>,
              channels: f32,
              padding: vec3<f32>,
            };
            @group(0) @binding(0) var tex : texture_2d<f32>;
            @group(0) @binding(1) var<uniform> params : PresentParams;
            @fragment
            fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
              let dims = textureDimensions(tex);
              let width = max(i32(dims.x), 1);
              let height = max(i32(dims.y), 1);
              let pixel = clamp(
                vec2<i32>(floor(fragCoord.xy)),
                vec2<i32>(0, 0),
                vec2<i32>(width - 1, height - 1)
              );
              let color = textureLoad(tex, pixel, 0);
              let adjusted = (color - params.minVals) * params.invRange;
              let channels = params.channels;
              var rgb: vec3<f32>;
              var alpha: f32;
              if (channels < 1.5) {
                let gray = adjusted.x;
                rgb = vec3<f32>(gray, gray, gray);
                alpha = 1.0;
              } else if (channels < 2.5) {
                let gray = adjusted.x;
                rgb = vec3<f32>(gray, gray, gray);
                alpha = adjusted.y;
              } else if (channels < 3.5) {
                rgb = adjusted.xyz;
                alpha = 1.0;
              } else {
                rgb = adjusted.xyz;
                alpha = adjusted.w;
              }
              let clampedRgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
              let clampedAlpha = clamp(alpha, 0.0, 1.0);
              return vec4<f32>(clampedRgb, clampedAlpha);
            }
          `;
          const bindGroupLayout = device.createBindGroupLayout({
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'unfilterable-float' },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
              },
            ],
          });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
          context._renderPipeline = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module: context.createShaderModule(vs), entryPoint: 'main' },
            fragment: {
              module: context.createShaderModule(fs),
              entryPoint: 'main',
              targets: [{ format: context.presentationFormat }],
            },
            primitive: { topology: 'triangle-list' },
          });
        }
        if (!context._renderParamsArray || context._renderParamsArray.length !== PRESENT_PARAMS_FLOATS) {
          context._renderParamsArray = new Float32Array(PRESENT_PARAMS_FLOATS);
        }
        if (
          !context._renderParamsBuffer ||
          (context._renderParamsBuffer.size ?? context._renderParamsBuffer._size ?? 0) < PRESENT_PARAMS_SIZE
        ) {
          if (context._renderParamsBuffer?.destroy) {
            try {
              context._renderParamsBuffer.destroy();
            } catch (err) {
              void err;
            }
          }
          context._renderParamsBuffer = device.createBuffer({
            size: PRESENT_PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          context._renderParamsBuffer._size = PRESENT_PARAMS_SIZE;
        }
        const paramsArray = context._renderParamsArray;
        paramsArray.fill(0);
        paramsArray.set([0, 0, 0, 0], 0);
        paramsArray.set([1, 1, 1, 1], 4);
        paramsArray[8] = 4;
        queue.writeBuffer(
          context._renderParamsBuffer,
          0,
          paramsArray.buffer,
          paramsArray.byteOffset,
          paramsArray.byteLength,
        );
        const renderBindGroup = device.createBindGroup({
          layout: context._renderPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: state.finalView },
            { binding: 1, resource: { buffer: context._renderParamsBuffer } },
          ],
        });
        let resolvedTarget = presentationTargetOption;
        if (typeof resolvedTarget === 'function') {
          resolvedTarget = resolvedTarget();
        }
        let targetView = null;
        if (resolvedTarget) {
          targetView = typeof resolvedTarget.createView === 'function' ? resolvedTarget.createView() : resolvedTarget;
        } else if (context.gpu && typeof context.gpu.getCurrentTexture === 'function') {
          const swapTexture = context.gpu.getCurrentTexture();
          if (swapTexture && typeof swapTexture.createView === 'function') {
            targetView = swapTexture.createView();
          }
        }
        if (targetView) {
          const pass = context.beginRenderPass(encoder, targetView);
          pass.setPipeline(context._renderPipeline);
          pass.setBindGroup(0, renderBindGroup);
          pass.draw(6);
          pass.end();
        }
      }

      if (readback && state.finalTexture) {
        const bytesPerPixel = 16;
        const rowStride = width * bytesPerPixel;
        const bytesPerRow = alignTo(rowStride, 256);
        const bufferSize = bytesPerRow * height;
        const stagingBuffer = device.createBuffer({
          size: bufferSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        encoder.copyTextureToBuffer(
          { texture: state.finalTexture },
          { buffer: stagingBuffer, bytesPerRow, rowsPerImage: height },
          { width, height, depthOrArrayLayers: 1 },
        );
        state.readbackRequest = {
          buffer: stagingBuffer,
          size: bufferSize,
          bytesPerRow,
          rowStride,
          width,
          height,
          gpuMapModeRead,
        };
      }

      if (timestampInfo?.querySet && timestampInfo?.resolveBuffer) {
        encoder.writeTimestamp(timestampInfo.querySet, 1);
        encoder.resolveQuerySet(
          timestampInfo.querySet,
          0,
          timestampInfo.count || 2,
          timestampInfo.resolveBuffer,
          0,
        );
      }

      state.profileInfo = timestampInfo;

      return state;
    };

    const finalizeResult = async (state, autoSubmitted, dispatchMs, callStart) => {
      const hasPerf = typeof performance !== 'undefined' && performance && typeof performance.now === 'function';
      const result = {
        texture: state?.finalTexture || null,
        textureView: state?.finalView || null,
        width,
        height,
        readback: null,
        readbackBuffer: null,
        readbackLayout: null,
        gpuTimeMs: null,
        profileInfo: null,
        dispatchWallMs: typeof dispatchMs === 'number' && Number.isFinite(dispatchMs) ? dispatchMs : null,
        totalTimeMs: null,
        queueFlushed: false,
        usedTimestampQuery: Boolean(state?.profileInfo?.usedTimestampQuery),
      };

      if (!state) {
        return result;
      }

      if (state.readbackRequest) {
        if (autoSubmitted) {
          const { buffer, bytesPerRow, rowStride, size, width: rw, height: rh } = state.readbackRequest;
          try {
            await buffer.mapAsync(gpuMapModeRead, 0, size);
            const mapped = buffer.getMappedRange(0, size).slice(0);
            const src = new Float32Array(mapped);
            const floatsPerRow = bytesPerRow / 4;
            const floatsPerStride = rowStride / 4;
            const total = rw * rh * 4;
            const out = new Float32Array(total);
            if (bytesPerRow === rowStride) {
              out.set(src.subarray(0, total));
            } else {
              for (let y = 0; y < rh; y += 1) {
                const srcOffset = y * floatsPerRow;
                const dstOffset = y * floatsPerStride;
                out.set(src.subarray(srcOffset, srcOffset + floatsPerStride), dstOffset);
              }
            }
            result.readback = out;
          } finally {
            try {
              state.readbackRequest.buffer.unmap();
            } catch (err) {
              void err;
            }
            if (state.readbackRequest.buffer.destroy) {
              try {
                state.readbackRequest.buffer.destroy();
              } catch (err) {
                void err;
              }
            }
          }
          result.queueFlushed = true;
          if (context) {
            context._pendingDispatch = false;
          }
        } else {
          result.readbackBuffer = state.readbackRequest.buffer;
          result.readbackLayout = {
            size: state.readbackRequest.size,
            bytesPerRow: state.readbackRequest.bytesPerRow,
            rowStride: state.readbackRequest.rowStride,
            width: state.readbackRequest.width,
            height: state.readbackRequest.height,
            mapMode: gpuMapModeRead,
          };
        }
      }

      if (state.profileInfo?.usedTimestampQuery) {
        if (autoSubmitted) {
          const { resolveBuffer, count = 2 } = state.profileInfo;
          if (resolveBuffer) {
            const byteLength = Math.max(16, Math.ceil(count) * 8);
            try {
              await resolveBuffer.mapAsync(gpuMapModeRead, 0, byteLength);
              const copy = resolveBuffer.getMappedRange(0, byteLength).slice(0);
              const timestamps = new BigUint64Array(copy);
              const begin = timestamps[0] || 0n;
              const end = timestamps[1] || 0n;
              const diff = end > begin ? end - begin : 0n;
              const period = Number(context.device?.limits?.timestampPeriod || 1);
              const gpuMs = Number(diff) * (period / 1e6);
              if (Number.isFinite(gpuMs)) {
                result.gpuTimeMs = gpuMs;
              }
            } finally {
              try {
                resolveBuffer.unmap();
              } catch (err) {
                void err;
              }
            }
            result.queueFlushed = true;
            if (context) {
              context._pendingDispatch = false;
            }
          }
        } else {
          result.profileInfo = state.profileInfo;
        }
      }

      const wantsGpuTiming = Boolean(context?.profilingOptions?.timestampQueries);
      if (
        autoSubmitted &&
        !result.queueFlushed &&
        wantsGpuTiming &&
        (!state.profileInfo || !state.profileInfo.usedTimestampQuery) &&
        context.queue?.onSubmittedWorkDone
      ) {
        const waitStart = hasPerf ? performance.now() : 0;
        try {
          await context.queue.onSubmittedWorkDone();
        } finally {
          if (context) {
            context._pendingDispatch = false;
          }
        }
        if (waitStart && hasPerf) {
          const elapsed = performance.now() - waitStart;
          if (Number.isFinite(elapsed)) {
            result.gpuTimeMs = elapsed;
          }
        }
        result.queueFlushed = true;
      }

      if (context?.profile) {
        if (Number.isFinite(result.gpuTimeMs)) {
          context.profile.lastGPUTime = result.gpuTimeMs;
          context.profile.webgpu = result.gpuTimeMs;
        } else {
          context.profile.lastGPUTime = 0;
          context.profile.webgpu = 0;
        }
        if (Number.isFinite(result.dispatchWallMs)) {
          context.profile.lastDispatchMs = result.dispatchWallMs;
        } else {
          context.profile.lastDispatchMs = 0;
        }
        context.profile.timestampQueryEnabled = Boolean(result.usedTimestampQuery && result.gpuTimeMs !== null);
        context.profile.parityReadback = Boolean(readback);
      }

      if (hasPerf && callStart) {
        result.totalTimeMs = performance.now() - callStart;
      }

      return result;
    };

    const hasPerf = typeof performance !== 'undefined' && performance && typeof performance.now === 'function';
    const callStart = hasPerf ? performance.now() : 0;

    if (autoSubmit) {
      const dispatchStart = hasPerf ? performance.now() : 0;
      const state = await context.withEncoder((encoder) => encode(encoder));
      const dispatchMs = hasPerf ? performance.now() - dispatchStart : null;
      return finalizeResult(state, true, dispatchMs, callStart);
    }

    const dispatchStart = hasPerf ? performance.now() : 0;
    const state = await encode(providedEncoder);
    const dispatchMs = hasPerf ? performance.now() - dispatchStart : null;
    return finalizeResult(state, false, dispatchMs, callStart);
  }

  dispose() {
    this.destroy();
  }

  destroy() {
    if (Array.isArray(this.uniformBuffers)) {
      for (const holder of this.uniformBuffers) {
        if (holder?.release) {
          try {
            holder.release();
          } catch (err) {
            void err;
          }
        }
      }
    }
    this.uniformBuffers = [];
    if (Array.isArray(this.stages)) {
      for (const descriptor of this.stages) {
        if (!descriptor) continue;
        if (descriptor.pipeline && typeof descriptor.pipeline.destroy === 'function') {
          try {
            descriptor.pipeline.destroy();
          } catch (err) {
            void err;
          }
        }
        if (descriptor._maskGPUBuffer && typeof descriptor._maskGPUBuffer.destroy === 'function') {
          try {
            descriptor._maskGPUBuffer.destroy();
          } catch (err) {
            void err;
          }
        }
        descriptor.pipeline = null;
        descriptor.bindGroupLayout = null;
        descriptor.pipelineLayout = null;
        descriptor._maskGPUBuffer = null;
        descriptor._maskGPUBufferSize = 0;
        descriptor._maskData = null;
      }
    }
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

function buildUniformBufferPair(stage, ctx) {
  const layout = stage.uniformLayout;
  if (!layout || !layout.size) {
    return { entry: null, views: [] };
  }
  const buildFallback = () => {
    const buffers = [new ArrayBuffer(layout.size), new ArrayBuffer(layout.size)];
    const views = buffers.map((buffer, index) => ({
      index,
      view: new DataView(buffer),
      arrayBuffer: buffer,
      gpuBuffer: null,
      flush: () => {},
    }));
    return { entry: null, views, release: () => {} };
  };
  if (!ctx) {
    return buildFallback();
  }
  let entry = null;
  try {
    entry = ctx.acquireUniformBufferPair(layout.size);
  } catch {
    entry = null;
  }
  if (!entry) {
    return buildFallback();
  }
  const views = [0, 1].map((index) => {
    const view = typeof entry.getView === 'function' ? entry.getView(index) : null;
    const arrayBuffer =
      typeof entry.getArrayBuffer === 'function'
        ? entry.getArrayBuffer(index)
        : view?.buffer || new ArrayBuffer(layout.size);
    const flush = () => {
      if (typeof entry.flush === 'function') {
        entry.flush(index);
      } else if (typeof entry.flushAll === 'function') {
        entry.flushAll();
      }
    };
    const gpuBuffer =
      typeof entry.getGPUBuffer === 'function' ? entry.getGPUBuffer(index) : null;
    return {
      index,
      view: view || new DataView(arrayBuffer),
      arrayBuffer,
      gpuBuffer,
      flush,
    };
  });
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    if (entry && typeof ctx.releaseUniformBufferPair === 'function') {
      ctx.releaseUniformBufferPair(entry);
    }
    entry = null;
  };
  return { entry, views, release };
}

function normalizeResourceType(resourceType) {
  if (typeof resourceType !== 'string') {
    return 'unknown';
  }
  const trimmed = resourceType.trim();
  if (!trimmed) {
    return 'unknown';
  }
  const camelToKebab = trimmed.replace(/([a-z0-9])([A-Z])/g, '$1-$2');
  return camelToKebab.replace(/[\s_]+/g, '-').toLowerCase();
}

function buildBindingSignature(descriptor) {
  const parts = [];
  const bindings = descriptor.bindings || {};
  if (bindings.hasUniform) parts.push('uniform');
  if (bindings.hasFrameUniform) parts.push('frame');
  if (bindings.readsTexture) parts.push('read-storage');
  if (bindings.writesTexture) parts.push('write-storage');
  if (Array.isArray(bindings.auxiliary)) {
    for (const aux of bindings.auxiliary) {
      const type = normalizeResourceType(aux?.resourceType || 'unknown');
      const opt = aux?.optional ? 'opt' : 'req';
      const accessRaw = typeof aux?.access === 'string' ? aux.access.trim().toLowerCase() : '';
      const access = accessRaw || (type === 'storage-buffer' ? 'read-write' : '');
      const suffix = access && type === 'storage-buffer' && access !== 'read-write' ? `:${access}` : '';
      parts.push(`aux:${type}:${opt}${suffix}`);
    }
  }
  return parts.length ? parts.join('|') : 'none';
}

function buildBindGroupLayoutEntries(descriptor) {
  const entries = [];
  const bindings = descriptor.bindings || {};
  const visibility =
    typeof GPUShaderStage !== 'undefined' ? GPUShaderStage.COMPUTE : COMPUTE_STAGE_VISIBILITY;
  if (visibility === 0) {
    return entries;
  }
  if (bindings.hasUniform) {
    entries.push({
      binding: 0,
      visibility,
      buffer: { type: 'uniform' },
    });
  }
  if (bindings.hasFrameUniform) {
    entries.push({
      binding: 1,
      visibility,
      buffer: { type: 'uniform' },
    });
  }
  if (bindings.readsTexture) {
    entries.push({
      binding: 2,
      visibility,
      storageTexture: { access: 'read-only', format: STORAGE_TEXTURE_FORMAT, viewDimension: '2d' },
    });
  }
  if (bindings.writesTexture) {
    entries.push({
      binding: 3,
      visibility,
      storageTexture: { access: 'write-only', format: STORAGE_TEXTURE_FORMAT, viewDimension: '2d' },
    });
  }
  const auxiliaries = Array.isArray(bindings.auxiliary) ? bindings.auxiliary : [];
  auxiliaries.forEach((aux, idx) => {
    const binding = 4 + idx;
    const entry = buildAuxiliaryLayoutEntry(aux, binding, visibility);
    if (entry) {
      entries.push(entry);
    }
  });
  return entries;
}

function buildAuxiliaryLayoutEntry(aux, binding, visibility) {
  if (!aux) return null;
  const normalizedType = normalizeResourceType(aux.resourceType || 'texture');
  const collapsedType = normalizedType.replace(/-/g, '');
  switch (collapsedType) {
    case 'sampler':
      return { binding, visibility, sampler: {} };
    case 'storagebuffer': {
      const accessRaw = typeof aux.access === 'string' ? aux.access.trim().toLowerCase() : '';
      const bufferType = accessRaw === 'read-only' ? 'read-only-storage' : 'storage';
      return { binding, visibility, buffer: { type: bufferType } };
    }
    case 'buffer':
    case 'arraybuffer':
    case 'typedarray':
      return { binding, visibility, buffer: { type: 'read-only-storage' } };
    case 'null':
      return null;
    default:
      return {
        binding,
        visibility,
        texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
      };
  }
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
    if (effect.__effectName === 'palette' && typeof params.name === 'string') {
      // Palette shader uniforms resolve palette coefficients at runtime. The
      // string parameter is preserved in the stage snapshot metadata but is
      // removed here so GPU support detection is not blocked by non-numeric
      // values.
      delete params.name;
    }
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
  const isMultiresGenerator = isMultiresGeneratorStage(stage);
  const descriptor = {
    index,
    stageType: stage.stageType,
    category: stage.category,
    label: stage.label,
    signature: `${stage.category}:${stage.name}`,
    shaderId: resolveShaderId(stage),
    shaderSource: null,
    shaderFilename: null,
    pipeline: null,
    bindingSignature: null,
    bindGroupLayout: null,
    pipelineLayout: null,
    uniformLayout: null,
    uniformDefaults: {},
    resourceParams: [],
    normalizationShaderFilename: null,
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

  const paramAnalysis = isMultiresGenerator
    ? analyseMultiresGeneratorStage(stage.params || {})
    : analyseStageParams(stage.params || {});
  descriptor.uniformLayout = paramAnalysis.layout;
  descriptor.uniformDefaults = paramAnalysis.defaults;
  descriptor.resourceParams = paramAnalysis.resources;
  descriptor.bindings.hasUniform = Boolean(descriptor.uniformLayout);
  descriptor.bindings.auxiliary = paramAnalysis.resources.map((r) => ({
    name: r.name,
    resourceType: r.resourceType,
    optional: r.value === null || r.value === undefined,
    access: r.access,
  }));
  descriptor.issues.push(...paramAnalysis.issues);
  if (paramAnalysis.unsupported) {
    descriptor.gpuSupported = false;
  }

  if (isMultiresGenerator) {
    specializeMultiresDescriptor(descriptor, stage);
  } else if (stage.stageType === 'effect') {
    specializeEffectDescriptor(descriptor, stage);
  }

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

  descriptor.bindingSignature = buildBindingSignature(descriptor);
  if (ctx && typeof ctx.getCachedBindGroupLayout === 'function') {
    descriptor.bindGroupLayout = ctx.getCachedBindGroupLayout(
      descriptor.bindingSignature,
      () => buildBindGroupLayoutEntries(descriptor),
    );
  }
  if (descriptor.bindGroupLayout && ctx && typeof ctx.getCachedPipelineLayout === 'function') {
    descriptor.pipelineLayout = ctx.getCachedPipelineLayout(
      descriptor.shaderId,
      descriptor.bindingSignature,
      [descriptor.bindGroupLayout],
    );
  }

  if (!descriptor.shaderSource) {
    descriptor.gpuSupported = false;
  }

  return descriptor;
}

function resolveShaderId(stage) {
  if (!stage || stage.unsupported) return null;
  if (stage.stageType === 'generator') {
    if (isMultiresGeneratorStage(stage)) {
      return 'MULTIRES_WGSL';
    }
    return null;
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
  const templateFilename = getShaderFilename(template);
  if (templateFilename && !descriptor.shaderFilename) {
    descriptor.shaderFilename = templateFilename;
  }
  let source = template;
  if (sharedResources && typeof sharedResources.specializeShaderSource === 'function') {
    const specialized = sharedResources.specializeShaderSource(template, descriptor);
    if (!specialized) {
      return specialized;
    }
    source = inheritShaderFilename(specialized, template);
  }
  const resolvedFilename = getShaderFilename(source);
  if (resolvedFilename && !descriptor.shaderFilename) {
    descriptor.shaderFilename = resolvedFilename;
  }
  return source;
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

function getStageGeneratorName(stage) {
  if (!stage) return null;
  const candidate =
    (typeof stage.name === 'string' && stage.name) ||
    (stage.params && typeof stage.params.generator === 'string' && stage.params.generator);
  if (!candidate) return null;
  return candidate.toLowerCase();
}

function isMultiresGeneratorStage(stage) {
  if (!stage || stage.stageType !== 'generator') {
    return false;
  }
  const name = getStageGeneratorName(stage);
  return !name || name === 'multires';
}

function analyseMultiresGeneratorStage(params) {
  const issues = [];
  let unsupported = false;
  const flagUnsupported = (name) => {
    unsupported = true;
    issues.push({
      level: 'error',
      message: `multires generator parameter "${name}" is not yet supported on the GPU path`,
    });
  };
  if (params && typeof params === 'object') {
    const maskParam = resolveParam(params, ['mask']);
    const maskValue = normalizeMaskValue(maskParam);
    if (maskParam !== undefined && maskParam !== null && maskValue === null) {
      flagUnsupported('mask');
    }
    const hueDistribValue = normalizeOverrideDistribution(
      resolveParam(params, ['hueDistrib', 'hue_distrib']),
    );
    if (hueDistribValue === null) {
      unsupported = true;
      issues.push({
        level: 'error',
        message: 'multires generator parameter "hueDistrib" is not supported on the GPU path',
      });
    }

    const saturationDistribValue = normalizeOverrideDistribution(
      resolveParam(params, ['saturationDistrib', 'saturation_distrib']),
    );
    if (saturationDistribValue === null) {
      unsupported = true;
      issues.push({
        level: 'error',
        message: 'multires generator parameter "saturationDistrib" is not supported on the GPU path',
      });
    }

    const brightnessDistribValue = normalizeOverrideDistribution(
      resolveParam(params, ['brightnessDistrib', 'brightness_distrib']),
    );
    if (brightnessDistribValue === null) {
      unsupported = true;
      issues.push({
        level: 'error',
        message: 'multires generator parameter "brightnessDistrib" is not supported on the GPU path',
      });
    }

    const brightnessFreqValue = resolveParam(params, ['brightnessFreq', 'brightness_freq']);
    if (!isBrightnessFrequencyParamSupported(brightnessFreqValue)) {
      unsupported = true;
      issues.push({
        level: 'error',
        message: 'multires generator parameter "brightnessFreq" is not supported on the GPU path',
      });
    }
    if (params.withSupersample) flagUnsupported('withSupersample');
    if (params.withFxaa) flagUnsupported('withFxaa');
    if (params.withAi) flagUnsupported('withAi');
    if (params.withUpscale) flagUnsupported('withUpscale');
    const octaveEffects = Array.isArray(params.octaveEffects) ? params.octaveEffects : [];
    if (octaveEffects.length) flagUnsupported('octaveEffects');
    const postEffects = Array.isArray(params.postEffects) ? params.postEffects : [];
    if (postEffects.length) flagUnsupported('postEffects');
    const finalEffects = Array.isArray(params.finalEffects) ? params.finalEffects : [];
    if (finalEffects.length) flagUnsupported('finalEffects');
  }

  const fields = [
    { name: 'freq', scalarType: 'f32', components: 2, defaultValue: [1, 1] },
    { name: 'speed', scalarType: 'f32', components: 1, defaultValue: 1 },
    { name: 'sin_amount', scalarType: 'f32', components: 1, defaultValue: 0 },
    { name: 'color_params0', scalarType: 'f32', components: 4, defaultValue: [0.125, 0, 1, 0] },
    { name: 'color_params1', scalarType: 'f32', components: 4, defaultValue: [0, 0, 0, 0] },
    {
      name: 'options0',
      scalarType: 'u32',
      components: 4,
      defaultValue: [1, OctaveBlending.falloff >>> 0, 3, 0],
    },
    {
      name: 'options1',
      scalarType: 'u32',
      components: 4,
      defaultValue: [0, ValueDistribution.simplex >>> 0, ColorSpace.hsv >>> 0, 0],
    },
    { name: 'options2', scalarType: 'u32', components: 4, defaultValue: [0, 0, 0, 0] },
    { name: 'options3', scalarType: 'u32', components: 4, defaultValue: [0, 0, 0, 0] },
  ];
  const layout = buildStd140Layout(fields, issues);
  const defaults = {
    freq: fields[0].defaultValue.slice(),
    speed: fields[1].defaultValue,
    sin_amount: fields[2].defaultValue,
    color_params0: fields[3].defaultValue.slice(),
    color_params1: fields[4].defaultValue.slice(),
    options0: fields[5].defaultValue.slice(),
    options1: fields[6].defaultValue.slice(),
    options2: fields[7].defaultValue.slice(),
    options3: fields[8].defaultValue.slice(),
  };

  return { layout, defaults, resources: [], issues, requiresUniforms: true, unsupported };
}

function analyseStageParams(params) {
  const defaults = {};
  const resources = [];
  const issues = [];
  let unsupported = false;
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
      if (classification.resourceType === 'string') {
        unsupported = true;
        issues.push({
          level: 'error',
          message: `Parameter "${name}" with type string is not supported on the GPU path.`,
        });
      }
      resources.push({ name, resourceType: classification.resourceType, value });
    } else {
      issues.push({ level: 'error', message: classification.reason });
    }
  }

  const layout = uniformFields.length ? buildStd140Layout(uniformFields, issues) : null;

  return { layout, defaults, resources, issues, requiresUniforms, unsupported };
}

function getStageName(stage) {
  if (!stage) return '';
  const raw = stage.name || stage.params?.generator || '';
  return typeof raw === 'string' ? raw.toLowerCase() : '';
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

function specializeMultiresDescriptor(descriptor, stage) {
  descriptor.shaderId = 'MULTIRES_WGSL';
  const metadata = buildMultiresUniformMetadata(stage);
  if (!metadata?.layout) {
    descriptor.gpuSupported = false;
    descriptor.issues.push({
      level: 'error',
      message: 'Failed to build multires uniform layout.',
    });
    return;
  }
  descriptor.uniformLayout = metadata.layout;
  descriptor.uniformDefaults = metadata.defaults;
  descriptor.resourceParams = [];
  descriptor.bindings.hasUniform = true;
  descriptor.bindings.auxiliary = [];
  const normalizationResource = {
    name: 'normalizationState',
    resourceType: 'storage-buffer',
    size: 16,
    access: 'read-write',
  };
  descriptor.resourceParams.push(normalizationResource);
  descriptor.bindings.auxiliary.push({
    name: normalizationResource.name,
    resourceType: normalizationResource.resourceType,
    optional: false,
    access: normalizationResource.access,
  });
  descriptor.requiresNormalizationResolve = true;
  descriptor.normalizationShaderSource = SHADERS.MULTIRES_NORMALIZE_WGSL;
  descriptor.normalizationShaderFilename =
    getShaderFilename(descriptor.normalizationShaderSource) || descriptor.normalizationShaderFilename || null;
  const workgroupSize = Array.isArray(descriptor.specialization?.workgroupSize)
    ? descriptor.specialization.workgroupSize.slice()
    : [...DEFAULT_WORKGROUP_SIZE];
  descriptor.normalizationWorkgroupSize = workgroupSize;
  const sinStateResource = {
    name: 'sinNormalizationState',
    resourceType: 'storage-buffer',
    size: 16,
    access: 'read-write',
  };
  descriptor.resourceParams.push(sinStateResource);
  descriptor.bindings.auxiliary.push({
    name: sinStateResource.name,
    resourceType: sinStateResource.resourceType,
    optional: false,
    access: sinStateResource.access,
  });
  const maskResource = {
    name: 'maskData',
    resourceType: 'storage-buffer',
    size: 0,
    access: 'read-only',
  };
  descriptor.resourceParams.push(maskResource);
  descriptor.bindings.auxiliary.push({
    name: maskResource.name,
    resourceType: maskResource.resourceType,
    optional: false,
    access: maskResource.access,
  });
  const permutationResource = {
    name: 'permutationTables',
    resourceType: 'storage-buffer',
    size: 0,
    access: 'read-only',
  };
  descriptor.resourceParams.push(permutationResource);
  descriptor.bindings.auxiliary.push({
    name: permutationResource.name,
    resourceType: permutationResource.resourceType,
    optional: false,
    access: permutationResource.access,
  });
  descriptor.multiresBaseParams = cloneStageParams(stage?.params || {});
  descriptor.multiresMaskConfig = extractMultiresMaskConfig(stage?.params || {});
  ensureMaskDataHolder(descriptor);
  descriptor.resolveUniformParams = (params) =>
    resolveMultiresUniformParams(params, descriptor);
  descriptor.specialization = descriptor.specialization || {
    workgroupSize: [...DEFAULT_WORKGROUP_SIZE],
    constants: {},
  };
  descriptor.specialization.constants.channelCount = metadata.defaults?.options0?.[2] ?? 4;
  if (Array.isArray(metadata.issues) && metadata.issues.length) {
    descriptor.issues.push(...metadata.issues);
  }
}

function specializeEffectDescriptor(descriptor, stage) {
  if (!descriptor || !stage) return;
  const stageName = getStageName(stage);
  switch (stageName) {
    case 'palette':
      specializePaletteDescriptor(descriptor);
      break;
    default:
      break;
  }
}

function toVec4(values, fallback) {
  const base = Array.isArray(fallback) ? fallback.slice(0, 4) : [0, 0, 0, 0];
  const src = Array.isArray(values) ? values : [];
  for (let i = 0; i < 4; i += 1) {
    if (i < src.length && Number.isFinite(Number(src[i]))) {
      base[i] = Number(src[i]);
    } else if (!Number.isFinite(Number(base[i]))) {
      base[i] = 0;
    }
  }
  if (base.length < 4) {
    base.length = 4;
  }
  return base.slice(0, 4);
}

function specializePaletteDescriptor(descriptor) {
  if (!descriptor) return;
  const defaults = {
    width: 1,
    height: 1,
    channels: 4,
    blend: 0,
    amp: [0.5, 0.5, 0.5, 0],
    freq: [1, 1, 1, 0],
    offset: [0.5, 0.5, 0.5, 0],
    phase: [0, 0, 0, 0],
  };
  const fields = [
    { name: 'width', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.width },
    { name: 'height', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.height },
    { name: 'channels', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.channels },
    { name: 'blend', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.blend },
    { name: 'amp', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.amp },
    { name: 'freq', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.freq },
    { name: 'offset', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.offset },
    { name: 'phase', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.phase },
  ];
  descriptor.uniformLayout = buildStd140Layout(fields, descriptor.issues);
  descriptor.uniformDefaults = defaults;
  descriptor.bindings.hasUniform = true;
  descriptor.resolveUniformParams = (params = {}) => {
    const width = Math.max(1, Math.floor(Number(params.width) || defaults.width));
    const height = Math.max(1, Math.floor(Number(params.height) || defaults.height));
    const channels = Math.max(1, Math.floor(Number(params.channels) || Number(params.channelCount) || defaults.channels));
    let blend = defaults.blend;
    if (typeof params.alpha === 'number' && Number.isFinite(params.alpha)) {
      const angle = params.alpha * Math.PI;
      blend = Math.fround((1 - Math.cos(angle)) * 0.5);
    }
    const paletteName = typeof params.name === 'string' ? params.name : null;
    const palette = paletteName && Object.prototype.hasOwnProperty.call(PALETTES, paletteName)
      ? PALETTES[paletteName]
      : null;
    const amp = toVec4(palette?.amp, defaults.amp);
    const freq = toVec4(palette?.freq, defaults.freq);
    const offset = toVec4(palette?.offset, defaults.offset);
    const phaseBase = toVec4(palette?.phase, defaults.phase);
    const time = Number(params.time) || 0;
    for (let i = 0; i < 3; i += 1) {
      phaseBase[i] = Math.fround((phaseBase[i] ?? 0) + time);
    }
    if (!palette || !paletteName || channels < 3) {
      blend = 0;
    }
    return {
      width,
      height,
      channels,
      blend,
      amp,
      freq,
      offset,
      phase: phaseBase,
    };
  };
}

function buildMultiresUniformMetadata(stage) {
  const issues = [];
  const params = (stage && stage.params) || {};
  const freq = normalizeVec2(resolveParam(params, ['freq', 'frequency']), [1, 1]);
  const speed = toNumber(resolveParam(params, ['speed']), 1);
  const sinAmount = toNumber(resolveParam(params, ['sin', 'sinAmount']), 0);
  const hueRange = toNumber(resolveParam(params, ['hueRange', 'hue_range']), 0.125);
  const hueRotation = toNumber(resolveParam(params, ['hueRotation', 'hue_rotation']), 0);
  const saturation = toNumber(resolveParam(params, ['saturation']), 1);
  const splineOrder = toUint(
    resolveParam(params, ['splineOrder', 'spline_order']),
    InterpolationType.bicubic,
  );
  const octaves = toUint(resolveParam(params, ['octaves']), 1);
  const octaveBlending = toUint(
    resolveParam(params, ['octaveBlending', 'octave_blending']),
    OctaveBlending.falloff,
  );
  const ridges = toBoolean(resolveParam(params, ['ridges', 'ridged']));
  const distrib = toUint(resolveParam(params, ['distrib', 'distribution']), ValueDistribution.simplex);
  const seedOffset = toUint(resolveParam(params, ['seedOffset', 'seed_offset', 'seed']), 0);
  const colorSpace = resolveColorSpaceValue(
    resolveParam(params, ['color_space', 'colorSpace']),
    ColorSpace.hsv,
  );
  const withAlpha = toBoolean(resolveParam(params, ['withAlpha', 'with_alpha']));
  const channelCount = computeChannelCount(colorSpace, withAlpha, octaveBlending);
  const latticeDrift = toNumber(resolveParam(params, ['latticeDrift', 'lattice_drift']), 0);
  const cornersFlag = toBoolean(resolveParam(params, ['corners']));

  const hueDistribValue = normalizeOverrideDistribution(
    resolveParam(params, ['hueDistrib', 'hue_distrib']),
  );
  if (hueDistribValue === null) {
    issues.push({
      level: 'error',
      message: 'Unsupported hueDistrib override for multires generator.',
    });
  }
  const safeHueDistrib = hueDistribValue === null ? 0 : hueDistribValue >>> 0;

  const saturationDistribValue = normalizeOverrideDistribution(
    resolveParam(params, ['saturationDistrib', 'saturation_distrib']),
  );
  if (saturationDistribValue === null) {
    issues.push({
      level: 'error',
      message: 'Unsupported saturationDistrib override for multires generator.',
    });
  }
  const safeSaturationDistrib =
    saturationDistribValue === null ? 0 : saturationDistribValue >>> 0;

  const brightnessDistribValue = normalizeOverrideDistribution(
    resolveParam(params, ['brightnessDistrib', 'brightness_distrib']),
  );
  if (brightnessDistribValue === null) {
    issues.push({
      level: 'error',
      message: 'Unsupported brightnessDistrib override for multires generator.',
    });
  }
  let safeBrightnessDistrib =
    brightnessDistribValue === null ? 0 : brightnessDistribValue >>> 0;

  const brightnessFreqRaw = resolveParam(params, ['brightnessFreq', 'brightness_freq']);
  let brightnessFreqFlag = 0;
  if (brightnessFreqRaw !== undefined && brightnessFreqRaw !== null) {
    if (isBrightnessFrequencyParamSupported(brightnessFreqRaw)) {
      brightnessFreqFlag = 1;
    } else {
      issues.push({
        level: 'error',
        message: 'Unsupported brightnessFreq override for multires generator.',
      });
    }
  }
  if (brightnessFreqFlag && safeBrightnessDistrib === 0) {
    safeBrightnessDistrib = ValueDistribution.simplex >>> 0;
  }

  const defaults = {
    freq,
    speed,
    sin: sinAmount,
    colorParams0: [hueRange, hueRotation, saturation, 0],
    colorParams1: [0, 0, latticeDrift, splineOrder],
    options0: [octaves, octaveBlending, channelCount, ridges ? 1 : 0],
    options1: [seedOffset, distrib, colorSpace, withAlpha ? 1 : 0],
    options2: [safeHueDistrib, safeSaturationDistrib, safeBrightnessDistrib, brightnessFreqFlag >>> 0],
    options3: [cornersFlag ? 1 : 0, 0, 0, 0],
  };

  const fields = [
    { name: 'freq', scalarType: 'f32', components: 2, bool: false, defaultValue: defaults.freq },
    { name: 'speed', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.speed },
    { name: 'sin', scalarType: 'f32', components: 1, bool: false, defaultValue: defaults.sin },
    { name: 'colorParams0', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.colorParams0 },
    { name: 'colorParams1', scalarType: 'f32', components: 4, bool: false, defaultValue: defaults.colorParams1 },
    { name: 'options0', scalarType: 'u32', components: 4, bool: false, defaultValue: defaults.options0 },
    { name: 'options1', scalarType: 'u32', components: 4, bool: false, defaultValue: defaults.options1 },
    { name: 'options2', scalarType: 'u32', components: 4, bool: false, defaultValue: defaults.options2 },
    { name: 'options3', scalarType: 'u32', components: 4, bool: false, defaultValue: defaults.options3 },
  ];
  const layout = buildStd140Layout(fields, issues);
  return { layout, defaults, issues };
}

function resolveParam(params, names) {
  if (!params) return undefined;
  const list = Array.isArray(names) ? names : [names];
  for (const name of list) {
    if (Object.prototype.hasOwnProperty.call(params, name) && params[name] !== undefined) {
      return params[name];
    }
  }
  return undefined;
}

function normalizeVec2(value, fallback) {
  const out = Array.isArray(value)
    ? value.map((v) => Number(v))
    : Number.isFinite(value)
    ? [Number(value), Number(value)]
    : [];
  const a = Number.isFinite(out[0]) ? out[0] : fallback[0];
  const b = Number.isFinite(out[1]) ? out[1] : fallback[1] ?? a;
  return [Number.isFinite(a) ? a : 1, Number.isFinite(b) ? b : Number.isFinite(a) ? a : 1];
}

function toNumber(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function toUint(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback >>> 0;
  return Math.max(0, Math.trunc(num)) >>> 0;
}

function toBoolean(value) {
  if (value === undefined || value === null) return false;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') {
    const lowered = value.toLowerCase();
    return lowered === 'true' || lowered === 'yes' || lowered === 'on' || lowered === '1';
  }
  return Boolean(value);
}

function resolveColorSpaceValue(value, fallback) {
  if (typeof value === 'string') {
    const key = value.toLowerCase();
    if (Object.prototype.hasOwnProperty.call(ColorSpace, key)) {
      return ColorSpace[key];
    }
  }
  const num = Number(value);
  if (Number.isFinite(num) && num > 0) {
    return num;
  }
  return fallback;
}

function computeChannelCount(colorSpace, withAlpha, octaveBlending) {
  const space = Number.isFinite(colorSpace) ? colorSpace : ColorSpace.hsv;
  let channels = space === ColorSpace.grayscale ? 1 : 3;
  if (withAlpha) {
    channels += 1;
  }
  const blending = Number.isFinite(octaveBlending) ? octaveBlending : OctaveBlending.falloff;
  if (blending === OctaveBlending.alpha && (channels === 1 || channels === 3)) {
    channels += 1;
  }
  return channels;
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

function resolveMultiresUniformParams(params, descriptor) {
  const merged = mergeStageParams(descriptor?.multiresBaseParams, params);
  const width = coerceNumber(
    merged.width ?? (Array.isArray(merged.shape) ? merged.shape[1] : undefined),
    0,
  );
  const height = coerceNumber(
    merged.height ?? (Array.isArray(merged.shape) ? merged.shape[0] : undefined),
    0,
  );
  const freqValue = merged.freq ?? merged.frequency;
  const freq = computeMultiresFrequency(freqValue, width, height, merged.shape);

  const speed = coerceNumber(merged.speed, 1);
  const sinValue = coerceNumber(merged.sin ?? merged.sinAmount, 0);
  const hueRange = coerceNumber(merged.hueRange ?? merged.hue_range, 0.125);
  const hueRotationRaw = merged.hueRotation ?? merged.hue_rotation;
  const hueRotation = coerceNumber(hueRotationRaw, 0);
  const saturation = coerceNumber(merged.saturation, 1);
  const splineOrder = coerceInt(
    merged.splineOrder ?? merged.spline_order,
    InterpolationType.bicubic,
  );

  const colorSpace = coerceColorSpace(merged.colorSpace ?? merged.color_space);
  const withAlpha = coerceBool(merged.withAlpha ?? merged.with_alpha);
  const ridges = coerceBool(merged.ridges);
  const octaves = Math.max(1, coerceInt(merged.octaves, 1));
  const octaveBlending = coerceOctaveBlending(
    merged.octaveBlending ?? merged.octave_blending,
  );
  const distrib = coerceValueDistribution(merged.distrib);
  const seedOffset = coerceInt(merged.seedOffset ?? merged.seed_offset, 0);
  const latticeDrift = coerceNumber(merged.latticeDrift ?? merged.lattice_drift, 0);
  const cornersFlag = coerceBool(merged.corners);

  const channelCount = computeMultiresChannelCount(
    colorSpace,
    withAlpha,
    octaveBlending,
  );

  const hueDistribValue = normalizeOverrideDistribution(
    merged.hueDistrib ?? merged.hue_distrib,
  );
  const hueDistrib = hueDistribValue === null ? 0 : hueDistribValue >>> 0;

  const saturationDistribValue = normalizeOverrideDistribution(
    merged.saturationDistrib ?? merged.saturation_distrib,
  );
  const saturationDistrib =
    saturationDistribValue === null ? 0 : saturationDistribValue >>> 0;

  let brightnessDistribValue = normalizeOverrideDistribution(
    merged.brightnessDistrib ?? merged.brightness_distrib,
  );
  if (brightnessDistribValue === null) {
    brightnessDistribValue = 0;
  }

  const brightnessFreqRaw = merged.brightnessFreq ?? merged.brightness_freq;
  let brightnessFreqFlag = 0;
  let brightnessFreqVec = [0, 0];
  if (brightnessFreqRaw !== undefined && brightnessFreqRaw !== null) {
    if (isBrightnessFrequencyParamSupported(brightnessFreqRaw)) {
      const freqVec = computeMultiresFrequency(
        brightnessFreqRaw,
        width,
        height,
        merged.shape,
      );
      if (Array.isArray(freqVec) && freqVec.length >= 2) {
        brightnessFreqVec = [freqVec[0], freqVec[1]];
        brightnessFreqFlag = 1;
      }
    }
  }

  let brightnessDistrib = brightnessDistribValue >>> 0;
  if (brightnessFreqFlag && brightnessDistrib === 0) {
    brightnessDistrib = ValueDistribution.simplex >>> 0;
  }
  if (!brightnessFreqFlag && brightnessDistrib === 0) {
    brightnessFreqVec = [0, 0];
  }

  const hasBrightnessOverride =
    brightnessDistrib !== 0 || brightnessFreqFlag !== 0;
  if (!hasBrightnessOverride) {
    brightnessFreqVec = [0, 0];
    brightnessFreqFlag = 0;
  }

  const maskConfig = extractMultiresMaskConfig(merged);
  const maskInfo = computeMultiresMaskData({
    descriptor,
    maskConfig,
    freq,
    width,
    height,
    octaves,
    channelCount,
    distrib,
    cornersFlag,
    splineOrder,
    time: merged.time ?? 0,
    speed,
    frameIndex: merged.frameIndex ?? 0,
  });
  if (descriptor) {
    descriptor.multiresMaskConfig = maskConfig;
  }

  if (descriptor) {
    descriptor._multiresPermutationPlan = {
      baseFreq: Array.isArray(freq) ? freq.slice(0, 2) : [freq, freq],
      octaves,
      channelCount,
      hasHueOverride: Boolean(hueDistrib),
      hasSaturationOverride: Boolean(saturationDistrib),
      hasBrightnessOverride,
      hasLatticeDrift: latticeDrift !== 0,
      seedOffset: seedOffset >>> 0,
    };
  }

  return {
    freq,
    speed,
    sin: sinValue,
    colorParams0: [hueRange, hueRotation, saturation, 0],
    colorParams1: [brightnessFreqVec[0], brightnessFreqVec[1], latticeDrift, splineOrder],
    options0: [octaves, octaveBlending, channelCount, ridges ? 1 : 0],
    options1: [seedOffset, distrib, colorSpace, withAlpha ? 1 : 0],
    options2: [hueDistrib, saturationDistrib, brightnessDistrib, brightnessFreqFlag >>> 0],
    options3: [
      cornersFlag ? 1 : 0,
      maskInfo.enabled ? 1 : 0,
      maskInfo.alpha ? 1 : 0,
      maskInfo.octaveCount,
    ],
  };
}

function mergeStageParams(base, overrides) {
  const out = {};
  if (base && typeof base === 'object') {
    for (const [key, value] of Object.entries(base)) {
      if (value !== undefined) {
        out[key] = value;
      }
    }
  }
  if (overrides && typeof overrides === 'object') {
    for (const [key, value] of Object.entries(overrides)) {
      if (value !== undefined) {
        out[key] = value;
      }
    }
  }
  return out;
}

function coerceNumber(value, fallback = 0) {
  if (value === null || value === undefined) return fallback;
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : fallback;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return fallback;
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  if (Array.isArray(value) && value.length) {
    return coerceNumber(value[0], fallback);
  }
  if (ArrayBuffer.isView(value) && value.length) {
    return coerceNumber(value[0], fallback);
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function coerceInt(value, fallback = 0) {
  const num = coerceNumber(value, fallback);
  if (!Number.isFinite(num)) return fallback;
  return Math.trunc(num);
}

function coerceBool(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'false' || normalized === '0' || normalized === 'off') {
      return false;
    }
    if (normalized === 'true' || normalized === '1' || normalized === 'on') {
      return true;
    }
  }
  if (value === null || value === undefined) return false;
  const num = coerceNumber(value, NaN);
  if (Number.isFinite(num)) {
    return num !== 0;
  }
  return Boolean(value);
}

function coerceOctaveBlending(value) {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'reduce_max' || normalized === 'reduce-max') {
      return OctaveBlending.reduce_max;
    }
    if (normalized === 'alpha') {
      return OctaveBlending.alpha;
    }
    if (normalized === 'falloff') {
      return OctaveBlending.falloff;
    }
  }
  const numeric = coerceInt(value, OctaveBlending.falloff);
  if (
    numeric === OctaveBlending.falloff ||
    numeric === OctaveBlending.reduce_max ||
    numeric === OctaveBlending.alpha
  ) {
    return numeric;
  }
  return OctaveBlending.falloff;
}

function coerceValueDistribution(value) {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'exp' || normalized === 'exponential') {
      return ValueDistribution.exp;
    }
    if (normalized === 'simplex') {
      return ValueDistribution.simplex;
    }
  }
  const numeric = coerceInt(value, ValueDistribution.simplex);
  if (numeric === ValueDistribution.exp) {
    return ValueDistribution.exp;
  }
  return ValueDistribution.simplex;
}

function coerceColorSpace(value) {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'grayscale' || normalized === 'gray' || normalized === 'greyscale') {
      return ColorSpace.grayscale;
    }
    if (normalized === 'rgb') {
      return ColorSpace.rgb;
    }
    if (normalized === 'oklab') {
      return ColorSpace.oklab;
    }
    if (normalized === 'hsv') {
      return ColorSpace.hsv;
    }
  }
  const numeric = coerceInt(value, ColorSpace.hsv);
  if (
    numeric === ColorSpace.grayscale ||
    numeric === ColorSpace.rgb ||
    numeric === ColorSpace.hsv ||
    numeric === ColorSpace.oklab
  ) {
    return numeric;
  }
  return ColorSpace.hsv;
}

function computeMultiresFrequency(freqValue, width, height, shape) {
  if (Array.isArray(freqValue) || ArrayBuffer.isView(freqValue)) {
    const arr = Array.from(freqValue).map((v) => coerceNumber(v, 0));
    if (arr.length === 1) {
      const coerced = Math.max(1, Math.floor(arr[0] || 1));
      return [coerced, coerced];
    }
    if (arr.length >= 2) {
      const freqY = Math.max(1, Math.floor(arr[0] || 1));
      const freqX = Math.max(1, Math.floor(arr[1] || arr[0] || 1));
      return [freqX, freqY];
    }
  }

  const fallbackFreq = coerceNumber(freqValue, 1) || 1;
  let effectiveWidth = width;
  let effectiveHeight = height;
  if ((!effectiveWidth || !effectiveHeight) && Array.isArray(shape) && shape.length >= 2) {
    const shapeHeight = coerceNumber(shape[0], 0);
    const shapeWidth = coerceNumber(shape[1], 0);
    if (!effectiveHeight && shapeHeight) effectiveHeight = shapeHeight;
    if (!effectiveWidth && shapeWidth) effectiveWidth = shapeWidth;
  }
  const dims = freqForDimensions(fallbackFreq, effectiveWidth, effectiveHeight);
  const freqX = Math.max(1, Math.floor(dims[1] || 1));
  const freqY = Math.max(1, Math.floor(dims[0] || 1));
  return [freqX, freqY];
}

function freqForDimensions(freq, width, height) {
  const safeFreq = Math.max(1, Math.floor(coerceNumber(freq, 1) || 1));
  const safeWidth = Math.max(1, Math.floor(coerceNumber(width, 1) || 1));
  const safeHeight = Math.max(1, Math.floor(coerceNumber(height, 1) || 1));
  if (safeHeight === safeWidth) {
    return [safeFreq, safeFreq];
  }
  if (safeHeight < safeWidth) {
    const freqX = Math.max(1, Math.floor((safeFreq * safeWidth) / safeHeight));
    return [safeFreq, freqX];
  }
  const freqY = Math.max(1, Math.floor((safeFreq * safeHeight) / safeWidth));
  return [freqY, safeFreq];
}

function computeMultiresChannelCount(colorSpace, withAlpha, octaveBlending) {
  let baseChannels = colorSpace === ColorSpace.grayscale ? 1 : 3;
  if (withAlpha) {
    baseChannels += 1;
  }
  if (
    octaveBlending === OctaveBlending.alpha &&
    (baseChannels === 1 || baseChannels === 3)
  ) {
    baseChannels += 1;
  }
  return Math.max(1, Math.min(4, baseChannels));
}

const OVERRIDE_DISTRIBUTION_NAME_MAP = new Map([
  ['simplex', ValueDistribution.simplex >>> 0],
  ['exp', ValueDistribution.exp >>> 0],
  ['exponential', ValueDistribution.exp >>> 0],
  ['ones', ValueDistribution.ones >>> 0],
  ['one', ValueDistribution.ones >>> 0],
  ['mids', ValueDistribution.mids >>> 0],
  ['mid', ValueDistribution.mids >>> 0],
  ['zeros', ValueDistribution.zeros >>> 0],
  ['zero', ValueDistribution.zeros >>> 0],
  ['none', 0],
  ['off', 0],
]);

const ALLOWED_OVERRIDE_DISTRIBUTIONS = new Set([
  0,
  ValueDistribution.simplex >>> 0,
  ValueDistribution.exp >>> 0,
  ValueDistribution.ones >>> 0,
  ValueDistribution.mids >>> 0,
  ValueDistribution.zeros >>> 0,
]);

function normalizeOverrideDistribution(value) {
  if (value === undefined || value === null) {
    return 0;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    const normalized = value >>> 0;
    return ALLOWED_OVERRIDE_DISTRIBUTIONS.has(normalized) ? normalized : null;
  }
  if (typeof value === 'string') {
    const key = value.trim().toLowerCase();
    if (OVERRIDE_DISTRIBUTION_NAME_MAP.has(key)) {
      return OVERRIDE_DISTRIBUTION_NAME_MAP.get(key);
    }
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return normalizeOverrideDistribution(numeric);
    }
    return null;
  }
  return null;
}

function normalizeMaskValue(value) {
  if (value === undefined || value === null) {
    return 0;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    return Math.max(0, Math.trunc(value)) >>> 0;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return 0;
    const lowered = trimmed.toLowerCase();
    if (Object.prototype.hasOwnProperty.call(ValueMask, lowered)) {
      return ValueMask[lowered];
    }
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      return normalizeMaskValue(numeric);
    }
    return null;
  }
  return null;
}

function extractMultiresMaskConfig(params) {
  const maskValue = normalizeMaskValue(resolveParam(params, ['mask']));
  const enabled = Boolean(maskValue);
  if (!enabled) {
    return {
      enabled: false,
      value: 0,
      inverse: false,
      static: true,
    };
  }
  const inverse = toBoolean(resolveParam(params, ['maskInverse', 'mask_inverse'])) || false;
  const maskStatic = toBoolean(resolveParam(params, ['maskStatic', 'mask_static'])) || false;
  return {
    enabled,
    value: maskValue >>> 0,
    inverse,
    static: maskStatic,
  };
}

function computeMultiresOctaveFrequency(baseFreq, octaveIndex) {
  const multiplier = Math.pow(2, octaveIndex);
  let fx = Math.floor(baseFreq[0] * 0.5 * multiplier);
  let fy = Math.floor(baseFreq[1] * 0.5 * multiplier);
  if (!Number.isFinite(fx)) fx = 1;
  if (!Number.isFinite(fy)) fy = 1;
  fx = Math.max(1, fx);
  fy = Math.max(1, fy);
  return [fx, fy];
}

function pinCornersArray(data, width, height, channels, freqY, freqX, corners) {
  const fy = Math.max(1, Math.floor(freqY));
  const shouldOffset = (!corners && fy % 2 === 0) || (corners && fy % 2 === 1);
  if (!shouldOffset) {
    return data;
  }
  const fx = Math.max(1, Math.floor(freqX));
  const xOff = Math.floor((width / fx) * 0.5);
  const yOff = Math.floor((height / fy) * 0.5);
  const out = new Float32Array(width * height * channels);
  for (let y = 0; y < height; y += 1) {
    const sy = (y + yOff + height) % height;
    for (let x = 0; x < width; x += 1) {
      const sx = (x + xOff + width) % width;
      const dst = (y * width + x) * channels;
      const src = (sy * width + sx) * channels;
      for (let c = 0; c < channels; c += 1) {
        out[dst + c] = data[src + c];
      }
    }
  }
  return out;
}

function ensureMaskDataHolder(descriptor) {
  if (!descriptor) return null;
  if (!descriptor._maskData) {
    descriptor._maskData = {
      array: new Float32Array([1, 1, 1, 1]),
      width: 1,
      height: 1,
      octaveCount: 1,
      alpha: false,
      enabled: false,
    };
    descriptor._maskDataDirty = true;
  }
  return descriptor._maskData;
}

function buildMaskCacheKey(options) {
  const {
    maskConfig,
    freq,
    width,
    height,
    octaves,
    channelCount,
    distrib,
    cornersFlag,
    splineOrder,
  } = options;
  if (!maskConfig?.enabled) {
    return 'mask:disabled';
  }
  const freqX = Number.isFinite(freq?.[0]) ? Number(freq[0]) : 0;
  const freqY = Number.isFinite(freq?.[1]) ? Number(freq[1]) : 0;
  const widthInt = Math.max(0, Math.floor(Number(width) || 0));
  const heightInt = Math.max(0, Math.floor(Number(height) || 0));
  const octaveCount = Math.max(0, Math.floor(Number(octaves) || 0));
  const key = {
    value: maskConfig.value >>> 0,
    inverse: maskConfig.inverse ? 1 : 0,
    static: maskConfig.static ? 1 : 0,
    freqX,
    freqY,
    width: widthInt,
    height: heightInt,
    octaves: octaveCount,
    channels: channelCount >>> 0,
    distrib: distrib >>> 0,
    corners: cornersFlag ? 1 : 0,
    spline: Number.isFinite(splineOrder) ? Math.trunc(splineOrder) : 0,
  };
  return JSON.stringify(key);
}

function computeMultiresMaskData(options) {
  const {
    descriptor,
    maskConfig,
    freq,
    width,
    height,
    octaves,
    channelCount,
    distrib,
    cornersFlag,
    splineOrder,
    time = 0,
    speed = 1,
    frameIndex = 0,
  } = options;
  const holder = ensureMaskDataHolder(descriptor);
  const cacheKey = descriptor ? buildMaskCacheKey(options) : 'mask:none';
  const dynamicMask = Boolean(maskConfig?.enabled && !maskConfig.static);
  const effectiveTime = dynamicMask ? Number(time) || 0 : 0;
  const effectiveSpeed = Number.isFinite(Number(speed)) ? Number(speed) : 1;
  const frameId = Number.isFinite(Number(frameIndex))
    ? Math.max(0, Math.trunc(Number(frameIndex)))
    : 0;

  if (!maskConfig?.enabled) {
    holder.array = holder.array && holder.array.length ? holder.array : new Float32Array([1, 1, 1, 1]);
    holder.width = 1;
    holder.height = 1;
    holder.octaveCount = 0;
    holder.alpha = false;
    holder.enabled = false;
    if (descriptor) {
      descriptor._maskCacheKey = cacheKey;
      descriptor._maskCacheInfo = { enabled: false, alpha: false, octaveCount: 0 };
      descriptor._maskDynamicState = null;
    }
    descriptor._maskDataDirty = true;
    return { enabled: false, alpha: false, octaveCount: 0 };
  }

  if (descriptor && holder) {
    if (dynamicMask) {
      const last = descriptor._maskDynamicState;
      if (
        last &&
        last.frameIndex === frameId &&
        last.time === effectiveTime &&
        last.speed === effectiveSpeed &&
        descriptor._maskCacheInfo
      ) {
        const info = descriptor._maskCacheInfo;
        holder.enabled = info.enabled;
        holder.alpha = info.alpha;
        holder.octaveCount = info.octaveCount;
        return { enabled: info.enabled, alpha: info.alpha, octaveCount: info.octaveCount };
      }
    } else if (
      descriptor._maskCacheKey === cacheKey &&
      descriptor._maskCacheInfo
    ) {
      const info = descriptor._maskCacheInfo;
      holder.enabled = info.enabled;
      holder.alpha = info.alpha;
      holder.octaveCount = info.octaveCount;
      return { enabled: info.enabled, alpha: info.alpha, octaveCount: info.octaveCount };
    }
  }

  const baseFreq = [
    Math.max(1, Math.floor(freq[0] || 1)),
    Math.max(1, Math.floor(freq[1] || freq[0] || 1)),
  ];
  const pixelCount = Math.max(1, Math.floor(width) * Math.max(1, Math.floor(height)));
  const targetChannels = channelCount === 2 || channelCount === 4 ? 1 : Math.min(channelCount, 3);
  const alphaMode = channelCount === 2 || channelCount === 4;
  const arrays = [];
  let octaveCount = 0;
  for (let octave = 1; octave <= octaves; octave += 1) {
    const [fx, fy] = computeMultiresOctaveFrequency(baseFreq, octave);
    if (fx > width && fy > height) {
      break;
    }
    const glyphShape = [fy, fx, targetChannels];
    const maskResult = maskValues(maskConfig.value, glyphShape, {
      inverse: maskConfig.inverse,
      time: effectiveTime,
      speed: effectiveSpeed,
    });
    let maskTensor = maskResult && Array.isArray(maskResult) ? maskResult[0] : null;
    if (!maskTensor) {
      continue;
    }
    const targetHeight = Math.max(1, Math.floor(height));
    const targetWidth = Math.max(1, Math.floor(width));
    const needsResample =
      !isNativeSize(distrib) ||
      maskTensor.shape[0] !== targetHeight ||
      maskTensor.shape[1] !== targetWidth;
    if (needsResample) {
      maskTensor = resample(maskTensor, [targetHeight, targetWidth, targetChannels], splineOrder);
    }
    const data = maskTensor.read();
    const adjusted = pinCornersArray(
      data,
      targetWidth || maskTensor.shape[1],
      targetHeight || maskTensor.shape[0],
      targetChannels,
      fy,
      fx,
      cornersFlag,
    );
    arrays.push(adjusted);
    octaveCount += 1;
  }

  if (!octaveCount) {
    holder.array = new Float32Array([1, 1, 1, 1]);
    holder.width = 1;
    holder.height = 1;
    holder.octaveCount = 0;
    holder.alpha = alphaMode;
    holder.enabled = false;
    descriptor._maskDataDirty = true;
    if (descriptor) {
      descriptor._maskDynamicState = dynamicMask
        ? { frameIndex: frameId, time: effectiveTime, speed: effectiveSpeed }
        : null;
      descriptor._maskCacheKey = dynamicMask ? null : cacheKey;
      descriptor._maskCacheInfo = { enabled: false, alpha: alphaMode, octaveCount: 0 };
    }
    return { enabled: false, alpha: alphaMode, octaveCount: 0 };
  }

  const stride = 4;
  const out = new Float32Array(Math.max(1, pixelCount) * stride * octaveCount);
  const widthInt = Math.max(1, Math.floor(width));
  const heightInt = Math.max(1, Math.floor(height));
  const pixels = widthInt * heightInt;
  for (let octave = 0; octave < octaveCount; octave += 1) {
    const src = arrays[octave];
    const base = octave * pixels * stride;
    for (let i = 0; i < pixels; i += 1) {
      const dst = base + i * stride;
      const srcBase = i * targetChannels;
      const v0 = src[srcBase] ?? 1;
      const v1 = targetChannels > 1 ? src[srcBase + 1] : v0;
      const v2 = targetChannels > 2 ? src[srcBase + 2] : v0;
      out[dst] = alphaMode ? v0 : v0;
      out[dst + 1] = alphaMode ? v0 : v1;
      out[dst + 2] = alphaMode ? v0 : v2;
      out[dst + 3] = alphaMode ? v0 : 1;
    }
  }

  holder.array = out;
  holder.width = widthInt;
  holder.height = heightInt;
  holder.octaveCount = octaveCount;
  holder.alpha = alphaMode;
  holder.enabled = true;
  descriptor._maskDataDirty = true;
  if (descriptor) {
    descriptor._maskCacheKey = dynamicMask ? null : cacheKey;
    descriptor._maskCacheInfo = { enabled: true, alpha: alphaMode, octaveCount };
    descriptor._maskDynamicState = dynamicMask
      ? { frameIndex: frameId, time: effectiveTime, speed: effectiveSpeed }
      : null;
  }
  return { enabled: true, alpha: alphaMode, octaveCount };
}

function computePermutationSeeds(plan, width, height, frameSeed) {
  if (!plan) {
    return [];
  }
  const baseFreqRaw = Array.isArray(plan.baseFreq) ? plan.baseFreq : [plan.baseFreq, plan.baseFreq];
  const baseFreq = [
    Math.max(1, Math.floor(Number(baseFreqRaw[0]) || 1)),
    Math.max(1, Math.floor(Number(baseFreqRaw[1] ?? baseFreqRaw[0]) || 1)),
  ];
  const octaves = Math.max(1, Math.floor(Number(plan.octaves) || 1));
  const channelCount = Math.max(1, Math.min(4, Math.floor(Number(plan.channelCount) || 1)));
  const hasHue = Boolean(plan.hasHueOverride);
  const hasSaturation = Boolean(plan.hasSaturationOverride);
  const hasBrightness = Boolean(plan.hasBrightnessOverride);
  const hasLattice = Boolean(plan.hasLatticeDrift);
  const seedOffset = Math.trunc(Number(plan.seedOffset) || 0) >>> 0;
  const callsPerOctave =
    1 + (hasHue ? 1 : 0) + (hasSaturation ? 1 : 0) + (hasBrightness ? 1 : 0) + (hasLattice ? 2 : 0);
  const frameSeedU32 = frameSeed >>> 0;

  const seeds = [];
  const seen = new Set();

  for (let octave = 1; octave <= octaves; octave += 1) {
    const [freqX, freqY] = computeMultiresOctaveFrequency(baseFreq, octave);
    if (freqX > width && freqY > height) {
      break;
    }
    const octaveOffset = (octave - 1) * callsPerOctave;
    let baseSeed = addUint32(frameSeedU32, seedOffset);
    baseSeed = addUint32(baseSeed, octaveOffset >>> 0);
    if (frameSeedU32 !== 0) {
      baseSeed = addUint32(baseSeed, 1);
    }
    const layerSeed = baseSeed;
    for (let channel = 0; channel < channelCount; channel += 1) {
      const channelSeed = addUint32(layerSeed, (channel * CHANNEL_SEED_DELTA) >>> 0);
      if (!seen.has(channelSeed)) {
        seen.add(channelSeed);
        seeds.push(channelSeed);
      }
    }
    let seedCursor = addUint32(baseSeed, 1);
    if (hasHue) {
      if (!seen.has(seedCursor)) {
        seen.add(seedCursor);
        seeds.push(seedCursor);
      }
      seedCursor = addUint32(seedCursor, 1);
    }
    if (hasSaturation) {
      if (!seen.has(seedCursor)) {
        seen.add(seedCursor);
        seeds.push(seedCursor);
      }
      seedCursor = addUint32(seedCursor, 1);
    }
    if (hasBrightness) {
      if (!seen.has(seedCursor)) {
        seen.add(seedCursor);
        seeds.push(seedCursor);
      }
      seedCursor = addUint32(seedCursor, 1);
    }
    if (hasLattice) {
      if (!seen.has(seedCursor)) {
        seen.add(seedCursor);
        seeds.push(seedCursor);
      }
      seedCursor = addUint32(seedCursor, 1);
      if (!seen.has(seedCursor)) {
        seen.add(seedCursor);
        seeds.push(seedCursor);
      }
      seedCursor = addUint32(seedCursor, 1);
    }
  }

  return seeds;
}

function addUint32(a, b) {
  return (Math.trunc(a) + Math.trunc(b)) >>> 0;
}

function getPermutationTables(seed) {
  const key = seed >>> 0;
  if (PERMUTATION_TABLE_CACHE.has(key)) {
    return PERMUTATION_TABLE_CACHE.get(key);
  }
  const simplex = new OpenSimplex(key);
  const perm = new Uint32Array(PERMUTATION_TABLE_SIZE);
  const grad = new Uint32Array(PERMUTATION_TABLE_SIZE);
  for (let i = 0; i < PERMUTATION_TABLE_SIZE; i += 1) {
    perm[i] = simplex.perm[i] >>> 0;
    grad[i] = simplex.permGradIndex3D[i] >>> 0;
  }
  const record = { perm, grad };
  PERMUTATION_TABLE_CACHE.set(key, record);
  return record;
}

function coerceFrequencyInput(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? [value] : null;
  }
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    const arr = Array.from(value);
    if (!arr.every((entry) => Number.isFinite(Number(entry)))) {
      return null;
    }
    return arr.map((entry) => Number(entry));
  }
  return null;
}

function isBrightnessFrequencyParamSupported(value) {
  if (value === undefined || value === null) {
    return true;
  }
  return coerceFrequencyInput(value) !== null;
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

export function buildTopologySignatureFromPreset(preset) {
  const stages = collectPresetStages(preset);
  return stages.map((stage) => `${stage.category}:${stage.name}`).join('|');
}

