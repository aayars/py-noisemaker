import * as SHADERS from './shaders.js';

const DEFAULT_WORKGROUP_SIZE = [8, 8, 1];
const COMPUTE_STAGE_VISIBILITY =
  typeof GPUShaderStage !== 'undefined' ? GPUShaderStage.COMPUTE : 0;
const STORAGE_TEXTURE_FORMAT = 'rgba32float';
const PRESENT_PARAMS_FLOATS = 16;
const PRESENT_PARAMS_SIZE = PRESENT_PARAMS_FLOATS * 4;

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
            pipeline = await context.createComputePipeline(descriptor.shaderSource);
            descriptor.pipeline = pipeline;
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
          const layout =
            descriptor.bindGroupLayout || (pipeline.getBindGroupLayout ? pipeline.getBindGroupLayout(0) : null);
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

        pingPong.swap();
        state.finalTexture = pingPong.readTex;
        state.finalView = pingPong.readFbo;

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

      if (
        autoSubmitted &&
        !result.queueFlushed &&
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
        descriptor.pipeline = null;
        descriptor.bindGroupLayout = null;
        descriptor.pipelineLayout = null;
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

function buildBindingSignature(descriptor) {
  const parts = [];
  const bindings = descriptor.bindings || {};
  if (bindings.hasUniform) parts.push('uniform');
  if (bindings.hasFrameUniform) parts.push('frame');
  if (bindings.readsTexture) parts.push('read-storage');
  if (bindings.writesTexture) parts.push('write-storage');
  if (Array.isArray(bindings.auxiliary)) {
    for (const aux of bindings.auxiliary) {
      const type = aux?.resourceType || 'unknown';
      const opt = aux?.optional ? 'opt' : 'req';
      parts.push(`aux:${type}:${opt}`);
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
  const type = aux.resourceType || 'texture';
  switch (type) {
    case 'sampler':
      return { binding, visibility, sampler: {} };
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
    bindingSignature: null,
    bindGroupLayout: null,
    pipelineLayout: null,
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

export function buildTopologySignatureFromPreset(preset) {
  const stages = collectPresetStages(preset);
  return stages.map((stage) => `${stage.category}:${stage.name}`).join('|');
}

