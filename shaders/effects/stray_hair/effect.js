import meta from './meta.json' with { type: 'json' };
import { OpenSimplex } from '../../../../js/noisemaker/simplex.js';

const RGBA_CHANNEL_COUNT = 4;
const PARAM_FLOAT_LENGTH = Number(meta.resources?.params?.size ?? 32) / Float32Array.BYTES_PER_ELEMENT;
const FLOATS_PER_AGENT = 9;
const PARAM_FLOAT_COUNT = 20;
const SPEED_MIN = Number(meta.parameters?.find((p) => p.name === 'speed')?.min ?? 0.0);
const SPEED_MAX = Number(meta.parameters?.find((p) => p.name === 'speed')?.max ?? 3.0);

// Stray hair worm parameters (very sparse, long strands)
const DEFAULT_DENSITY = 0.003;  // Very low: 0.0025-0.00375
const DEFAULT_STRIDE = 0.5;  // Larger strides for longer hairs
const DEFAULT_STRIDE_DEVIATION = 0.25;
const DEFAULT_INTENSITY = 1.0;  // No fading - hairs stay solid
const DEFAULT_LIFETIME = 0.0;  // Static - one burst and done
const DURATION_MIN = 8;
const DURATION_MAX = 16;
const FLOW_FREQ_MIN = 2;
const FLOW_FREQ_MAX = 4;
const BRIGHTNESS_FREQ = 32;

const PARAM_INDEX = Object.freeze({
    width: 0,
    height: 1,
    channelCount: 2,
    behavior: 4,
    density: 5,
    stride: 6,
    strideDeviation: 8,
    alpha: 9,
    kink: 10,
    quantize: 12,
    time: 13,
    intensity: 15,
    inputIntensity: 16,
    lifetime: 17,
});

const hasOwn = (object, property) => Object.prototype.hasOwnProperty.call(object, property);

const PARAMETER_BINDINGS = meta.parameterBindings ?? {};

function getBindingOffset(name) {
    const binding = PARAMETER_BINDINGS[name];
    if (!binding || binding.buffer !== 'params') {
        return undefined;
    }
    const offset = Number(binding.offset);
    if (!Number.isFinite(offset)) {
        throw new Error(`StrayHair parameter '${name}' binding offset must be a finite number.`);
    }
    return offset;
}

function clamp(value, minValue, maxValue) {
    let result = value;
    if (Number.isFinite(minValue)) {
        result = Math.max(result, minValue);
    }
    if (Number.isFinite(maxValue)) {
        result = Math.min(result, maxValue);
    }
    return result;
}

function mulberry32(seed) {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let x = t;
        x = Math.imul(x ^ (x >>> 15), x | 1);
        x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
}

function randomIntInclusive(rng, minValue, maxValue) {
    let start = minValue;
    let end = maxValue;
    if (end < start) {
        [start, end] = [end, start];
    }
    const span = Math.max(end - start + 1, 1);
    return Math.floor(rng() * span) + start;
}

function computeAgentCount(width, height, density) {
    const maxDim = Math.max(width, height);
    if (maxDim <= 0 || density <= 0) {
        return 1;
    }
    return Math.max(1, Math.floor(maxDim * density));
}

function normalizedStride(stride, width, height) {
    const scale = Math.max(width, height) / 1024;
    const base = stride * (scale <= 0 ? 1 : scale);
    return Math.max(0.1, base);
}

function seedAgents(width, height, agentCount, strideConfig, rng) {
    const data = new Float32Array(agentCount * FLOATS_PER_AGENT);
    const strideValue = normalizedStride(strideConfig.stride, width, height);
    const strideDeviation = strideConfig.deviation;

    for (let i = 0; i < agentCount; i += 1) {
        const offset = i * FLOATS_PER_AGENT;
        data[offset + 0] = rng() * width; // x
        data[offset + 1] = rng() * height; // y
        data[offset + 2] = rng() * Math.PI * 2; // rotation
        const strideVariation = strideValue * (1 + (rng() - 0.5) * 2 * strideDeviation);
        data[offset + 3] = Math.max(0.1, strideVariation);

        // Agent luminance - bright white strands
        const luminance = 0.8 + rng() * 0.2;  // 0.8-1.0 for visible white strands
        data[offset + 4] = luminance;
        data[offset + 5] = luminance;
        data[offset + 6] = luminance;
        data[offset + 7] = rng() * 1_000_000; // seed
        data[offset + 8] = 0.0; // age=0 means immediately active (not -1)
    }

    return data;
}

function toFiniteNumber(value) {
    if (value === undefined || value === null || value === '') {
        return null;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

class StrayHairEffect {
    static id = meta.id;
    static label = meta.label;
    static metadata = meta;

    constructor({ helpers } = {}) {
        this.helpers = helpers ?? {};
        this.device = null;
        this.width = 0;
        this.height = 0;
        this.resources = null;

        this.userState = {
            speed: Number(meta.parameters?.find((p) => p.name === 'speed')?.default ?? 1.0),
            enabled: Boolean(meta.parameters?.find((p) => p.name === 'enabled')?.default ?? true),
        };

        this.agentBuffers = null;
        this.noiseTexture = null;
        this.noiseTextureView = null;
        this.noiseData = null;
        this.noiseFloatsPerRow = 0;
        this.noiseSeed = null;
    this.brightnessTexture = null;
    this.brightnessTextureView = null;
    this.brightnessData = null;
    this.brightnessFloatsPerRow = 0;
    this.brightnessSeed = null;
        this.wormParamsState = null;
        this.wormParamsBuffer = null;
        this.wormParamsDirty = false;
        this.wormPipelines = null;
        this.wormBindGroupLayout = null;
        this.wormPipelineLayout = null;
        this.wormComputeBindGroup = null;
        this.wormComputeSwapBindGroup = null;
        this.wormComputePasses = null;
        this.wormDispatchConfig = null;

        this.#timeSeconds = 0;
        this.#lastTimestamp = null;
        this.hairGenerated = false;  // Track if we've already generated static hair
    }

    async ensureResources({ device, width, height, multiresResources }) {
        if (!device) {
            throw new Error('StrayHairEffect requires a GPUDevice.');
        }
        if (!multiresResources?.outputTexture) {
            throw new Error('StrayHairEffect requires multires output texture.');
        }

        if (!this.userState.enabled) {
            this.#destroyTransientResources();
            this.resources = this.#createDisabledResources(width, height);
            return this.resources;
        }

        if (this.device && this.device !== device) {
            this.invalidateResources();
            this.#destroyTransientResources();
        }

        this.device = device;
        this.width = width;
        this.height = height;

        if (this.resources) {
            const sizeMatches = this.resources.textureWidth === width && this.resources.textureHeight === height;
            const sameFeedback = this.resources.boundWormTexture === this.wormFeedbackTexture;
            const sameBrightness = this.resources.boundBrightnessTexture === this.brightnessTexture;
            if (sizeMatches && sameFeedback && sameBrightness) {
                this.resources.computePasses = this.#composeComputePasses();
                this.resources.enabled = true;
                return this.resources;
            }
            this.invalidateResources();
        }

        await this.#createResources({ device, width, height, multiresResources });
        return this.resources;
    }

    async updateParams(updates = {}) {
        if (!updates || typeof updates !== 'object') {
            throw new TypeError('StrayHairEffect.updateParams expects an object.');
        }

        const changed = [];

        if (hasOwn(updates, 'enabled')) {
            const enabled = Boolean(updates.enabled);
            if (enabled !== this.userState.enabled) {
                this.userState.enabled = enabled;
                changed.push('enabled');
                this.invalidateResources();
            }
        }

        if (hasOwn(updates, 'speed')) {
            const numeric = toFiniteNumber(updates.speed);
            if (numeric !== null) {
                const clamped = clamp(numeric, SPEED_MIN, SPEED_MAX);
                if (clamped !== this.userState.speed) {
                    this.userState.speed = clamped;
                    if (this.resources?.paramsState) {
                        const offset = this.resources.bindingOffsets.speed;
                        if (typeof offset === 'number') {
                            this.resources.paramsState[offset] = clamped;
                            this.resources.paramsDirty = true;
                        }
                    }
                    changed.push('speed');
                }
            }
        }

        return { updated: changed };
    }

    getUIState() {
        return {
            enabled: this.userState.enabled,
            speed: this.userState.speed,
        };
    }

    beforeDispatch({ device, encoder }) {
        const resources = this.resources;
        if (!resources?.enabled) {
            return;
        }

        const bindingOffsets = resources.bindingOffsets ?? {};
        const currentSeed = typeof bindingOffsets.seed === 'number'
            ? resources.paramsState[bindingOffsets.seed]
            : 0;

        const seedChanged = this.noiseSeed === null || this.noiseSeed !== currentSeed;

        if (seedChanged || !this.wormParamsState) {
            const updatedParams = this.#createInitialWormParams(this.width, this.height, currentSeed);
            if (this.wormParamsState) {
                this.wormParamsState.set(updatedParams);
            } else {
                this.wormParamsState = updatedParams;
            }
            this.wormParamsDirty = true;

            const targetDensity = this.wormConfig?.density ?? DEFAULT_DENSITY;
            const targetAgentCount = computeAgentCount(this.width, this.height, targetDensity);
            const reuseBuffers = Boolean(this.agentBuffers) && this.agentBuffers.count === targetAgentCount;
            this.#seedAgents(device, this.width, this.height, currentSeed, reuseBuffers, targetAgentCount);

            this.#refreshNoiseTexture(device, this.width, this.height, currentSeed);
            this.#refreshBrightnessTexture(device, this.width, this.height, currentSeed);
            if (this.resources) {
                this.resources.boundBrightnessTexture = this.brightnessTexture;
            }
            if (typeof bindingOffsets.seed === 'number') {
                resources.paramsState[bindingOffsets.seed] = currentSeed;
                resources.paramsDirty = true;
            }
            this.noiseSeed = currentSeed;
            this.brightnessSeed = currentSeed;
            this.hairGenerated = false;
            this.wormDispatchConfig = null;
        }

        if (this.wormParamsState) {
            this.wormParamsState[PARAM_INDEX.time] = 0;
            this.wormParamsDirty = true;
        }

        if (typeof bindingOffsets.time === 'number') {
            resources.paramsState[bindingOffsets.time] = 0;
            resources.paramsDirty = true;
        }

        if (this.wormParamsDirty && this.wormParamsBuffer) {
            device.queue.writeBuffer(this.wormParamsBuffer, 0, this.wormParamsState);
            this.wormParamsDirty = false;
        }

        if (seedChanged || !this.wormComputeBindGroup) {
            this.#rebindWormPasses(device);
        }

        if (resources.paramsDirty && resources.paramsBuffer) {
            device.queue.writeBuffer(resources.paramsBuffer, 0, resources.paramsState);
            resources.paramsDirty = false;
        }

        if (this.hairGenerated) {
            return;
        }

        if (!encoder) {
            throw new Error('StrayHairEffect requires a GPUCommandEncoder to bake its static worm trails.');
        }

    this.#runStaticStrayHairBurst(encoder);
        this.hairGenerated = true;
        resources.computePasses = this.#composeComputePasses();
    }

    afterDispatch() {
        const resources = this.resources;
        if (!resources?.enabled) {
            return;
        }

        if (!this.agentBuffers) {
            return;
        }
        this.agentBuffers.current = 'a';
    }

    invalidateResources() {
        if (!this.resources) {
            return;
        }

        try {
            this.resources.resourceSet?.destroyAll?.();
        } catch (error) {
            this.helpers.logWarn?.('Failed to destroy stray hair resource set:', error);
        }

        try {
            this.resources.outputTexture?.destroy?.();
        } catch (error) {
            this.helpers.logWarn?.('Failed to destroy stray hair output texture:', error);
        }

        this.resources = null;
        this.hairGenerated = false;
        this.wormDispatchConfig = null;
    }

    destroy() {
        this.invalidateResources();
        this.#destroyTransientResources();
        this.device = null;
    }

    async #createResources({ device, width, height, multiresResources }) {
        const preservedSeed = this.noiseSeed ?? 0;
        this.#destroyTransientResources();
        this.noiseSeed = preservedSeed;

        const {
            getShaderDescriptor,
            getShaderMetadataCached,
            warnOnNonContiguousBindings,
            createShaderResourceSet,
            createBindGroupEntriesFromResources,
            getOrCreateBindGroupLayout,
            getOrCreatePipelineLayout,
            getOrCreateComputePipeline,
            getBufferToTexturePipeline,
        } = this.helpers;

        this.#refreshNoiseTexture(device, width, height, this.noiseSeed ?? 0);
    this.#refreshBrightnessTexture(device, width, height, this.noiseSeed ?? 0);

        const seed = this.noiseSeed ?? 0;
        const initialParams = this.#createInitialWormParams(width, height, seed);
        this.wormParamsState = initialParams;

        const initialAgentCount = computeAgentCount(width, height, this.wormConfig?.density ?? 0.375);
        this.#seedAgents(device, width, height, seed, false, initialAgentCount);

        const wormDescriptor = getShaderDescriptor('worms');
        const wormMetadata = await getShaderMetadataCached('worms');
        warnOnNonContiguousBindings?.(wormMetadata.bindings, wormDescriptor.id);

        this.wormBindGroupLayout = getOrCreateBindGroupLayout(device, wormDescriptor.id, 'compute', wormMetadata);
        this.wormPipelineLayout = getOrCreatePipelineLayout(device, wormDescriptor.id, 'compute', this.wormBindGroupLayout);

        const initDescriptor = getShaderDescriptor('worms/init_from_prev');
        const moveDescriptor = getShaderDescriptor('worms/agent_move');
        const finalDescriptor = getShaderDescriptor('worms/final_blend');
        const bufferToTextureDescriptor = getShaderDescriptor('worms/buffer_to_texture');

        const initPipeline = await getOrCreateComputePipeline(device, initDescriptor.id, this.wormPipelineLayout, initDescriptor.entryPoint ?? 'main');
        const movePipeline = await getOrCreateComputePipeline(device, moveDescriptor.id, this.wormPipelineLayout, moveDescriptor.entryPoint ?? 'main');
        const finalPipeline = await getOrCreateComputePipeline(device, finalDescriptor.id, this.wormPipelineLayout, finalDescriptor.entryPoint ?? 'main');

        const bufferToTextureMetadata = await getShaderMetadataCached(bufferToTextureDescriptor.id);
        const wormsBufferToTextureLayout = getOrCreateBindGroupLayout(device, bufferToTextureDescriptor.id, 'compute', bufferToTextureMetadata);
        const bufferToTexturePipelineLayout = getOrCreatePipelineLayout(device, bufferToTextureDescriptor.id, 'compute', wormsBufferToTextureLayout);
        const trailsPipeline = await getOrCreateComputePipeline(device, bufferToTextureDescriptor.id, bufferToTexturePipelineLayout, bufferToTextureDescriptor.entryPoint ?? 'main');

        this.wormPipelines = {
            init: initPipeline,
            move: movePipeline,
            final: finalPipeline,
            trails: trailsPipeline,
            pixelWorkgroupSize: [8, 8, 1],
            agentWorkgroupSize: [64, 1, 1],
            bufferWorkgroupSize: [8, 8, 1],
        };

        const pixelCount = Math.max(width * height, 1);
        const outputBufferSize = pixelCount * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
        this.wormOutputBuffer = device.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.wormFeedbackTexture = device.createTexture({
            size: { width, height, depthOrArrayLayers: 1 },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        });
        this.wormFeedbackView = this.wormFeedbackTexture.createView();

        this.wormParamsBuffer = device.createBuffer({
            size: this.wormParamsState.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.wormParamsBuffer, 0, this.wormParamsState);

        this.wormComputeBindGroup = device.createBindGroup({
            layout: this.wormBindGroupLayout,
            entries: [
                { binding: 0, resource: this.noiseTextureView },
                { binding: 1, resource: { buffer: this.wormOutputBuffer } },
                { binding: 2, resource: { buffer: this.wormParamsBuffer } },
                { binding: 3, resource: this.wormFeedbackView },
                { binding: 4, resource: { buffer: this.agentBuffers.a } },
                { binding: 5, resource: { buffer: this.agentBuffers.b } },
            ],
        });

        this.wormComputeSwapBindGroup = device.createBindGroup({
            layout: this.wormBindGroupLayout,
            entries: [
                { binding: 0, resource: this.noiseTextureView },
                { binding: 1, resource: { buffer: this.wormOutputBuffer } },
                { binding: 2, resource: { buffer: this.wormParamsBuffer } },
                { binding: 3, resource: this.wormFeedbackView },
                { binding: 4, resource: { buffer: this.agentBuffers.b } },
                { binding: 5, resource: { buffer: this.agentBuffers.a } },
            ],
        });

        this.trailBufferToTextureBindGroup = device.createBindGroup({
            layout: wormsBufferToTextureLayout,
            entries: [
                { binding: 0, resource: { buffer: this.wormOutputBuffer } },
                { binding: 1, resource: this.wormFeedbackView },
                { binding: 2, resource: { buffer: this.wormParamsBuffer } },
            ],
        });

        this.#updateWormComputePasses(width, height);

        const descriptor = getShaderDescriptor(meta.id);
        const shaderMetadata = await getShaderMetadataCached(meta.id);
        warnOnNonContiguousBindings?.(shaderMetadata.bindings, descriptor.id);

        const resourceSet = createShaderResourceSet(device, descriptor, shaderMetadata, width, height, {
            inputTextures: {
                input_texture: multiresResources.outputTexture,
                worm_texture: this.wormFeedbackTexture,
                brightness_texture: this.brightnessTexture,
            },
        });

        const bindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
        const pipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', bindGroupLayout);
        const combinePipeline = await getOrCreateComputePipeline(device, descriptor.id, pipelineLayout, descriptor.entryPoint ?? 'main');
        const combineBindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: createBindGroupEntriesFromResources(shaderMetadata.bindings, resourceSet),
        });

        const paramsBuffer = resourceSet.buffers.params;
        const outputBuffer = resourceSet.buffers.output_buffer;

        const paramsState = new Float32Array(PARAM_FLOAT_LENGTH);
        const bindingOffsets = {
            width: getBindingOffset('width'),
            height: getBindingOffset('height'),
            channel_count: getBindingOffset('channel_count'),
            channelCount: getBindingOffset('channel_count'),
            time: getBindingOffset('time'),
            speed: getBindingOffset('speed'),
            seed: getBindingOffset('seed'),
        };

        if (typeof bindingOffsets.width === 'number') {
            paramsState[bindingOffsets.width] = width;
        }
        if (typeof bindingOffsets.height === 'number') {
            paramsState[bindingOffsets.height] = height;
        }
        if (typeof bindingOffsets.channel_count === 'number') {
            paramsState[bindingOffsets.channel_count] = RGBA_CHANNEL_COUNT;
        }
        if (typeof bindingOffsets.time === 'number') {
            paramsState[bindingOffsets.time] = 0;
        }
        if (typeof bindingOffsets.speed === 'number') {
            paramsState[bindingOffsets.speed] = this.userState.speed;
        }
        if (typeof bindingOffsets.seed === 'number') {
            paramsState[bindingOffsets.seed] = this.noiseSeed ?? 0;
        }

        device.queue.writeBuffer(paramsBuffer, 0, paramsState);

        const outputTexture = device.createTexture({
            size: { width, height, depthOrArrayLayers: 1 },
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        const storageView = outputTexture.createView();
        const sampleView = outputTexture.createView();

        const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureLayout } = await getBufferToTexturePipeline(device);
        const bufferToTextureBindGroup = device.createBindGroup({
            layout: bufferToTextureLayout,
            entries: [
                { binding: 0, resource: { buffer: outputBuffer } },
                { binding: 1, resource: storageView },
                { binding: 2, resource: { buffer: paramsBuffer } },
            ],
        });

        const blitBindGroup = device.createBindGroup({
            layout: multiresResources.blitBindGroupLayout,
            entries: [{ binding: 0, resource: sampleView }],
        });

        this.finalPass = {
            pipeline: combinePipeline,
            bindGroup: combineBindGroup,
            workgroupSize: shaderMetadata.workgroupSize ?? [8, 8, 1],
            getDispatch: () => [
                Math.ceil(width / Math.max((shaderMetadata.workgroupSize?.[0] ?? 8), 1)),
                Math.ceil(height / Math.max((shaderMetadata.workgroupSize?.[1] ?? 8), 1)),
                1,
            ],
        };

        this.resources = {
            descriptor,
            shaderMetadata,
            resourceSet,
            computePipeline: null,
            computeBindGroup: null,
            paramsBuffer,
            paramsState,
            paramsDirty: false,
            outputBuffer,
            outputTexture,
            storageView,
            sampleView,
            bufferToTexturePipeline,
            bufferToTextureBindGroup,
            bufferToTextureWorkgroupSize: [8, 8, 1],
            blitBindGroup,
            workgroupSize: this.finalPass.workgroupSize,
            enabled: true,
            textureWidth: width,
            textureHeight: height,
            bindingOffsets,
            device,
            computePasses: this.#composeComputePasses(),
            boundWormTexture: this.wormFeedbackTexture,
            boundBrightnessTexture: this.brightnessTexture,
            shouldCopyOutputToPrev: false,
        };
    }

    #composeComputePasses() {
        if (!this.finalPass) {
            return [];
        }
        return [this.finalPass];
    }

    #updateWormComputePasses(width, height) {
        if (!this.wormPipelines || !this.agentBuffers) {
            this.wormDispatchConfig = null;
            if (this.resources) {
                this.resources.computePasses = this.#composeComputePasses();
            }
            return;
        }

        const {
            pixelWorkgroupSize,
            agentWorkgroupSize,
            bufferWorkgroupSize,
        } = this.wormPipelines;

        const pixelWorkgroup = pixelWorkgroupSize ?? [8, 8, 1];
        const agentWorkgroup = agentWorkgroupSize ?? [64, 1, 1];
        const bufferWorkgroup = bufferWorkgroupSize ?? [8, 8, 1];

        const safeWidth = Math.max(width, 1);
        const safeHeight = Math.max(height, 1);
        const agentCount = Math.max(this.agentBuffers.count, 1);

        const pixelDispatch = [
            Math.ceil(safeWidth / Math.max(pixelWorkgroup[0], 1)),
            Math.ceil(safeHeight / Math.max(pixelWorkgroup[1], 1)),
            1,
        ];
        const agentDispatch = [
            Math.max(1, Math.ceil(agentCount / Math.max(agentWorkgroup[0], 1))),
            1,
            1,
        ];
        const bufferDispatch = [
            Math.ceil(safeWidth / Math.max(bufferWorkgroup[0], 1)),
            Math.ceil(safeHeight / Math.max(bufferWorkgroup[1], 1)),
            1,
        ];

        this.wormDispatchConfig = {
            pixel: pixelDispatch,
            agent: agentDispatch,
            buffer: bufferDispatch,
        };

        if (this.resources) {
            this.resources.computePasses = this.#composeComputePasses();
        }
    }

    #computeFrequencies(baseFrequency, width, height) {
        const safeBase = Math.max(1, Math.round(baseFrequency));
        const safeWidth = Math.max(1, Math.round(width));
        const safeHeight = Math.max(1, Math.round(height));

        if (safeHeight === safeWidth) {
            return { x: safeBase, y: safeBase };
        }

        if (safeHeight < safeWidth) {
            const freqX = Math.max(1, Math.round(safeBase * safeWidth / safeHeight));
            return { x: freqX, y: safeBase };
        }

        const freqY = Math.max(1, Math.round(safeBase * safeHeight / safeWidth));
        return { x: safeBase, y: freqY };
    }

    #calculateWormSteps(width, height) {
        const safeWidth = Math.max(width, 1);
        const safeHeight = Math.max(height, 1);
        const duration = Math.max(this.wormConfig?.duration ?? DURATION_MIN, 1);
        const estimate = Math.floor(Math.sqrt(Math.min(safeWidth, safeHeight)) * duration);
        return Math.max(1, estimate);
    }

    #runStaticStrayHairBurst(encoder) {
        if (!this.wormPipelines || !this.agentBuffers || !this.trailBufferToTextureBindGroup) {
            return;
        }

        if (!this.wormDispatchConfig) {
            this.#updateWormComputePasses(this.width, this.height);
        }

        const dispatch = this.wormDispatchConfig;
        if (!dispatch) {
            return;
        }

        const {
            init: initPipeline,
            move: movePipeline,
            final: finalPipeline,
            trails: trailsPipeline,
        } = this.wormPipelines;

        if (!initPipeline || !movePipeline || !finalPipeline || !trailsPipeline) {
            return;
        }

        const wormSteps = this.#calculateWormSteps(this.width, this.height);
        if (wormSteps <= 0) {
            return;
        }

        const computePass = encoder.beginComputePass();

        computePass.setPipeline(initPipeline);
        computePass.setBindGroup(0, this.wormComputeBindGroup);
        computePass.dispatchWorkgroups(dispatch.pixel[0], dispatch.pixel[1], dispatch.pixel[2]);

        let useSwap = false;
        for (let step = 0; step < wormSteps; step += 1) {
            const bindGroup = useSwap && this.wormComputeSwapBindGroup
                ? this.wormComputeSwapBindGroup
                : this.wormComputeBindGroup;

            computePass.setPipeline(movePipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(dispatch.agent[0], dispatch.agent[1], dispatch.agent[2]);

            if (this.wormComputeSwapBindGroup) {
                useSwap = !useSwap;
            }
        }

        const finalBindGroup = useSwap && this.wormComputeSwapBindGroup
            ? this.wormComputeSwapBindGroup
            : this.wormComputeBindGroup;

        computePass.setPipeline(finalPipeline);
        computePass.setBindGroup(0, finalBindGroup);
        computePass.dispatchWorkgroups(dispatch.pixel[0], dispatch.pixel[1], dispatch.pixel[2]);

        computePass.setPipeline(trailsPipeline);
        computePass.setBindGroup(0, this.trailBufferToTextureBindGroup);
        computePass.dispatchWorkgroups(dispatch.buffer[0], dispatch.buffer[1], dispatch.buffer[2]);

        computePass.end();
    }

    #createInitialWormParams(width, height, seed) {
        const params = new Float32Array(PARAM_FLOAT_COUNT);
        params[PARAM_INDEX.width] = width;
        params[PARAM_INDEX.height] = height;
        params[PARAM_INDEX.channelCount] = RGBA_CHANNEL_COUNT;
        params[3] = 0;

    const rng = mulberry32(Math.floor(seed) ^ 0x9e3779b9);
        
    // Stray hair: sparse, long strands with high kink
    const density = 0.0025 + rng() * 0.00125;  // Match Python's sparse range
    const duration = randomIntInclusive(rng, DURATION_MIN, DURATION_MAX);
    const stride = DEFAULT_STRIDE;
    const behavior = 3;  // Unruly
    const kink = randomIntInclusive(rng, 5, 50);  // High kink: 5-50

        params[PARAM_INDEX.behavior] = behavior;
        params[PARAM_INDEX.density] = density;
        params[PARAM_INDEX.stride] = stride;
        params[PARAM_INDEX.strideDeviation] = DEFAULT_STRIDE_DEVIATION;
        params[PARAM_INDEX.alpha] = 1.0;
        params[PARAM_INDEX.kink] = kink;
        params[PARAM_INDEX.quantize] = 0.0;
        params[PARAM_INDEX.time] = 0.0;
        params[14] = 0;
        params[PARAM_INDEX.intensity] = 1.0;  // No fading - keep all trails at full brightness
        params[PARAM_INDEX.inputIntensity] = 0.0;
        params[PARAM_INDEX.lifetime] = DEFAULT_LIFETIME;
        params[18] = 0;
        params[19] = 0;

        this.wormConfig = {
            density,
            stride,
            deviation: DEFAULT_STRIDE_DEVIATION,
            duration,
        };

        return params;
    }

    #advanceTime() {
        const now = typeof performance !== 'undefined' && typeof performance.now === 'function'
            ? performance.now()
            : Date.now();

        if (this.#lastTimestamp === null) {
            this.#lastTimestamp = now;
            return this.#timeSeconds;
        }

        const deltaSeconds = Math.max((now - this.#lastTimestamp) / 1000, 0);
        this.#lastTimestamp = now;
        this.#timeSeconds = (this.#timeSeconds + deltaSeconds) % 1_000_000;
        return this.#timeSeconds;
    }

    #refreshNoiseTexture(device, width, height, seed) {
        if (width <= 0 || height <= 0) {
            return;
        }

        if (!this.noiseTexture || this.width !== width || this.height !== height) {
            this.noiseTexture?.destroy?.();
            this.noiseTexture = device.createTexture({
                size: { width, height, depthOrArrayLayers: 1 },
                format: 'rgba32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.noiseTextureView = this.noiseTexture.createView();
        }

        const floatsPerPixel = RGBA_CHANNEL_COUNT;
        const rowDataLength = width * floatsPerPixel;
        const paddedFloatsPerRow = Math.max(64, Math.ceil(rowDataLength / 64) * 64);
        const totalFloats = paddedFloatsPerRow * height;

        if (!this.noiseData || this.noiseData.length !== totalFloats) {
            this.noiseData = new Float32Array(totalFloats);
            this.noiseFloatsPerRow = paddedFloatsPerRow;
        }

        const rng = mulberry32(Math.floor(seed) ^ 0x517cc1b7 ^ (width << 4) ^ (height << 7));
        const baseFrequency = Math.max(FLOW_FREQ_MIN, Math.min(FLOW_FREQ_MAX, randomIntInclusive(rng, FLOW_FREQ_MIN, FLOW_FREQ_MAX)));
        const { x: freqX, y: freqY } = this.#computeFrequencies(baseFrequency, width, height);
        const safeWidth = Math.max(width, 1);
        const safeHeight = Math.max(height, 1);
        const scaleX = freqX / safeWidth;
        const scaleY = freqY / safeHeight;
        const offsetX = rng();
        const offsetY = rng();
        const simplexSeed = (Math.floor(seed) ^ 0x4c3a2f19) >>> 0;
        const simplex = new OpenSimplex(simplexSeed);

        for (let y = 0; y < height; y += 1) {
            const rowOffset = y * paddedFloatsPerRow;
            const sampleY = (y * scaleY) + offsetY;
            for (let x = 0; x < width; x += 1) {
                const sampleX = (x * scaleX) + offsetX;
                const noiseValue = simplex.noise2D(sampleX, sampleY);
                const normalized = Math.min(1, Math.max(0, (noiseValue + 1) * 0.5));
                const base = rowOffset + x * floatsPerPixel;
                this.noiseData[base + 0] = normalized;
                this.noiseData[base + 1] = normalized;
                this.noiseData[base + 2] = normalized;
                this.noiseData[base + 3] = 1.0;
            }
            for (let p = rowDataLength; p < paddedFloatsPerRow; p += 1) {
                this.noiseData[rowOffset + p] = 0.0;
            }
        }

        device.queue.writeTexture(
            { texture: this.noiseTexture },
            this.noiseData,
            {
                bytesPerRow: paddedFloatsPerRow * Float32Array.BYTES_PER_ELEMENT,
                rowsPerImage: height,
            },
            { width, height, depthOrArrayLayers: 1 },
        );

        this.noiseSeed = seed;
    }

    #refreshBrightnessTexture(device, width, height, seed) {
        if (width <= 0 || height <= 0) {
            return;
        }

        if (!this.brightnessTexture || this.width !== width || this.height !== height) {
            this.brightnessTexture?.destroy?.();
            this.brightnessTexture = device.createTexture({
                size: { width, height, depthOrArrayLayers: 1 },
                format: 'rgba32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.brightnessTextureView = this.brightnessTexture.createView();
        }

        const floatsPerPixel = RGBA_CHANNEL_COUNT;
        const rowDataLength = width * floatsPerPixel;
        const paddedFloatsPerRow = Math.max(64, Math.ceil(rowDataLength / 64) * 64);
        const totalFloats = paddedFloatsPerRow * height;

        if (!this.brightnessData || this.brightnessData.length !== totalFloats) {
            this.brightnessData = new Float32Array(totalFloats);
            this.brightnessFloatsPerRow = paddedFloatsPerRow;
        }

        const safeWidth = Math.max(width, 1);
        const safeHeight = Math.max(height, 1);
        const simplexSeed = (Math.floor(seed) ^ 0x6f5d2ec3) >>> 0;
        const simplex = new OpenSimplex(simplexSeed);

        const rng = mulberry32(Math.floor(seed) ^ 0x3d1b58f1 ^ (width << 2) ^ (height << 3));
        const { x: freqX, y: freqY } = this.#computeFrequencies(BRIGHTNESS_FREQ, width, height);
        const scaleX = freqX / safeWidth;
        const scaleY = freqY / safeHeight;
        const offsetX = rng();
        const offsetY = rng();

        for (let y = 0; y < height; y += 1) {
            const rowOffset = y * paddedFloatsPerRow;
            const sampleY = (y * scaleY) + offsetY;
            for (let x = 0; x < width; x += 1) {
                const sampleX = (x * scaleX) + offsetX;
                const rawValue = simplex.noise2D(sampleX, sampleY);
                const normalized = Math.min(1, Math.max(0, (rawValue + 1) * 0.5));
                const base = rowOffset + x * floatsPerPixel;
                this.brightnessData[base + 0] = normalized;
                this.brightnessData[base + 1] = normalized;
                this.brightnessData[base + 2] = normalized;
                this.brightnessData[base + 3] = 1.0;
            }
            for (let p = rowDataLength; p < paddedFloatsPerRow; p += 1) {
                this.brightnessData[rowOffset + p] = 0.0;
            }
        }

        device.queue.writeTexture(
            { texture: this.brightnessTexture },
            this.brightnessData,
            {
                bytesPerRow: paddedFloatsPerRow * Float32Array.BYTES_PER_ELEMENT,
                rowsPerImage: height,
            },
            { width, height, depthOrArrayLayers: 1 },
        );

        this.brightnessSeed = seed;
    }

    #initializeWormTextureWithNoise(device, width, height, seed) {
        // Generate value noise to initialize the worm feedback texture
        // This matches Python's: mask = value.values(freq=2-4, ...)
        const rng = mulberry32(Math.floor(seed) ^ 0xfeedbeef);
        
        // Use Float32Array for rgba16float texture format
        const floatsPerPixel = 4;  // RGBA
        const totalFloats = width * height * floatsPerPixel;
        const noiseData = new Float32Array(totalFloats);
        
        // Generate value noise at low frequency (similar to freq=2-4 in Python)
        // For simplicity, we'll use random values per pixel
        // A proper implementation would use actual value noise
        for (let i = 0; i < totalFloats; i += floatsPerPixel) {
            const value = 0.25 + rng() * 0.5;  // 0.25 to 0.75 range
            noiseData[i + 0] = value;
            noiseData[i + 1] = value;
            noiseData[i + 2] = value;
            noiseData[i + 3] = 1.0;
        }
        
        // Write to the worm feedback texture
        const bytesPerRow = width * floatsPerPixel * Float32Array.BYTES_PER_ELEMENT;
        device.queue.writeTexture(
            { texture: this.wormFeedbackTexture },
            noiseData,
            { bytesPerRow, rowsPerImage: height },
            { width, height, depthOrArrayLayers: 1 },
        );
    }

    #seedAgents(device, width, height, seed, reuseBuffers, agentCountOverride) {
        const rng = mulberry32(Math.floor(seed) ^ 0x1a976f87);
        const density = this.wormConfig?.density ?? 0.375;
        const stride = this.wormConfig?.stride ?? 0.75;
        const strideConfig = {
            stride,
            deviation: this.wormConfig?.deviation ?? 0.5,
        };
        const agentCount = agentCountOverride ?? computeAgentCount(width, height, density);
        const data = seedAgents(width, height, agentCount, strideConfig, rng);

        const needsReallocate = !reuseBuffers
            || !this.agentBuffers
            || this.agentBuffers.byteLength !== data.byteLength;

        if (needsReallocate) {
            const bufferA = device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            const bufferB = device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });

            device.queue.writeBuffer(bufferA, 0, data);
            device.queue.writeBuffer(bufferB, 0, data);

            this.agentBuffers?.a?.destroy?.();
            this.agentBuffers?.b?.destroy?.();

            this.agentBuffers = {
                a: bufferA,
                b: bufferB,
                current: 'a',
                count: agentCount,
                byteLength: data.byteLength,
            };
            return;
        }

        device.queue.writeBuffer(this.agentBuffers.a, 0, data);
        device.queue.writeBuffer(this.agentBuffers.b, 0, data);
        this.agentBuffers.count = agentCount;
        this.agentBuffers.byteLength = data.byteLength;
        this.agentBuffers.current = 'a';

        this.#updateWormComputePasses(width, height);
    }

    #rebindWormPasses(device) {
        if (!this.agentBuffers || !this.wormBindGroupLayout) {
            return;
        }

        this.wormComputeBindGroup = device.createBindGroup({
            layout: this.wormBindGroupLayout,
            entries: [
                { binding: 0, resource: this.noiseTextureView },
                { binding: 1, resource: { buffer: this.wormOutputBuffer } },
                { binding: 2, resource: { buffer: this.wormParamsBuffer } },
                { binding: 3, resource: this.wormFeedbackView },
                { binding: 4, resource: { buffer: this.agentBuffers.a } },
                { binding: 5, resource: { buffer: this.agentBuffers.b } },
            ],
        });

        this.wormComputeSwapBindGroup = device.createBindGroup({
            layout: this.wormBindGroupLayout,
            entries: [
                { binding: 0, resource: this.noiseTextureView },
                { binding: 1, resource: { buffer: this.wormOutputBuffer } },
                { binding: 2, resource: { buffer: this.wormParamsBuffer } },
                { binding: 3, resource: this.wormFeedbackView },
                { binding: 4, resource: { buffer: this.agentBuffers.b } },
                { binding: 5, resource: { buffer: this.agentBuffers.a } },
            ],
        });

        this.#updateWormComputePasses(this.width, this.height);
    }

    #createDisabledResources(width, height) {
        return {
            descriptor: null,
            shaderMetadata: null,
            resourceSet: {
                destroyAll() {},
            },
            computePipeline: null,
            computeBindGroup: null,
            paramsBuffer: null,
            paramsState: null,
            paramsDirty: false,
            outputBuffer: null,
            outputTexture: null,
            storageView: null,
            sampleView: null,
            bufferToTexturePipeline: null,
            bufferToTextureBindGroup: null,
            bufferToTextureWorkgroupSize: [8, 8, 1],
            blitBindGroup: null,
            workgroupSize: [8, 8, 1],
            enabled: false,
            textureWidth: width,
            textureHeight: height,
            bindingOffsets: {},
            device: this.device,
            computePasses: [],
            shouldCopyOutputToPrev: false,
            boundWormTexture: null,
            boundBrightnessTexture: null,
        };
    }

    #destroyTransientResources() {
        this.agentBuffers?.a?.destroy?.();
        this.agentBuffers?.b?.destroy?.();
        this.agentBuffers = null;

        this.wormOutputBuffer?.destroy?.();
        this.wormOutputBuffer = null;

        this.wormFeedbackTexture?.destroy?.();
        this.wormFeedbackTexture = null;
        this.wormFeedbackView = null;

        this.wormParamsBuffer?.destroy?.();
        this.wormParamsBuffer = null;
        this.wormParamsState = null;
        this.wormConfig = null;

        this.wormComputeBindGroup = null;
        this.wormComputeSwapBindGroup = null;
        this.wormComputePasses = null;
        this.trailBufferToTextureBindGroup = null;
        this.finalPass = null;
    this.wormDispatchConfig = null;

        this.noiseTexture?.destroy?.();
        this.noiseTexture = null;
        this.noiseTextureView = null;
        this.noiseData = null;
        this.noiseFloatsPerRow = 0;
        this.noiseSeed = null;

    this.brightnessTexture?.destroy?.();
    this.brightnessTexture = null;
    this.brightnessTextureView = null;
    this.brightnessData = null;
    this.brightnessFloatsPerRow = 0;
    this.brightnessSeed = null;

        this.#timeSeconds = 0;
        this.#lastTimestamp = null;
        this.hairGenerated = false;
    }

    #timeSeconds;
    #lastTimestamp;
}

export default StrayHairEffect;

