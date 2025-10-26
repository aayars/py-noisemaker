# SimpleComputeEffect Implementation Guide

**Complete reference for implementing WGSL shader effects in the Noisemaker GPU pipeline**

Last Updated: October 18, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Parameter Binding System](#parameter-binding-system)
4. [Single-Pass Effect Tutorial](#single-pass-effect-tutorial)
5. [Multi-Pass Effect Tutorial](#multi-pass-effect-tutorial)
6. [Testing Guide](#testing-guide)
7. [Common Pitfalls](#common-pitfalls)
8. [Reference](#reference)

---

## Overview

### What is SimpleComputeEffect?

`SimpleComputeEffect` is a base class that provides standard scaffolding for GPU effects in the Noisemaker shader pipeline. It offers:

- **Automatic parameter management** from metadata
- **GPU resource lifecycle** (buffers, textures, pipelines)
- **Uniform buffer binding** with offset mapping
- **UI integration** through parameter definitions
- **Hot-reloading** and state preservation

### Effect Module Structure

Every effect lives in its own directory under `/shaders/effects/`:

```
shaders/effects/<effect-name>/
├── effect.js           # Effect class implementation
├── meta.json          # Metadata: parameters, bindings, resources
└── <effect-name>.wgsl # WGSL compute shader
```

### The Effect Contract

Your effect class should:

1. Either extend `SimpleComputeEffect` OR implement the lifecycle interface directly
2. Define `static metadata` referencing `meta.json`
3. Implement `ensureResources()`, `updateParams()`, `getUIState()`, and `destroy()`

**Note:** Complex effects (worms, wormhole, DLA, frame, fibers, aberration, false_color) implement their own lifecycle without extending SimpleComputeEffect. Simple effects (vignette, snow, sobel, etc.) can extend SimpleComputeEffect for automatic handling.

---

## Architecture

### Data Flow

```
User Input → UI Controls → updateParams() → Uniform Buffer → GPU → Output Texture/Buffer
```

### Lifecycle

```
constructor()
    ↓
ensureResources(context)
    ↓
updateParams(changes)
    ↓
execute(encoder, context)
    ↓
destroy()
```

### Key Concepts

**Uniform Buffer**: GPU-side memory holding shader parameters (read-only from shader)  
**Storage Buffer**: GPU-side memory for read/write data (often used for output)  
**Binding**: Connection between JavaScript resources and WGSL shader variables  
**Offset**: Index into the uniform buffer Float32Array where a parameter lives

---

## Parameter Binding System

### The Critical Rule

**Parameter bindings support two formats:**

1. **Explicit offset format** (recommended for SimpleComputeEffect):
```json
{
  "parameterBindings": {
    "my_param": { "buffer": "params", "offset": 3 }
  }
}
```

2. **Dot-notation format** (legacy, used by custom effect classes):
```json
{
  "parameterBindings": {
    "my_param": "params.my_vector.x"
  }
}
```

Effects extending SimpleComputeEffect should use explicit offsets. Custom effect classes (worms, wormhole, etc.) parse dot-notation bindings manually.

### Correct Binding Format

In `meta.json`:

```json
{
  "parameterBindings": {
    "width": { "buffer": "params", "offset": 0 },
    "height": { "buffer": "params", "offset": 1 },
    "channel_count": { "buffer": "params", "offset": 2 },
    "my_param": { "buffer": "params", "offset": 3 },
    "another_param": { "buffer": "params", "offset": 4 },
    "time": { "buffer": "params", "offset": 5 },
    "speed": { "buffer": "params", "offset": 6 },
    "enabled": { "kind": "toggle" }
  }
}
```

### Automatic Bindings

Three parameters are automatically bound if not specified:

- `width` → offset 0
- `height` → offset 1  
- `channel_count` → offset 2

You can override these by providing explicit bindings.

### The WGSL Struct

Your shader struct **must** match the offsets:

```wgsl
struct MyEffectParams {
    width : f32,           // offset 0
    height : f32,          // offset 1
    channel_count : f32,   // offset 2
    my_param : f32,        // offset 3
    another_param : f32,   // offset 4
    time : f32,            // offset 5
    speed : f32,           // offset 6
    _pad0 : f32,          // offset 7 (padding to align)
};
```

### Buffer Size Requirements

Uniform buffers must be sized in multiples of 4 bytes (1 float). The runtime automatically rounds up to at least 256 bytes for GPU alignment:

```json
{
  "resources": {
    "params": {
      "kind": "uniformBuffer",
      "size": 32  // 8 floats × 4 bytes = 32 bytes (will be allocated as 256 minimum)
    }
  }
}
```

**Implementation note:** While you can specify any multiple of 4, the GPU allocator in `main.js` ensures a minimum of 256 bytes and aligns to 256-byte boundaries internally.

### Special Binding Types

**Toggle (boolean)**: Not stored in uniform buffer, used for enable/disable logic

```json
{
  "enabled": { "kind": "toggle" }
}
```

---

## Single-Pass Effect Tutorial

We'll implement a simple **brightness adjustment** effect that multiplies RGB values by a factor.

### Step 1: Create Directory Structure

```bash
mkdir -p shaders/effects/adjust_brightness
cd shaders/effects/adjust_brightness
```

### Step 2: Define Metadata (`meta.json`)

```json
{
  "id": "adjust_brightness",
  "label": "Adjust Brightness",
  "stage": "effect",
  "shader": {
    "entryPoint": "main",
    "url": "/shaders/effects/adjust_brightness/adjust_brightness.wgsl"
  },
  "resources": {
    "input_texture": {
      "kind": "sampledTexture",
      "format": "rgba32float"
    },
    "output_buffer": {
      "kind": "storageBuffer",
      "size": "pixel-f32x4"
    },
    "params": {
      "kind": "uniformBuffer",
      "size": 32
    }
  },
  "parameters": [
    {
      "name": "enabled",
      "type": "boolean",
      "default": true,
      "label": "Enabled",
      "description": "Toggle brightness adjustment on or off."
    },
    {
      "name": "brightness",
      "type": "float",
      "default": 1.0,
      "min": 0.0,
      "max": 3.0,
      "step": 0.1,
      "label": "Brightness",
      "description": "Brightness multiplier (1.0 = original)."
    }
  ],
  "parameterBindings": {
    "width": { "buffer": "params", "offset": 0 },
    "height": { "buffer": "params", "offset": 1 },
    "channel_count": { "buffer": "params", "offset": 2 },
    "brightness": { "buffer": "params", "offset": 3 },
    "time": { "buffer": "params", "offset": 4 },
    "speed": { "buffer": "params", "offset": 5 },
    "_pad0": { "buffer": "params", "offset": 6 },
    "_pad1": { "buffer": "params", "offset": 7 },
    "enabled": { "kind": "toggle" }
  }
}
```

**Key Points:**
- `brightness` is at offset 3 (after width, height, channel_count)
- We add padding to reach 32 bytes (8 floats)
- `enabled` is a toggle, not in the buffer

### Step 3: Write WGSL Shader (`adjust_brightness.wgsl`)

```wgsl
// Adjust Brightness: Multiply RGB channels by a brightness factor
struct BrightnessParams {
    width : f32,
    height : f32,
    channel_count : f32,
    brightness : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : BrightnessParams;

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    // Bounds check - critical for GPU safety
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    // Read input pixel
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    
    // Apply brightness to RGB, preserve alpha
    let brightness_factor : f32 = params.brightness;
    let adjusted_rgb : vec3<f32> = texel.rgb * brightness_factor;
    let final_color : vec4<f32> = vec4<f32>(adjusted_rgb, texel.a);

    // Write output
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, final_color);
}
```

**Key Points:**
- Struct members match `meta.json` offsets exactly
- End struct members with `,` not `;` (WGSL requirement)
- **Always** bounds-check first: `if (gid.x >= width || gid.y >= height) return;`
- Use `@workgroup_size(8, 8, 1)` for standard effects
- Preserve alpha channel (don't modify transparency)

### Step 4: Create Effect Class (`effect.js`)

```javascript
import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import meta from './meta.json' with { type: 'json' };

export default class AdjustBrightnessEffect extends SimpleComputeEffect {
  static metadata = meta;
}
```

**That's it!** `SimpleComputeEffect` handles everything else:
- Loading the shader
- Creating uniform buffers
- Mapping parameters to offsets
- Updating uniforms when sliders change
- Resource cleanup

### Step 5: Register Effect

Add to `/demo/gpu-effects/shader-registry.js`:

```javascript
import adjustBrightnessMeta from '../../shaders/effects/adjust_brightness/meta.json' with { type: 'json' };

export const SHADER_REGISTRY = {
  // ... existing effects
  adjust_brightness: metaToDescriptor(adjustBrightnessMeta),
};
```

### Step 6: Test

Load `http://localhost:8000/demo/gpu-effects/` and select "Adjust Brightness" from the dropdown. The brightness slider should appear and work.

---

## Multi-Pass Effect Tutorial

We'll examine the **worms** effect, which requires multiple GPU passes:

1. **Initialization pass**: Set up flow field
2. **Iteration passes**: Simulate particle movement (N iterations)
3. **Output pass**: Write final result

### Understanding Multi-Pass Architecture

Some effects can't be computed in a single shader dispatch because they:

- Require iterative algorithms (erosion, DLA, particle systems)
- Need intermediate state between passes
- Depend on previous frame results

### Step 1: Define Metadata (`meta.json`)

```json
{
  "id": "worms",
  "label": "Worms",
  "stage": "effect",
  "shader": {
    "entryPoint": "main",
    "url": "/shaders/effects/worms/worms.wgsl"
  },
  "resources": {
    "input_texture": {
      "kind": "sampledTexture",
      "format": "rgba32float"
    },
    "output_buffer": {
      "kind": "storageBuffer",
      "size": "pixel-f32x4"
    },
    "state_buffer": {
      "kind": "storageBuffer",
      "size": "custom"
    },
    "params": {
      "kind": "uniformBuffer",
      "size": 64
    }
  },
  "parameters": [
    {
      "name": "enabled",
      "type": "boolean",
      "default": true,
      "label": "Enabled"
    },
    {
      "name": "density",
      "type": "float",
      "default": 0.1,
      "min": 0.001,
      "max": 1.0,
      "step": 0.001,
      "label": "Density"
    },
    {
      "name": "iterations",
      "type": "int",
      "default": 50,
      "min": 1,
      "max": 200,
      "step": 1,
      "label": "Iterations"
    }
  ],
  "parameterBindings": {
    "width": { "buffer": "params", "offset": 0 },
    "height": { "buffer": "params", "offset": 1 },
    "channel_count": { "buffer": "params", "offset": 2 },
    "density": { "buffer": "params", "offset": 3 },
    "iterations": { "buffer": "params", "offset": 4 },
    "time": { "buffer": "params", "offset": 5 },
    "speed": { "buffer": "params", "offset": 6 },
    "alpha": { "buffer": "params", "offset": 7 },
    "enabled": { "kind": "toggle" }
  }
}
```

### Step 2: Custom Effect Class with Multi-Pass Logic

**CRITICAL:** Multi-pass effects do NOT override `execute()`. The runtime drives passes from the `resources.computePasses` array.

```javascript
import meta from './meta.json' with { type: 'json' };

const RGBA_CHANNEL_COUNT = 4;

class WormsEffect {
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
      density: 0.1,
      enabled: true,
    };
    this.agentBuffers = null;
  }

  async ensureResources({ device, width, height, multiresResources }) {
    if (!device) {
      throw new Error('WormsEffect.ensureResources requires a GPUDevice.');
    }
    if (!multiresResources?.outputTexture) {
      throw new Error('WormsEffect.ensureResources requires multires output texture.');
    }

    // Check if we can reuse existing resources
    if (this.resources &&
        this.resources.textureWidth === width &&
        this.resources.textureHeight === height) {
      return this.resources;
    }

    // Clean up old resources
    this.invalidateResources();

    // Create custom agent buffers for particle data
    const agentCount = Math.floor(Math.max(width, height) * this.userState.density);
    const bufferSize = agentCount * 8 * Float32Array.BYTES_PER_ELEMENT;
    
    const bufferA = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const bufferB = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    this.agentBuffers = { a: bufferA, b: bufferB, current: 'a', agentCount };

    // Create standard resources using helpers
    const {
      getShaderDescriptor,
      getShaderMetadataCached,
      getOrCreateBindGroupLayout,
      getOrCreatePipelineLayout,
      getOrCreateComputePipeline,
      getBufferToTexturePipeline,
    } = this.helpers;

    const descriptor = getShaderDescriptor('worms');
    const shaderMetadata = await getShaderMetadataCached('worms');
    
    // Get multi-pass shader descriptors
    const initFromPrevDescriptor = getShaderDescriptor('worms/init_from_prev');
    const agentMoveDescriptor = getShaderDescriptor('worms/agent_move');
    const finalBlendDescriptor = getShaderDescriptor('worms/final_blend');

    // Create pipelines for each pass
    const computeBindGroupLayout = getOrCreateBindGroupLayout(device, descriptor.id, 'compute', shaderMetadata);
    const computePipelineLayout = getOrCreatePipelineLayout(device, descriptor.id, 'compute', computeBindGroupLayout);
    
    const initFromPrevPipeline = await getOrCreateComputePipeline(
      device, 'worms/init_from_prev', computePipelineLayout, 'main'
    );
    const agentMovePipeline = await getOrCreateComputePipeline(
      device, 'worms/agent_move', computePipelineLayout, 'main'
    );
    const finalBlendPipeline = await getOrCreateComputePipeline(
      device, 'worms/final_blend', computePipelineLayout, 'main'
    );

    // Create other required resources (params buffer, output buffer, textures, etc.)
    const paramsBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const outputBufferSize = width * height * RGBA_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT;
    const outputBuffer = device.createBuffer({
      size: outputBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const feedbackTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });

    const outputTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    // Create bind group (will be recreated in beforeDispatch for buffer swapping)
    const computeBindGroup = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: feedbackTexture.createView() },
        { binding: 4, resource: { buffer: this.agentBuffers.a } },
        { binding: 5, resource: { buffer: this.agentBuffers.b } },
      ],
    });

    const { pipeline: bufferToTexturePipeline, bindGroupLayout: bufferToTextureBindGroupLayout } = 
      await getBufferToTexturePipeline(device);
    
    const bufferToTextureBindGroup = device.createBindGroup({
      layout: bufferToTextureBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: outputTexture.createView() },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const blitBindGroup = device.createBindGroup({
      layout: multiresResources.blitBindGroupLayout,
      entries: [{ binding: 0, resource: outputTexture.createView() }],
    });

    // **KEY PATTERN:** Define computePasses array to drive multi-pass execution
    const agentWorkgroupSize = [64, 1, 1];
    const blendWorkgroupSize = [8, 8, 1];
    
    const computePasses = [
      // Pass 0: Copy prev_texture to output_buffer (pixel-parallel)
      {
        pipeline: initFromPrevPipeline,
        bindGroup: computeBindGroup,
        workgroupSize: blendWorkgroupSize,
        getDispatch: ({ width, height }) => [
          Math.ceil(width / blendWorkgroupSize[0]),
          Math.ceil(height / blendWorkgroupSize[1]),
          1
        ],
      },
      // Pass 1: Move agents and deposit trails (agent-parallel)
      {
        pipeline: agentMovePipeline,
        bindGroup: computeBindGroup,
        workgroupSize: agentWorkgroupSize,
        getDispatch: () => [
          Math.ceil(agentCount / agentWorkgroupSize[0]),
          1,
          1
        ],
      },
      // Pass 2: Final blend with input (pixel-parallel)
      {
        pipeline: finalBlendPipeline,
        bindGroup: computeBindGroup,
        workgroupSize: blendWorkgroupSize,
        getDispatch: ({ width, height }) => [
          Math.ceil(width / blendWorkgroupSize[0]),
          Math.ceil(height / blendWorkgroupSize[1]),
          1
        ],
      },
    ];

    // Initialize params state
    const paramsState = new Float32Array(16);
    paramsState[0] = width;
    paramsState[1] = height;
    paramsState[2] = RGBA_CHANNEL_COUNT;
    // ... set other parameters
    device.queue.writeBuffer(paramsBuffer, 0, paramsState);

    // Return resources object with computePasses array
    this.resources = {
      computePipeline: agentMovePipeline, // Dummy, not used when computePasses is present
      computeBindGroup,
      computePasses,  // **This drives multi-pass execution in main.js**
      paramsBuffer,
      paramsState,
      outputBuffer,
      outputTexture,
      feedbackTexture,
      bufferToTexturePipeline,
      bufferToTextureBindGroup,
      blitBindGroup,
      workgroupSize: blendWorkgroupSize,
      enabled: this.userState.enabled,
      textureWidth: width,
      textureHeight: height,
      paramsDirty: false,
      device,
      computeBindGroupLayout,
      shouldCopyOutputToPrev: true,
    };

    return this.resources;
  }

  // Called before each frame to swap agent buffers
  beforeDispatch({ device, multiresResources }) {
    if (!this.agentBuffers || !this.resources) return;

    // Swap buffers
    const currentIsA = this.agentBuffers.current === 'a';
    const inputBuffer = currentIsA ? this.agentBuffers.a : this.agentBuffers.b;
    const outputBuffer = currentIsA ? this.agentBuffers.b : this.agentBuffers.a;

    // Recreate bind group with swapped buffers
    const newBindGroup = device.createBindGroup({
      layout: this.resources.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: multiresResources.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.resources.outputBuffer } },
        { binding: 2, resource: { buffer: this.resources.paramsBuffer } },
        { binding: 3, resource: this.resources.feedbackTexture.createView() },
        { binding: 4, resource: { buffer: inputBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
      ],
    });
    
    this.resources.computeBindGroup = newBindGroup;
    
    // Update bind groups in all compute passes
    if (this.resources.computePasses && Array.isArray(this.resources.computePasses)) {
      this.resources.computePasses.forEach(pass => {
        pass.bindGroup = newBindGroup;
      });
    }
  }

  // Called after each frame to update buffer marker
  afterDispatch() {
    if (!this.agentBuffers) return;
    this.agentBuffers.current = this.agentBuffers.current === 'a' ? 'b' : 'a';
  }

  async updateParams(updates = {}) {
    // Implementation similar to SimpleComputeEffect
    const updated = [];
    // ... handle parameter updates
    return { updated };
  }

  getUIState() {
    return { ...this.userState };
  }

  invalidateResources() {
    // Clean up GPU resources
    if (this.agentBuffers) {
      [this.agentBuffers.a, this.agentBuffers.b].forEach((buffer) => {
        if (buffer?.destroy) buffer.destroy();
      });
      this.agentBuffers = null;
    }
    // ... clean up other resources
    this.resources = null;
  }

  destroy() {
    this.invalidateResources();
  }
}

export default WormsEffect;
```

**Key Multi-Pass Concepts:**

1. **computePasses Array**: Define an array of pass configurations in `resources.computePasses`
2. **Pass Configuration**: Each pass specifies `pipeline`, `bindGroup`, `workgroupSize`, and `getDispatch` function
3. **Runtime Execution**: `main.js` iterates through `computePasses` and dispatches each one automatically (lines 2261-2330)
4. **Buffer Swapping**: Use `beforeDispatch()` and `afterDispatch()` hooks for double-buffering
5. **NO execute() Method**: The runtime never calls `execute()`—all dispatch logic is driven by the resources object

### Step 3: Multi-Pass WGSL Shader

```wgsl
struct WormsParams {
    width : f32,
    height : f32,
    channel_count : f32,
    density : f32,
    iterations : f32,
    time : f32,
    speed : f32,
    alpha : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;
@group(0) @binding(3) var<storage, read_write> state_buffer : array<f32>;

// Shared memory for workgroup cooperation
var<workgroup> shared_state : array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Read from state buffer
    let particle_idx : u32 = gid.y * dims.x + gid.x;
    let state_base : u32 = particle_idx * 4u;
    
    var particle_x : f32 = state_buffer[state_base + 0u];
    var particle_y : f32 = state_buffer[state_base + 1u];
    
    // Update particle position (simplified)
    particle_x += 0.1;
    particle_y += 0.1;
    
    // Write back to state
    state_buffer[state_base + 0u] = particle_x;
    state_buffer[state_base + 1u] = particle_y;
    
    // Accumulate to output
    let pixel_idx : u32 = gid.y * dims.x + gid.x;
    let output_base : u32 = pixel_idx * 4u;
    
    output_buffer[output_base + 0u] += 0.01; // Accumulate red
    output_buffer[output_base + 1u] += 0.01;
    output_buffer[output_base + 2u] += 0.01;
    output_buffer[output_base + 3u] = 1.0;
}
```

**Multi-Pass Shader Features:**

- **State buffer**: Persistent storage across iterations
- **Workgroup shared memory**: For cooperation within a workgroup
- **Accumulation**: Build up result over multiple passes
- **Atomic operations**: If needed (use `atomicAdd`, etc.)

---

## Testing Guide

### Manual Testing

1. Start local server: `python -m http.server 8000`
2. Open `http://localhost:8000/demo/gpu-effects/`
3. Select your effect from dropdown
4. Verify:
   - Effect appears in dropdown
   - Controls render correctly
   - Sliders respond
   - No console errors
   - Output looks correct

### Automated Testing with Playwright

Create a test file `test/<effect-name>-test.mjs`:

```javascript
#!/usr/bin/env node

import { chromium } from 'playwright';

(async () => {
  console.log('=== Adjust Brightness Test ===\n');
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  const errors = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  
  page.on('pageerror', err => {
    errors.push(err.message);
  });

  await page.goto('http://localhost:8000/demo/gpu-effects/index.html');
  await page.waitForTimeout(2000);
  
  console.log('✓ Page loaded');
  
  // Select effect
  await page.selectOption('select#effect-selector', 'adjust_brightness');
  await page.waitForTimeout(1500);
  
  console.log('✓ Effect selected');
  
  // Check if controls exist
  const brightnessSlider = await page.evaluate(() => {
    const slider = document.querySelector('input[name="brightness"]');
    return slider ? {
      value: slider.value,
      min: slider.min,
      max: slider.max
    } : null;
  });
  
  if (!brightnessSlider) {
    console.error('✗ Brightness slider not found');
    await browser.close();
    process.exit(1);
  }
  
  console.log(`✓ Brightness slider found: ${brightnessSlider.value} (${brightnessSlider.min}-${brightnessSlider.max})`);
  
  // Test slider interaction
  console.log('\nTesting slider...');
  await page.locator('input[name="brightness"]').evaluate(el => {
    el.value = '2.0';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(300);
  console.log('  Set brightness to 2.0');
  
  await page.locator('input[name="brightness"]').evaluate(el => {
    el.value = '0.5';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(300);
  console.log('  Set brightness to 0.5');
  
  await page.locator('input[name="brightness"]').evaluate(el => {
    el.value = '1.0';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(300);
  console.log('  Set brightness to 1.0');
  
  await page.waitForTimeout(1000);
  
  console.log(`\n=== RESULTS ===`);
  console.log(`✓ Effect loads without errors`);
  console.log(`✓ Slider responsive`);
  console.log(`✓ Console errors: ${errors.length}`);
  
  if (errors.length > 0) {
    console.log('\n⚠ Errors detected:');
    errors.forEach(e => console.log(`  - ${e}`));
    await browser.close();
    process.exit(1);
  }
  
  console.log('\n✅ All tests passed!');
  
  await browser.close();
  process.exit(0);
})();
```

### Key Testing Patterns

**1. Find sliders by name attribute:**
```javascript
const slider = page.locator('input[name="my_param"]');
```

**2. Set slider value:**
```javascript
await slider.evaluate(el => {
  el.value = '0.5';
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
```

**3. Check for console errors:**
```javascript
page.on('console', msg => {
  if (msg.type() === 'error') errors.push(msg.text());
});
page.on('pageerror', err => errors.push(err.message));
```

**4. Verify effect appears in dropdown:**
```javascript
const options = await page.evaluate(() => {
  const select = document.querySelector('#effect-selector');
  return Array.from(select.options).map(o => o.value);
});
expect(options).toContain('my_effect');
```

### Running Tests

```bash
# Make executable
chmod +x test/my-effect-test.mjs

# Run test
node test/my-effect-test.mjs
```

---

## Common Pitfalls

### 1. Wrong Parameter Binding Format

**❌ WRONG:**
```json
{
  "parameterBindings": {
    "brightness": "params.size_brightness.w"
  }
}
```

**✅ CORRECT:**
```json
{
  "parameterBindings": {
    "brightness": { "buffer": "params", "offset": 3 }
  }
}
```

### 2. Struct/Binding Mismatch

**❌ WRONG:**
```wgsl
struct Params {
    width : f32,      // offset 0
    height : f32,     // offset 1
    brightness : f32, // offset 2 - WRONG!
};
```

```json
{
  "parameterBindings": {
    "brightness": { "offset": 3 }  // Says offset 3!
  }
}
```

**✅ CORRECT:** Offsets must match exactly.

### 3. Forgetting Bounds Check

**❌ WRONG:**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(input_texture, coords, 0); // May read out of bounds!
}
```

**✅ CORRECT:**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(input_texture, 0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;  // Critical!
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(input_texture, coords, 0);
}
```

### 4. Single-Threaded GPU Code

**❌ WRONG (very slow):**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Only thread (0,0) does all the work
    if (gid.x != 0u || gid.y != 0u) {
        return;
    }
    
    for (var y = 0u; y < height; y++) {
        for (var x = 0u; x < width; x++) {
            // Process pixel...
        }
    }
}
```

**✅ CORRECT (parallel):**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Each thread processes one pixel
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    // Process this thread's pixel
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    // ...
}
```

### 5. Modifying Alpha Channel Unintentionally

**❌ WRONG:**
```wgsl
let brightness_vec = vec4<f32>(brightness); // Applies to all 4 channels!
let result = texel * brightness_vec;
```

**✅ CORRECT:**
```wgsl
let brightness_rgb = vec3<f32>(brightness);
let result_rgb = texel.rgb * brightness_rgb;
let result = vec4<f32>(result_rgb, texel.a); // Preserve alpha
```

### 6. Using Semicolons in Struct

**❌ WRONG:**
```wgsl
struct Params {
    width : f32;  // WRONG! Will not parse
    height : f32;
};
```

**✅ CORRECT:**
```wgsl
struct Params {
    width : f32,  // Commas, not semicolons
    height : f32,
};
```

### 7. Missing Padding

**❌ WRONG (31 bytes):**
```json
{
  "resources": {
    "params": { "size": 31 }  // Not aligned!
  }
}
```

**✅ CORRECT (32 bytes):**
```json
{
  "resources": {
    "params": { "size": 32 }  // Multiple of 16
  }
}
```

---

## Reference

### SimpleComputeEffect API

#### Constructor

```javascript
constructor({ helpers } = {})
```

#### Static Properties

```javascript
static metadata  // References meta.json
```

#### Instance Properties

```javascript
this.userState          // Current parameter values (Map)
this.resources          // GPU resources (buffers, pipelines, bind groups)
this.paramOffsets       // Parameter name → buffer offset mapping
this.enabledParamName   // Name of the enabled/disabled toggle parameter
```

#### Methods

```javascript
// Lifecycle
async ensureResources(context)  // Create/update GPU resources
destroy()                        // Clean up resources
invalidateResources()           // Force resource recreation

// Parameters
async updateParams(updates)     // Update parameter values
getUIState()                    // Get current UI state

// Execution (override for custom behavior)
async execute(encoder, context) // Encode GPU commands
```

### Meta.json Schema

```javascript
{
  "id": string,              // Unique effect identifier
  "label": string,           // Display name
  "stage": "effect" | "generator",
  "shader": {
    "entryPoint": string,    // Usually "main"
    "url": string           // Path to .wgsl file
  },
  "resources": {
    "input_texture"?: {
      "kind": "sampledTexture",
      "format": "rgba32float"
    },
    "output_buffer"?: {
      "kind": "storageBuffer",
      "size": "pixel-f32x4" | number
    },
    "params"?: {
      "kind": "uniformBuffer",
      "size": number  // Bytes (multiple of 16)
    }
  },
  "parameters": [
    {
      "name": string,
      "type": "boolean" | "float" | "int",
      "default": any,
      "min"?: number,
      "max"?: number,
      "step"?: number,
      "label": string,
      "description": string
    }
  ],
  "parameterBindings": {
    [name: string]: {
      "buffer": "params",
      "offset": number
    } | {
      "kind": "toggle"
    }
  }
}
```

### Common WGSL Patterns

#### Standard Effect Structure

```wgsl
struct EffectParams {
    width : f32,
    height : f32,
    channel_count : f32,
    // ... your parameters
    _pad0 : f32,  // Add padding as needed
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : EffectParams;

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let input_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    
    // Your effect logic here
    let output_color : vec4<f32> = process(input_color);

    let pixel_index : u32 = gid.y * dims.x + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, output_color);
}
```

### File Paths Reference

```
/shaders/common/
  └── simple-compute-effect.js    # Base class

/shaders/effects/<your-effect>/
  ├── effect.js                   # Your effect class
  ├── meta.json                   # Metadata
  └── <your-effect>.wgsl          # WGSL shader

/demo/gpu-effects/
  ├── index.html                  # Main demo page
  ├── main.js                     # WebGPU setup
  ├── shader-registry.js          # Effect registration
  ├── effect-manager.js           # Effect lifecycle
  └── effect-ui-generator.js      # UI generation

/test/
  └── <your-effect>-test.mjs      # Playwright test
```

---

## Complete Working Example: Vignette Effect

See the vignette effect for a complete, production-ready SimpleComputeEffect example:

- `/shaders/effects/vignette/meta.json` - Metadata with explicit offset bindings
- `/shaders/effects/vignette/vignette.wgsl` - WGSL shader
- `/shaders/effects/vignette/effect.js` - Effect class (extends SimpleComputeEffect)
- `/test/vignette-final-check.mjs` - Full test suite

The vignette effect demonstrates:
- ✅ Correct explicit offset parameter bindings
- ✅ RGB-only processing (preserving alpha)
- ✅ Proper GPU parallelization
- ✅ Full test coverage with Playwright
- ✅ Working sliders and UI integration

## Complete Working Example: Worms Effect

See the worms effect for a complete, production-ready custom multi-pass effect:

- `/shaders/effects/worms/meta.json` - Metadata with dot-notation bindings
- `/shaders/effects/worms/*.wgsl` - Multi-pass WGSL shaders
- `/shaders/effects/worms/effect.js` - Custom effect class (does NOT extend SimpleComputeEffect)
- `/test/test-worms-simple.js` - Test suite

The worms effect demonstrates:
- ✅ Multi-pass execution via `computePasses` array
- ✅ Custom agent buffer management with double-buffering
- ✅ `beforeDispatch()` and `afterDispatch()` hooks
- ✅ Dot-notation parameter bindings parsed manually
- ✅ Temporal feedback with `prev_texture`

---

## Getting Help

1. **Check existing effects** in `/shaders/effects/` for patterns
2. **Run tests** to catch binding errors early
3. **Use browser DevTools** to inspect WebGPU errors
4. **Verify struct alignment** matches meta.json offsets
5. **Test with simple values** before complex logic

**Remember:** Parameter binding is the #1 source of errors. Always use explicit offsets!

---

**End of Guide**
