# GPU Effects Agent Instructions

You are implementing WGSL shaders in `/demo/gpu-effects/` following the swappable module architecture.

**READ FIRST**: Before implementing any effect, read `/shaders/IMPLEMENTATION_GUIDE.md` for complete architecture documentation, tutorials, and working examples.

## Critical Requirements

**YOU ARE FORBIDDEN** from implementing "passthrough shaders".

**YOU ARE NOT DONE** until:
1. Page loads at `/demo/gpu-effects/index.html`
2. Shader appears in dropdown with correct name
3. Executes without console errors or performance degradation

## Implementation Rules

### Architecture
- **Simple effects**: Extend `SimpleComputeEffect` (see `/shaders/effects/vignette/effect.js`)
  - Use explicit offset parameter bindings in `meta.json`
  - Follow single-pass tutorial in IMPLEMENTATION_GUIDE.md
- **Complex/multi-pass effects**: Implement custom class (see `/shaders/effects/worms/effect.js`)
  - Define `resources.computePasses` array to drive execution
  - Runtime never calls `execute()`—passes driven by resources object
  - Use `beforeDispatch()` and `afterDispatch()` hooks for buffer swapping
  - Follow multi-pass tutorial in IMPLEMENTATION_GUIDE.md
- Reference Python implementation in `/noisemaker/effects.py` for faithful port
- Audit and verify efficient threading in all GPU workgroups

### Key Architecture Points
- **Runtime never calls `execute()`**—passes are driven by `resources.computePasses` array in `main.js`
- **Two binding formats**: explicit offsets (for SimpleComputeEffect) or dot-notation (for custom classes)
- **Uniform buffers**: Size must be multiple of 4 bytes; runtime upscales to minimum 256 bytes
- **Multi-pass**: Each pass specifies `pipeline`, `bindGroup`, `workgroupSize`, and `getDispatch` function

### Forbidden Actions
- Disabling shaders or tests
- Using 2D canvas contexts
- Obscuring or masking real problems
- Exposing `width`, `height`, `channels`, or `time` as UI controls
- Exposing filenames in UI
- Creating unrequested documentation
- Using emojis

### Testing & Verification
- Use Playwright tests to verify functionality
- Shaders must run without errors or warnings
- Use visual diffs to verify outputs: `compare -metric RMSE`
  - Test with vanilla multires input using same frame/time/seed params
- Never ask user for manual log validation—automate it
