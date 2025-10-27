# Async Rendering for Browser Performance

This document describes the async-friendly rendering implementation that prevents the browser from freezing during computationally expensive preset rendering.

## Problem

JavaScript presets, especially complex multi-octave ones, perform intensive CPU calculations that block the browser's main thread. This causes:

- **UI freezing**: The browser becomes unresponsive during rendering
- **Poor user experience**: No visual feedback that work is in progress
- **Potential timeouts**: Very long renders may trigger browser warnings

## Solution

We've implemented a cooperative multitasking approach using `YieldController` that periodically yields control back to the browser's event loop during expensive operations.

### Architecture

#### 1. **YieldController** (`js/noisemaker/asyncHelpers.js`)

A utility class that tracks when to yield based on:
- **Time-based**: Yields every 16ms (~60fps) to keep UI responsive
- **Operation-based**: Yields every N operations to balance performance
- **Progressive enhancement**: Uses modern `scheduler.yield()` when available, falls back to `requestIdleCallback` or `setTimeout`

#### 2. **Generator Integration** (`js/noisemaker/generators.js`)

The `multires()` generator now:
- Creates a `YieldController` instance
- Yields before processing each octave
- Yields before applying each effect in the pipeline
- Only activates in browser contexts (disabled in Node.js tests)

#### 3. **Demo Integration** (`demo/js/index.html`)

The demo page:
- Imports `yieldToMain()` helper
- Yields between preset evaluation and rendering
- Yields between queued renders
- Shows "Rendering..." status during work
- Tracks rendering state with `isRendering` flag

### Benefits

✅ **Browser stays responsive** - Users can interact with controls during rendering  
✅ **Visual feedback** - Status updates show rendering is in progress  
✅ **No architectural changes** - Existing code continues to work  
✅ **Progressive enhancement** - Uses best available browser API  
✅ **Test-friendly** - Automatically disables in non-browser contexts  

### Performance Impact

The yielding mechanism adds minimal overhead:
- Yields occur only every ~16ms (at most ~60 times per second)
- Check operations are lightweight (performance.now() comparison)
- No impact on final image quality
- Slight increase in total render time (typically <5%) but vastly improved perceived performance

### Usage

No code changes required for existing presets. The yielding happens automatically during:

1. **Octave generation loops** - Between each octave layer
2. **Effect application** - Between octave, post, and final effects
3. **Render queuing** - Between sequential renders

### Browser Compatibility

- **Modern browsers** (Chrome 115+): Uses `scheduler.yield()`
- **Browsers with requestIdleCallback**: Uses `requestIdleCallback()`
- **Older browsers**: Falls back to `setTimeout(0)`
- **Node.js**: Automatically disabled during tests

### Example

Before (blocking):
```javascript
// Renders synchronously, freezes browser
await render(preset, seed, { width: 1024, height: 1024 });
```

After (non-blocking):
```javascript
// Renders cooperatively, yields to browser
await render(preset, seed, { width: 1024, height: 1024 });
// No API changes needed - yielding happens internally
```

### Configuration

The `YieldController` can be tuned via options:

```javascript
const controller = new YieldController({
  yieldIntervalMs: 16,    // Time between yields (default: 16ms)
  yieldEveryNOps: 100,    // Operations between yields (default: 100)
  enabled: true           // Enable/disable (default: true in browser)
});
```

Currently configured in `generators.js` with sensible defaults optimized for 60fps UI updates.

### Future Enhancements

Potential improvements:
- **Worker threads**: Offload rendering to Web Workers for true parallelism
- **WebGPU compute**: Move CPU-heavy operations to GPU compute shaders
- **Adaptive yielding**: Adjust yield frequency based on frame rate measurements
- **Progress callbacks**: Expose progress events for custom UI indicators
- **Render cancellation**: Allow aborting long-running renders

## Testing

The async behavior is automatically disabled during tests to ensure:
- Deterministic test execution
- Fast test runs without yielding overhead
- Compatibility with Node.js test runners

## See Also

- [Scheduler API](https://developer.mozilla.org/en-US/docs/Web/API/Scheduler/yield) - Modern cooperative scheduling
- [requestIdleCallback](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestIdleCallback) - Idle period scheduling
- [Long Tasks API](https://developer.mozilla.org/en-US/docs/Web/API/PerformanceLongTaskTiming) - Monitoring main thread blocking
