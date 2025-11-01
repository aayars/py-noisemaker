/**
 * Noisemaker Live Examples Runtime
 * 
 * Automatically renders all noisemaker-live-canvas elements on the page
 * using the bundled Noisemaker.js library.
 */

(function() {
    'use strict';
    
    // Wait for both DOM and Noisemaker to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLiveExamples);
    } else {
        initLiveExamples();
    }
    
    async function initLiveExamples() {
        // Wait a bit for the bundle to load if not ready yet
        let attempts = 0;
        while (!window.Noisemaker && attempts < 50) {
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }
        
        if (!window.Noisemaker) {
            console.error('Noisemaker bundle not loaded after 5 seconds');
            return;
        }
        
        // Find all canvas elements
        const canvases = document.querySelectorAll('.noisemaker-live-canvas');
        
        console.log(`Found ${canvases.length} Noisemaker live example(s)`);
        
        // Set up lazy loading with IntersectionObserver
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !entry.target.dataset.rendered) {
                    entry.target.dataset.rendered = 'true';
                    renderCanvas(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, {
            rootMargin: '200px',
            threshold: 0.5
        });
        
        // Render each canvas (lazy or immediate based on data attribute)
        for (const canvas of canvases) {
            const isLazy = canvas.dataset.lazy === 'true';
            
            if (isLazy) {
                // Use IntersectionObserver for lazy loading
                observer.observe(canvas);
            } else {
                // Render immediately
                await renderCanvas(canvas);
            }
            
            // Attach random button handler
            const wrapper = canvas.closest('.noisemaker-live-canvas-wrapper');
            const randomBtn = wrapper?.querySelector('.noisemaker-live-random');
            if (randomBtn) {
                randomBtn.addEventListener('click', () => {
                    // Generate new random seed
                    const newSeed = Math.floor(Math.random() * 100000);
                    canvas.dataset.seed = newSeed;
                    renderCanvas(canvas);
                });
            }
        }
    }
    
    async function renderCanvas(canvas) {
        const wrapper = canvas.closest('.noisemaker-live-canvas-wrapper');
        const loadingDiv = wrapper?.querySelector('.noisemaker-live-loading');
        const errorDiv = wrapper?.querySelector('.noisemaker-live-error');
        
        try {
            // Get parameters from data attributes
            const presetName = canvas.dataset.preset;
            const effectName = canvas.dataset.effect;
            const inputName = canvas.dataset.input || 'basic';
            const seed = parseInt(canvas.dataset.seed, 10) || 42;
            const width = parseInt(canvas.dataset.width, 10) || 512;
            const height = parseInt(canvas.dataset.height, 10) || 512;
            const time = parseFloat(canvas.dataset.time) || 0.0;
            const frame = parseFloat(canvas.dataset.frame) || 0.0;
            
            if (!presetName && !effectName) {
                throw new Error('No preset or effect specified');
            }
            
            // Show loading state
            if (loadingDiv) loadingDiv.style.display = 'block';
            canvas.style.opacity = '0.3';
            
            // Create preset and render
            // Note: The bundled API requires loading the preset table first
            const { Preset, PRESETS } = window.Noisemaker;
            
            const PRESET_TABLE = PRESETS();
            
            let tensor;
            const startTime = performance.now();
            
            // If rendering an effect, we need to render the input then apply the effect
            if (effectName) {
                // Check if effect exists (EFFECTS is the registry from effectsRegistry.js)
                const { EFFECTS, Effect } = window.Noisemaker;
                
                // Try both dash and underscore versions (e.g., "adjust-hue" and "adjust_hue")
                const effectNameSnake = effectName.replace(/-/g, '_');
                if (!EFFECTS || (!EFFECTS[effectName] && !EFFECTS[effectNameSnake])) {
                    throw new Error(`Effect "${effectName}" is not available in the JavaScript implementation yet. This effect exists in Python but hasn't been ported.`);
                }
                
                // Step 1: Render the input preset to get a tensor
                if (!PRESET_TABLE[inputName]) {
                    throw new Error(`Input preset "${inputName}" not found. Available presets: ${Object.keys(PRESET_TABLE).slice(0, 5).join(', ')}...`);
                }
                
                const inputPreset = new Preset(inputName, PRESET_TABLE, {}, seed, { debug: false });
                const inputTensor = await inputPreset.render(seed, {
                    width: width,
                    height: height,
                    time: time,
                    speed: 1.0,
                });
                
                // Step 2: Create an effect function with default parameters
                const effectFunc = Effect(effectName, {});
                
                // Step 3: Apply the effect function to the tensor
                // Note: effects might be async, and shape needs channels
                const shape = inputTensor.shape; // Use the input tensor's shape which has [h, w, c]
                const result = effectFunc(inputTensor, shape, 0, 1);
                
                // Effects might return a Promise
                tensor = result && typeof result.then === 'function' ? await result : result;
            } else {
                // Just render a preset directly
                if (!PRESET_TABLE[presetName]) {
                    throw new Error(`Preset "${presetName}" not found. Available presets: ${Object.keys(PRESET_TABLE).slice(0, 5).join(', ')}...`);
                }
                
                const preset = new Preset(presetName, PRESET_TABLE, {}, seed, { debug: false });
                tensor = await preset.render(seed, {
                    width: width,
                    height: height,
                    time: time,
                    speed: 1.0,
                });
            }

            
            // Convert tensor to ImageData and draw on canvas
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(width, height);
            const dataMaybe = tensor.read();
            const data = dataMaybe && typeof dataMaybe.then === 'function'
                ? await dataMaybe
                : dataMaybe;
            const channels = tensor.shape?.[2] || 3;
            
            for (let i = 0; i < width * height; i++) {
                const base = i * channels;
                const red = data?.[base] ?? 0;
                const green = data?.[base + 1] ?? red;
                const blue = data?.[base + 2] ?? red;
                const r = Math.floor(red * 255);
                const g = Math.floor(green * 255);
                const b = Math.floor(blue * 255);
                
                imageData.data[i * 4 + 0] = r;
                imageData.data[i * 4 + 1] = g;
                imageData.data[i * 4 + 2] = b;
                imageData.data[i * 4 + 3] = 255;
            }
            
            ctx.putImageData(imageData, 0, 0);
            
            // Show canvas, hide loading
            if (loadingDiv) loadingDiv.style.display = 'none';
            
            // Add a subtle fade-in effect
            canvas.style.transition = 'opacity 0.3s ease-in';
            canvas.style.opacity = '1';
            
        } catch (error) {
            console.error(`Error rendering canvas:`, error);
            
            // Show error message
            if (errorDiv) {
                errorDiv.textContent = `Error: ${error.message}`;
                errorDiv.style.display = 'block';
            }
            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
            
            // Draw error on canvas
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffe6e6';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#cc0000';
            ctx.font = '14px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Render Error', canvas.width / 2, canvas.height / 2 - 10);
            ctx.font = '12px monospace';
            ctx.fillText(error.message, canvas.width / 2, canvas.height / 2 + 10);
            
            canvas.style.opacity = '1';
            if (loadingDiv) loadingDiv.style.display = 'none';
        }
    }
    
    // Expose render function for manual control if needed
    window.renderNoisemakerCanvas = renderCanvas;
    
})();
