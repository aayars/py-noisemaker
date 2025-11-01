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
        
        // Render each canvas
        for (const canvas of canvases) {
            await renderCanvas(canvas);
            
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
            const seed = parseInt(canvas.dataset.seed, 10) || 42;
            const width = parseInt(canvas.dataset.width, 10) || 512;
            const height = parseInt(canvas.dataset.height, 10) || 512;
            const time = parseFloat(canvas.dataset.time) || 0.0;
            const frame = parseFloat(canvas.dataset.frame) || 0.0;
            
            if (!presetName) {
                throw new Error('No preset specified');
            }
            
            console.log(`Rendering preset "${presetName}" (seed: ${seed}, ${width}x${height})`);
            
            // Show loading state
            if (loadingDiv) loadingDiv.style.display = 'block';
            canvas.style.opacity = '0.3';
            
            // Create preset and render
            // Note: The bundled API requires loading the preset table first
            const { Preset, PRESETS } = window.Noisemaker;
            const PRESET_TABLE = PRESETS();
            
            if (!PRESET_TABLE[presetName]) {
                throw new Error(`Preset "${presetName}" not found. Available presets: ${Object.keys(PRESET_TABLE).slice(0, 5).join(', ')}...`);
            }
            
            const preset = new Preset(presetName, PRESET_TABLE, {}, seed, { debug: false });
            
            const startTime = performance.now();
            
            // Render the preset - returns a tensor
            const tensor = await preset.render(seed, {
                width: width,
                height: height,
                time: time,
                speed: 1.0,
            });
            
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
            
            const elapsed = performance.now() - startTime;
            console.log(`Rendered "${presetName}" in ${elapsed.toFixed(0)}ms`);
            
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
