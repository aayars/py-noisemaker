/**
 * Embedded Preset Viewer for Composer docs
 * Lightweight version of demo/js for inline documentation
 */

(function() {
    'use strict';

    // Will be populated when noisemaker bundle loads
    let Preset = null;
    let PRESETS = null;
    let render = null;
    let Context = null;
    let rng = null;
    let presetSource = {};

    function formatPreset(obj, indent = 0) {
        const pad = '  '.repeat(indent);
        if (typeof obj === 'function') {
            if (obj.length === 0) {
                try {
                    const result = obj();
                    if (result && typeof result === 'object') {
                        return formatPreset(result, indent);
                    }
                } catch (e) {
                    // ignore invocation failures
                }
            }
            return obj.toString();
        }
        if (obj && typeof obj === 'object') {
            if (Array.isArray(obj)) {
                return '[\n' + obj.map((v) => pad + '  ' + formatPreset(v, indent + 1)).join(',\n') + '\n' + pad + ']';
            }
            return '{\n' + Object.entries(obj).map(([k, v]) => pad + '  ' + JSON.stringify(k) + ': ' + formatPreset(v, indent + 1)).join(',\n') + '\n' + pad + '}';
        }
        return JSON.stringify(obj);
    }

    function toSnake(str) {
        return str.replace(/([A-Z])/g, (m) => '_' + m.toLowerCase());
    }

    function toSnakeKeys(obj) {
        const result = {};
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                result[toSnake(key)] = obj[key];
            }
        }
        return result;
    }

    function serializePreset(preset, seed) {
        const toPlain = (effects) =>
            effects.map((fn) => ({
                effect: toSnake(fn.__effectName || fn.name || 'fn'),
                ...toSnakeKeys(fn.__params || {}),
            }));

        return {
            name: preset.name || 'unknown',
            layers: preset.layers || [],
            settings: preset.settings || {},
            generator: preset.generator ? {
                name: preset.generator.name || 'basic',
                ...toSnakeKeys(preset.generator.kwargs || {}),
            } : null,
            octaves: preset.octaves ? toPlain(preset.octaves) : [],
            post: preset.post ? toPlain(preset.post) : [],
            final: preset.final ? toPlain(preset.final) : [],
        };
    }

    async function initPresetViewer(container) {
        const canvas = container.querySelector('.preset-viewer-canvas');
        const select = container.querySelector('.preset-viewer-select');
        const codeArea = container.querySelector('.preset-viewer-code');
        const loadingIndicator = container.querySelector('.preset-viewer-loading');
        const randomButton = container.querySelector('.preset-viewer-random');

        if (!canvas || !select || !codeArea) {
            console.error('Preset viewer: missing required elements');
            return;
        }

        // Lazy load noisemaker bundle
        if (!Preset) {
            try {
                // Load from _static (copied during build)
                const module = await import('./noisemaker.js');
                
                // Import exactly like javascript.rst shows
                Preset = module.Preset;
                PRESETS = module.PRESETS();
                render = module.render;
                Context = module.Context;
                rng = module.rng;
                
                console.log('PRESETS type:', typeof PRESETS);
                console.log('PRESETS keys:', Object.keys(PRESETS).slice(0, 10));
                console.log('bloom preset:', PRESETS['bloom']);
                
                // Load preset source text from _static
                const response = await fetch('_static/presets.dsl');
                const dslText = await response.text();
                
                // Extract preset source exactly like demo/js does
                function extractPresetSource(source, name) {
                    const quotedName = `"${name}"`;
                    let searchStart = 0;

                    while (searchStart < source.length) {
                        const nameIdx = source.indexOf(quotedName, searchStart);
                        if (nameIdx === -1) return '';

                        let idx = nameIdx + quotedName.length;
                        while (idx < source.length && /\s/.test(source[idx])) idx++;

                        if (idx >= source.length || source[idx] !== ':') {
                            searchStart = nameIdx + quotedName.length;
                            continue;
                        }

                        idx++;
                        let braceIdx = idx;
                        while (braceIdx < source.length) {
                            const ch = source[braceIdx];
                            if (/\s/.test(ch)) {
                                braceIdx++;
                                continue;
                            }
                            if (ch === '/' && source[braceIdx + 1] === '/') {
                                braceIdx += 2;
                                while (braceIdx < source.length && source[braceIdx] !== '\n') braceIdx++;
                                continue;
                            }
                            if (ch === '/' && source[braceIdx + 1] === '*') {
                                braceIdx += 2;
                                while (braceIdx + 1 < source.length &&
                                    !(source[braceIdx] === '*' && source[braceIdx + 1] === '/')) {
                                    braceIdx++;
                                }
                                if (braceIdx + 1 < source.length) braceIdx += 2;
                                continue;
                            }
                            break;
                        }

                        if (braceIdx >= source.length || source[braceIdx] !== '{') {
                            searchStart = nameIdx + quotedName.length;
                            continue;
                        }

                        let end = braceIdx;
                        let depth = 0;
                        let inString = false;
                        let escaped = false;

                        while (end < source.length) {
                            const ch = source[end++];
                            if (inString) {
                                if (escaped) {
                                    escaped = false;
                                    continue;
                                }
                                if (ch === '\\') {
                                    escaped = true;
                                } else if (ch === '"') {
                                    inString = false;
                                }
                                continue;
                            }
                            if (ch === '"') {
                                inString = true;
                            } else if (ch === '{') {
                                depth++;
                            } else if (ch === '}') {
                                depth--;
                                if (depth === 0) {
                                    return source.slice(nameIdx, end);
                                }
                            }
                        }
                        return '';
                    }
                    return '';
                }

                presetSource = {};
                for (const name of Object.keys(PRESETS)) {
                    presetSource[name] = extractPresetSource(dslText, name);
                }
            } catch (error) {
                console.error('Failed to load noisemaker:', error);
                codeArea.textContent = 'Error loading Noisemaker module\n' + error.message + '\n\nMake sure to run "npm run build" before building docs.';
                return;
            }
        }

        // Validate that presets were loaded
        if (!PRESETS || Object.keys(PRESETS).length === 0) {
            codeArea.textContent = 'Error: No presets found in bundle.\n\nPRESSTS object is: ' + JSON.stringify(PRESETS, null, 2);
            select.innerHTML = '<option>No presets available</option>';
            return;
        }

        // Clear placeholder and populate preset dropdown
        select.innerHTML = '';
        const presetNames = Object.keys(PRESETS).sort();
        
        if (presetNames.length === 0) {
            codeArea.textContent = 'Error: PRESETS object is empty';
            select.innerHTML = '<option>No presets available</option>';
            return;
        }
        
        presetNames.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        });

        // Set initial preset (use first one or 'cool-water' if available)
        const initialPreset = presetNames.includes('cool-water') ? 'cool-water' : presetNames[0];
        
        if (!initialPreset) {
            codeArea.textContent = 'Error: Could not determine initial preset';
            return;
        }
        
        select.value = initialPreset;

        async function renderPreset(presetName, seed = 42) {
            if (!presetName) {
                codeArea.textContent = `Error: No preset name provided`;
                return;
            }
            
            if (!PRESETS[presetName]) {
                codeArea.textContent = `Preset "${presetName}" not found\n\nAvailable presets: ${Object.keys(PRESETS).join(', ')}`;
                return;
            }

            // Show loading - exactly like demo/js
            if (loadingIndicator) {
                loadingIndicator.textContent = 'Rendering (0%)';
                loadingIndicator.style.display = 'block';
            }
            canvas.style.opacity = '0.5';

            // Update progress callback - exactly like demo/js
            function updateRenderProgress(percent) {
                const percentInt = Math.floor(Math.min(100, Math.max(0, percent)));
                if (loadingIndicator) {
                    loadingIndicator.textContent = `Rendering (${percentInt}%)`;
                }
            }

            try {
                // Set seed
                if (rng && rng.setSeed) {
                    rng.setSeed(seed);
                }

                // Create preset exactly like demo/js does
                const presetTable = PRESETS;
                const preset = new Preset(presetName, presetTable, {}, seed, { debug: false });
                
                // If it's an effect-only preset, render on top of basic
                let renderPreset = preset;
                let renderPresets = presetTable;
                if (!preset.is_generator() && preset.is_effect()) {
                    renderPresets = PRESETS;
                    const basePreset = new Preset('basic', renderPresets, {}, seed, { debug: false });
                    basePreset.name = `${presetName}-on-basic`;
                    const combinedFinal = Array.isArray(basePreset.final_effects)
                        ? basePreset.final_effects.slice()
                        : [];
                    combinedFinal.push(preset);
                    basePreset.final_effects = combinedFinal;
                    renderPreset = basePreset;
                }
                
                // Render exactly like demo/js does
                updateRenderProgress(30);
                const ctx = new Context(canvas);
                await render(renderPreset, seed, {
                    width: canvas.width,
                    height: canvas.height,
                    ctx: ctx,
                    presets: renderPresets,
                    debug: false,
                    time: 1.0,
                    frameIndex: 0,
                    progressCallback: updateRenderProgress,
                });
                updateRenderProgress(100);

                // Show original source
                let sourceText = presetSource[presetName] || '// Source not available';
                
                // Trim 2 leading spaces from all lines except the first
                const lines = sourceText.split('\n');
                if (lines.length > 1) {
                    sourceText = lines[0] + '\n' + lines.slice(1).map(line => 
                        line.startsWith('  ') ? line.slice(2) : line
                    ).join('\n');
                }
                
                codeArea.textContent = sourceText;

                canvas.style.opacity = '1';
            } catch (error) {
                console.error('Render error:', error);
                codeArea.textContent = `Error rendering preset:\n${error.message}`;
            } finally {
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
            }
        }

        // Handle preset selection
        select.addEventListener('change', () => {
            renderPreset(select.value);
        });

        // Handle random button - exactly like javascript.rst
        if (randomButton) {
            randomButton.addEventListener('click', () => {
                const newSeed = Math.floor(Math.random() * 1000000);
                renderPreset(select.value, newSeed);
            });
        }

        // Initial render
        await renderPreset(initialPreset);
    }

    // Initialize all preset viewers on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }

    function initAll() {
        const viewers = document.querySelectorAll('.preset-viewer-container');
        viewers.forEach(viewer => {
            initPresetViewer(viewer);
        });
    }

})();
