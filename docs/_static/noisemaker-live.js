/**
 * Noisemaker Live Examples Runtime
 * 
 * Automatically renders all noisemaker-live-canvas elements on the page
 * using the bundled Noisemaker.js library.
 */

(function() {
    'use strict';
    
    const ENUM_PARAM_MAP = {
        distMetric: 'DistanceMetric',
        sobel_metric: 'DistanceMetric',
        sobelMetric: 'DistanceMetric',
        diagramType: 'VoronoiDiagramType',
        colorSpace: 'ColorSpace',
        color_space: 'ColorSpace',
        behavior: 'WormBehavior',
        pointDistrib: 'PointDistribution',
        mask: 'ValueMask',
        spline_order: 'InterpolationType',
        splineOrder: 'InterpolationType'
    };

    function getEnumForParam(paramName, effectName, defaultValue) {
        if (!window.Noisemaker) {
            return null;
        }

        if (paramName === 'distrib' || paramName === 'hue_distrib' || paramName === 'brightness_distrib' || paramName === 'saturation_distrib') {
            if (typeof defaultValue === 'number' && defaultValue >= 1000000) {
                return 'PointDistribution';
            }
            return 'ValueDistribution';
        }

        if (paramName === 'name' && effectName === 'palette') {
            return 'PALETTES';
        }

        return ENUM_PARAM_MAP[paramName] ?? null;
    }

    function buildEnumOptions(enumSource) {
        const nm = window.Noisemaker;
        if (enumSource === 'PALETTES') {
            const palettes = nm?.PALETTES ?? {};
            return Object.keys(palettes)
                .sort((a, b) => a.localeCompare(b))
                .map(name => ({ value: name, label: humanizeLabel(name) }));
        }

        const enumObject = nm?.[enumSource] ?? nm?.enums?.[enumSource];
        if (!enumObject) {
            return [];
        }

        return Object.entries(enumObject)
            .sort((a, b) => {
                const [, valueA] = a;
                const [, valueB] = b;
                if (typeof valueA === 'number' && typeof valueB === 'number') {
                    return valueA - valueB;
                }
                return String(valueA).localeCompare(String(valueB));
            })
            .map(([key, value]) => ({
                value: String(value),
                label: humanizeLabel(key)
            }));
    }

    function humanizeLabel(value) {
        return String(value)
            .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
            .replace(/[_-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .toLowerCase();
    }

    function resolveValue(rawValue, defaultValue) {
        if (rawValue === undefined) {
            return defaultValue;
        }

        if (typeof defaultValue === 'boolean') {
            return rawValue === 'true';
        }

        if (typeof defaultValue === 'number') {
            const parsed = Number(rawValue);
            return Number.isFinite(parsed) ? parsed : defaultValue;
        }

        if (defaultValue === null || defaultValue === undefined) {
            return rawValue;
        }

        return rawValue;
    }

    function coerceParamValue(rawValue, defaultValue, enumSource) {
        if (rawValue === undefined) {
            return undefined;
        }

        if (enumSource === 'PALETTES') {
            return rawValue;
        }

        if (enumSource) {
            const parsedEnumValue = Number(rawValue);
            return Number.isNaN(parsedEnumValue) ? undefined : parsedEnumValue;
        }

        if (typeof defaultValue === 'boolean') {
            return rawValue === 'true';
        }

        if (typeof defaultValue === 'number') {
            const parsedNumber = Number(rawValue);
            return Number.isNaN(parsedNumber) ? undefined : parsedNumber;
        }

        if (typeof defaultValue === 'string') {
            return rawValue;
        }

        if (defaultValue === null || defaultValue === undefined) {
            if (rawValue === 'true' || rawValue === 'false') {
                return rawValue === 'true';
            }
            const parsedFallback = Number(rawValue);
            return Number.isNaN(parsedFallback) ? rawValue : parsedFallback;
        }

        return rawValue;
    }

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
            const example = canvas.closest('.noisemaker-live-example');
            const randomBtn = wrapper?.querySelector('.noisemaker-live-random');
            const controlsContainer = example?.querySelector('.noisemaker-live-controls');
            
            if (randomBtn) {
                randomBtn.addEventListener('click', () => {
                    // Generate new random seed
                    const newSeed = Math.floor(Math.random() * 100000);
                    canvas.dataset.seed = newSeed;
                    renderCanvas(canvas);
                });
            }
            
            // Generate parameter controls for effects or generators
            const effectName = canvas.dataset.effect;
            const generatorName = canvas.dataset.generator;
            
            if (effectName && controlsContainer && window.Noisemaker?.EFFECTS) {
                const effectNameSnake = effectName.replace(/-/g, '_');
                const effect = window.Noisemaker.EFFECTS[effectName] || window.Noisemaker.EFFECTS[effectNameSnake];
                
                if (effect) {
                    // Clear existing controls (except structure)
                    controlsContainer.innerHTML = '';
                    
                    // Get parameter names (everything except 'func')
                    const paramNames = Object.keys(effect).filter(k => k !== 'func');
                    
                    if (paramNames.length > 0) {
                        paramNames.forEach(paramName => {
                            const defaultValue = effect[paramName];
                            const { dataset } = canvas;
                            const rawDatasetValue = dataset[paramName];

                            const control = document.createElement('div');
                            control.className = 'noisemaker-live-control';

                            const label = document.createElement('label');
                            label.textContent = humanizeLabel(paramName);
                            control.appendChild(label);

                            const enumSource = getEnumForParam(paramName, effectName, defaultValue);
                            let resolvedValue = resolveValue(rawDatasetValue, defaultValue);
                            let input;

                            if (enumSource) {
                                input = document.createElement('select');
                                input.className = `control-${paramName}`;

                                const options = buildEnumOptions(enumSource);

                                // Add a null/auto option if the default is null or undefined
                                if (defaultValue === null || defaultValue === undefined) {
                                    const noneOption = document.createElement('option');
                                    noneOption.value = '';
                                    noneOption.textContent = 'auto';
                                    input.appendChild(noneOption);
                                    if (rawDatasetValue === undefined) {
                                        resolvedValue = '';
                                    }
                                }

                                options.forEach(({ value, label: optionLabel }) => {
                                    const option = document.createElement('option');
                                    option.value = value;
                                    option.textContent = optionLabel;
                                    input.appendChild(option);
                                });

                                input.value = String(resolvedValue ?? '');

                                input.addEventListener('change', (e) => {
                                    const newValue = e.target.value;
                                    if (newValue === '') {
                                        delete dataset[paramName];
                                    } else {
                                        dataset[paramName] = newValue;
                                    }
                                    renderCanvas(canvas);
                                });

                                control.appendChild(input);
                            } else if (typeof defaultValue === 'boolean') {
                                // Checkbox for boolean
                                input = document.createElement('input');
                                input.type = 'checkbox';
                                input.className = `control-${paramName}`;
                                input.checked = resolvedValue;
                                
                                input.addEventListener('change', (e) => {
                                    dataset[paramName] = String(e.target.checked);
                                    renderCanvas(canvas);
                                });
                                
                                control.appendChild(input);
                            } else if (typeof defaultValue === 'string') {
                                // Text input for string
                                input = document.createElement('input');
                                input.type = 'text';
                                input.className = `control-${paramName}`;
                                input.value = resolvedValue ?? '';
                                input.style.width = '120px';
                                
                                input.addEventListener('change', (e) => {
                                    dataset[paramName] = e.target.value;
                                    renderCanvas(canvas);
                                });
                                
                                control.appendChild(input);
                            } else if (typeof defaultValue === 'number') {
                                // Determine if integer or float
                                const isInteger = Number.isInteger(defaultValue);
                                
                                // Range slider for number with smart defaults
                                input = document.createElement('input');
                                input.type = 'range';
                                input.className = `control-${paramName}`;
                                
                                // Determine sensible ranges based on parameter name patterns and values
                                let min, max, step;
                                
                                // Alpha/opacity parameters: 0-1
                                if (paramName === 'alpha') {
                                    min = 0; max = 1; step = 0.01;
                                }
                                // Frequency parameters: 1-20
                                else if (paramName.includes('freq') || paramName.includes('Freq')) {
                                    min = 1; max = 20; step = 1;
                                }
                                // Octaves: 1-8
                                else if (paramName === 'octaves') {
                                    min = 1; max = 8; step = 1;
                                }
                                // Iterations: 1-200
                                else if (paramName === 'iterations') {
                                    min = 1; max = 200; step = 1;
                                }
                                // Displacement: -2 to 2 for most, special cases handled
                                else if (paramName === 'displacement') {
                                    if (defaultValue >= 10) { min = 0; max = 100; step = 1; }
                                    else { min = -2; max = 2; step = 0.01; }
                                }
                                // Amount parameters: typically 0-2
                                else if (paramName === 'amount') {
                                    min = 0; max = 2; step = 0.01;
                                }
                                // Density: 0-100
                                // Angle: 0-360
                                else if (paramName === 'angle') {
                                    min = 0; max = 360; step = 1;
                                }
                                // Sides: 3-12
                                else if (paramName === 'sides' || paramName === 'sdfSides') {
                                    min = 3; max = 12; step = 1;
                                }
                                // Levels: 2-32
                                else if (paramName === 'levels') {
                                    min = 2; max = 32; step = 1;
                                }
                                // Kink: 0-5
                                else if (paramName === 'kink') {
                                    min = 0; max = 5; step = 0.1;
                                }
                                // Spline order: 0-5
                                else if (paramName === 'splineOrder') {
                                    min = 0; max = 5; step = 1;
                                }
                                // Zoom: 0.1-5
                                else if (paramName === 'zoom') {
                                    min = 0.1; max = 5; step = 0.1;
                                }
                                // Saturation/hue ranges: 0-2
                                else if (paramName === 'saturation' || paramName === 'hueRange' || paramName === 'hueRotation') {
                                    min = 0; max = 2; step = 0.01;
                                }
                                // Smoothstep parameters: 0-1
                                else if (paramName === 'a' || paramName === 'b') {
                                    min = 0; max = 1; step = 0.01;
                                }
                                // Large integers (distrib, kernel, etc): use value-based range
                                else if (defaultValue > 100) {
                                    min = 0; 
                                    max = Math.max(10000, defaultValue * 2); 
                                    step = isInteger ? Math.max(1, Math.floor(defaultValue / 100)) : Math.max(0.1, defaultValue / 100);
                                }
                                // 0-1 normalized values
                                else if (defaultValue >= 0 && defaultValue <= 1) {
                                    min = 0; max = 1; step = isInteger ? 1 : 0.01;
                                }
                                // Small integers 1-10
                                else if (defaultValue >= 1 && defaultValue <= 10 && isInteger) {
                                    min = 1; max = 20; step = 1;
                                }
                                // Larger positive numbers
                                else if (defaultValue > 10) {
                                    min = 0; 
                                    max = Math.ceil(defaultValue * 3); 
                                    step = isInteger ? Math.max(1, Math.floor(defaultValue / 10)) : Math.max(0.1, defaultValue / 10);
                                }
                                // Negative or around zero
                                else {
                                    min = Math.floor(defaultValue - Math.abs(defaultValue) * 2 - 1);
                                    max = Math.ceil(defaultValue + Math.abs(defaultValue) * 2 + 1);
                                    step = isInteger ? 1 : 0.01;
                                }
                                
                                input.min = String(min);
                                input.max = String(max);
                                input.step = String(step);
                                
                                const currentValue = typeof resolvedValue === 'number' ? resolvedValue : defaultValue;
                                input.value = String(currentValue);
                                
                                const valueDisplay = document.createElement('span');
                                valueDisplay.className = 'control-value';
                                const decimals = isInteger || step >= 1 ? 0 : (step >= 0.1 ? 1 : 2);
                                valueDisplay.textContent = currentValue.toFixed(decimals);
                                
                                input.addEventListener('input', (e) => {
                                    const newValue = parseFloat(e.target.value);
                                    valueDisplay.textContent = newValue.toFixed(decimals);
                                });
                                
                                input.addEventListener('change', (e) => {
                                    const rawValue = parseFloat(e.target.value);
                                    const newValue = isInteger ? Math.round(rawValue) : rawValue;
                                    dataset[paramName] = String(newValue);
                                    renderCanvas(canvas);
                                });
                                
                                control.appendChild(input);
                                control.appendChild(valueDisplay);
                            } else {
                                // Null or object - show type and value
                                const valueSpan = document.createElement('span');
                                valueSpan.className = 'control-value';
                                valueSpan.textContent = defaultValue === null ? 'null' : `[${typeof defaultValue}]`;
                                control.appendChild(valueSpan);
                            }
                            
                            controlsContainer.appendChild(control);
                        });
                        
                        if (controlsContainer.children.length === 0) {
                            controlsContainer.innerHTML = '<p class="no-params-message">No adjustable parameters</p>';
                        }
                    } else {
                        controlsContainer.innerHTML = '<p class="no-params-message">No adjustable parameters</p>';
                    }
                }
            } else if (generatorName && controlsContainer && window.Noisemaker) {
                // Generator controls
                const generator = window.Noisemaker[generatorName];
                
                if (generator) {
                    // Define generator parameter defaults
                    const GENERATOR_PARAMS = {
                        basic: {
                            ridges: false,
                            sin: 0.0,
                            spline_order: 3,
                            distrib: 1,
                            corners: false,
                            mask: null,
                            mask_inverse: false,
                            mask_static: false,
                            lattice_drift: 0.0,
                            color_space: 21,
                            hue_range: 0.125,
                            hue_rotation: null,
                            saturation: 1.0,
                            hue_distrib: null,
                            brightness_distrib: null,
                            brightness_freq: null,
                            saturation_distrib: null
                        },
                        multires: {
                            freq: 3,
                            octaves: 1,
                            ridges: false,
                            sin: 0.0,
                            spline_order: 3,
                            distrib: 1,
                            corners: false,
                            mask: null,
                            mask_inverse: false,
                            mask_static: false,
                            lattice_drift: 0.0,
                            color_space: 21,
                            hue_range: 0.125,
                            hue_rotation: null,
                            saturation: 1.0,
                            hue_distrib: null,
                            saturation_distrib: null,
                            brightness_distrib: null,
                            brightness_freq: null
                        }
                    };
                    
                    const params = GENERATOR_PARAMS[generatorName];
                    if (params) {
                        controlsContainer.innerHTML = '';
                        
                        Object.keys(params).forEach(paramName => {
                            const defaultValue = params[paramName];
                            const { dataset } = canvas;
                            const rawDatasetValue = dataset[paramName];

                            const control = document.createElement('div');
                            control.className = 'noisemaker-live-control';

                            const label = document.createElement('label');
                            label.textContent = humanizeLabel(paramName);
                            control.appendChild(label);

                            const enumSource = getEnumForParam(paramName, generatorName, defaultValue);
                            let resolvedValue = resolveValue(rawDatasetValue, defaultValue);
                            let input;

                            if (enumSource) {
                                input = document.createElement('select');
                                input.className = `control-${paramName}`;

                                const options = buildEnumOptions(enumSource);

                                if (defaultValue === null || defaultValue === undefined) {
                                    const noneOption = document.createElement('option');
                                    noneOption.value = '';
                                    noneOption.textContent = 'auto';
                                    input.appendChild(noneOption);
                                    if (rawDatasetValue === undefined) {
                                        resolvedValue = '';
                                    }
                                }

                                options.forEach(({ value, label: optionLabel }) => {
                                    const option = document.createElement('option');
                                    option.value = value;
                                    option.textContent = optionLabel;
                                    input.appendChild(option);
                                });

                                input.value = String(resolvedValue ?? '');

                                input.addEventListener('change', (e) => {
                                    const newValue = e.target.value;
                                    if (newValue === '') {
                                        delete dataset[paramName];
                                    } else {
                                        dataset[paramName] = newValue;
                                    }
                                    renderCanvas(canvas);
                                });

                                control.appendChild(input);
                            } else if (typeof defaultValue === 'boolean') {
                                input = document.createElement('input');
                                input.type = 'checkbox';
                                input.className = `control-${paramName}`;
                                input.checked = resolvedValue;
                                
                                input.addEventListener('change', (e) => {
                                    dataset[paramName] = String(e.target.checked);
                                    renderCanvas(canvas);
                                });
                                
                                control.appendChild(input);
                            } else if (typeof defaultValue === 'number') {
                                const isInteger = Number.isInteger(defaultValue);
                                
                                input = document.createElement('input');
                                input.type = 'range';
                                input.className = `control-${paramName}`;
                                
                                let min, max, step;
                                
                                if (paramName === 'sin' || paramName === 'lattice_drift') {
                                    min = 0; max = 1; step = 0.01;
                                } else if (paramName === 'hue_range' || paramName === 'saturation') {
                                    min = 0; max = 2; step = 0.01;
                                } else if (paramName === 'freq') {
                                    min = 1; max = 20; step = 1;
                                } else if (paramName === 'octaves') {
                                    min = 1; max = 8; step = 1;
                                } else if (defaultValue >= 0 && defaultValue <= 1) {
                                    min = 0; max = 1; step = isInteger ? 1 : 0.01;
                                } else if (defaultValue >= 1 && defaultValue <= 10 && isInteger) {
                                    min = 1; max = 20; step = 1;
                                } else {
                                    min = 0; 
                                    max = Math.ceil(defaultValue * 3); 
                                    step = isInteger ? Math.max(1, Math.floor(defaultValue / 10)) : 0.01;
                                }
                                
                                input.min = String(min);
                                input.max = String(max);
                                input.step = String(step);
                                
                                const currentValue = typeof resolvedValue === 'number' ? resolvedValue : defaultValue;
                                input.value = String(currentValue);
                                
                                const valueDisplay = document.createElement('span');
                                valueDisplay.className = 'control-value';
                                const decimals = isInteger || step >= 1 ? 0 : (step >= 0.1 ? 1 : 2);
                                valueDisplay.textContent = currentValue.toFixed(decimals);
                                
                                input.addEventListener('input', (e) => {
                                    const newValue = parseFloat(e.target.value);
                                    valueDisplay.textContent = newValue.toFixed(decimals);
                                });
                                
                                input.addEventListener('change', (e) => {
                                    const rawValue = parseFloat(e.target.value);
                                    const newValue = isInteger ? Math.round(rawValue) : rawValue;
                                    dataset[paramName] = String(newValue);
                                    renderCanvas(canvas);
                                });
                                
                                control.appendChild(input);
                                control.appendChild(valueDisplay);
                            }
                            
                            controlsContainer.appendChild(control);
                        });
                    }
                }
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
            const generatorName = canvas.dataset.generator;
            const inputName = canvas.dataset.input || 'basic';
            const seed = parseInt(canvas.dataset.seed, 10) || 42;
            const width = parseInt(canvas.dataset.width, 10) || 512;
            const height = parseInt(canvas.dataset.height, 10) || 512;
            const time = parseFloat(canvas.dataset.time) || 1.0;
            const frame = parseFloat(canvas.dataset.frame) || 0.0;
            
            if (!presetName && !effectName && !generatorName) {
                throw new Error('No preset, effect, or generator specified');
            }
            
            // Update progress callback - exactly like demo/js
            function updateRenderProgress(percent) {
                const percentInt = Math.floor(Math.min(100, Math.max(0, percent)));
                if (loadingDiv) {
                    loadingDiv.textContent = `Rendering (${percentInt}%)`;
                }
            }
            
            // Show loading state - exactly like demo/js
            if (loadingDiv) {
                loadingDiv.textContent = 'Rendering (0%)';
                loadingDiv.style.display = 'block';
            }
            canvas.style.opacity = '0.3';
            
            // Create preset and render
            // Note: The bundled API requires loading the preset table first
            const { Preset, PRESETS } = window.Noisemaker;
            
            const PRESET_TABLE = PRESETS();
            
            let tensor;
            const startTime = performance.now();
            
            // If rendering a generator directly
            if (generatorName) {
                const generator = window.Noisemaker[generatorName];
                if (!generator) {
                    throw new Error(`Generator "${generatorName}" not found`);
                }
                
                const { dataset } = canvas;
                const opts = {};
                
                // Collect custom parameters from dataset
                const GENERATOR_PARAMS = {
                    basic: ['ridges', 'sin', 'spline_order', 'distrib', 'corners', 'mask', 'mask_inverse', 'mask_static', 'lattice_drift', 'color_space', 'hue_range', 'hue_rotation', 'saturation', 'hue_distrib', 'brightness_distrib', 'brightness_freq', 'saturation_distrib'],
                    multires: ['freq', 'octaves', 'ridges', 'sin', 'spline_order', 'distrib', 'corners', 'mask', 'mask_inverse', 'mask_static', 'lattice_drift', 'color_space', 'hue_range', 'hue_rotation', 'saturation', 'hue_distrib', 'saturation_distrib', 'brightness_distrib', 'brightness_freq']
                };
                
                const paramNames = GENERATOR_PARAMS[generatorName] || [];
                paramNames.forEach(paramName => {
                    if (dataset[paramName] === undefined) {
                        return;
                    }
                    
                    const defaultValue = paramName === 'ridges' ? false :
                                       paramName === 'corners' ? false :
                                       paramName === 'mask_inverse' ? false :
                                       paramName === 'mask_static' ? false :
                                       paramName === 'freq' ? 3 :
                                       paramName === 'octaves' ? 1 :
                                       paramName === 'sin' ? 0.0 :
                                       paramName === 'lattice_drift' ? 0.0 :
                                       paramName === 'hue_range' ? 0.125 :
                                       paramName === 'saturation' ? 1.0 :
                                       paramName === 'distrib' ? 1 :
                                       paramName === 'spline_order' ? 3 :
                                       paramName === 'color_space' ? 21 :
                                       paramName === 'mask' ? null :
                                       paramName === 'hue_rotation' ? null :
                                       paramName === 'hue_distrib' ? null :
                                       paramName === 'brightness_distrib' ? null :
                                       paramName === 'brightness_freq' ? null :
                                       paramName === 'saturation_distrib' ? null : null;
                    const enumSource = getEnumForParam(paramName, generatorName, defaultValue);
                    const coercedValue = coerceParamValue(dataset[paramName], defaultValue, enumSource);
                    
                    if (coercedValue !== undefined && coercedValue !== null) {
                        // Skip setting null-defaulted enum params when empty
                        if (enumSource && defaultValue === null && (coercedValue === '' || coercedValue === null)) {
                            return;
                        }
                        // Convert snake_case to camelCase for JS API
                        const jsParamName = paramName.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
                        opts[jsParamName] = coercedValue;
                    }
                });
                
                opts.time = time;
                opts.seed = seed;
                
                const shape = [height, width, 3];
                const freqValue = Number(opts.freq) || 3;
                const freq = [freqValue, freqValue];
                delete opts.freq;
                
                tensor = await generator(freq, shape, opts);
            }
            // If rendering an effect, we need to render the input then apply the effect
            else if (effectName) {
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
                updateRenderProgress(30);
                const inputTensor = await inputPreset.render(seed, {
                    width: width,
                    height: height,
                    time: time,
                    speed: 1.0,
                    progressCallback: updateRenderProgress,
                });
                updateRenderProgress(70);
                
                // Step 2: Collect custom parameters from canvas dataset
                const effect = EFFECTS[effectName] || EFFECTS[effectNameSnake];
                const customParams = {};

                // Get all parameter names from the effect (excluding 'func')
                const paramNames = Object.keys(effect).filter(k => k !== 'func');
                paramNames.forEach(paramName => {
                    const rawValue = canvas.dataset[paramName];
                    if (rawValue === undefined) {
                        return;
                    }

                    const defaultValue = effect[paramName];
                    const enumSource = getEnumForParam(paramName, effectName, defaultValue);
                    const coercedValue = coerceParamValue(rawValue, defaultValue, enumSource);

                    if (coercedValue !== undefined && coercedValue !== null && !(enumSource && rawValue === '')) {
                        customParams[paramName] = coercedValue;
                    }
                });
                
                // Step 3: Create an effect function with custom parameters
                const effectFunc = Effect(effectName, customParams);
                
                // Step 4: Apply the effect function to the tensor
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
                updateRenderProgress(30);
                tensor = await preset.render(seed, {
                    width: width,
                    height: height,
                    time: time,
                    speed: 1.0,
                    progressCallback: updateRenderProgress,
                });
                updateRenderProgress(100);
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
