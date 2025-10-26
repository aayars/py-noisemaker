class EffectUIGenerator {
  constructor(container, options = {}) {
    if (!container) {
      throw new Error('EffectUIGenerator requires a container element.');
    }

    this.container = container;
    this.onChange = options.onChange ?? null;
    this.controls = new Map();
    this.metadata = null;
  }

  render(metadata, initialValues = {}) {
    this.metadata = metadata ?? {};
    const params = Array.isArray(this.metadata.parameters) ? this.metadata.parameters : [];

    this.container.textContent = '';
    this.controls.clear();

    // Forbidden parameters that should not be exposed as UI controls
    const forbiddenParams = new Set(['enabled', 'width', 'height', 'channels', 'time']);

    params.forEach((param) => {
      if (forbiddenParams.has(param.name)) return; // Skip forbidden parameters

      const value = this.#resolveInitialValue(param, initialValues[param.name]);
      const control = this.#createControl(param, value);
      this.container.appendChild(control.group);
      this.controls.set(param.name, control);
    });
  }

  applyValues(values = {}, options = {}) {
    const { silent = true } = options;
    Object.entries(values).forEach(([name, rawValue]) => {
      this.setValue(name, rawValue, { silent });
    });
  }

  setValue(name, rawValue, options = {}) {
    const control = this.controls.get(name);
    if (!control) {
      return;
    }

    const value = control.coerce(rawValue);
    this.#setControlValue(control, value);

    const { silent = true } = options;
    if (!silent && typeof this.onChange === 'function') {
      try {
        const result = this.onChange(name, value, { source: 'programmatic' });
        if (result?.catch) {
          result.catch((error) => console.error(`EffectUIGenerator onChange failed for ${name}:`, error));
        }
      } catch (error) {
        console.error(`EffectUIGenerator onChange failed for ${name}:`, error);
      }
    }
  }

  getCurrentValues() {
    const values = {};
    this.controls.forEach((control, name) => {
      values[name] = control.coerce(control.input.value);
    });
    return values;
  }

  getDefaults() {
    const defaults = {};
    if (!this.metadata) {
      return defaults;
    }
    const params = Array.isArray(this.metadata.parameters) ? this.metadata.parameters : [];
    params.forEach((param) => {
      defaults[param.name] = this.#resolveInitialValue(param, undefined);
    });
    return defaults;
  }

  #createControl(param, value) {
    const group = document.createElement('div');
    group.className = 'control-group';

    const header = document.createElement('div');
    header.className = 'control-header';

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = param.label ?? this.#toTitleCase(param.name);

    const valueEl = document.createElement('span');
    valueEl.className = 'control-value';

    header.appendChild(label);
    header.appendChild(valueEl);
    group.appendChild(header);

    const input = this.#createInput(param);
    group.appendChild(input);

    const control = {
      param,
      group,
      header,
      label,
      valueEl,
      input,
      decimals: this.#inferDecimalPlaces(param),
      coerce: (raw) => this.#coerceValue(param, raw),
      format: (val) => this.#formatValue(param, val),
      min: this.#resolveNumericBoundary(param.min),
      max: this.#resolveNumericBoundary(param.max),
    };

    this.#setControlValue(control, control.coerce(value));

    input.addEventListener('input', (event) => {
      const coerced = control.coerce(event.target.value);
      this.#setControlValue(control, coerced);
      if (typeof this.onChange === 'function') {
        try {
          const result = this.onChange(param.name, coerced, { source: 'input' });
          if (result?.catch) {
            result.catch((error) => console.error(`EffectUIGenerator onChange failed for ${param.name}:`, error));
          }
        } catch (error) {
          console.error(`EffectUIGenerator onChange failed for ${param.name}:`, error);
        }
      }
    });

    return control;
  }

  #createInput(param) {
    switch (param.type) {
      case 'boolean':
        return this.#createBooleanInput(param);
      case 'float':
      case 'number':
      case 'int':
      default:
        return this.#createNumberInput(param);
    }
  }

  #createBooleanInput(param) {
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1';
    input.step = '1';
    input.className = 'control-slider';
    input.name = param.name;
    input.setAttribute('aria-label', param.label ?? param.name);
    return input;
  }

  #createNumberInput(param) {
    const input = document.createElement('input');
    input.type = 'range';

    const min = Number.isFinite(param.min) ? param.min : 0;
    const max = Number.isFinite(param.max) ? param.max : 1;
    const step = Number.isFinite(param.step) && param.step > 0 ? param.step : 0.01;

    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.className = 'control-slider';
    input.name = param.name;
    input.setAttribute('aria-label', param.label ?? param.name);
    return input;
  }

  #setControlValue(control, value) {
    const { input, valueEl, format, param } = control;

    let displayValue = value;
    if (param.type === 'boolean') {
      input.value = value ? '1' : '0';
    } else {
      let numericValue = Number(value);
      if (Number.isFinite(control.min)) {
        numericValue = Math.max(control.min, numericValue);
      }
      if (Number.isFinite(control.max)) {
        numericValue = Math.min(control.max, numericValue);
      }
      displayValue = numericValue;
      input.value = String(numericValue);
    }

    valueEl.textContent = format(displayValue);
  }

  #coerceValue(param, rawValue) {
    switch (param.type) {
      case 'boolean':
        if (typeof rawValue === 'boolean') {
          return rawValue;
        }
        if (typeof rawValue === 'number') {
          return rawValue >= 0.5;
        }
        return rawValue === '1' || rawValue === 'true' || rawValue === 1;
      case 'int':
        return Math.round(Number(rawValue));
      case 'float':
      case 'number':
      default: {
        const numeric = Number(rawValue);
        return Number.isFinite(numeric) ? numeric : 0;
      }
    }
  }

  #formatValue(param, value) {
    switch (param.type) {
      case 'boolean':
        return value ? 'On' : 'Off';
      case 'int':
        return String(Math.round(value));
      case 'float':
      case 'number':
      default: {
        const decimals = this.#inferDecimalPlaces(param);
        return Number(value).toFixed(decimals);
      }
    }
  }

  #resolveInitialValue(param, explicitValue) {
    if (typeof explicitValue !== 'undefined') {
      return explicitValue;
    }
    if (typeof param.default !== 'undefined') {
      return param.default;
    }
    switch (param.type) {
      case 'boolean':
        return false;
      case 'int':
      case 'number':
      case 'float':
      default:
        return 0;
    }
  }

  #inferDecimalPlaces(param) {
    if (param.type === 'boolean' || param.type === 'int') {
      return 0;
    }
    const step = Number(param.step);
    if (Number.isFinite(step) && step > 0) {
      const decimals = Math.round(Math.log10(1 / step));
      return Math.max(0, Math.min(decimals, 6));
    }
    return 3;
  }

  #resolveNumericBoundary(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : undefined;
  }

  #toTitleCase(value) {
    return String(value ?? '')
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/(^|\s)\w/g, (m) => m.toUpperCase());
  }
}

export default EffectUIGenerator;
