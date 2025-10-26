export class UnusedKeys extends Error {}

function toCamel(str) {
  return str.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

export class SettingsDict {
  constructor(obj = {}) {
    this.__data = { ...obj };
    this.__accessed = {};
    const handler = {
      get: (target, prop, receiver) => {
        if (prop in target) {
          return Reflect.get(target, prop, receiver);
        }
        const key = typeof prop === 'string' ? toCamel(prop) : prop;
        target.__accessed[key] = true;
        return target.__data[key];
      },
      set: (target, prop, value) => {
        if (prop in target) {
          target[prop] = value;
        } else {
          const key = typeof prop === 'string' ? toCamel(prop) : prop;
          target.__data[key] = value;
        }
        return true;
      },
      has: (target, prop) => {
        const key = typeof prop === 'string' ? toCamel(prop) : prop;
        return prop in target || key in target.__data;
      },
      deleteProperty: (target, prop) => {
        const key = typeof prop === 'string' ? toCamel(prop) : prop;
        delete target.__data[key];
        delete target.__accessed[key];
        return true;
      },
      ownKeys: (target) => {
        return Reflect.ownKeys(target.__data);
      },
      getOwnPropertyDescriptor: (target, prop) => {
        const key = typeof prop === 'string' ? toCamel(prop) : prop;
        if (key in target.__data) {
          return Object.getOwnPropertyDescriptor(target.__data, key);
        }
        return undefined;
      },
    };
    return new Proxy(this, handler);
  }

  wasAccessed(key) {
    return Object.prototype.hasOwnProperty.call(this.__accessed, key);
  }

  raiseIfUnaccessed(unusedOkay = []) {
    const keys = [];
    for (const key of Object.keys(this.__data)) {
      if (!this.wasAccessed(key) && !unusedOkay.includes(key)) {
        keys.push(key);
      }
    }
    if (keys.length === 1) {
      const k = keys[0];
      throw new UnusedKeys(`Settings key "${k}" (value: ${this.__data[k]}) is unused. This is usually human error.`);
    } else if (keys.length > 1) {
      throw new UnusedKeys(`Settings keys ${keys} are unused. This is usually human error.`);
    }
  }
}
