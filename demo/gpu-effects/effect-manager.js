const DEFAULT_EFFECT_EXPORT = 'default';

class EffectManager {
  constructor({ helpers } = {}) {
    this.helpers = helpers ?? {};
    this.effects = new Map();
    this.activeEffectId = null;
    this.activeEffectInstance = null;
  }

  registerEffect(descriptor) {
    const { id, label, metadata, loadModule } = descriptor ?? {};
    if (!id) {
      throw new Error('EffectManager.registerEffect requires an id.');
    }
    if (this.effects.has(id)) {
      throw new Error(`Effect '${id}' is already registered.`);
    }
    if (typeof loadModule !== 'function') {
      throw new Error(`Effect '${id}' must provide a loadModule() function.`);
    }

    this.effects.set(id, {
      id,
      label: label ?? metadata?.label ?? id,
      metadata: metadata ?? null,
      loadModule,
      module: null,
      effectClass: null,
    });

    if (!this.activeEffectId) {
      this.activeEffectId = id;
    }
  }

  getAvailableEffects() {
    return Array.from(this.effects.values()).map(({ id, label, metadata }) => ({
      id,
      label: label ?? id,
      metadata,
    }));
  }

  getActiveEffectId() {
    return this.activeEffectId;
  }

  getEffectDescriptor(effectId = this.activeEffectId) {
    if (!effectId) {
      return null;
    }
    return this.effects.get(effectId) ?? null;
  }

  getEffectMetadata(effectId = this.activeEffectId) {
    const descriptor = this.getEffectDescriptor(effectId);
    if (!descriptor) {
      return null;
    }
    if (descriptor.metadata) {
      return descriptor.metadata;
    }
    const effectClass = descriptor.effectClass;
    if (effectClass?.metadata) {
      descriptor.metadata = effectClass.metadata;
      descriptor.label = descriptor.label ?? effectClass.label ?? effectClass.metadata?.label ?? descriptor.id;
      return descriptor.metadata;
    }
    return null;
  }

  async ensureActiveEffect(effectId) {
    const targetId = effectId ?? this.activeEffectId ?? this.#getFirstRegisteredId();
    if (!targetId) {
      throw new Error('EffectManager.ensureActiveEffect called with no registered effects.');
    }

    if (this.activeEffectId !== targetId) {
      await this.setActiveEffect(targetId);
    }

    if (!this.activeEffectInstance) {
      await this.#instantiateActiveEffect(this.activeEffectId);
    }

    return this.activeEffectInstance;
  }

  async setActiveEffect(effectId) {
    if (!effectId) {
      throw new Error('EffectManager.setActiveEffect requires an effect id.');
    }
    if (!this.effects.has(effectId)) {
      throw new Error(`Effect '${effectId}' is not registered.`);
    }
    if (this.activeEffectId === effectId && this.activeEffectInstance) {
      return this.activeEffectInstance;
    }

    await this.#disposeActiveEffect();
    this.activeEffectId = effectId;
    return this.ensureActiveEffect(effectId);
  }

  async updateActiveParams(updates = {}) {
    if (!updates || typeof updates !== 'object') {
      throw new TypeError('EffectManager.updateActiveParams expects an object.');
    }
    const effect = await this.ensureActiveEffect();
    if (!effect || typeof effect.updateParams !== 'function') {
      throw new Error('Active effect does not support parameter updates.');
    }
    return effect.updateParams(updates);
  }

  async getActiveUIState() {
    const effect = await this.ensureActiveEffect();
    if (effect && typeof effect.getUIState === 'function') {
      return effect.getUIState();
    }
    return {};
  }

  invalidateActiveEffectResources() {
    if (!this.activeEffectInstance) {
      return;
    }
    const { logWarn } = this.helpers ?? {};
    const effect = this.activeEffectInstance;
    if (typeof effect.invalidateResources === 'function') {
      try {
        effect.invalidateResources();
      } catch (error) {
        logWarn?.(`Failed to invalidate resources for effect '${this.activeEffectId}':`, error);
      }
    }
  }

  #getFirstRegisteredId() {
    const iterator = this.effects.keys();
    const { value } = iterator.next();
    return value ?? null;
  }

  async #instantiateActiveEffect(effectId) {
    const descriptor = this.effects.get(effectId);
    if (!descriptor) {
      throw new Error(`Cannot instantiate unregistered effect '${effectId}'.`);
    }

    if (!descriptor.effectClass) {
      const module = await descriptor.loadModule();
      descriptor.module = module;
      const exported = module?.[DEFAULT_EFFECT_EXPORT] ?? module?.default ?? module;
      if (typeof exported !== 'function') {
        throw new Error(`Effect module for '${effectId}' does not export a class.`);
      }
      descriptor.effectClass = exported;
      if (!descriptor.metadata && exported.metadata) {
        descriptor.metadata = exported.metadata;
      }
      descriptor.label = descriptor.label ?? descriptor.metadata?.label ?? exported.label ?? effectId;
    }

    this.activeEffectInstance = new descriptor.effectClass({ helpers: this.helpers });
    return this.activeEffectInstance;
  }

  async #disposeActiveEffect() {
    if (!this.activeEffectInstance) {
      return;
    }

    const effect = this.activeEffectInstance;
    const { logWarn } = this.helpers ?? {};

    if (typeof effect.invalidateResources === 'function') {
      try {
        effect.invalidateResources();
      } catch (error) {
        logWarn?.(`Failed to invalidate resources for effect '${this.activeEffectId}':`, error);
      }
    }

    if (typeof effect.destroy === 'function') {
      try {
        effect.destroy();
      } catch (error) {
        logWarn?.(`Failed to destroy effect '${this.activeEffectId}':`, error);
      }
    }

    this.activeEffectInstance = null;
  }
}

export default EffectManager;
