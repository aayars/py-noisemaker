// Multires is a generator, not a standard effect
export default class MultiresGenerator {
  constructor() {
    this.id = 'multires';
  }
}

// Export additional pass descriptors if this is a multi-pass generator
export const additionalPasses = {};
