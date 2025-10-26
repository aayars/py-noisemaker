import assert from 'assert';
import { Context } from '../js/noisemaker/context.js';

const ctx = new Context(null);
let submits = 0;
ctx.device = {
  createCommandEncoder() {
    return {
      beginComputePass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        };
      },
      beginRenderPass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          draw() {},
          end() {},
        };
      },
      finish() {
        return {};
      },
    };
  },
  pushErrorScope() {},
  popErrorScope() { return null; },
};
ctx.queue = { submit() { submits++; } };
ctx.createComputePipeline = async () => ({});
ctx.createBindGroup = () => ({});

await ctx.withEncoder(async () => {
  await ctx.runCompute('a', [], 1);
  await ctx.runCompute('b', [], 1);
});

assert.strictEqual(submits, 1);
console.log('encoder tests passed');
