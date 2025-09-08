import { ValueMask } from './constants.js';

// Minimal hard-coded bitmap masks
export const Masks = {
  [ValueMask.chess]: [
    [0, 1],
    [1, 0],
  ],
  [ValueMask.waffle]: [
    [0, 1],
    [1, 1],
  ],
  [ValueMask.square]: [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  [ValueMask.h_hex]: [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
  ],
  [ValueMask.v_hex]: [
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
  ],
};

export function maskShape(mask) {
  const m = Masks[mask];
  const height = m.length;
  const width = m[0].length;
  const channels = Array.isArray(m[0][0]) ? m[0][0].length : 1;
  return [height, width, channels];
}
