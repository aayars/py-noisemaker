const TAU = Math.PI * 2;

export const PALETTES = {
  grayscale: {
    amp: [0.5, 0.5, 0.5],
    freq: [2.0, 2.0, 2.0],
    offset: [0.5, 0.5, 0.5],
    phase: [1.0, 1.0, 1.0],
  },
  rainbow: {
    amp: [0.5, 0.5, 0.5],
    freq: [1.0, 1.0, 1.0],
    offset: [0.5, 0.5, 0.5],
    phase: [0.0, 0.33, 0.67],
  },
};

export function samplePalette(name, steps = 256) {
  const p = PALETTES[name];
  if (!p) throw new Error(`Unknown palette ${name}`);
  const colors = [];
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const r = p.offset[0] + p.amp[0] * Math.cos(TAU * (p.freq[0] * t * 0.875 + 0.0625 + p.phase[0]));
    const g = p.offset[1] + p.amp[1] * Math.cos(TAU * (p.freq[1] * t * 0.875 + 0.0625 + p.phase[1]));
    const b = p.offset[2] + p.amp[2] * Math.cos(TAU * (p.freq[2] * t * 0.875 + 0.0625 + p.phase[2]));
    colors.push([r, g, b]);
  }
  return colors;
}
