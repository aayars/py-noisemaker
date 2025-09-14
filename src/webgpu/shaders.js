import fs from 'fs';

export const VORONOI_WGSL = fs.readFileSync(new URL('./voronoi.wgsl', import.meta.url), 'utf8');
export const EROSION_WORMS_WGSL = fs.readFileSync(new URL('./erosion-worms.wgsl', import.meta.url), 'utf8');
export const WORMS_WGSL = fs.readFileSync(new URL('./worms.wgsl', import.meta.url), 'utf8');
