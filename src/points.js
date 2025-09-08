import { PointDistribution, ValueMask } from './constants.js';
import { Masks, maskShape } from './masks.js';
import { random as simplexRandom } from './simplex.js';
import { random as seededRandom, setSeed as setUtilSeed } from './util.js';

function isGrid(distrib) {
  return distrib >= PointDistribution.square && distrib < PointDistribution.spiral;
}

function isCircular(distrib) {
  return distrib >= PointDistribution.circular;
}

export function pointCloud(
  freq,
  {
    distrib = PointDistribution.random,
    shape = null,
    corners = false,
    generations = 1,
    drift = 0,
    time = 0,
    speed = 1,
    seed,
  } = {}
) {
  if (!freq) return [[], []];

  if (seed !== undefined) setUtilSeed(seed);

  const x = [];
  const y = [];

  let width, height;
  if (!shape) {
    width = 1;
    height = 1;
  } else {
    height = shape[0];
    width = shape[1];
  }

  if (typeof distrib === 'string') {
    if (PointDistribution[distrib] !== undefined) {
      distrib = PointDistribution[distrib];
    } else if (ValueMask[distrib] !== undefined) {
      distrib = ValueMask[distrib];
    }
  }

  const isPoint = Object.values(PointDistribution).includes(distrib);
  const isMask = Object.values(ValueMask).includes(distrib);

  const rangeX = width * 0.5;
  const rangeY = height * 0.5;

  if (isMask) {
    const mask = Masks[distrib];
    const [maskHeight, maskWidth] = maskShape(distrib);
    const xSpace = width / maskWidth;
    const ySpace = height / maskHeight;
    const xMargin = xSpace * 0.5;
    const yMargin = ySpace * 0.5;

    for (let mx = 0; mx < maskWidth; mx++) {
      for (let my = 0; my < maskHeight; my++) {
        let pixel = mask[my][mx];
        if (Array.isArray(pixel)) pixel = pixel.reduce((a, b) => a + b, 0);

        let xDrift = 0;
        let yDrift = 0;
        if (drift) {
          xDrift = simplexRandom(time, undefined, speed) * drift / freq * width;
          yDrift = simplexRandom(time, undefined, speed) * drift / freq * height;
        }

        if (pixel !== 0) {
          x.push(Math.floor(xMargin + mx * xSpace + xDrift));
          y.push(Math.floor(yMargin + my * ySpace + yDrift));
        }
      }
    }

    return [x, y];
  }

  let pointFunc = rand;
  const seen = new Set();
  const active = [];

  if (isGrid(distrib)) {
    pointFunc = squareGrid;
    active.push({ x: 0, y: 0, gen: 1 });
  } else if (distrib === PointDistribution.spiral) {
    pointFunc = spiral;
    active.push({ x: rangeY, y: rangeX, gen: 1 });
  } else if (isCircular(distrib)) {
    pointFunc = circular;
    active.push({ x: rangeY, y: rangeX, gen: 1 });
  } else {
    active.push({ x: rangeY, y: rangeX, gen: 1 });
  }

  seen.add(`${active[0].x},${active[0].y}`);

  while (active.length) {
    const { x: cx, y: cy, gen } = active.pop();
    if (gen <= generations) {
      const multiplier = Math.max(2 * (gen - 1), 1);
      const [xs, ys] = pointFunc({
        freq,
        distrib,
        corners,
        centerX: cx,
        centerY: cy,
        rangeX: rangeX / multiplier,
        rangeY: rangeY / multiplier,
        width,
        height,
        generation: gen,
        time,
        speed: speed * 0.1,
      });

      for (let i = 0; i < xs.length; i++) {
        let xPoint = xs[i];
        let yPoint = ys[i];
        const key = `${xPoint},${yPoint}`;
        if (seen.has(key)) continue;
        seen.add(key);
        active.push({ x: xPoint, y: yPoint, gen: gen + 1 });

        let xDrift = 0;
        let yDrift = 0;
        if (drift) {
          xDrift = simplexRandom(time, undefined, speed) * drift;
          yDrift = simplexRandom(time, undefined, speed) * drift;
        }

        if (!shape) {
          xPoint = (xPoint + xDrift / freq) % 1;
          yPoint = (yPoint + yDrift / freq) % 1;
        } else {
          xPoint = Math.floor(xPoint + (xDrift / freq * width)) % width;
          yPoint = Math.floor(yPoint + (yDrift / freq * height)) % height;
        }

        x.push(xPoint);
        y.push(yPoint);
      }
    }
  }

  return [x, y];
}

export function rand({
  freq = 2,
  centerX = 0.5,
  centerY = 0.5,
  rangeX = 0.5,
  rangeY = 0.5,
  width = 1,
  height = 1,
  seed,
} = {}) {
  if (seed !== undefined) setUtilSeed(seed);
  const x = [];
  const y = [];
  for (let i = 0; i < freq * freq; i++) {
    const _x = (centerX + (seededRandom() * (rangeX * 2) - rangeX)) % width;
    const _y = (centerY + (seededRandom() * (rangeY * 2) - rangeY)) % height;
    x.push(_x);
    y.push(_y);
  }
  return [x, y];
}

export function squareGrid({
  freq = 1,
  distrib = PointDistribution.square,
  corners = false,
  centerX = 0,
  centerY = 0,
  rangeX = 1,
  rangeY = 1,
  width = 1,
  height = 1,
} = {}) {
  const x = [];
  const y = [];
  const driftAmount = 0.5 / freq;
  let drift;
  if (freq % 2 === 0) {
    drift = corners ? driftAmount : 0;
  } else {
    drift = corners ? 0 : driftAmount;
  }

  for (let a = 0; a < freq; a++) {
    for (let b = 0; b < freq; b++) {
      if (distrib === PointDistribution.waffle && b % 2 === 0 && a % 2 === 0) continue;
      if (distrib === PointDistribution.chess && a % 2 === b % 2) continue;

      let xDrift = 0;
      let yDrift = 0;
      if (distrib === PointDistribution.h_hex) {
        xDrift = b % 2 === 1 ? driftAmount : 0;
      }
      if (distrib === PointDistribution.v_hex) {
        yDrift = a % 2 === 1 ? 0 : driftAmount;
      }

      const _x = (centerX + (((a / freq) + drift + xDrift) * rangeX * 2)) % width;
      const _y = (centerY + (((b / freq) + drift + yDrift) * rangeY * 2)) % height;
      x.push(_x);
      y.push(_y);
    }
  }
  return [x, y];
}

export function spiral({
  freq = 1,
  centerX = 0,
  centerY = 0,
  rangeX = 1,
  rangeY = 1,
  width = 1,
  height = 1,
  seed,
} = {}) {
  if (seed !== undefined) setUtilSeed(seed);
  const kink = 0.5 + seededRandom() * 0.5;
  const x = [];
  const y = [];
  const count = freq * freq;
  for (let i = 0; i < count; i++) {
    const fract = i / count;
    const degrees = fract * 360 * (Math.PI / 180) * kink;
    x.push((centerX + Math.sin(degrees) * fract * rangeX) % width);
    y.push((centerY + Math.cos(degrees) * fract * rangeY) % height);
  }
  return [x, y];
}

export function circular({
  freq = 1,
  distrib = PointDistribution.concentric,
  centerX = 0,
  centerY = 0,
  rangeX = 1,
  rangeY = 1,
  width = 1,
  height = 1,
  seed,
} = {}) {
  if (seed !== undefined) setUtilSeed(seed);
  const x = [];
  const y = [];
  const ringCount = freq;
  const dotCount = freq;
  x.push(centerX);
  y.push(centerY);
  const rotation = (1 / dotCount) * 360 * (Math.PI / 180);
  const kink = 0.5 + seededRandom() * 0.5;
  for (let i = 1; i <= ringCount; i++) {
    const distFract = i / ringCount;
    for (let j = 1; j <= dotCount; j++) {
      let rads = j * rotation;
      if (distrib === PointDistribution.circular) {
        rads += rotation * 0.5 * i;
      }
      if (distrib === PointDistribution.rotating) {
        rads += rotation * distFract * kink;
      }
      const xPoint = centerX + Math.sin(rads) * distFract * rangeX;
      const yPoint = centerY + Math.cos(rads) * distFract * rangeY;
      x.push(xPoint % width);
      y.push(yPoint % height);
    }
  }
  return [x, y];
}
