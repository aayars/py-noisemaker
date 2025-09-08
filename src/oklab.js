import { Tensor } from './tensor.js';
import { srgbToLin, linToSRGB } from './util.js';

export function rgbToOklab(tensor) {
  const [h, w, c] = tensor.shape;
  if (c !== 3) throw new Error('rgbToOklab expects 3-channel tensor');
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const r = srgbToLin(src[i * 3]);
    const g = srgbToLin(src[i * 3 + 1]);
    const b = srgbToLin(src[i * 3 + 2]);
    const l = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    const m = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    const s = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;
    const l_ = Math.cbrt(l);
    const m_ = Math.cbrt(m);
    const s_ = Math.cbrt(s);
    out[i * 3] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    out[i * 3 + 1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    out[i * 3 + 2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function oklabToRgb(tensor) {
  const [h, w, c] = tensor.shape;
  if (c !== 3) throw new Error('oklabToRgb expects 3-channel tensor');
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const L = src[i * 3];
    const a = src[i * 3 + 1];
    const b = src[i * 3 + 2];
    const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
    const l = l_ * l_ * l_;
    const m = m_ * m_ * m_;
    const s = s_ * s_ * s_;
    const r = 4.0767245293 * l - 3.3072168827 * m + 0.2307590544 * s;
    const g = -1.2681437731 * l + 2.6093323231 * m - 0.3411344290 * s;
    const bLin = -0.0041119885 * l - 0.7034763098 * m + 1.7068625689 * s;
    out[i * 3] = linToSRGB(r);
    out[i * 3 + 1] = linToSRGB(g);
    out[i * 3 + 2] = linToSRGB(bLin);
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

