import { Tensor } from './tensor.js';
import { srgbToLin, linToSRGB, withTensorData } from './util.js';

export function rgbToOklab(tensor) {
  if (tensor && typeof tensor.then === 'function') {
    return tensor.then(rgbToOklab);
  }
  const [h, w, c] = tensor.shape;
  if (c !== 3) throw new Error('rgbToOklab expects 3-channel tensor');
  return withTensorData(tensor, (src) => {
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
  });
}

export function oklabToRgb(tensor) {
  if (tensor && typeof tensor.then === 'function') {
    return tensor.then(oklabToRgb);
  }
  const [h, w, c] = tensor.shape;
  if (c !== 3) throw new Error('oklabToRgb expects 3-channel tensor');
  return withTensorData(tensor, (src) => {
    const out = new Float32Array(h * w * 3);
    const f32 = Math.fround;
    const c1 = f32(0.3963377774);
    const c2 = f32(0.2158037573);
    const c3 = f32(-0.1055613458);
    const c4 = f32(-0.0638541728);
    const c5 = f32(-0.0894841775);
    const c6 = f32(-1.291485548);
    const r1 = f32(4.0767245293);
    const r2 = f32(-3.3072168827);
    const r3 = f32(0.2307590544);
    const g1 = f32(-1.2681437731);
    const g2 = f32(2.6093323231);
    const g3 = f32(-0.3411344290);
    const b1 = f32(-0.0041119885);
    const b2 = f32(-0.7034763098);
    const b3 = f32(1.7068625689);
    const scratch = new Float32Array(1);
    const fmul = (x, y) => {
      scratch[0] = x * y;
      return scratch[0];
    };
    const fadd = (x, y) => {
      scratch[0] = x + y;
      return scratch[0];
    };
    for (let i = 0; i < h * w; i++) {
      const base = i * 3;
      const L = f32(src[base]);
      const a = f32(src[base + 1]);
      const b = f32(src[base + 2]);
      const l_ = fadd(fadd(L, fmul(c1, a)), fmul(c2, b));
      const m_ = fadd(fadd(L, fmul(c3, a)), fmul(c4, b));
      const s_ = fadd(fadd(L, fmul(c5, a)), fmul(c6, b));
      const lSq = fmul(l_, l_);
      const mSq = fmul(m_, m_);
      const sSq = fmul(s_, s_);
      const l = fmul(lSq, l_);
      const m = fmul(mSq, m_);
      const s = fmul(sSq, s_);
      const r = fadd(fadd(fmul(r1, l), fmul(r2, m)), fmul(r3, s));
      const g = fadd(fadd(fmul(g1, l), fmul(g2, m)), fmul(g3, s));
      const bLin = fadd(fadd(fmul(b1, l), fmul(b2, m)), fmul(b3, s));
      out[i * 3] = linToSRGB(r);
      out[i * 3 + 1] = linToSRGB(g);
      out[i * 3 + 2] = linToSRGB(bLin);
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
  });
}

