import { Tensor } from './tensor.js';
import {
  ValueDistribution,
  DistanceMetric,
  InterpolationType,
} from './constants.js';
import { maskValues } from './masks.js';

export const FULLSCREEN_VS = `#version 300 es
precision highp float;
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

function fract(x) {
  return x - Math.floor(x);
}

// GPU fragment shader implementations operate in 32bit float precision. The
// original hash used very large constants which, when combined with the limited
// precision of GLSL `sin`, produced visible banding artifacts.  Using smaller
// constants keeps the argument to `sin` within a sane range while still
// providing a deterministic pseudo-random distribution.
function rand2D(x, y, seed = 0, time = 0, speed = 1) {
  // Keep the original high‑precision hash for CPU paths to preserve existing
  // test fixtures. The GPU shader uses a reduced version to avoid float
  // precision issues.
  const s =
    x * 12.9898 +
    y * 78.233 +
    seed * 37.719 +
    time * speed * 0.1;
  return fract(Math.sin(s) * 43758.5453);
}

const GPU_DISTRIBS = new Set([
  ValueDistribution.uniform,
  ValueDistribution.exp,
  ValueDistribution.column_index,
  ValueDistribution.row_index,
  ValueDistribution.center_circle,
]);

export function freqForShape(freq, shape) {
  const [height, width] = shape;
  if (height === width) {
    return [freq, freq];
  }
  if (height < width) {
    return [freq, Math.floor((freq * width) / height)];
  }
  return [Math.floor((freq * height) / width), freq];
}

/**
 * Generate value noise or other simple distributions.
 *
 * @param {number|[number, number]} freq Frequency of the grid (scalar or `[fx, fy]`).
 * @param {[number, number, number]} shape Output tensor shape `[height,width,channels]`.
 * @param {Object} opts Options object.
 * @param {ValueDistribution} [opts.distrib=ValueDistribution.uniform] Distribution type.
 * @param {boolean} [opts.corners=false] Wrap noise coordinates for seamless tiling.
 * @param {ValueMask} [opts.mask] Optional mask to multiply the result with.
 * @param {boolean} [opts.maskInverse=false] Invert the mask values.
 * @param {boolean} [opts.maskStatic=false] Ignore time/speed when generating masks.
 * @param {InterpolationType} [opts.splineOrder=InterpolationType.bicubic]
 *   Interpolation type.
 * @param {number} [opts.time=0] Animation time value.
 * @param {number} [opts.seed=0] Random seed.
 * @param {number} [opts.speed=1] Time multiplier for animation.
 * @returns {Tensor} Generated tensor.
 */
export function values(freq, shape, opts = {}) {
  const [height, width, channels = 1] = shape;
  const freqX = Array.isArray(freq) ? freq[0] : freq;
  const freqY = Array.isArray(freq) ? freq[1] : freq;
  const {
    ctx = null,
    distrib = ValueDistribution.uniform,
    corners = false,
    mask,
    maskInverse = false,
    maskStatic = false,
    splineOrder = InterpolationType.bicubic,
    time = 0,
    seed = 0,
    speed = 1,
  } = opts;
  const gpuDistrib = GPU_DISTRIBS.has(distrib);
  let maskData = null;
  let maskTex = null;
  let maskWidth = 0;
  let maskHeight = 0;
  let maskChannels = 0;
  const interp = (t) => {
    switch (splineOrder) {
      case InterpolationType.linear:
        return t;
      case InterpolationType.cosine:
        return 0.5 - Math.cos(t * Math.PI) * 0.5;
      default:
        return t * t * (3 - 2 * t);
    }
  };
  if (mask !== undefined && mask !== null) {
    const maskShape = [
      Math.max(1, Math.floor(freqX)),
      Math.max(1, Math.floor(freqY)),
      1,
    ];
    const [maskTensor] = maskValues(mask, maskShape, {
      inverse: maskInverse,
      time: maskStatic ? 0 : time,
      speed,
    });
    [maskHeight, maskWidth, maskChannels] = maskTensor.shape;
    if (ctx && !ctx.isCPU && gpuDistrib) {
      maskTex = Tensor.fromArray(ctx, maskTensor.read(), maskTensor.shape);
    } else {
      maskData = maskTensor.read();
    }
  }

  if (ctx && !ctx.isCPU && gpuDistrib) {
    const gl = ctx.gl;
    const fs = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 u_freq;
uniform float u_seed;
uniform float u_time;
uniform float u_speed;
uniform float u_corners;
uniform int u_interp;
uniform int u_distrib;
uniform sampler2D u_mask;
uniform int u_useMask;
float rand2D(float x,float y,float seed,float time,float speed){
 float s=x*12.9898+y*78.233+mod(seed,65536.0)*37.719+time*speed*0.1;
 return fract(sin(s)*43758.5453);
}
float interp(float t){
 if(u_interp==${InterpolationType.linear}) return t;
 if(u_interp==${InterpolationType.cosine}) return 0.5 - cos(t*3.141592653589793)*0.5;
 return t*t*(3.0-2.0*t);
}
void main(){
 float val=0.0;
 if(u_distrib==${ValueDistribution.exp}){
  float r=rand2D(gl_FragCoord.x,gl_FragCoord.y,u_seed,u_time,u_speed);
  val=pow(r,3.0);
 }else if(u_distrib==${ValueDistribution.column_index}){
  if(float(${width})<=1.0) val=0.0; else val=gl_FragCoord.x/float(${width - 1});
 }else if(u_distrib==${ValueDistribution.row_index}){
  if(float(${height})<=1.0) val=0.0; else val=gl_FragCoord.y/float(${height - 1});
 }else if(u_distrib==${ValueDistribution.center_circle}){
  float dx=(gl_FragCoord.x+0.5)/float(${width})-0.5;
  float dy=(gl_FragCoord.y+0.5)/float(${height})-0.5;
  float d=sqrt(dx*dx+dy*dy);
  val=max(0.0,1.0-d*2.0);
 }else{
  float u=(gl_FragCoord.x/float(${width}))*u_freq.x;
  float v=(gl_FragCoord.y/float(${height}))*u_freq.y;
  float x0=floor(u);
  float y0=floor(v);
  float xf=fract(u);
  float yf=fract(v);
  float fx=max(1.0,floor(u_freq.x));
  float fy=max(1.0,floor(u_freq.y));
  float xb=u_corners>0.5?mod(x0,fx):x0;
  float yb=u_corners>0.5?mod(y0,fy):y0;
  float x1=u_corners>0.5?mod(xb+1.0,fx):xb+1.0;
  float y1=u_corners>0.5?mod(yb+1.0,fy):yb+1.0;
  float r00=rand2D(xb,yb,u_seed,u_time,u_speed);
  float r10=rand2D(x1,yb,u_seed,u_time,u_speed);
  float r01=rand2D(xb,y1,u_seed,u_time,u_speed);
  float r11=rand2D(x1,y1,u_seed,u_time,u_speed);
  if(u_interp==${InterpolationType.constant}){
   val=r00;
  }else{
   float sx=interp(xf);
   float sy=interp(yf);
   float nx0=mix(r00,r10,sx);
   float nx1=mix(r01,r11,sx);
   val=mix(nx0,nx1,sy);
  }
 }
 if(u_useMask==1){
  vec2 msize=vec2(textureSize(u_mask,0));
  float mu=(gl_FragCoord.x/float(${width}))*msize.x;
  float mv=(gl_FragCoord.y/float(${height}))*msize.y;
  float mx0=floor(mu);
  float my0=floor(mv);
  float mxf=fract(mu);
  float myf=fract(mv);
  vec2 uv00=(vec2(mx0,my0)+0.5)/msize;
  vec2 uv10=(vec2(mx0+1.0,my0)+0.5)/msize;
  vec2 uv01=(vec2(mx0,my0+1.0)+0.5)/msize;
  vec2 uv11=(vec2(mx0+1.0,my0+1.0)+0.5)/msize;
  float m00=texture(u_mask,uv00).r;
  float m10=texture(u_mask,uv10).r;
  float m01=texture(u_mask,uv01).r;
  float m11=texture(u_mask,uv11).r;
  float mval;
  if(u_interp==${InterpolationType.constant}){
   mval=m00;
  }else{
   float sx=interp(mxf);
   float sy=interp(myf);
   float mx0v=mix(m00,m10,sx);
   float mx1v=mix(m01,m11,sx);
   mval=mix(mx0v,mx1v,sy);
  }
  val*=mval;
 }
 outColor=vec4(val);
}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(width, height);
    gl.useProgram(prog);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_freq'), freqX, freqY);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_seed'), seed);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_time'), time);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_speed'), speed);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_corners'), corners ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_interp'), splineOrder);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_distrib'), distrib);
    if (maskTex) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, maskTex.handle);
      gl.uniform1i(gl.getUniformLocation(prog, 'u_mask'), 0);
      gl.uniform1i(gl.getUniformLocation(prog, 'u_useMask'), 1);
    } else {
      gl.uniform1i(gl.getUniformLocation(prog, 'u_useMask'), 0);
    }
    ctx.bindFramebuffer(pp.writeFbo, width, height);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [height, width, channels]);
  }

  const data = new Float32Array(height * width * channels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0;
      switch (distrib) {
        case ValueDistribution.ones:
          val = 1;
          break;
        case ValueDistribution.mids:
          val = 0.5;
          break;
        case ValueDistribution.zeros:
          val = 0;
          break;
        case ValueDistribution.column_index:
          val = width === 1 ? 0 : x / (width - 1);
          break;
        case ValueDistribution.row_index:
          val = height === 1 ? 0 : y / (height - 1);
          break;
        case ValueDistribution.center_circle:
        case ValueDistribution.center_triangle:
        case ValueDistribution.center_diamond:
        case ValueDistribution.center_square:
        case ValueDistribution.center_pentagon:
        case ValueDistribution.center_hexagon:
        case ValueDistribution.center_heptagon:
        case ValueDistribution.center_octagon:
        case ValueDistribution.center_nonagon:
        case ValueDistribution.center_decagon:
        case ValueDistribution.center_hendecagon:
        case ValueDistribution.center_dodecagon: {
          const dx = (x + 0.5) / width - 0.5;
          const dy = (y + 0.5) / height - 0.5;
          let metric = DistanceMetric.euclidean;
          let sdfSides = 5;
          switch (distrib) {
            case ValueDistribution.center_triangle:
              metric = DistanceMetric.triangular;
              break;
            case ValueDistribution.center_diamond:
              metric = DistanceMetric.manhattan;
              break;
            case ValueDistribution.center_square:
              metric = DistanceMetric.chebyshev;
              break;
            case ValueDistribution.center_pentagon:
              metric = DistanceMetric.sdf;
              sdfSides = 5;
              break;
            case ValueDistribution.center_hexagon:
              metric = DistanceMetric.hexagram;
              break;
            case ValueDistribution.center_heptagon:
              metric = DistanceMetric.sdf;
              sdfSides = 7;
              break;
            case ValueDistribution.center_octagon:
              metric = DistanceMetric.octagram;
              break;
            case ValueDistribution.center_nonagon:
              metric = DistanceMetric.sdf;
              sdfSides = 9;
              break;
            case ValueDistribution.center_decagon:
              metric = DistanceMetric.sdf;
              sdfSides = 10;
              break;
            case ValueDistribution.center_hendecagon:
              metric = DistanceMetric.sdf;
              sdfSides = 11;
              break;
            case ValueDistribution.center_dodecagon:
              metric = DistanceMetric.sdf;
              sdfSides = 12;
              break;
            default:
              metric = DistanceMetric.euclidean;
          }
          const d = distance(dx, dy, metric, sdfSides);
          val = Math.max(0, 1 - d * 2);
          break;
        }
        case ValueDistribution.exp: {
          const r = rand2D(x, y, seed, time, speed);
          val = Math.pow(r, 3);
          break;
        }
        case ValueDistribution.uniform:
        default: {
          const u = (x / width) * freqX;
          const v = (y / height) * freqY;
          const x0 = Math.floor(u);
          const y0 = Math.floor(v);
          const xf = u - x0;
          const yf = v - y0;
          const fx = Math.max(1, Math.floor(freqX));
          const fy = Math.max(1, Math.floor(freqY));
          const xb = corners ? x0 % fx : x0;
          const yb = corners ? y0 % fy : y0;
          const x1 = corners ? (xb + 1) % fx : xb + 1;
          const y1 = corners ? (yb + 1) % fy : yb + 1;
          const r00 = rand2D(xb, yb, seed, time, speed);
          const r10 = rand2D(x1, yb, seed, time, speed);
          const r01 = rand2D(xb, y1, seed, time, speed);
          const r11 = rand2D(x1, y1, seed, time, speed);
          if (splineOrder === InterpolationType.constant) {
            val = r00;
            break;
          }
          const sx = interp(xf);
          const sy = interp(yf);
          const nx0 = r00 * (1 - sx) + r10 * sx;
          const nx1 = r01 * (1 - sx) + r11 * sx;
          val = nx0 * (1 - sy) + nx1 * sy;
          break;
        }
      }
      const idx = (y * width + x) * channels;
      if (maskData) {
        const mu = (x / width) * maskWidth;
        const mv = (y / height) * maskHeight;
        const mx0 = Math.floor(mu);
        const my0 = Math.floor(mv);
        let m;
        if (splineOrder === InterpolationType.constant) {
          m = maskData[(my0 * maskWidth + mx0) * maskChannels];
        } else {
          const mx1 = Math.min(mx0 + 1, maskWidth - 1);
          const my1 = Math.min(my0 + 1, maskHeight - 1);
          const xf = mu - mx0;
          const yf = mv - my0;
          const sx = interp(xf);
          const sy = interp(yf);
          const m00 = maskData[(my0 * maskWidth + mx0) * maskChannels];
          const m10 = maskData[(my0 * maskWidth + mx1) * maskChannels];
          const m01 = maskData[(my1 * maskWidth + mx0) * maskChannels];
          const m11 = maskData[(my1 * maskWidth + mx1) * maskChannels];
          const mx0v = m00 * (1 - sx) + m10 * sx;
          const mx1v = m01 * (1 - sx) + m11 * sx;
          m = mx0v * (1 - sy) + mx1v * sy;
        }
        if (channels === 2) {
          data[idx] = val;
          data[idx + 1] = m;
          continue;
        } else if (channels === 4) {
          data[idx] = val;
          data[idx + 1] = val;
          data[idx + 2] = val;
          data[idx + 3] = m;
          continue;
        } else {
          val *= m;
        }
      }
      for (let c = 0; c < channels; c++) {
        data[idx + c] = val;
      }
    }
  }
  return Tensor.fromArray(null, data, [height, width, channels]);
}

export function downsample(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const ctx = tensor.ctx;
  const nh = Math.floor(h / factor);
  const nw = Math.floor(w / factor);
  if (ctx && !ctx.isCPU && factor === 2) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform vec2 u_texel;\nout vec4 outColor;\nvoid main(){\n vec2 uv = (gl_FragCoord.xy*2.0 - vec2(1.0)) * u_texel;\n vec4 sum = texture(u_tex, uv) + texture(u_tex, uv + vec2(u_texel.x,0.0)) + texture(u_tex, uv + vec2(0.0,u_texel.y)) + texture(u_tex, uv + u_texel);\n outColor = sum * 0.25;\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(nw, nh);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_texel'), 1 / w, 1 / h);
    ctx.bindFramebuffer(pp.writeFbo, nw, nh);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [nh, nw, c]);
  }
  const src = tensor.read();
  const out = new Float32Array(nh * nw * c);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        for (let yy = 0; yy < factor; yy++) {
          for (let xx = 0; xx < factor; xx++) {
            const idx = ((y * factor + yy) * w + (x * factor + xx)) * c + k;
            sum += src[idx];
          }
        }
        out[(y * nw + x) * c + k] = sum / (factor * factor);
      }
    }
  }
  return Tensor.fromArray(ctx, out, [nh, nw, c]);
}

export function upsample(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const ctx = tensor.ctx;
  const nh = h * factor;
  const nw = w * factor;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_factor;\nuniform vec2 u_srcSize;\nout vec4 outColor;\nvoid main(){\n vec2 uv = (floor(gl_FragCoord.xy / u_factor) + 0.5) / u_srcSize;\n outColor = texture(u_tex, uv);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(nw, nh);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_factor'), factor);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_srcSize'), w, h);
    ctx.bindFramebuffer(pp.writeFbo, nw, nh);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [nh, nw, c]);
  }
  const src = tensor.read();
  const out = new Float32Array(nh * nw * c);
  for (let y = 0; y < nh; y++) {
    const y0 = Math.floor(y / factor);
    for (let x = 0; x < nw; x++) {
      const x0 = Math.floor(x / factor);
      for (let k = 0; k < c; k++) {
        out[(y * nw + x) * c + k] = src[(y0 * w + x0) * c + k];
      }
    }
  }
  return Tensor.fromArray(ctx, out, [nh, nw, c]);
}

export function warp(tensor, flow, amount = 1) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const flowData = flow.read();
  const out = new Float32Array(h * w * c);
  function sample(x, y, k) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    const xFract = x - x0;
    const yFract = y - y0;
    const top = src[(y0 * w + x0) * c + k] * (1 - xFract) + src[(y0 * w + x1) * c + k] * xFract;
    const bottom = src[(y1 * w + x0) * c + k] * (1 - xFract) + src[(y1 * w + x1) * c + k] * xFract;
    return top * (1 - yFract) + bottom * yFract;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = flowData[(y * w + x) * 2] * amount;
      const dy = flowData[(y * w + x) * 2 + 1] * amount;
      for (let k = 0; k < c; k++) {
        out[(y * w + x) * c + k] = sample(x + dx, y + dy, k);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function blend(a, b, t) {
  const [h, w, c] = a.shape;
  const ctx = a.ctx;
  const bc = b.shape[2];
  if (ctx && !ctx.isCPU && b.ctx === ctx && typeof t === 'number' && bc === c) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_a;\nuniform sampler2D u_b;\nuniform float u_t;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec4 ca = texture(u_a, uv);\n vec4 cb = texture(u_b, uv);\n outColor = mix(ca, cb, u_t);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, a.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_a'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, b.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_b'), 1);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_t'), t);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [h, w, c]);
  }
  const da = a.read();
  const db = b.read();
  const dt = typeof t === 'number' ? null : t.read();
  const tc = dt ? t.shape[2] : 0;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const baseA = i * c;
    const baseB = i * bc;
    const baseT = dt ? i * tc : 0;
    for (let k = 0; k < c; k++) {
      const aVal = da[baseA + k];
      const bVal = db[baseB + (k < bc ? k : 0)];
      const tVal = dt ? dt[baseT + (k < tc ? k : 0)] : t;
      out[baseA + k] = aVal * (1 - tVal) + bVal * tVal;
    }
  }
  return Tensor.fromArray(ctx, out, [h, w, c]);
}

export function normalize(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let k = 0; k < c; k++) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = k; i < src.length; i += c) {
      const v = src[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    for (let i = k; i < src.length; i += c) {
      out[i] = (src[i] - min) / range;
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function clamp01(tensor) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = Math.min(1, Math.max(0, src[i]));
  }
  return Tensor.fromArray(tensor.ctx, out, tensor.shape);
}

export function distance(
  dx,
  dy,
  metric = DistanceMetric.euclidean,
  sdfSides = 5
) {
  switch (metric) {
    case DistanceMetric.manhattan:
      return Math.abs(dx) + Math.abs(dy);
    case DistanceMetric.chebyshev:
      return Math.max(Math.abs(dx), Math.abs(dy));
    case DistanceMetric.octagram:
      return Math.max(
        (Math.abs(dx) + Math.abs(dy)) / Math.SQRT2,
        Math.max(Math.abs(dx), Math.abs(dy))
      );
    case DistanceMetric.triangular:
      return Math.max(Math.abs(dx) - dy * 0.5, dy);
    case DistanceMetric.hexagram:
      return Math.max(
        Math.max(Math.abs(dx) - dy * 0.5, dy),
        Math.max(Math.abs(dx) + dy * 0.5, -dy)
      );
    case DistanceMetric.sdf: {
      const arctan = Math.atan2(dx, -dy) + Math.PI;
      const r = (Math.PI * 2) / sdfSides;
      return (
        Math.cos(Math.floor(0.5 + arctan / r) * r - arctan) *
        Math.sqrt(dx * dx + dy * dy)
      );
    }
    case DistanceMetric.euclidean:
    default:
      return Math.sqrt(dx * dx + dy * dy);
  }
}

export function sobel(tensor) {
  const [h, w, c] = tensor.shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform vec2 u_texel;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec2 t = u_texel;\n vec4 s00 = texture(u_tex, uv + vec2(-t.x,-t.y));\n vec4 s10 = texture(u_tex, uv + vec2(0.0,-t.y));\n vec4 s20 = texture(u_tex, uv + vec2(t.x,-t.y));\n vec4 s01 = texture(u_tex, uv + vec2(-t.x,0.0));\n vec4 s21 = texture(u_tex, uv + vec2(t.x,0.0));\n vec4 s02 = texture(u_tex, uv + vec2(-t.x,t.y));\n vec4 s12 = texture(u_tex, uv + vec2(0.0,t.y));\n vec4 s22 = texture(u_tex, uv + vec2(t.x,t.y));\n vec4 gx = -s00 + s20 - 2.0*s01 + 2.0*s21 - s02 + s22;\n vec4 gy = -s00 - 2.0*s10 - s20 + s02 + 2.0*s12 + s22;\n outColor = sqrt(gx*gx + gy*gy);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_texel'), 1 / w, 1 / h);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [h, w, c]);
  }
  const src = tensor.read();
  const out = new Float32Array(src.length);
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  function get(x, y, k) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    return src[(y * w + x) * c + k];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let gx = 0, gy = 0, idx = 0;
        for (let yy = -1; yy <= 1; yy++) {
          for (let xx = -1; xx <= 1; xx++) {
            const v = get(x + xx, y + yy, k);
            gx += gxKernel[idx] * v;
            gy += gyKernel[idx] * v;
            idx++;
          }
        }
        out[(y * w + x) * c + k] = Math.sqrt(gx * gx + gy * gy);
      }
    }
  }
  return Tensor.fromArray(ctx, out, [h, w, c]);
}

export function hsvToRgb(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const H = src[i * c];
    const S = src[i * c + 1];
    const V = src[i * c + 2];
    const C = V * S;
    const hPrime = (H * 6) % 6;
    const X = C * (1 - Math.abs((hPrime % 2) - 1));
    let r1, g1, b1;
    switch (Math.floor(hPrime)) {
      case 0: r1 = C; g1 = X; b1 = 0; break;
      case 1: r1 = X; g1 = C; b1 = 0; break;
      case 2: r1 = 0; g1 = C; b1 = X; break;
      case 3: r1 = 0; g1 = X; b1 = C; break;
      case 4: r1 = X; g1 = 0; b1 = C; break;
      case 5: r1 = C; g1 = 0; b1 = X; break;
      default: r1 = 0; g1 = 0; b1 = 0; break;
    }
    const m = V - C;
    out[i * 3] = r1 + m;
    out[i * 3 + 1] = g1 + m;
    out[i * 3 + 2] = b1 + m;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function rgbToHsv(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const r = src[i * c];
    const g = src[i * c + 1];
    const b = src[i * c + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let hVal;
    if (d === 0) {
      hVal = 0;
    } else if (max === r) {
      hVal = ((g - b) / d) % 6;
    } else if (max === g) {
      hVal = (b - r) / d + 2;
    } else {
      hVal = (r - g) / d + 4;
    }
    hVal /= 6;
    if (hVal < 0) hVal += 1;
    const sVal = max === 0 ? 0 : d / max;
    out[i * 3] = hVal;
    out[i * 3 + 1] = sVal;
    out[i * 3 + 2] = max;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function adjustHue(tensor, amount) {
  const hsv = rgbToHsv(tensor);
  const data = hsv.read();
  for (let i = 0; i < data.length; i += 3) {
    data[i] = (data[i] + amount) % 1;
    if (data[i] < 0) data[i] += 1;
  }
  return hsvToRgb(Tensor.fromArray(hsv.ctx, data, hsv.shape));
}

export function randomHue(tensor, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}

export function valueMap(tensor, palette) {
  const [h, w, c] = tensor.shape;
  if (c !== 1) throw new Error('valueMap expects single-channel tensor');
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  const n = palette.length;
  for (let i = 0; i < h * w; i++) {
    const idx = Math.min(n - 1, Math.max(0, Math.floor(src[i] * (n - 1))));
    const [r, g, b] = palette[idx];
    out[i * 3] = r;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = b;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function ridge(tensor) {
  const data = tensor.read();
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = 1 - Math.abs(data[i] * 2 - 1);
  }
  return Tensor.fromArray(tensor.ctx, out, tensor.shape);
}

export function convolution(tensor, kernel, opts = {}) {
  const { normalize: doNormalize = true, alpha = 1 } = opts;
  const [h, w, c] = tensor.shape;
  const kh = kernel.length;
  const kw = kernel[0].length;
  let maxVal = -Infinity;
  let minVal = Infinity;
  for (const row of kernel) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
    }
  }
  const scale = Math.max(Math.abs(maxVal), Math.abs(minVal)) || 1;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const halfH = Math.floor(kh / 2);
  const halfW = Math.floor(kw / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        for (let j = 0; j < kh; j++) {
          for (let i = 0; i < kw; i++) {
            const yy = (y + j - halfH + h) % h;
            const xx = (x + i - halfW + w) % w;
            const val = src[(yy * w + xx) * c + k];
            sum += (kernel[j][i] / scale) * val;
          }
        }
        out[(y * w + x) * c + k] = sum;
      }
    }
  }
  let result = Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  if (doNormalize) result = normalize(result);
  if (alpha !== 1) result = blend(tensor, result, alpha);
  return result;
}

export function refract(tensor, referenceX = null, referenceY = null, displacement = 0.5) {
  const [h, w, c] = tensor.shape;
  const rx = (referenceX || tensor).read();
  const ry = (referenceY || tensor).read();
  const flowData = new Float32Array(h * w * 2);
  for (let i = 0; i < h * w; i++) {
    flowData[i * 2] = rx[i] * 2 - 1;
    flowData[i * 2 + 1] = ry[i] * 2 - 1;
  }
  const flow = Tensor.fromArray(tensor.ctx, flowData, [h, w, 2]);
  return warp(tensor, flow, displacement);
}

export function fft(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c * 2);
  for (let k = 0; k < c; k++) {
    for (let u = 0; u < h; u++) {
      for (let v = 0; v < w; v++) {
        let re = 0, im = 0;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const angle = -2 * Math.PI * ((u * y) / h + (v * x) / w);
            const val = src[(y * w + x) * c + k];
            re += val * Math.cos(angle);
            im += val * Math.sin(angle);
          }
        }
        const idx = (u * w + v) * c * 2 + k * 2;
        out[idx] = re;
        out[idx + 1] = im;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c * 2]);
}

export function ifft(tensor) {
  const [h, w, c2] = tensor.shape;
  const c = c2 / 2;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let k = 0; k < c; k++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let re = 0;
        for (let u = 0; u < h; u++) {
          for (let v = 0; v < w; v++) {
            const idx = (u * w + v) * c * 2 + k * 2;
            const real = src[idx];
            const imag = src[idx + 1];
            const angle = 2 * Math.PI * ((u * y) / h + (v * x) / w);
            re += real * Math.cos(angle) - imag * Math.sin(angle);
          }
        }
        out[(y * w + x) * c + k] = re / (h * w);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function rotate(tensor, angle) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const rx = (x - cx) * cos + (y - cy) * sin + cx;
      const ry = -(x - cx) * sin + (y - cy) * cos + cy;
      const ix = Math.round(rx);
      const iy = Math.round(ry);
      for (let k = 0; k < c; k++) {
        let val = 0;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
          val = src[(iy * w + ix) * c + k];
        }
        out[(y * w + x) * c + k] = val;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function zoom(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const rx = (x - cx) / factor + cx;
      const ry = (y - cy) / factor + cy;
      const ix = Math.round(rx);
      const iy = Math.round(ry);
      for (let k = 0; k < c; k++) {
        let val = 0;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
          val = src[(iy * w + ix) * c + k];
        }
        out[(y * w + x) * c + k] = val;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function fxaa(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const lumWeights = [0.299, 0.587, 0.114];
  function reflect(i, n) {
    if (n === 1) return 0;
    const m = (2 * n - 2);
    i = ((i % m) + m) % m;
    return i < n ? i : m - i;
  }
  function idx(x, y, k) {
    x = reflect(x, w);
    y = reflect(y, h);
    return (y * w + x) * c + k;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (c === 1) {
        const lC = src[idx(x, y, 0)];
        const lN = src[idx(x, y - 1, 0)];
        const lS = src[idx(x, y + 1, 0)];
        const lW = src[idx(x - 1, y, 0)];
        const lE = src[idx(x + 1, y, 0)];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lC - lN));
        const wS = Math.exp(-Math.abs(lC - lS));
        const wW = Math.exp(-Math.abs(lC - lW));
        const wE = Math.exp(-Math.abs(lC - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        out[idx(x, y, 0)] =
          (lC * wC + lN * wN + lS * wS + lW * wW + lE * wE) / sum;
      } else if (c === 2) {
        const lum = src[idx(x, y, 0)];
        const alpha = src[idx(x, y, 1)];
        const lN = src[idx(x, y - 1, 0)];
        const lS = src[idx(x, y + 1, 0)];
        const lW = src[idx(x - 1, y, 0)];
        const lE = src[idx(x + 1, y, 0)];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lum - lN));
        const wS = Math.exp(-Math.abs(lum - lS));
        const wW = Math.exp(-Math.abs(lum - lW));
        const wE = Math.exp(-Math.abs(lum - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        out[idx(x, y, 0)] =
          (lum * wC + lN * wN + lS * wS + lW * wW + lE * wE) / sum;
        out[idx(x, y, 1)] = alpha;
      } else if (c === 3 || c === 4) {
        const rgbC = [src[idx(x, y, 0)], src[idx(x, y, 1)], src[idx(x, y, 2)]];
        const rgbN = [src[idx(x, y - 1, 0)], src[idx(x, y - 1, 1)], src[idx(x, y - 1, 2)]];
        const rgbS = [src[idx(x, y + 1, 0)], src[idx(x, y + 1, 1)], src[idx(x, y + 1, 2)]];
        const rgbW = [src[idx(x - 1, y, 0)], src[idx(x - 1, y, 1)], src[idx(x - 1, y, 2)]];
        const rgbE = [src[idx(x + 1, y, 0)], src[idx(x + 1, y, 1)], src[idx(x + 1, y, 2)]];
        const lC = rgbC[0] * lumWeights[0] + rgbC[1] * lumWeights[1] + rgbC[2] * lumWeights[2];
        const lN = rgbN[0] * lumWeights[0] + rgbN[1] * lumWeights[1] + rgbN[2] * lumWeights[2];
        const lS = rgbS[0] * lumWeights[0] + rgbS[1] * lumWeights[1] + rgbS[2] * lumWeights[2];
        const lW = rgbW[0] * lumWeights[0] + rgbW[1] * lumWeights[1] + rgbW[2] * lumWeights[2];
        const lE = rgbE[0] * lumWeights[0] + rgbE[1] * lumWeights[1] + rgbE[2] * lumWeights[2];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lC - lN));
        const wS = Math.exp(-Math.abs(lC - lS));
        const wW = Math.exp(-Math.abs(lC - lW));
        const wE = Math.exp(-Math.abs(lC - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        for (let k = 0; k < 3; k++) {
          out[idx(x, y, k)] =
            (rgbC[k] * wC + rgbN[k] * wN + rgbS[k] * wS + rgbW[k] * wW + rgbE[k] * wE) / sum;
        }
        if (c === 4) out[idx(x, y, 3)] = src[idx(x, y, 3)];
      } else {
        for (let k = 0; k < c; k++) out[idx(x, y, k)] = src[idx(x, y, k)];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function gaussianBlur(tensor, radius = 1) {
  const size = radius * 2 + 1;
  const sigma = radius / 2 || 1;
  const kernel = [];
  for (let y = -radius; y <= radius; y++) {
    const row = [];
    for (let x = -radius; x <= radius; x++) {
      row.push(Math.exp(-(x * x + y * y) / (2 * sigma * sigma)));
    }
    kernel.push(row);
  }
  return convolution(tensor, kernel, { normalize: false });
}
