import { Tensor } from './tensor.js';
import {
  ValueDistribution,
  DistanceMetric,
  InterpolationType,
  isNativeSize,
} from './constants.js';
import { maskValues } from './masks.js';
import { random } from './util.js';
import { simplex as simplexNoise, setSeed as setSimplexSeed } from './simplex.js';

let _seed = 0x12345678;
let _opCounter = 0;

export function setSeed(s) {
  _seed = s >>> 0;
  setSimplexSeed(s);
}

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
  // precision issues.  Floor coordinates to integers to match the GPU shader
  // and avoid fractional artifacts when called with non‑integer inputs.
  const sx = Math.floor(x);
  const sy = Math.floor(y);
  const s =
    sx * 12.9898 +
    sy * 78.233 +
    seed * 37.719 +
    time * speed * 0.1;
  return fract(Math.sin(s) * 43758.5453);
}

function offsetTensor(tensor, shape, x = 0, y = 0) {
  const [h, w, c] = shape;
  if (x === 0 && y === 0) return tensor;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let yy = 0; yy < h; yy++) {
    const sy = (yy + y + h) % h;
    for (let xx = 0; xx < w; xx++) {
      const sx = (xx + x + w) % w;
      const dst = (yy * w + xx) * c;
      const sIdx = (sy * w + sx) * c;
      for (let k = 0; k < c; k++) {
        out[dst + k] = src[sIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

function pinCorners(tensor, shape, freq, corners) {
  if (
    (!corners && freq[0] % 2 === 0) ||
    (corners && freq[0] % 2 === 1)
  ) {
    const xOff = Math.floor((shape[1] / freq[1]) * 0.5);
    const yOff = Math.floor((shape[0] / freq[0]) * 0.5);
    return offsetTensor(tensor, shape, xOff, yOff);
  }
  return tensor;
}

// All ValueDistribution members are supported on the GPU path.  Keep the set
// in sync with the enumeration so any new distributions automatically opt in.
const GPU_DISTRIBS = new Set(Object.values(ValueDistribution));
GPU_DISTRIBS.delete(ValueDistribution.simplex);

export function valueNoise(count, freq = 8) {
  // RNG: freq+1 calls to random for lattice points 0..freq
  const lattice = Array.from({ length: freq + 1 }, () => random());
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const x = (i / count) * freq;
    const xi = Math.floor(x);
    const xf = x - xi;
    const t = xf * xf * (3 - 2 * xf);
    out[i] = lattice[xi] * (1 - t) + lattice[xi + 1] * t;
  }
  return out;
}

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
 * @param {ValueDistribution} [opts.distrib=ValueDistribution.simplex] Distribution type.
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
  let freqX, freqY;
  if (Array.isArray(freq)) {
    // Python's freq argument is [height, width]; mirror that here
    [freqY, freqX] = freq;
  } else {
    // Match Python's freq_for_shape behavior for scalar input
    [freqY, freqX] = freqForShape(freq, [height, width]);
  }
  const {
    ctx = null,
    distrib = ValueDistribution.simplex,
    corners = false,
    mask,
    maskInverse = false,
    maskStatic = false,
    splineOrder = InterpolationType.bicubic,
    time = 0,
    seed,
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
      Math.max(1, Math.floor(freqY)),
      Math.max(1, Math.floor(freqX)),
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

  if (ctx && !ctx.isCPU && gpuDistrib && channels === 1) {
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
 float sx=floor(x);
 float sy=floor(y);
 float s=sx*12.9898+sy*78.233+mod(seed,65536.0)*37.719+time*speed*0.1;
 return fract(sin(s)*43758.5453);
}
float sdfPolygon(float dx,float dy,float sides){
 float an=atan(dx,-dy)+3.141592653589793;
 float r=6.283185307179586/sides;
 return cos(floor(0.5+an/r)*r-an)*sqrt(dx*dx+dy*dy);
}
float interp(float t){
 if(u_interp==${InterpolationType.linear}) return t;
 if(u_interp==${InterpolationType.cosine}) return 0.5 - cos(t*3.141592653589793)*0.5;
 return t*t*(3.0-2.0*t);
}
void main(){
 float val=0.0;
 float dx=(gl_FragCoord.x+0.5)/float(${width})-0.5;
 float dy=(gl_FragCoord.y+0.5)/float(${height})-0.5;
 if(u_distrib==${ValueDistribution.exp}){
  float r=rand2D(gl_FragCoord.x,gl_FragCoord.y,u_seed,u_time,u_speed);
  val=pow(r,3.0);
 }else if(u_distrib==${ValueDistribution.column_index}){
  float fy=max(1.0,floor(u_freq.y));
  float y=(gl_FragCoord.y/float(${height}))*fy;
  if((u_corners<0.5 && mod(fy,2.0)==0.0) || (u_corners>0.5 && mod(fy,2.0)==1.0)){
   y=mod(y+0.5,fy);
  }
  val=fy<=1.0?0.0:floor(y)/(fy-1.0);
 }else if(u_distrib==${ValueDistribution.row_index}){
  float fx=max(1.0,floor(u_freq.x));
  float x=(gl_FragCoord.x/float(${width}))*fx;
  if((u_corners<0.5 && mod(fx,2.0)==0.0) || (u_corners>0.5 && mod(fx,2.0)==1.0)){
   x=mod(x+0.5,fx);
  }
  val=fx<=1.0?0.0:floor(x)/(fx-1.0);
 }else if(u_distrib==${ValueDistribution.ones}){
  val=1.0;
 }else if(u_distrib==${ValueDistribution.mids}){
  val=0.5;
 }else if(u_distrib==${ValueDistribution.zeros}){
  val=0.0;
 }else if(u_distrib==${ValueDistribution.center_circle}){
  float d=sqrt(dx*dx+dy*dy);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_triangle}){
  float d=max(abs(dx)-dy*0.5,dy);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_diamond}){
  float d=abs(dx)+abs(dy);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_square}){
  float d=max(abs(dx),abs(dy));
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_pentagon}){
  float d=sdfPolygon(dx,dy,5.0);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_hexagon}){
  float d=max(max(abs(dx)-dy*0.5,dy),max(abs(dx)+dy*0.5,-dy));
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_heptagon}){
  float d=sdfPolygon(dx,dy,7.0);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_octagon}){
  float d=max((abs(dx)+abs(dy))/1.4142135623730951,max(abs(dx),abs(dy)));
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_nonagon}){
  float d=sdfPolygon(dx,dy,9.0);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_decagon}){
  float d=sdfPolygon(dx,dy,10.0);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_hendecagon}){
  float d=sdfPolygon(dx,dy,11.0);
  val=max(0.0,1.0-d*2.0);
 }else if(u_distrib==${ValueDistribution.center_dodecagon}){
  float d=sdfPolygon(dx,dy,12.0);
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
    gl.uniform1f(gl.getUniformLocation(prog, 'u_seed'), seed ?? _seed);
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

  const fx = Math.max(1, Math.floor(freqX));
  const fy = Math.max(1, Math.floor(freqY));
  const needsFullSize = isNativeSize(distrib);
  const initWidth = needsFullSize ? width : fx;
  const initHeight = needsFullSize ? height : fy;
  const size = initHeight * initWidth * channels;
  let tensor;
  if (distrib === ValueDistribution.simplex || distrib === ValueDistribution.exp) {
    tensor = simplexNoise([initHeight, initWidth, channels], { time, seed, speed });
    if (distrib === ValueDistribution.exp) {
      const data = tensor.read();
      for (let i = 0; i < data.length; i++) {
        data[i] = Math.pow(data[i], 4);
      }
      tensor = Tensor.fromArray(null, data, [initHeight, initWidth, channels]);
    }
  } else {
    const data = new Float32Array(size);
    for (let y = 0; y < initHeight; y++) {
      for (let x = 0; x < initWidth; x++) {
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
            val = initHeight === 1 ? 0 : y / (initHeight - 1);
            break;
          case ValueDistribution.row_index:
            val = initWidth === 1 ? 0 : x / (initWidth - 1);
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
            const dx = (x + 0.5) / initWidth - 0.5;
            const dy = (y + 0.5) / initHeight - 0.5;
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
          default:
            val = rand2D(x, y, seed, time, speed);
            break;
        }
        const idx = (y * initWidth + x) * channels;
        for (let c = 0; c < channels; c++) {
          data[idx + c] = val;
        }
      }
    }
    tensor = Tensor.fromArray(null, data, [initHeight, initWidth, channels]);
  }

  if (maskData) {
    let mTensor = Tensor.fromArray(null, maskData, [maskHeight, maskWidth, maskChannels]);
    if (needsFullSize) {
      mTensor = resample(mTensor, [height, width, maskChannels], splineOrder);
      mTensor = pinCorners(
        mTensor,
        [height, width, maskChannels],
        [freqY, freqX],
        corners
      );
    }
    const mArr = mTensor.read();
    const tArr = tensor.read();
    const mh = mTensor.shape[0];
    const mw = mTensor.shape[1];
    const total = mh * mw;
    if (channels === 2) {
      const out = new Float32Array(total * 2);
      for (let i = 0; i < total; i++) {
        out[i * 2] = tArr[i * channels];
        out[i * 2 + 1] = mArr[i];
      }
      tensor = Tensor.fromArray(null, out, [mh, mw, 2]);
    } else if (channels === 4) {
      const out = new Float32Array(total * 4);
      for (let i = 0; i < total; i++) {
        out[i * 4] = tArr[i * channels];
        out[i * 4 + 1] = tArr[i * channels + 1];
        out[i * 4 + 2] = tArr[i * channels + 2];
        out[i * 4 + 3] = mArr[i];
      }
      tensor = Tensor.fromArray(null, out, [mh, mw, 4]);
    } else {
      for (let i = 0; i < total; i++) {
        for (let c = 0; c < channels; c++) {
          tArr[i * channels + c] *= mArr[i];
        }
      }
      tensor = Tensor.fromArray(null, tArr, [mh, mw, channels]);
    }
  }

  if (!needsFullSize) {
    tensor = resample(tensor, [height, width, channels], splineOrder);
    tensor = pinCorners(
      tensor,
      [height, width, channels],
      [freqY, freqX],
      corners
    );
  }

  if (
    distrib !== ValueDistribution.ones &&
    distrib !== ValueDistribution.mids &&
    distrib !== ValueDistribution.zeros
  ) {
    tensor = normalize(tensor);
  }

  return tensor;
}



export function resample(tensor, shape, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  const [nh, nw, nc = c] = shape;
  const ctx = tensor.ctx;
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
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_srcSize;
uniform vec2 u_dstSize;
uniform int u_interp;
out vec4 outColor;
float interp(float t){
 if(u_interp==${InterpolationType.linear}) return t;
 if(u_interp==${InterpolationType.cosine}) return 0.5 - cos(t*3.141592653589793)*0.5;
 return t*t*(3.0-2.0*t);
}
void main(){
 vec2 uv = (gl_FragCoord.xy - 0.5) / u_dstSize;
 vec2 coord = uv * (u_srcSize - 1.0);
 vec2 c0 = floor(coord);
 vec2 f = coord - c0;
 vec2 c1 = min(c0 + 1.0, u_srcSize - 1.0);
 vec2 uv00 = (c0 + 0.5) / u_srcSize;
 vec2 uv10 = (vec2(c1.x, c0.y) + 0.5) / u_srcSize;
 vec2 uv01 = (vec2(c0.x, c1.y) + 0.5) / u_srcSize;
 vec2 uv11 = (c1 + 0.5) / u_srcSize;
 vec4 v00 = texture(u_tex, uv00);
 vec4 v10 = texture(u_tex, uv10);
 vec4 v01 = texture(u_tex, uv01);
 vec4 v11 = texture(u_tex, uv11);
 if(u_interp==${InterpolationType.constant}){
  outColor = v00;
 }else{
  float sx = interp(f.x);
  float sy = interp(f.y);
  vec4 mx0 = mix(v00, v10, sx);
  vec4 mx1 = mix(v01, v11, sx);
  outColor = mix(mx0, mx1, sy);
 }
}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(nw, nh);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_srcSize'), w, h);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_dstSize'), nw, nh);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_interp'), splineOrder);
    ctx.bindFramebuffer(pp.writeFbo, nw, nh);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [nh, nw, nc]);
  }
  const src = tensor.read();
  const out = new Float32Array(nh * nw * nc);

  function sampleWrapped(ix, iy, k) {
    ix = ((ix % w) + w) % w;
    iy = ((iy % h) + h) % h;
    return src[(iy * w + ix) * c + Math.min(k, c - 1)];
  }

  function cubic(a, b, c1, d, t) {
    const t2 = t * t;
    const t3 = t2 * t;
    const a0 = d - c1 - a + b;
    const a1 = a - b - a0;
    const a2 = c1 - a;
    const a3 = b;
    return a0 * t3 + a1 * t2 + a2 * t + a3;
  }

  for (let y = 0; y < nh; y++) {
    const gy = (y * h) / nh;
    const y0 = Math.floor(gy);
    const yf = gy - y0;
    for (let x = 0; x < nw; x++) {
      const gx = (x * w) / nw;
      const x0 = Math.floor(gx);
      const xf = gx - x0;
      for (let k = 0; k < nc; k++) {
        let val;
        if (splineOrder === InterpolationType.constant) {
          val = sampleWrapped(Math.round(gx), Math.round(gy), k);
        } else if (splineOrder === InterpolationType.linear || splineOrder === InterpolationType.cosine) {
          const v00 = sampleWrapped(x0, y0, k);
          const v10 = sampleWrapped(x0 + 1, y0, k);
          const v01 = sampleWrapped(x0, y0 + 1, k);
          const v11 = sampleWrapped(x0 + 1, y0 + 1, k);
          const sx = splineOrder === InterpolationType.cosine ? 0.5 - Math.cos(xf * Math.PI) * 0.5 : xf;
          const sy = splineOrder === InterpolationType.cosine ? 0.5 - Math.cos(yf * Math.PI) * 0.5 : yf;
          const mx0 = v00 * (1 - sx) + v10 * sx;
          const mx1 = v01 * (1 - sx) + v11 * sx;
          val = mx0 * (1 - sy) + mx1 * sy;
        } else {
          const rows = [];
          for (let m = -1; m < 3; m++) {
            rows[m + 1] = cubic(
              sampleWrapped(x0 - 1, y0 + m, k),
              sampleWrapped(x0, y0 + m, k),
              sampleWrapped(x0 + 1, y0 + m, k),
              sampleWrapped(x0 + 2, y0 + m, k),
              xf
            );
          }
          val = cubic(rows[0], rows[1], rows[2], rows[3], yf);
        }
        out[(y * nw + x) * nc + k] = val;
      }
    }
  }

  return Tensor.fromArray(tensor.ctx, out, [nh, nw, nc]);
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

export function proportionalDownsample(tensor, shape, newShape) {
  const [h, w, c] = shape;
  const [nh, nw] = newShape;
  const kH = Math.max(1, Math.floor(h / nh));
  const kW = Math.max(1, Math.floor(w / nw));
  const outH = Math.floor((h - kH) / kH + 1);
  const outW = Math.floor((w - kW) / kW + 1);
  const ctx = tensor.ctx;
  const src = tensor.read();
  const out = new Float32Array(outH * outW * c);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        for (let yy = 0; yy < kH; yy++) {
          for (let xx = 0; xx < kW; xx++) {
            const iy = y * kH + yy;
            const ix = x * kW + xx;
            const idx = (iy * w + ix) * c + k;
            sum += src[idx];
          }
        }
        out[(y * outW + x) * c + k] = sum / (kH * kW);
      }
    }
  }
  const down = Tensor.fromArray(ctx, out, [outH, outW, c]);
  return resample(down, [nh, nw, c]);
}

export function upsample(tensor, factor, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  return resample(tensor, [h * factor, w * factor, c], splineOrder);
}

function cubicInterpolate(a, b, c, d, t) {
  const t2 = t * t;
  const a0 = d - c - a + b;
  const a1 = a - b - a0;
  const a2 = c - a;
  const a3 = b;
  return a0 * t * t2 + a1 * t2 + a2 * t + a3;
}

export function warp(tensor, flow, amount = 1, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const flowData = flow.read();
  const out = new Float32Array(h * w * c);

  const wrap = (v, max) => {
    v %= max;
    return v < 0 ? v + max : v;
  };

  function sample(x, y, k) {
    x = wrap(x, w);
    y = wrap(y, h);
    if (splineOrder === InterpolationType.constant) {
      const ix = wrap(Math.round(x), w);
      const iy = wrap(Math.round(y), h);
      return src[(iy * w + ix) * c + k];
    }

    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;

    if (splineOrder === InterpolationType.bicubic) {
      const get = (ix, iy) => {
        ix = wrap(ix, w);
        iy = wrap(iy, h);
        return src[(iy * w + ix) * c + k];
      };
      const col = new Array(4);
      for (let m = -1; m < 3; m++) {
        const row = new Array(4);
        for (let n = -1; n < 3; n++) {
          row[n + 1] = get(x0 + n, y0 + m);
        }
        col[m + 1] = cubicInterpolate(row[0], row[1], row[2], row[3], fx);
      }
      return cubicInterpolate(col[0], col[1], col[2], col[3], fy);
    }

    const x1 = wrap(x0 + 1, w);
    const y1 = wrap(y0 + 1, h);
    const interp = splineOrder === InterpolationType.cosine
      ? (t) => 0.5 - Math.cos(t * Math.PI) * 0.5
      : (t) => t;
    const tx = interp(fx);
    const ty = interp(fy);
    const s00 = src[(y0 * w + x0) * c + k];
    const s10 = src[(y0 * w + x1) * c + k];
    const s01 = src[(y1 * w + x0) * c + k];
    const s11 = src[(y1 * w + x1) * c + k];
    const x_y0 = s00 * (1 - tx) + s10 * tx;
    const x_y1 = s01 * (1 - tx) + s11 * tx;
    return x_y0 * (1 - ty) + x_y1 * ty;
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = flowData[(y * w + x) * 2] * amount * w;
      const dy = flowData[(y * w + x) * 2 + 1] * amount * h;
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
  const bChannels = b.shape[2];
  if (ctx && !ctx.isCPU && b.ctx === ctx && typeof t === 'number' && bChannels === c) {
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
  const [bh, bw, bc] = b.shape;
  const [th, tw, tc] = dt ? t.shape : [0, 0, 0];
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    const by = y % bh;
    const ty = dt ? y % th : 0;
    for (let x = 0; x < w; x++) {
      const bx = x % bw;
      const tx = dt ? x % tw : 0;
      const baseA = (y * w + x) * c;
      const baseB = (by * bw + bx) * bc;
      const baseT = dt ? (ty * tw + tx) * tc : 0;
      for (let k = 0; k < c; k++) {
          const aVal = da[baseA + k];
          const bVal = db[baseB + (k < bc ? k : 0)];
          const tVal = dt ? dt[baseT + (k < tc ? k : 0)] : t;
          out[baseA + k] = Math.fround(
            aVal * (1 - tVal) + bVal * tVal,
          );
      }
    }
  }
  return Tensor.fromArray(ctx, out, [h, w, c]);
}

export function normalize(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  min = Math.fround(min);
  max = Math.fround(max);
  const range = Math.fround(max - min) || Math.fround(1);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const diff = Math.fround(src[i] - min);
    out[i] = Math.fround(diff / range);
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
    default: {
      const sum = Math.fround(Math.fround(dx * dx) + Math.fround(dy * dy));
      return Math.fround(Math.sqrt(sum));
    }
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
    const H = Math.fround(src[i * c]);
    const S = Math.fround(src[i * c + 1]);
    const V = Math.fround(src[i * c + 2]);
    const C = Math.fround(V * S);
    const hPrime = Math.fround(Math.fround(H * 6) % 6);
    const X = Math.fround(C * Math.fround(1 - Math.abs(Math.fround(hPrime % 2) - 1)));
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
    const m = Math.fround(V - C);
    out[i * 3] = Math.fround(r1 + m);
    out[i * 3 + 1] = Math.fround(g1 + m);
    out[i * 3 + 2] = Math.fround(b1 + m);
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
        hVal = Math.fround(((g - b) / d) % 6);
      } else if (max === g) {
        hVal = Math.fround((b - r) / d + 2);
      } else {
        hVal = Math.fround((r - g) / d + 4);
      }
      hVal = Math.fround(hVal / 6);
      if (hVal < 0) hVal += 1;
      const sVal = max === 0 ? 0 : Math.fround(d / max);
      out[i * 3] = Math.fround(hVal);
      out[i * 3 + 1] = Math.fround(sVal);
      out[i * 3 + 2] = Math.fround(max);
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
            const contrib = Math.fround(kernel[j][i] * val);
            sum = Math.fround(sum + contrib);
          }
        }
        out[(y * w + x) * c + k] = Math.fround(sum);
      }
    }
  }
  let result = Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  if (doNormalize) result = normalize(result);
  if (alpha !== 1) result = blend(tensor, result, alpha);
  return result;
}

export function refract(
  tensor,
  referenceX = null,
  referenceY = null,
  displacement = 0.5,
  splineOrder = InterpolationType.bicubic,
  signedRange = true,
) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const rx = (referenceX || tensor).read();
  const ry = (referenceY || tensor).read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let vx = rx[y * w + x];
      let vy = ry[y * w + x];
      if (signedRange) {
        vx = vx * 2 - 1;
        vy = vy * 2 - 1;
      } else {
        vx *= 2;
        vy *= 2;
      }
      const dx = Math.fround(vx * displacement * w);
      const dy = Math.fround(vy * displacement * h);
      let x0 = x + Math.trunc(dx);
      let y0 = y + Math.trunc(dy);
      let x1 = x0 + 1;
      let y1 = y0 + 1;
      x0 = ((x0 % w) + w) % w;
      x1 = ((x1 % w) + w) % w;
      y0 = ((y0 % h) + h) % h;
      y1 = ((y1 % h) + h) % h;
      const fx = Math.fround(dx - Math.floor(dx));
      const fy = Math.fround(dy - Math.floor(dy));
      const tx = splineOrder === InterpolationType.cosine
        ? Math.fround(0.5 - Math.cos(fx * Math.PI) * 0.5)
        : fx;
      const ty = splineOrder === InterpolationType.cosine
        ? Math.fround(0.5 - Math.cos(fy * Math.PI) * 0.5)
        : fy;
      for (let k = 0; k < c; k++) {
        const s00 = src[(y0 * w + x0) * c + k];
        const s10 = src[(y0 * w + x1) * c + k];
        const s01 = src[(y1 * w + x0) * c + k];
        const s11 = src[(y1 * w + x1) * c + k];
        const x_y0 = s00 * (1 - tx) + s10 * tx;
        const x_y1 = s01 * (1 - tx) + s11 * tx;
        out[(y * w + x) * c + k] = x_y0 * (1 - ty) + x_y1 * ty;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
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
