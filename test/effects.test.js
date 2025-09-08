import assert from 'assert';
import { Tensor } from '../src/tensor.js';
import {
  posterize,
  palette,
  invert,
  aberration,
  reindex,
  vignette,
  dither,
  grain,
  saturation,
  randomHue,
} from '../src/effects.js';
import { adjustHue, rgbToHsv, hsvToRgb, values, blend } from '../src/value.js';
import { setSeed, random } from '../src/util.js';
import { spawnSync } from 'child_process';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

// posterize regression
const posterData = new Float32Array([0.1, 0.5, 0.9, 0.3]);
const posterTensor = Tensor.fromArray(null, posterData, [2, 2, 1]);
const jsPoster = posterize(posterTensor, [2, 2, 1], 0, 1, 4).read();
const posterPy = spawnSync('python', ['-'], {
  input: `import json, tensorflow as tf\nfrom noisemaker.effects import posterize\nvals=${JSON.stringify(Array.from(posterData))}\ntex=tf.constant(vals, shape=[2,2,1], dtype=tf.float32)\nout=posterize(tex,[2,2,1],levels=4)\nprint(json.dumps(out.numpy().flatten().tolist()))`,
  encoding: 'utf8'
});
const posterExpected = JSON.parse(posterPy.stdout.trim());
arraysClose(Array.from(jsPoster), posterExpected);

// palette regression
const palData = new Float32Array([0.0, 0.5, 1.0, 0.25]);
const palTensor = Tensor.fromArray(null, palData, [2, 2, 1]);
const jsPal = palette(palTensor, [2, 2, 1], 0, 1, 'grayscale').read();
const palPy = spawnSync('python', ['-'], {
  input: `import json, math\nfrom noisemaker.palettes import PALETTES\nvals=[0.0,0.5,1.0,0.25]\np=PALETTES['grayscale']\nout=[]\nfor t in vals:\n    r=p['offset'][0]+p['amp'][0]*math.cos(math.tau*(p['freq'][0]*t*0.875+0.0625+p['phase'][0]))\n    g=p['offset'][1]+p['amp'][1]*math.cos(math.tau*(p['freq'][1]*t*0.875+0.0625+p['phase'][1]))\n    b=p['offset'][2]+p['amp'][2]*math.cos(math.tau*(p['freq'][2]*t*0.875+0.0625+p['phase'][2]))\n    out.extend([r,g,b])\nprint(json.dumps(out))`,
  encoding: 'utf8'
});
const palExpected = JSON.parse(palPy.stdout.trim());
arraysClose(Array.from(jsPal), palExpected);

// invert
const invData = new Float32Array([0.2, 0.5, 0.8]);
const invTensor = Tensor.fromArray(null, invData, [1, 3, 1]);
const invResult = invert(invTensor, [1, 3, 1], 0, 1).read();
arraysClose(Array.from(invResult), [0.8, 0.5, 0.2]);

// aberration deterministic check
setSeed(123);
const abShape = [1, 4, 3];
const abData = new Float32Array([
  0.1, 0.2, 0.3,
  0.4, 0.5, 0.6,
  0.7, 0.8, 0.9,
  0.2, 0.4, 0.6,
]);
const abTensor = Tensor.fromArray(null, abData, abShape);
const disp = Math.round(abShape[1] * 0.25 * random());
const hueShift = random() * 0.1 - 0.05;
const shifted = adjustHue(abTensor, hueShift).read();
const manual = new Float32Array(abShape[0] * abShape[1] * 3);
for (let x = 0; x < abShape[1]; x++) {
  const base = x * 3;
  const rIdx = Math.min(abShape[1] - 1, x + disp) * 3;
  const bIdx = Math.max(0, x - disp) * 3;
  manual[base] = shifted[rIdx];
  manual[base + 1] = shifted[base + 1];
  manual[base + 2] = shifted[bIdx + 2];
}
const expected = adjustHue(Tensor.fromArray(null, manual, abShape), -hueShift).read();
setSeed(123);
const abResult = aberration(abTensor, abShape, 0, 1, 0.25).read();
arraysClose(Array.from(abResult), Array.from(expected));

// reindex deterministic
const reData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const reTensor = Tensor.fromArray(null, reData, [2, 2, 1]);
const manualRe = new Float32Array(4);
const mod = 2;
for (let y = 0; y < 2; y++) {
  for (let x = 0; x < 2; x++) {
    const idx = y * 2 + x;
    const r = reData[idx];
    const xo = Math.floor((r * 0.5 * mod + r) % 2);
    const yo = Math.floor((r * 0.5 * mod + r) % 2);
    manualRe[idx] = reData[yo * 2 + xo];
  }
}
const jsRe = reindex(reTensor, [2, 2, 1], 0, 1, 0.5).read();
arraysClose(Array.from(jsRe), Array.from(manualRe));

// vignette regression (numpy impl)
const vigData = new Float32Array([0.1, 0.5, 0.3, 0.8]);
const vigTensor = Tensor.fromArray(null, vigData, [2, 2, 1]);
const jsVig = vignette(vigTensor, [2, 2, 1], 0, 1, 0.25, 0.5).read();
const vigPy = spawnSync('python', ['-'], {
  input: `import json, math, numpy as np\nvals=${JSON.stringify(Array.from(vigData))}\narr=np.array(vals).reshape(2,2,1)\nminv=arr.min(); maxv=arr.max(); norm=(arr-minv)/(maxv-minv)\ncx=(2-1)/2; cy=(2-1)/2; maxd=math.sqrt(cx*cx+cy*cy)\nmask=np.zeros((2,2,1))\nfor y in range(2):\n  for x in range(2):\n    dx=x-cx; dy=y-cy\n    dist=math.sqrt(dx*dx+dy*dy)/maxd\n    mask[y,x,0]=dist**2\nedges=np.ones((2,2,1))*0.25\nvig=norm*(1-mask)+edges*mask\nfinal=norm*(1-0.5)+vig*0.5\nprint(json.dumps(final.flatten().tolist()))`,
  encoding: 'utf8',
});
const vigExpected = JSON.parse(vigPy.stdout.trim());
arraysClose(Array.from(jsVig), vigExpected);

// dither deterministic
setSeed(1);
const ditData = new Float32Array([0.2, 0.4, 0.6, 0.8]);
const ditTensor = Tensor.fromArray(null, ditData, [2, 2, 1]);
setSeed(1);
const noise = values(Math.max(2,2), [2,2,1], { time:0, seed:0, speed:1000 });
const nData = noise.read();
const manualDit = new Float32Array(4);
for (let i=0;i<4;i++){let v=ditData[i]+(nData[i]-0.5)/2; v=Math.floor(Math.min(1,Math.max(0,v))*2)/2; manualDit[i]=v;}
setSeed(1);
const jsDit = dither(ditTensor, [2,2,1], 0,1,2).read();
arraysClose(Array.from(jsDit), Array.from(manualDit));

// grain deterministic
setSeed(2);
const grData = new Float32Array([0.3,0.6,0.9,0.0]);
const grTensor = Tensor.fromArray(null, grData, [2,2,1]);
setSeed(2);
const gn = values(Math.max(2,2), [2,2,1], { time:0, speed:200 });
const gnData = gn.read();
const blended = blend(grTensor, Tensor.fromArray(null, (()=>{const arr=new Float32Array(4); for(let i=0;i<4;i++) arr[i]=gnData[i]; return arr;})(), [2,2,1]), 0.25).read();
setSeed(2);
const jsGrain = grain(grTensor, [2,2,1], 0,1,0.25).read();
arraysClose(Array.from(jsGrain), Array.from(blended));

// saturation regression via python colorsys
const satData = new Float32Array([
  0.2,0.4,0.6,
  0.8,0.1,0.3,
]);
const satTensor = Tensor.fromArray(null, satData, [2,1,3]);
const jsSat = saturation(satTensor, [2,1,3],0,1,0.5).read();
const satPy = spawnSync('python',['-'],{
  input:`import json, colorsys\nvals=${JSON.stringify(Array.from(satData))}\nout=[]\nfor i in range(0,len(vals),3):\n r,g,b=vals[i:i+3]\n h,s,v=colorsys.rgb_to_hsv(r,g,b)\n s=min(1,max(0,s*0.5))\n r,g,b=colorsys.hsv_to_rgb(h,s,v)\n out.extend([r,g,b])\nprint(json.dumps(out))`,
  encoding:'utf8'
});
const satExpected=JSON.parse(satPy.stdout.trim());
arraysClose(Array.from(jsSat), satExpected);

// randomHue deterministic
setSeed(5);
const rhData = new Float32Array([
  0.1,0.2,0.3,
  0.4,0.5,0.6,
]);
const rhTensor = Tensor.fromArray(null, rhData, [2,1,3]);
setSeed(5);
const shift = random()*0.1-0.05;
const pyHue = spawnSync('python',['-'],{
  input:`import json, colorsys\nvals=${JSON.stringify(Array.from(rhData))}\nshift=${shift}\nout=[]\nfor i in range(0,len(vals),3):\n r,g,b=vals[i:i+3]\n h,s,v=colorsys.rgb_to_hsv(r,g,b)\n h=(h+shift)%1\n r,g,b=colorsys.hsv_to_rgb(h,s,v)\n out.extend([r,g,b])\nprint(json.dumps(out))`,
  encoding:'utf8'
});
const hueExpected=JSON.parse(pyHue.stdout.trim());
setSeed(5);
const jsHue = randomHue(rhTensor,[2,1,3],0,1,0.05).read();
arraysClose(Array.from(jsHue), hueExpected);

console.log('Effects tests passed');
