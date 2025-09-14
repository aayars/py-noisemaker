import { spawnSync } from 'child_process';
import { existsSync } from 'fs';

const envCandidates = [
  process.env.CHROME,
  process.env.CHROME_PATH,
  process.env.CHROME_BIN
];
let chrome = envCandidates.find(p => p && existsSync(p)) || null;
if (!chrome) {
  const whichCmd = process.platform === 'win32' ? 'where' : 'which';
  const candidates = [
    'google-chrome',
    'chromium-browser',
    'chromium',
    'chrome',
    'google-chrome-stable'
  ];
  for (const c of candidates) {
    const which = spawnSync(whichCmd, [c]);
    if (which.status === 0) { chrome = c; break; }
  }
}
if (!chrome) {
  console.log('Chrome not found, skipping WebGPU tests');
  process.exit(0);
}
const res = spawnSync(chrome, [
  '--headless',
  '--enable-unsafe-webgpu',
  '--disable-dawn-features=disallow_unsafe_apis',
  '--use-angle=swiftshader',
  '--disable-gpu',
  '--dump-dom',
  'test/webgpu.test.html'
], { encoding: 'utf8' });
if (res.status !== 0) {
  console.error(res.stderr || res.stdout);
  process.exit(res.status);
}
if (!res.stdout.includes('webgpu ok')) {
  console.error(res.stderr || res.stdout || 'WebGPU test failed');
  process.exit(1);
}
console.log('WebGPU tests passed (including wobble and CRT parity)');
