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
  console.log('Chrome not found, skipping Voronoi parity test');
  process.exit(0);
}
const res = spawnSync(chrome, [
  '--headless',
  '--enable-unsafe-webgpu',
  '--disable-dawn-features=disallow_unsafe_apis',
  '--use-angle=swiftshader',
  '--disable-gpu',
  '--dump-dom',
  'test/voronoi-parity.test.html'
], { encoding: 'utf8' });
if (res.status !== 0) {
  console.error(res.stderr || res.stdout);
  process.exit(res.status);
}
if (res.stdout.includes('no webgpu')) {
  console.log('WebGPU not available, skipping Voronoi parity test');
  process.exit(0);
}
if (!res.stdout.includes('voronoi parity ok')) {
  console.error('Voronoi parity test failed');
  process.exit(1);
}
console.log('Voronoi parity test passed');
