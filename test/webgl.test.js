import { spawnSync } from 'child_process';

const candidates = ['google-chrome', 'chromium-browser', 'chromium', 'chrome'];
let chrome = null;
for (const c of candidates) {
  const which = spawnSync('which', [c]);
  if (which.status === 0) {
    chrome = c;
    break;
  }
}

if (!chrome) {
  console.log('Chrome not found, skipping WebGL tests');
  process.exit(0);
}

const res = spawnSync(chrome, [
  '--headless',
  '--use-gl=swiftshader',
  '--disable-gpu',
  '--dump-dom',
  'test/webgl.test.html'
], { encoding: 'utf8' });

if (res.status !== 0) {
  console.error(res.stderr || res.stdout);
  process.exit(res.status);
}

if (!res.stdout.includes('webgl ok')) {
  console.error('WebGL test failed');
  process.exit(1);
}

console.log('WebGL tests passed');
