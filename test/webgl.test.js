import { spawnSync } from 'child_process';
import { existsSync } from 'fs';

// Allow overriding the Chrome path via environment variables
const envCandidates = [
  process.env.CHROME,
  process.env.CHROME_PATH,
  process.env.CHROME_BIN
];

let chrome = envCandidates.find(p => p && existsSync(p)) || null;

// Search common executable names in PATH
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
    if (which.status === 0) {
      chrome = c;
      break;
    }
  }
}

// macOS default install location
if (!chrome && process.platform === 'darwin') {
  const macPath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  if (existsSync(macPath)) {
    chrome = macPath;
  }
}

// Windows default install locations
if (!chrome && process.platform === 'win32') {
  const winPaths = [
    process.env['PROGRAMFILES'] && `${process.env['PROGRAMFILES']}\\Google\\Chrome\\Application\\chrome.exe`,
    process.env['PROGRAMFILES(X86)'] && `${process.env['PROGRAMFILES(X86)']}\\Google\\Chrome\\Application\\chrome.exe`
  ];
  for (const p of winPaths) {
    if (p && existsSync(p)) {
      chrome = p;
      break;
    }
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
