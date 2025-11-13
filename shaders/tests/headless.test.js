#!/usr/bin/env node
/*
Headless shader test harness using Puppeteer (headless Chrome + WebGPU).
- Serves the repo root statically so /demo/gpu-effects/index.html can fetch /shaders/manifest.json and shader files.
- Opens the GPU effects demo in headless Chrome with WebGPU enabled, selects specific shaders, renders one frame,
  reads back the 2D canvas pixels, and validates deterministic sanity metrics.
- No changes to /js or /noisemaker code.

Requirements:
- Node 18+
- Chrome with WebGPU support (Puppeteer will download a compatible Chromium by default).
- macOS/Linux CI: ensure `--enable-unsafe-webgpu` is allowed; Puppeteer args include this flag.

Usage:
  node scripts/shaders/headless.test.js [--shader simple_frame/simple_frame] [--out artifacts/] [--size 128] [--time 0.0] [--seed 0]

Exit status is non-zero on failure.
*/

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { createReadStream } from 'node:fs';
import puppeteer from 'puppeteer';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

function parseArgs(argv) {
  const out = { shader: null, outDir: null, size: 128, time: 0.0, seed: 0, all: false, retries: 2 };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--shader') out.shader = argv[++i];
    else if (a === '--out') out.outDir = argv[++i];
    else if (a === '--size') out.size = parseInt(argv[++i], 10) || out.size;
    else if (a === '--time') out.time = parseFloat(argv[++i]);
    else if (a === '--seed') out.seed = parseFloat(argv[++i]);
    else if (a === '--all') out.all = true;
    else if (a === '--retries') out.retries = Math.max(0, parseInt(argv[++i], 10) || 0);
  }
  return out;
}

function mimeType(p) {
  if (p.endsWith('.html')) return 'text/html; charset=utf-8';
  if (p.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (p.endsWith('.json')) return 'application/json; charset=utf-8';
  if (p.endsWith('.wgsl')) return 'text/plain; charset=utf-8';
  if (p.endsWith('.png')) return 'image/png';
  if (p.endsWith('.css')) return 'text/css; charset=utf-8';
  return 'application/octet-stream';
}

function startStaticServer(rootDir, port = 0) {
  const server = http.createServer((req, res) => {
    try {
      const u = new URL(req.url, 'http://localhost');
      let pathname = decodeURIComponent(u.pathname);
      if (pathname === '/') pathname = '/index.html';

      const filePath = path.join(rootDir, pathname);
      if (!filePath.startsWith(rootDir)) {
        res.writeHead(403).end('Forbidden');
        return;
      }
      fs.stat(filePath, (err, st) => {
        if (err) {
          res.writeHead(404).end('Not found');
          return;
        }
        if (st.isDirectory()) {
          const indexPath = path.join(filePath, 'index.html');
          if (fs.existsSync(indexPath)) {
            res.writeHead(200, { 'Content-Type': mimeType(indexPath) });
            createReadStream(indexPath).pipe(res);
          } else {
            res.writeHead(403).end('Forbidden');
          }
          return;
        }
        res.writeHead(200, { 'Content-Type': mimeType(filePath) });
        createReadStream(filePath).pipe(res);
      });
    } catch (e) {
      res.writeHead(500).end('Internal error');
    }
  });

  server.listen(port);
  return new Promise((resolve) => {
    server.on('listening', () => {
      const address = server.address();
      resolve({ server, port: address.port });
    });
  });
}

async function main() {
  const args = parseArgs(process.argv);

  const { server, port } = await startStaticServer(REPO_ROOT, 0);
  const base = `http://localhost:${port}`;

  async function launch(headlessMode) {
    const args = [
      '--enable-unsafe-webgpu',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ];
    if (process.platform === 'darwin') {
      // Prefer ANGLE Metal on macOS
      args.push('--use-angle=metal');
    } else {
      // EGL can help on Linux
      args.push('--use-gl=egl');
    }
    return puppeteer.launch({ headless: headlessMode, args });
  }

  let browser = await launch('new');

  try {
  let page = await browser.newPage();

    // Pipe page console logs for easier debugging
    page.on('console', (msg) => {
      const type = msg.type();
      const text = msg.text();
      // Reduce noise: only forward warnings/errors/logs
      if (type === 'warning' || type === 'error' || type === 'log') {
        console[type === 'warning' ? 'warn' : type](text);
      }
    });

    // Force a deterministic viewport and device scale so canvas pixels are exact.
    await page.setViewport({ width: 1200, height: 800, deviceScaleFactor: 1 });

  // Navigate directly to the GPU effects demo so relative fetches work
  await page.goto(`${base}/demo/gpu-effects/index.html`, { waitUntil: 'networkidle2' });

    // Ensure WebGPU is available
    let hasGPU = await page.evaluate(() => !!navigator.gpu);
    if (!hasGPU) {
      // Retry in non-headless mode for environments where headless WebGPU is blocked
      await browser.close();
      browser = await launch(false);
      const page2 = await browser.newPage();
      await page2.setViewport({ width: 1200, height: 800, deviceScaleFactor: 1 });
  await page2.goto(`${base}/demo/gpu-effects/index.html`, { waitUntil: 'networkidle2' });
      hasGPU = await page2.evaluate(() => !!navigator.gpu);
      if (!hasGPU) throw new Error('WebGPU not available in Chromium');
      // Use the second page for the remainder
      await page.close();
      page = page2;
    }

    await page.waitForFunction(() => {
      return Boolean(window.__demo__ && window.__demo__.setActiveEffect && window.__demo__.runMultiresOnce);
    }, { timeout: 15000 });

    // Load manifest and pick shaders
    const manifest = await page.evaluate(async () => {
      const res = await fetch('/shaders/manifest.json');
      return res.json();
    });

    const shadersToTest = args.shader
      ? manifest.filter((m) => m.label === args.shader || m.path.endsWith(args.shader + '.wgsl'))
      : args.all
        ? manifest
        : manifest.filter((m) => /simple_frame\/simple_frame|palette\/palette/.test(m.path));

    if (!shadersToTest.length) {
      throw new Error(`No shaders matched ${args.shader || '(default set)'}`);
    }

    const failures = [];
    let index = 0;

    for (const entry of shadersToTest) {
      index += 1;
      const label = entry.label;
      const pathRel = entry.path; // like /shaders/effects/simple_frame/simple_frame.wgsl
      let result = null;
      let attempt = 0;
      for (; attempt <= args.retries; attempt++) {
        result = await page.evaluate(async (shaderPath, size, t, seed) => {
          const demo = window.__demo__;
          if (!demo) {
            return { error: 'Demo helpers unavailable' };
          }

          const canvas = document.getElementById('canvas');
          if (!canvas) {
            return { error: 'Canvas not found' };
          }

          canvas.width = size;
          canvas.height = size;

          const segments = shaderPath.split('/');
          const effectId = segments.length >= 5 ? segments[3] : shaderPath;

          try {
            await demo.setActiveEffect(effectId);
            if (demo.updateActiveEffectParams) {
              await demo.updateActiveEffectParams({ enabled: true });
            }
          } catch (err) {
            return { error: `Failed to activate effect '${effectId}': ${err?.message ?? err}` };
          }

          try {
            await demo.runMultiresOnce({ seed, time: t, frameIndex: 0 });
          } catch (err) {
            return { error: `Render failed: ${err?.message ?? err}` };
          }

          let floats;
          try {
            floats = await demo.readActiveEffectOutputFloats();
          } catch (err) {
            return { error: `Readback failed: ${err?.message ?? err}` };
          }

          if (!floats || floats.length === 0) {
            return { error: 'Effect produced no output' };
          }

          let sum = 0;
          let sumSq = 0;
          let nonZero = 0;
          for (let i = 0; i < floats.length; i += 4) {
            const r = floats[i];
            const g = floats[i + 1];
            const b = floats[i + 2];
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum += luminance;
            sumSq += luminance * luminance;
            if (r !== 0 || g !== 0 || b !== 0) {
              nonZero += 1;
            }
          }

          const n = floats.length / 4;
          const mean = sum / n;
          const variance = sumSq / n - mean * mean;
          const stddev = Math.sqrt(Math.max(variance, 0));

          return { width: size, height: size, n, mean, stddev, nonZero };
        }, pathRel, args.size, args.time, args.seed);
        if (!result || result.error) {
          // brief backoff before retry
          await new Promise(r => setTimeout(r, 150));
          continue;
        }
        break;
      }

      if (result && result.error) {
        failures.push(`${label}: ${result.error}`);
        console.error(`${label}: ${result.error}`);
        continue;
      }

      // Basic sanity thresholds: avoid blank frames, avoid NaN/Inf
      const epsilon = 1e-6;
      if (!Number.isFinite(result.mean) || !Number.isFinite(result.stddev)) {
        failures.push(`${label}: non-finite stats`);
      } else if (result.nonZero < result.n * 0.01) {
        failures.push(`${label}: mostly blank output`);
      } else if (result.stddev < epsilon) {
        failures.push(`${label}: zero variance`);
      }

      if (args.outDir) {
        // Capture only the canvas pixels for debugging
  const pngB64 = await page.$eval('#canvas', (canvas) => canvas.toDataURL('image/png').split(',')[1]);
        const outDir = path.resolve(REPO_ROOT, args.outDir);
        fs.mkdirSync(outDir, { recursive: true });
        const baseName = `${index.toString().padStart(2, '0')}_${label.replace(/\//g,'-')}`;
        const outPng = path.join(outDir, `${baseName}.png`);
        fs.writeFileSync(outPng, Buffer.from(pngB64, 'base64'));
        const outJson = path.join(outDir, `${baseName}.json`);
        fs.writeFileSync(outJson, JSON.stringify({
          shaderLabel: label,
          shaderPath: pathRel,
          width: result.width,
          height: result.height,
          pixels: result.n,
          mean: result.mean,
          stddev: result.stddev,
          nonZero: result.nonZero,
          params: { size: args.size, time: args.time, seed: args.seed },
          timestamp: new Date().toISOString(),
        }, null, 2));
        console.log('wrote', outPng, 'and', outJson);
      }

      console.log(`${label}: mean=${result.mean.toFixed(3)} stddev=${result.stddev.toFixed(3)} nonZero=${result.nonZero}/${result.n}`);
    }

    if (failures.length) {
      console.error('Shader test failures:\n - ' + failures.join('\n - '));
      process.exit(1);
    }
  } finally {
    await browser.close();
    server.close();
    await once(server, 'close');
  }
}

main().catch((err) => {
  console.error(err.stack || err);
  process.exit(1);
});
