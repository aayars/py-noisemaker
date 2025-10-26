#!/usr/bin/env node
/**
 * Visual diff test for shader effects
 * 
 * Usage:
 *   node test/visual-diff-effect.js <effect-name> [options]
 * 
 * Options:
 *   --seed <number>     Seed value (default: 12345)
 *   --time <number>     Time value (default: 0.0)
 *   --frame <number>    Frame index (default: 0)
 *   --baseline <name>   Baseline effect to compare against (default: multires only)
 *   --headless          Run in headless mode (default: false for inspection)
 * 
 * Examples:
 *   node test/visual-diff-effect.js normalize
 *   node test/visual-diff-effect.js normalize --seed 999 --time 1.5
 *   node test/visual-diff-effect.js bloom --baseline aberration
 *   node test/visual-diff-effect.js ridge --headless
 */

import { chromium } from 'playwright';
import { createCanvas, loadImage } from 'canvas';
import { execSync } from 'child_process';
import { existsSync } from 'fs';

// Parse command line arguments
const args = process.argv.slice(2);
const effectName = args[0];

if (!effectName) {
  console.error('Error: Effect name is required');
  console.error('Usage: node test/visual-diff-effect.js <effect-name> [options]');
  process.exit(1);
}

const getArg = (flag, defaultValue) => {
  const index = args.indexOf(flag);
  return index !== -1 && args[index + 1] ? args[index + 1] : defaultValue;
};

const hasFlag = (flag) => args.includes(flag);

const SEED = parseInt(getArg('--seed', '12345'), 10);
const TIME = parseFloat(getArg('--time', '0.0'));
const FRAME = parseInt(getArg('--frame', '0'), 10);
const BASELINE = getArg('--baseline', null);
const HEADLESS = hasFlag('--headless');

console.log(`
╔════════════════════════════════════════════════════════════════════╗
║            SHADER EFFECT VISUAL DIFF TEST                          ║
╚════════════════════════════════════════════════════════════════════╝

Effect:   ${effectName}
Seed:     ${SEED}
Time:     ${TIME}
Frame:    ${FRAME}
Baseline: ${BASELINE || 'multires only'}
Mode:     ${HEADLESS ? 'headless' : 'interactive'}
`);

async function runTest() {
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext();
  const page = await context.newPage();

  const errors = [];
  const warnings = [];

  page.on('console', (msg) => {
    const text = msg.text();
    const type = msg.type();
    
    if (type === 'error') {
      errors.push(text);
      console.error(`  [ERROR] ${text}`);
    } else if (type === 'warning' && !text.includes('Non-contiguous')) {
      warnings.push(text);
    }
  });

  page.on('pageerror', (error) => {
    errors.push(error.message);
    console.error(`  [PAGE ERROR] ${error.message}`);
  });

  try {
    // Load demo
    console.log('Loading GPU effects demo...');
    await page.goto('http://localhost:9090/demo/gpu-effects/index.html', {
      waitUntil: 'networkidle',
      timeout: 30000
    });

    await page.waitForTimeout(2000);

    const hasWebGPU = await page.evaluate(() => 'gpu' in navigator);
    if (!hasWebGPU) {
      throw new Error('WebGPU is not supported');
    }

    await page.waitForSelector('#effect-selector', { timeout: 10000 });

    // Check if effect exists
    const effectExists = await page.evaluate((name) => {
      const select = document.querySelector('#effect-selector');
      return Array.from(select.options).some(opt => opt.value === name);
    }, effectName);

    if (!effectExists) {
      throw new Error(`Effect '${effectName}' not found in effect selector`);
    }

    // Function to render with fixed parameters
    const renderWithParams = async (effectToUse) => {
      await page.selectOption('#effect-selector', effectToUse);
      await page.evaluate(({ seed, time, frame }) => {
        if (window.__demo__?.runMultiresOnce) {
          window.__demo__.runMultiresOnce({ seed, time, frameIndex: frame });
        }
      }, { seed: SEED, time: TIME, frame: FRAME });
      await page.waitForTimeout(1000);
    };

    // Render baseline
    console.log(`\n${BASELINE ? `Rendering baseline (${BASELINE})` : 'Rendering baseline (multires only)'}...`);
    if (BASELINE) {
      await renderWithParams(BASELINE);
    } else {
      // Ensure no effect is active - just render multires
      await page.evaluate(() => {
        const select = document.querySelector('#effect-selector');
        if (select.options.length > 0) {
          select.selectedIndex = 0;
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
      await page.waitForTimeout(500);
    }
    const baselinePath = '/tmp/effect-test-baseline.png';
    await page.screenshot({ path: baselinePath });
    console.log(`  ✓ Saved to ${baselinePath}`);

    // Check for errors after baseline
    if (errors.length > 0) {
      throw new Error('Errors detected during baseline render');
    }

    // Render with effect
    console.log(`\nRendering with effect (${effectName})...`);
    await renderWithParams(effectName);
    const effectPath = '/tmp/effect-test-effect.png';
    await page.screenshot({ path: effectPath });
    console.log(`  ✓ Saved to ${effectPath}`);

    // Check for errors after effect
    if (errors.length > 0) {
      console.error(`\n❌ ERRORS DETECTED (${errors.length}):`);
      errors.forEach(err => console.error(`   ${err}`));
      await browser.close();
      process.exit(1);
    }

    console.log('  ✓ No console errors');

    if (!HEADLESS) {
      console.log('\nBrowser will remain open for visual inspection.');
      console.log('Press Ctrl+C when done.\n');
      await new Promise(() => {}); // Wait forever
    } else {
      await browser.close();
    }

    // Analyze pixel differences
    console.log('\n' + '═'.repeat(70));
    console.log('PIXEL ANALYSIS');
    console.log('═'.repeat(70));

    const baselineImg = await loadImage(baselinePath);
    const effectImg = await loadImage(effectPath);

    const baselineCanvas = createCanvas(baselineImg.width, baselineImg.height);
    const baselineCtx = baselineCanvas.getContext('2d');
    baselineCtx.drawImage(baselineImg, 0, 0);

    const effectCanvas = createCanvas(effectImg.width, effectImg.height);
    const effectCtx = effectCanvas.getContext('2d');
    effectCtx.drawImage(effectImg, 0, 0);

    const width = baselineImg.width;
    const height = baselineImg.height;

    // Sample multiple positions
    const positions = [
      [Math.floor(width / 4), Math.floor(height / 4)],
      [Math.floor(width / 2), Math.floor(height / 2)],
      [Math.floor(3 * width / 4), Math.floor(3 * height / 4)],
      [Math.floor(width / 3), Math.floor(2 * height / 3)],
      [Math.floor(2 * width / 3), Math.floor(height / 3)],
    ];

    let baselineMin = 255, baselineMax = 0;
    let effectMin = 255, effectMax = 0;
    let totalDiff = 0;
    let significantChanges = 0;

    console.log('\nSample pixels (baseline → effect):');
    for (const [x, y] of positions) {
      const basePx = baselineCtx.getImageData(x, y, 1, 1).data;
      const effPx = effectCtx.getImageData(x, y, 1, 1).data;

      // Track min/max (excluding alpha)
      for (let i = 0; i < 3; i++) {
        baselineMin = Math.min(baselineMin, basePx[i]);
        baselineMax = Math.max(baselineMax, basePx[i]);
        effectMin = Math.min(effectMin, effPx[i]);
        effectMax = Math.max(effectMax, effPx[i]);
      }

      const rDiff = Math.abs(effPx[0] - basePx[0]);
      const gDiff = Math.abs(effPx[1] - basePx[1]);
      const bDiff = Math.abs(effPx[2] - basePx[2]);
      const avgDiff = (rDiff + gDiff + bDiff) / 3;

      totalDiff += avgDiff;
      if (avgDiff > 5) significantChanges++;

      console.log(`  [${String(x).padStart(4)}, ${String(y).padStart(4)}]: RGB(${String(basePx[0]).padStart(3)}, ${String(basePx[1]).padStart(3)}, ${String(basePx[2]).padStart(3)}) → RGB(${String(effPx[0]).padStart(3)}, ${String(effPx[1]).padStart(3)}, ${String(effPx[2]).padStart(3)}) Δ=${avgDiff.toFixed(1)}`);
    }

    const baselineSpan = baselineMax - baselineMin;
    const effectSpan = effectMax - effectMin;
    const avgChange = totalDiff / positions.length;

    console.log('\n' + '─'.repeat(70));
    console.log(`Baseline range: [${baselineMin}, ${baselineMax}] (span: ${baselineSpan})`);
    console.log(`Effect range:   [${effectMin}, ${effectMax}] (span: ${effectSpan})`);
    console.log(`Average change: ${avgChange.toFixed(2)} per pixel`);
    console.log(`Significant:    ${significantChanges} of ${positions.length} samples`);

    // ImageMagick comparison if available
    if (existsSync('/usr/local/bin/compare') || existsSync('/usr/bin/compare') || existsSync('/opt/homebrew/bin/compare')) {
      try {
        const diffPath = '/tmp/effect-test-diff.png';
        const result = execSync(
          `compare -metric RMSE ${baselinePath} ${effectPath} ${diffPath} 2>&1`,
          { encoding: 'utf8' }
        ).trim();
        console.log(`\nImageMagick RMSE: ${result}`);
        console.log(`Diff saved to:    ${diffPath}`);
      } catch (e) {
        // compare exits with code 1 when images differ
        if (e.stdout) {
          console.log(`\nImageMagick RMSE: ${e.stdout.trim()}`);
          console.log(`Diff saved to:    /tmp/effect-test-diff.png`);
        }
      }
    }

    console.log('\n' + '═'.repeat(70));
    console.log('RESULT');
    console.log('═'.repeat(70));

    if (errors.length > 0) {
      console.log('❌ TEST FAILED: Shader errors detected');
      process.exit(1);
    } else if (significantChanges === 0 && avgChange < 1.0) {
      console.log('⚠️  WARNING: No significant changes detected');
      console.log('   Effect may not be processing correctly');
      console.log('   (or input may be unchanged by this effect)');
    } else {
      console.log('✅ TEST PASSED');
      console.log(`   • No shader errors`);
      console.log(`   • ${significantChanges} significant pixel changes`);
      console.log(`   • Average change: ${avgChange.toFixed(2)}`);
    }

    console.log('\nOutput files:');
    console.log(`  • ${baselinePath}`);
    console.log(`  • ${effectPath}`);
    if (existsSync('/tmp/effect-test-diff.png')) {
      console.log(`  • /tmp/effect-test-diff.png`);
    }
    console.log('');

  } catch (error) {
    console.error('\n❌ TEST FAILED:', error.message);
    await browser.close();
    process.exit(1);
  }
}

runTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
