#!/usr/bin/env node
/**
 * Build standalone Noisemaker CLI executables using Node.js SEA (Single Executable Apps).
 * Produces true native binaries for the current platform.
 */

import { readFile, writeFile, copyFile, mkdir, chmod } from 'node:fs/promises';
import { join } from 'node:path';
import { execSync, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = join(__dirname, '..');

const PLATFORMS = [
  { os: 'linux', arch: 'x64', ext: '' },
  { os: 'linux', arch: 'arm64', ext: '' },
  { os: 'darwin', arch: 'x64', ext: '' },
  { os: 'darwin', arch: 'arm64', ext: '' },
  { os: 'win32', arch: 'x64', ext: '.exe' },
];

async function createSEAConfig(entryPath, blobPath) {
  // Strip shebang from bundle - it's invalid in SEA context
  let bundleCode = await readFile(entryPath, 'utf-8');
  if (bundleCode.startsWith('#!')) {
    const firstNewline = bundleCode.indexOf('\n');
    bundleCode = bundleCode.substring(firstNewline + 1);
    await writeFile(entryPath, bundleCode, 'utf-8');
    console.log('  Stripped shebang from bundle for SEA compatibility');
  }

  const config = {
    main: entryPath,
    output: blobPath,
    disableExperimentalSEAWarning: true,
  };
  
  const configPath = join(repoRoot, 'dist', 'cli', 'sea-config.json');
  await writeFile(configPath, JSON.stringify(config, null, 2));
  return configPath;
}

async function buildPlatform(platform) {
  const distDir = join(repoRoot, 'dist', 'cli');
  const platformDir = join(distDir, `${platform.os}-${platform.arch}`);
  await mkdir(platformDir, { recursive: true });

  const bundlePath = join(distDir, 'noisemaker-bundle.cjs');
  const blobPath = join(distDir, 'sea-prep.blob');
  const outputPath = join(platformDir, `noisemaker${platform.ext}`);

  console.log(`Building SEA for ${platform.os}-${platform.arch}...`);

  // Create SEA config
  const configPath = await createSEAConfig(bundlePath, blobPath);

  // Generate SEA blob
  execSync(`node --experimental-sea-config "${configPath}"`, {
    cwd: repoRoot,
    stdio: 'inherit',
  });

  // Copy appropriate Node binary
  let nodeBinaryPath;
  if (platform.os === process.platform && platform.arch === process.arch) {
    // Current platform - use runtime Node
    nodeBinaryPath = process.execPath;
  } else {
    // Cross-platform build would require pre-downloaded Node binaries
    throw new Error(
      `Cross-platform builds not yet supported. ` +
      `Build on ${platform.os}-${platform.arch} to create that binary.`
    );
  }

  // Use cp command to preserve permissions and avoid EACCES
  const cpResult = spawnSync('cp', [nodeBinaryPath, outputPath], { stdio: 'inherit' });
  if (cpResult.status !== 0) {
    throw new Error(`Failed to copy Node binary: exit code ${cpResult.status}`);
  }

  // Ensure output is writable for postject
  await chmod(outputPath, 0o755);

  // On macOS, remove signature before injection
  if (platform.os === 'darwin') {
    try {
      execSync(`codesign --remove-signature "${outputPath}"`, { cwd: repoRoot, stdio: 'pipe' });
    } catch (err) {
      // Signature removal may fail if binary is unsigned; that's fine
    }
  }

  // Inject SEA blob using postject
  const sentinelFuse = 'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2';
  const postjectCmd = platform.os === 'darwin'
    ? `npx postject "${outputPath}" NODE_SEA_BLOB "${blobPath}" --sentinel-fuse ${sentinelFuse} --macho-segment-name NODE_SEA`
    : `npx postject "${outputPath}" NODE_SEA_BLOB "${blobPath}" --sentinel-fuse ${sentinelFuse}`;

  execSync(postjectCmd, { cwd: repoRoot, stdio: 'inherit' });

  // Sign binary on macOS
  if (platform.os === 'darwin') {
    try {
      execSync(`codesign --sign - "${outputPath}"`, { cwd: repoRoot, stdio: 'inherit' });
    } catch (err) {
      console.warn('Warning: codesign failed (this is expected in CI)');
    }
  }

  // Make executable on Unix
  if (platform.ext === '') {
    await chmod(outputPath, 0o755);
  }

  console.log(`✓ ${platform.os}-${platform.arch} build complete`);
}

async function buildStandalone() {
  console.log('Building standalone executables with Node SEA...');
  
  const distDir = join(repoRoot, 'dist', 'cli');
  const bundlePath = join(distDir, 'noisemaker-bundle.cjs');

  if (!existsSync(bundlePath)) {
    throw new Error('Run npm run build:cli first to generate the bundle');
  }

  // Build for current platform only (SEA doesn't easily cross-compile)
  const currentPlatform = PLATFORMS.find(
    p => p.os === process.platform && p.arch === process.arch
  );

  if (!currentPlatform) {
    throw new Error(
      `Current platform ${process.platform}-${process.arch} not in build targets`
    );
  }

  await buildPlatform(currentPlatform);
  
  console.log('\n✓ Standalone executable created');
  console.log('Note: SEA builds are platform-specific.');
  console.log('Run this build on each target platform to create all binaries.');
}

buildStandalone().catch(err => {
  console.error('Build failed:', err);
  process.exit(1);
});
