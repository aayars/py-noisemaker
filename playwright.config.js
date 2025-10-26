import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './test',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:8888',
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--enable-webgpu-developer-features',
            '--disable-gpu-sandbox',
            '--use-angle=vulkan',
          ],
        },
      },
    },
  ],

  webServer: {
    command: 'python3 -m http.server 8888 --directory .',
    url: 'http://localhost:8888',
    reuseExistingServer: !process.env.CI,
  },
});
