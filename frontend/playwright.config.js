import { defineConfig } from '@playwright/test';
export default defineConfig({
 testDir:'./e2e', timeout:90000, workers:1, fullyParallel:false,
 outputDir:'../artifacts/browser-results', reporter:'list',
 use:{baseURL:process.env.BASE_URL || 'http://127.0.0.1:5000',headless:true,
  launchOptions:process.env.PLAYWRIGHT_EXECUTABLE_PATH ? {executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH} : {},
  screenshot:'only-on-failure',trace:'retain-on-failure'},
});
