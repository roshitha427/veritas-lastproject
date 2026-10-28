import { test, expect } from '@playwright/test';

test('switching text, image, and video clears the previous modality input', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Sign in', exact: true }).first().click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Email address').fill('qa-admin@example.invalid');
  await dialog.getByLabel('Password', { exact: true }).fill('local-browser-qa-only-password');
  await dialog.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(dialog).not.toBeVisible();

  const picker = page.locator('#sidebar').getByRole('button', { name: 'Check content' });
  await picker.click();
  await page.getByRole('menuitem', { name: /Text/ }).click();
  const content = page.locator('#news-content');
  await expect(content).toBeVisible();
  await content.fill('The first text claim should not carry into an image or video check.');

  await page.getByRole('tab', { name: 'Image', exact: true }).click();
  await expect(content).toHaveCount(0);
  await page.getByRole('button', { name: 'Add optional claim or caption', exact: true }).click();
  await expect(content).toHaveValue('');
  await content.fill('This is an image caption that belongs only to this image check.');

  await page.getByRole('tab', { name: 'Video', exact: true }).click();
  await expect(content).toHaveCount(0);
  await page.getByRole('button', { name: 'Add optional claim or caption', exact: true }).click();
  await expect(content).toHaveValue('');
  await expect(page.getByText('This is an image caption that belongs only to this image check.', { exact: true })).toHaveCount(0);
});
