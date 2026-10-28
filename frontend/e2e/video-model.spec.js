import { test, expect } from '@playwright/test';

test('video report keeps one experimental origin result and omits competing module scores', async ({ page }) => {
  const report = {
    id: 'video-adaptation-fixture', type: 'video', status: 'complete',
    title: 'AI video diagnostic', content: '', created_at: '2026-10-22T06:00:00Z',
    result: {
      ai_percentage: 20, fake_percentage: 20, real_percentage: 80, confidence: 80,
      media_origin: { verdict: 'inconclusive', basis: 'video_consistency_features', scores_publishable: false,
        headline: 'AI/manipulation-like video signals (experimental)', reason: 'Training fit only.' },
      video_assessment: { status: 'experimental', synthetic_or_manipulated_percentage: 93.1,
        real_class_percentage: 6.9, sampled_frames: 12, training_example: true,
        scope: 'Synthetic or manipulated video signal.', note: 'This clip was used for training; this result measures training fit.' },
      verification: { status: 'not_applicable', evidence: [] },
      report_summary: { schema_version: 5, mode: 'media_only',
        headline: 'Experimental video signals', summary: 'Video diagnostic.' },
    },
  };
  await page.route('**/api/session', route => route.fulfill({ json: {
    user: { id: 'qa', name: 'Video QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
  } }));
  await page.route('**/api/reports**', route => route.fulfill({ json:
    new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report,
  }));
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
  const card=page.locator('.report-card');
  await expect(card.getByRole('region', { name: 'Experimental video assessment' })).toHaveCount(0);
  await expect(card.getByRole('region', { name: 'Model metrics', exact:true })).toBeVisible();
  await expect(card.locator('.metric-label')).toHaveText(['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence']);
  await expect(card.locator('.metric strong')).toHaveText(['80.0%', '20.0%', '80.0%']);
  await expect(card.locator('.metric')).toHaveCount(3);
  await expect(card.locator('details')).toHaveCount(0);
  await expect(card).not.toContainText(/93.1%|6.9%|training fit|independent accuracy|Likely real video|More details/);

});

test('user-confirmed AI video shows provenance and style without inventing a score', async ({ page }) => {
  const report = { id: 'confirmed-video', type: 'video', status: 'complete', title: 'Confirmed video', content: '', created_at: '2026-10-22T06:00:00Z',
    result: { ai_percentage: 46.7, real_percentage: 53.3, confidence: 53.3,
      media_origin: { verdict: 'inconclusive' }, media_inspection: { description: 'Rendered characters in a scene.' },
      media_labels: { confirmed_origin: 'ai_generated', provenance: 'user_confirmed', origin_label: 'AI-generated', style: 'Animated' },
      report_summary: { schema_version: 5, mode: 'media_only' } } };
  await page.route('**/api/session', route => route.fulfill({ json: { user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa' } }));
  await page.route('**/api/reports**', route => route.fulfill({ json: new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
  await expect(page.locator('.media-inspection-heading strong')).toHaveText('AI-generated');
  await expect(page.locator('.media-inspection-heading strong')).toHaveAttribute('title', /supplied reference label/i);
  await expect(page.getByRole('region', { name: 'Model metrics', exact: true }).locator('.metric strong'))
    .toHaveText(['53.3%', '46.7%', '53.3%']);
  await expect(page.locator('.report-card .metric')).toHaveCount(3);
  await expect(page.locator('.report-card details')).toHaveCount(0);
  await expect(page.locator('.report-card')).not.toContainText('100.0%');
  await expect(page.locator('.report-card')).not.toContainText('The origin could not be confirmed.');
});
