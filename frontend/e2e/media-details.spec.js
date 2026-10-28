import { test, expect } from '@playwright/test';

for (const type of ['image', 'video']) {
test(`${type} claim keeps one factual result without scene or secondary model sections`, async ({ page }) => {
  const report = {
    id: 'media-details-fixture', type, status: 'complete', title: 'News poster',
    created_at: '2026-10-20T12:00:00Z',
    result: {
      ai_percentage: 99.9, fake_percentage: 99.9, real_percentage: 0.1, confidence: 99.9,
      media_origin: { verdict: 'inconclusive', score: 99.9 },
      media_details: { width: 800, height: 600, format: 'PNG' },
      temporal_assessment: type === 'video' ? { status: 'ready', fake_percentage: 25,
        real_percentage: 75, sampled_frames: 12, scope: 'Face-swap consistency over sampled frames.' } : undefined,
      media_inspection: { status: 'completed', visual_type: 'photograph',
        description: 'People at a table with a printed sign.',
        visible_features: ['Four people seated around a wooden table.', 'A large window behind the table.'],
        observed_actions: ['People face each other.'], visible_text: 'The city opened a new public library.' },
      report_summary: { schema_version: 5, verdict: 'unverified', headline: 'Review source evidence', summary: 'Evidence is incomplete.' },
      media_text_analysis: { content: 'The city opened a new public library.', ai_percentage: 35,
        ai_assessment: { experimental_input: true },
        text_model_assessment: { status: 'assessed', fake_percentage: 20, real_percentage: 80, confidence: 80 },
        text_metrics: { word_count: 7, character_count: 37 } },
    },
  };
  await page.route('**/api/session', route => route.fulfill({ json: {
    user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
  } }));
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
  await page.route('**/api/reports**', route => route.fulfill({ json:
    new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
  const card = page.locator('.report-card');
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.claim-assessment-text')).toContainText('The city opened a new public library.');
  await expect(card.getByRole('region', { name: 'Model metrics', exact: true })).toHaveCount(1);
  await expect(card.locator('.metric')).toHaveCount(3);
  await expect(card.locator('.metric-label')).toHaveText(type === 'image'
    ? ['Real image pattern score', 'AI image pattern score', 'Image model confidence']
    : ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence']);
  await expect(card.locator('.metric strong')).toHaveText(['0.1%', '99.9%', '99.9%']);
  await expect(card.locator('details')).toHaveCount(0);
  await expect(card).not.toContainText(/People at a table|Four people|large window|People face each other|Photograph|OCR|35.0%|25.0%|80.0%|sampled frames|More details/);
  await expect(card.getByRole('link',{name:'Download PDF',exact:true})).toHaveCount(1);

});
}
