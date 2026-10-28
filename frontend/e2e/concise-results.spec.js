import { test, expect } from '@playwright/test';

for (const type of ['text', 'image', 'video']) {
  test(`${type} unresolved report keeps the result short and metrics intact`, async ({ page }) => {
    const report = { id: 'concise-fixture', type, title: 'Sample report', status: 'complete',
      content: type === 'text' ? 'A claim awaiting verification.' : '', created_at: '2026-10-22T08:00:00Z',
      result: { ai_percentage: 99.9, real_percentage: .1, fake_percentage: 99.9, confidence: 99.9,
        media_inspection: { status: 'completed', visual_type: 'photograph', description: 'A person stands beside a blue car.',
          visible_features: ['A blue car beside the road.'], visible_text: 'A sign near the road.' },
        media_origin: { verdict: 'inconclusive', reason: 'The detector has not been locally validated. Internal model failure.' },
        text_model_assessment: { status: 'assessed', reason: 'Long classifier limitations and training faults.' },
        verification: { status: 'unverified', evidence: [] },
        report_summary: { schema_version: 5, verdict: 'unverified', headline: 'Internal provider error',
          summary: 'Excessive failed-model explanation.', limitations: ['A long technical fault report.'],
          gemini_model_assessment: { status: 'completed', answer: 'An optional answer.', reason: 'Not independently verified.' } },
      } };
    await page.route('**/api/session', route => route.fulfill({ json: {
      user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
    } }));
    await page.route('**/api/reports**', route => route.fulfill({ json:
      new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report,
    }));
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const card = page.locator('.report-card');
    expect(await card.innerText()).not.toMatch(/Internal provider error|Internal model failure|Long classifier limitations|Excessive failed-model/);
    await expect(card.locator('details')).toHaveCount(0);
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card.locator('.metric-label')).toHaveText(type === 'text'
      ? ['Real news pattern score', 'Fake news pattern score', 'Text model confidence']
      : type === 'image' ? ['Real image pattern score', 'AI image pattern score', 'Image model confidence']
      : ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence']);
    await expect(card.locator('.metric strong')).toHaveText(['0.1%', '99.9%', '99.9%']);
    await expect(card.getByRole('link', {name:/audit/i})).toHaveCount(0);
    if (type === 'text') {
      await expect(page.locator('.report-summary .summary-verdict')).toContainText('UNVERIFIED');
      await expect(page.getByText('An optional answer.', { exact: true })).toBeVisible();
      await expect(card.locator('.report-knowledge-note')).toContainText('general knowledge');
      await expect(card.getByRole('region', { name: 'Model metrics', exact: true })).toBeVisible();
    } else {
      await expect(page.getByRole('region', { name: 'Media result', exact: true }).getByText('A person stands beside a blue car.', { exact: true })).toBeVisible();
      await expect(page.getByRole('region', { name: 'Model metrics' })).toContainText('99.9%');
      await expect(page.locator('.media-inspection-card + section[aria-label="Model metrics"]')).toBeVisible();
      await expect(page.locator('.report-summary')).toHaveCount(0);
      await expect(page.getByText('A sign near the road.', { exact: true })).not.toBeVisible();
      await expect(page.getByText('A blue car beside the road.', { exact: true })).toBeHidden();
      await expect(card.locator(':scope > section[aria-label="Model metrics"] .metric')).toHaveCount(3);
      await page.screenshot({ path: `../artifacts/concise-${type}-report.png`, fullPage: true });
      await expect(card).not.toContainText('A blue car beside the road.');
      await expect(card).not.toContainText('A sign near the road.');
    }
  });
}

for (const [verdict, headline, score] of [['ai_generated', 'Likely AI-generated', 74.8], ['likely_real', 'Real image', 25.2]]) {
  test(`${verdict} retains one origin result without feature or OCR sections`, async ({ page }) => {
    const report = { id: 'resolved-origin', type: 'image', title: 'Resolved image', status: 'complete', content: '', created_at: '2026-10-22T08:00:00Z',
      result: { ai_percentage: score, real_percentage: 100-score, confidence: 74.8,
        media_origin: { verdict, score },
        media_inspection: { status: 'completed', visual_type: 'photograph', description: 'A woman in a green dress.', visible_features: ['long green dress', 'stone courtyard'], observed_actions: ['standing'], visible_text: 'Courtyard sign' },
        report_summary: { schema_version: 5, mode: 'media_only', verdict: 'unverified' } } };
    await page.route('**/api/session', route => route.fulfill({ json: { user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa' } }));
    await page.route('**/api/reports**', route => route.fulfill({ json: new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    await expect(page.locator('.media-inspection-heading strong')).toHaveText(headline);
    await expect(page.getByText('long green dress', { exact: true })).toBeHidden();
    await expect(page.getByText('Courtyard sign', { exact: true })).toBeHidden();
    await expect(page.getByRole('region', { name: 'Model metrics' })).toContainText(score.toFixed(1) + '%');
    await expect(page.locator('.report-card .metric')).toHaveCount(3);
    await expect(page.locator('.report-card details')).toHaveCount(0);
    await expect(page.locator('.report-card')).not.toContainText('long green dress');
    await expect(page.locator('.report-card')).not.toContainText('Courtyard sign');
    await expect(page.locator('.report-card')).not.toContainText('The origin could not be confirmed.');
  });
}

for (const verdict of ['true', 'false']) {
  test(`${verdict} shows an overall explanation, verdict reasoning and direct evidence link`, async ({ page }) => {
    const fullSummary = 'A detailed source comparison. '.repeat(20) + 'Final supporting detail.';
    const report = { id: 'verified-claim', type: 'text', title: 'Verified claim', content: 'Example claim.', status: 'complete', created_at: '2026-10-22T08:00:00Z',
      result: { report_summary: { schema_version: 5, verdict, headline: 'Claim assessed', summary: fullSummary, reason: 'Reason tied to source evidence.',
        summary_source_ids: [1], sources: [{ id: 1, title: 'Source article', publisher: 'Example', url: 'https://example.org/article', snippet: 'The complete source excerpt.' }],
        key_points: [{ text: 'The dated original record addresses the same event and location.', source_ids: [1] },
          { text: 'UNSUPPORTED_EXTRA_DETAIL', source_ids: [99] }] } } };
    await page.route('**/api/session', route => route.fulfill({ json: { user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa' } }));
    await page.route('**/api/reports**', route => route.fulfill({ json: new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const card = page.locator('.report-card');
    expect((await card.locator('.summary-body').innerText()).length).toBeGreaterThan(220);
    expect((await card.locator('.summary-body').innerText()).length).toBeLessThanOrEqual(1200);
    await expect(card.locator('.summary-verdict')).toHaveText(verdict.toUpperCase());
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card.locator('details')).toHaveCount(0);
    await expect(card.locator('.full-summary-text, .summary-reason')).toHaveCount(0);
    await expect(card.locator('.summary-citations a')).toHaveAttribute('href', 'https://example.org/article');
    await expect(card).toContainText('Final supporting detail.');
    await expect(card.locator('.report-verdict-reason')).toContainText('Reason tied to source evidence.');
    await expect(card.locator('.report-key-evidence li')).toHaveCount(1);
    await expect(card.locator('.report-key-evidence')).toContainText('The dated original record addresses the same event and location.');
    await expect(card).not.toContainText('UNSUPPORTED_EXTRA_DETAIL');
    await expect(card.getByRole('link', {name:'Download PDF', exact:true})).toHaveCount(1);
  });
}
