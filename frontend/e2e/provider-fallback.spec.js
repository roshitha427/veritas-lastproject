import { test, expect } from '@playwright/test';

for (const needsCurrent of [false, true]) {
  test(`${needsCurrent ? 'Current claim' : 'Background context'} stays separate from the source verdict`, async ({ page }) => {
    const report = {
      id: 'provider-fallback-fixture', type: 'text', status: 'complete', title: 'A short question',
      content: 'The Riverford bridge opened in 1985.', created_at: '2026-10-21T06:00:00Z',
      result: {
        text_metrics: { word_count: 7, character_count: 40, source_count: 0 },
        verification: { status: 'unverified', evidence: [] },
        report_summary: { schema_version: 5, verdict: 'unverified', knowledge_policy_version: 2,
          headline: 'The available sources do not settle this claim', summary: 'No direct source evidence was retrieved.',
          gemini_analysis: { status: 'completed', role: 'No independent source verification was possible.' },
          sources: [{id:1, url:'https://example.gov.in/notice', publisher:'Example Government', reliability:'official_primary_source'}],
          gemini_model_assessment: { status: needsCurrent ? 'needs_current_sources' : 'completed', verdict: 'unverified',
            user_message: 'This claim needs up-to-date evidence.',
            answer: needsCurrent ? 'An outdated current-fact answer.' : 'A cautious explanation from model knowledge.', reason: 'This answer needs independent verification.' } },
      },
    };
    await page.route('**/api/reports**', route => route.fulfill({ json:
      new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
    await page.route('**/api/session', route => route.fulfill({ json: {
      user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
    } }));
    await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const explanation = page.getByRole('region', { name: 'Additional explanation' });
    await expect(explanation).toBeHidden();
    await expect(page.locator('.report-summary .summary-verdict')).toContainText('UNVERIFIED');
    await expect(page.locator('.report-more-details')).toHaveCount(0);
    await expect(page.locator('.report-card')).not.toContainText(/Gemini|Mistral/);
    if (needsCurrent) {
      await expect(page.locator('.report-general-explanation')).toHaveCount(0);
      await expect(page.locator('.report-card')).not.toContainText('An outdated current-fact answer.');
    } else {
      await expect(page.locator('.report-general-explanation')).toHaveText('A cautious explanation from model knowledge.');
      await expect(page.locator('.report-knowledge-note')).toContainText('retrieved sources have not independently confirmed it');
      await expect(page.getByRole('heading', { name: /general knowledge|additional explanation|model.only/i })).toHaveCount(0);
    }
    await expect(page.locator('.summary-citations a')).toHaveCount(0);
    await expect(page.locator('.summary-body')).toHaveText('No direct source evidence was retrieved.');
    await expect(page.getByRole('link', { name: 'Download PDF', exact: true }))
      .toHaveAttribute('href', '/api/reports/provider-fallback-fixture/pdf');
  });
}
