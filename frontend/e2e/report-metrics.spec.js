import { test, expect } from '@playwright/test';

const expectedLabels = {
  text: ['Real news pattern score', 'Fake news pattern score', 'Text model confidence'],
  image: ['Real image pattern score', 'AI image pattern score', 'Image model confidence'],
  video: ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence'],
};

for (const sample of [
  { name: 'short English text', type: 'text', score: 31, confidence: 69, words: 8 },
  { name: 'English blog article', type: 'text', score: 18, confidence: 82, words: 140, ai: 84 },
  { name: 'source-contradicted text with high Real score', type: 'text', score: 1.3, confidence: 98.7, words: 12, verdict: 'false' },
  { name: 'unsupported-language text', type: 'text', score: null, confidence: null, words: 6 },
  { name: 'text with invalid model scores', type: 'text', score: -1, confidence: 101, words: 6 },
  { name: 'image', type: 'image', score: 20, confidence: 80 },
  { name: 'image with distinct legacy AI score', type: 'image', score: 20, confidence: 80, ai: 91 },
  { name: 'image with an unvalidated high AI score', type: 'image', score: 99.9, confidence: 99.9 },
  { name: 'real photograph with an unvalidated 100% AI score', type: 'image', score: 100, confidence: 100, visual: 'photograph' },
  { name: 'cartoon with an unvalidated high AI score', type: 'image', score: 99.7, confidence: 99.7, visual: 'cartoon_or_illustration' },
  { name: 'AI declaration despite a low classifier AI score', type: 'image', score: 19.1, confidence: 80.9, visual: 'cartoon_or_illustration', declared: true },
  { name: 'video', type: 'video', score: 74, confidence: 74 },
]) {
  test(`${sample.name} names the three model scores by their actual scope`, async ({ page }) => {
    const report = {
      id: 'metrics-fixture-' + sample.type, type: sample.type, status: 'complete',
      title: sample.name, content: sample.verdict === 'false'
        ? 'Geethanjali Institute of Science and Technology is located in Mumbai.'
        : 'A test claim for checking report metrics.',
      created_at: '2026-10-20T03:00:00Z',
      result: {
        fake_percentage: sample.score, real_percentage: sample.score === null ? null : 100 - sample.score,
        confidence: sample.confidence, ai_percentage: sample.ai ?? (sample.type === 'text' ? null : sample.score),
        text_metrics: { word_count: sample.words, source_count: 0 },
        media_inspection: sample.visual ? { status: 'completed', visual_type: sample.visual, description: 'Visible content test.' } : undefined,
        media_origin: sample.type === 'text' ? undefined : {
          verdict: sample.declared ? 'ai_content_declared' : 'inconclusive', score: sample.score,
          headline: sample.declared ? 'AI-generated content declared' : undefined,
          basis: sample.declared ? 'content_credentials' : undefined,
          credential_scope: sample.declared ? 'source_component' : undefined,
          reason: 'The detector has not been locally validated for origin classification.',
        },
        text_model_assessment: sample.type === 'text' ? {
          status: sample.score === null ? 'unsupported_language' : 'assessed',
          experimental_input: true, fake_percentage: sample.score, confidence: sample.confidence,
          reason: sample.score === null ? 'This language is not supported by the classifier.' : 'Short-text scores are experimental, not a fact-check.',
        } : undefined,
        verification: { evidence: [], status: 'unverified' },
        report_summary: { schema_version: 5, verdict: sample.verdict || 'unverified', headline: 'Review the evidence',
          mode: sample.type === 'text' ? undefined : 'media_only',
          summary: sample.verdict === 'false' ? 'The institution website records its address in Nellore, contradicting the Mumbai claim.' : 'A report metric test fixture.',
          summary_source_ids: sample.verdict === 'false' ? [1] : [],
          sources: sample.verdict === 'false' ? [{ id: 1, publisher: 'GIST', url: 'https://gist.edu.in/gist/gist-home/',
            snippet: 'The institution is located in Gangavaram, Nellore.' }] : [],
        },
      },
    };
    await page.route('**/api/session', route => route.fulfill({ json: {
      user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
    } }));
    await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
    await page.route('**/api/reports**', route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/reports') return route.fulfill({ json: { items: [report] } });
      if (pathname === '/api/reports/' + report.id) return route.fulfill({ json: report });
      return route.continue();
    });
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const card = page.locator('.report-card');
    const metrics = card.getByRole('region', { name: 'Model metrics', exact:true }).locator('.metric-grid');
    await expect(metrics).toBeVisible();
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card.locator('details')).toHaveCount(0);
    await expect(metrics.locator('.metric-label')).toHaveText(expectedLabels[sample.type]);
    await expect(card.locator('.metric-label').filter({ hasText: /^(Words analyzed|Characters analyzed|Sentences analyzed|Source excerpts)$/ })).toHaveCount(0);
    await expect(card).not.toContainText(/AI-writing likelihood|Human-writing likelihood|Held-out dataset|More details|Local text classifier/);
    if (sample.type === 'text') {
      await expect(metrics.locator('.metric strong')).toHaveText(sample.score === null || sample.score < 0
        ? ['Not assessed', 'Not assessed', 'Not assessed']
        : [(100 - sample.score).toFixed(1) + '%', sample.score.toFixed(1) + '%', sample.confidence.toFixed(1) + '%']);
      await expect(card.locator('.summary-verdict')).toHaveText((sample.verdict || 'unverified').toUpperCase());
      await expect(card.getByText('Real/Fake describe news-writing patterns; they do not verify this claim.', { exact: true })).toBeVisible();
      if (sample.verdict === 'false') {
        await expect(card.locator('.summary-body')).toContainText('contradicting the Mumbai claim');
        await expect(card.locator('.summary-citations a')).toHaveAttribute('href', 'https://gist.edu.in/gist/gist-home/');
      }
      await expect(metrics).not.toContainText(/Model confidence|Fake-news pattern|Real-news pattern/);
    } else {
      await expect(card.getByText('Real/Fake describe news-writing patterns; they do not verify this claim.', { exact: true })).toHaveCount(0);
      await expect(metrics.locator('.metric strong')).toHaveText([
        (100 - sample.score).toFixed(1) + '%', sample.score.toFixed(1) + '%', sample.confidence.toFixed(1) + '%',
      ]);
      await expect(page.locator('.media-inspection-heading')).toContainText(sample.declared ? 'AI-generated content declared' : sample.visual === 'photograph' ? 'Photograph' : sample.visual === 'cartoon_or_illustration' ? 'Cartoon or illustration' : 'Origin inconclusive');
      await expect(card).not.toContainText('Likely AI-generated');
      await expect(card).not.toContainText('Experimental AI-origin estimate:');
    }
    if (['short English text', 'image', 'video'].includes(sample.name)) {
      const region = card.getByRole('region', { name: 'Model metrics', exact: true });
      await region.screenshot({ path: `../artifacts/metric-labels-${sample.type}-desktop.png` });
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(metrics.locator('.metric-label')).toHaveText(expectedLabels[sample.type]);
      expect(await metrics.locator('.metric-label').evaluateAll(labels => labels.every(label =>
        label.scrollWidth <= label.clientWidth + 1 && label.scrollHeight <= label.clientHeight + 1))).toBeTruthy();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
      await region.screenshot({ path: `../artifacts/metric-labels-${sample.type}-mobile.png` });
    }
  });
}

for (const type of ['text', 'image', 'video']) {
  test(`${type} leaves absent model scores unavailable`, async ({ page }) => {
    const report = { id: 'missing-scores', type, status: 'complete', title: 'Missing scores',
      content: type === 'text' ? 'A claim.' : '', created_at: '2026-10-27T08:00:00Z',
      result: { ai_percentage: 91, real_percentage: null, fake_percentage: null, confidence: null,
        report_summary: { schema_version: 5, mode: type === 'text' ? undefined : 'media_only', verdict: 'unverified' } } };
    await page.route('**/api/session', route => route.fulfill({ json: {
      user: { id: 'qa', name: 'QA', role: 'user' }, csrf_token: 'qa',
    } }));
    await page.route('**/api/reports**', route => route.fulfill({ json:
      new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report,
    }));
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const metrics = page.getByRole('region', { name: 'Model metrics', exact: true });
    await expect(metrics.locator('.metric-label')).toHaveText(expectedLabels[type]);
    await expect(metrics.locator('.metric strong')).toHaveText([
      'Not assessed', type === 'text' ? 'Not assessed' : '91.0%', 'Not assessed',
    ]);
  });
}
