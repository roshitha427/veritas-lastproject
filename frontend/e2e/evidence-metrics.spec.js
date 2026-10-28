import { test, expect } from '@playwright/test';

function reportFixture(verdict = 'true', metrics = {}) {
  return {
    id: 'evidence-metrics-fixture', type: 'text', status: 'complete',
    title: 'Geethanjali Institute of Science and Technology exists in Gangavaram, Nellore.',
    content: 'Geethanjali Institute of Science and Technology exists in Gangavaram, Nellore.',
    created_at: '2026-10-23T08:00:00Z',
    result: {
      fake_percentage: 38, real_percentage: 62, confidence: 62, ai_percentage: 12.2,
      text_model_assessment: { status: 'assessed', experimental_input: true,
        reason: 'The original news-pattern score is 38.0%, with 62.0% model confidence.' },
      text_metrics: { word_count: 13, character_count: 89, sentence_count: 1, source_count: 1 },
      evidence_metrics: {
        status: 'assessed', supported_percentage: verdict === 'true' ? 100 : 0,
        contradicted_percentage: verdict === 'false' ? 100 : 0, unverified_percentage: 0,
        supported_claim_count: verdict === 'true' ? 1 : 0,
        contradicted_claim_count: verdict === 'false' ? 1 : 0,
        unverified_claim_count: 0, assessed_claim_count: 1, unassessed_claim_count: 0,
        total_claim_count: 1, verdict_agreement_percentage: 100, ...metrics,
      },
      report_summary: { schema_version: 5, verdict, headline: 'College location checked',
        summary: 'The institution website records the college address.', reason: 'The cited source settles the location.',
        summary_source_ids: [1], sources: [{ id: 1, publisher: 'gist.edu.in',
          url: 'https://gist.edu.in/gist/gist-home/', snippet: 'Gangavaram, Nellore.' }],
        claims: [{ text: 'The college exists in Gangavaram, Nellore.', verdict,
          reason: 'The official institution records its location.', supporting_evidence: verdict === 'true' ? [1] : [],
          contradicting_evidence: verdict === 'false' ? [1] : [] }],
      },
    },
  };
}

async function openReport(page, report) {
  await page.route('**/api/session', route => route.fulfill({ json: {
    user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
  } }));
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
  await page.route('**/api/reports**', route => route.fulfill({ json:
    new URL(route.request().url()).pathname === '/api/reports' ? { items: [report] } : report,
  }));
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
}

for (const verdict of ['true', 'false']) {
  test(`${verdict} shows model metrics alongside one factual verdict`, async ({ page }) => {
    await openReport(page, reportFixture(verdict));
    const card = page.locator('.report-card');
    const metrics = card.getByRole('region', { name: 'Model metrics', exact: true });
    await expect(metrics.locator('.metric-label')).toHaveText(['Real news pattern score', 'Fake news pattern score', 'Text model confidence']);
    await expect(metrics.locator('.metric strong')).toHaveText(['62.0%', '38.0%', '62.0%']);
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card.locator('.summary-verdict')).toHaveText(verdict.toUpperCase());
    await expect(card).not.toContainText(/Model confidence|AI-writing|98.5%|More details|Evidence agreement/);
    await expect(card.locator('details')).toHaveCount(0);
    if (verdict === 'true') await page.screenshot({ path: '../artifacts/evidence-metrics-true.png', fullPage: true });
  });
}

test('mixed evidence does not replace the model scores with evidence coverage', async ({ page }) => {
  await openReport(page, reportFixture('true', {
    supported_percentage: 50, contradicted_percentage: 25, unverified_percentage: 25,
    supported_claim_count: 2, contradicted_claim_count: 1, unverified_claim_count: 1,
    assessed_claim_count: 3, unassessed_claim_count: 1, total_claim_count: 4,
    verdict_agreement_percentage: 50,
  }));
  const metrics = page.getByRole('region', { name: 'Model metrics', exact: true });
  await expect(metrics.locator('.metric')).toHaveCount(3);
  await expect(metrics.locator('.metric strong')).toHaveText(['62.0%', '38.0%', '62.0%']);
  await expect(metrics).not.toContainText(/50.0%|25.0%|Claims supported|Claims contradicted|Claims unresolved|Evidence agreement/);
});

test('unassessed evidence preserves available text model scores', async ({ page }) => {
  await openReport(page, reportFixture('unverified', {
    status: 'not_assessed', supported_percentage: null, contradicted_percentage: null,
    unverified_percentage: null, verdict_agreement_percentage: null,
  }));
  const card = page.locator('.report-card');
  await expect(card.locator('.metric strong')).toHaveText(['62.0%', '38.0%', '62.0%']);
  await expect(card).not.toContainText(/12.2%|Model confidence|AI-writing/);
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
});

for (const type of ['text', 'image']) {
  test(`assessed unverified ${type} has one verdict and one model metric group`, async ({ page }) => {
    const report = reportFixture('unverified', {
      status: 'assessed', supported_percentage: 0, contradicted_percentage: 0,
      unverified_percentage: 100, supported_claim_count: 0, contradicted_claim_count: 0,
      unverified_claim_count: 1, total_claim_count: 1, verdict_agreement_percentage: null,
    });
    report.type = type;
    if (type === 'image') {
      report.result.claim_text = report.content;
      report.result.media_origin = { verdict: 'inconclusive', reason: 'The visual model cannot settle the factual claim.' };
      report.result.media_inspection = { description: 'A graphic containing the claim.' };
      report.result.media_text_analysis = { content: report.content,
        text_model_assessment: { status: 'assessed', fake_percentage: 38, real_percentage: 62, confidence: 62 } };
    }
    await openReport(page, report);
    const card = page.locator('.report-card');
    const metrics = card.getByRole('region', { name: 'Model metrics', exact: true });
    await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
    await expect(metrics.locator('.metric')).toHaveCount(3);
    await expect(metrics.locator('.metric-label')).toHaveText(type === 'text'
      ? ['Real news pattern score', 'Fake news pattern score', 'Text model confidence']
      : ['Real image pattern score', 'AI image pattern score', 'Image model confidence']);
    await expect(metrics.locator('.metric strong')).toHaveText(['62.0%', '38.0%', '62.0%']);
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card).not.toContainText(/100.0%|More details|AI-writing|Model confidence/);
    await expect(card.getByRole('region', { name: 'Model metrics', exact: true })).toHaveCount(1);
  });
}

test('saved report refresh keeps the three model metrics', async ({ page }) => {
  const current = reportFixture();
  const saved = { ...current, result: { ...current.result } };
  delete saved.result.evidence_metrics;
  await page.route('**/api/session', route => route.fulfill({ json: {
    user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
  } }));
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
  let reads = 0;
  await page.route('**/api/reports**', route => {
    if (new URL(route.request().url()).pathname === '/api/reports') return route.fulfill({ json: { items: [saved] } });
    reads += 1;
    return route.fulfill({ json: reads === 1 ? saved : current });
  });
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Model metrics', exact: true }).locator('.metric strong'))
    .toHaveText(['62.0%', '38.0%', '62.0%']);
  expect(reads).toBe(2);
});
