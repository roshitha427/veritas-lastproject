import { test, expect } from '@playwright/test';

const suggestions = [
  { name: 'PIB Fact Check', url: 'https://factcheck.pib.gov.in/', kind: 'government',
    description: 'Check official clarifications about Government of India claims.' },
  { name: 'National Portal of India', url: 'https://www.india.gov.in/', kind: 'government',
    description: 'Find the responsible ministry and its official announcements.' },
];

function fixture(overrides = {}) {
  const { summary = {}, ...reportOverrides } = overrides;
  return { id: 'recommended-sources-fixture', type: 'text', title: 'An announcement needs verification',
    content: 'The government has announced a new programme.', status: 'complete', created_at: '2026-10-24T08:00:00Z',
    ...reportOverrides,
    result: { ai_percentage: 12.2, real_percentage: 62, fake_percentage: 38, confidence: 62,
      evidence_metrics: { status: 'not_assessed' },
      text_model_assessment: { status: 'assessed', reason: 'A model writing-pattern score.' },
      report_summary: { schema_version: 5, verdict: 'unverified', headline: 'Claim unverified',
        summary: 'The claim has not been verified.', sources: [], recommended_sources: suggestions,
        recommendation_policy_version: 2, verification_review: { policy_version: 2 }, ...summary },
    },
  };
}

async function openReport(page, report, refreshed) {
  await page.route('**/api/session', route => route.fulfill({ json: {
    user: { id: 'qa', name: 'QA', email: 'qa@example.invalid', role: 'user' }, csrf_token: 'qa',
  } }));
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok' } }));
  const requests = [];
  await page.route('**/api/reports**', route => {
    if (new URL(route.request().url()).pathname === '/api/reports') return route.fulfill({ json: { items: [report] } });
    requests.push(route.request().method());
    return route.fulfill({ json: requests.length > 1 && refreshed ? refreshed : report });
  });
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
  return requests;
}

test('unverified claim highlights related links with a reason for checking each source', async ({ page }) => {
  await openReport(page, fixture());
  const card = page.locator('.report-card');
  const followups = card.getByRole('region', { name: 'Check this claim yourself', exact: true });
  const links = followups.locator('a.followup-link');
  await expect(links).toHaveCount(2);
  for (const [index, source] of suggestions.entries()) {
    await expect(links.nth(index)).toHaveText(source.name);
    await expect(links.nth(index)).toHaveAttribute('href', source.url);
    await expect(links.nth(index)).toHaveAttribute('target', '_blank');
    await expect(links.nth(index)).toHaveAttribute('rel', 'noopener noreferrer');
    await expect(followups.getByText(source.description, { exact: true })).toBeVisible();
  }
  expect(await links.first().evaluate(link => parseFloat(getComputedStyle(link).fontSize))).toBeGreaterThanOrEqual(15);
  await links.first().focus();
  await expect(links.first()).toBeFocused();
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card.locator('.metric')).toHaveCount(3);
  await expect(card.locator('.metric-label')).toHaveText(['Real news pattern score', 'Fake news pattern score', 'Text model confidence']);
  await expect(card.locator('.metric strong')).toHaveText(['62.0%', '38.0%', '62.0%']);
  await expect(card).not.toContainText(/12.2%|Model confidence|More details/);
  await expect(card.getByRole('link', {name: 'Download PDF', exact: true})).toHaveCount(1);
  await followups.screenshot({ path: '../artifacts/recommended-links-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(links.first()).toBeVisible();
  await expect(links.last()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await followups.screenshot({ path: '../artifacts/recommended-links-mobile.png' });
});

test('manual checks deduplicate normalized URLs and show at most three justified sources', async ({ page }) => {
  const extra = [
    { name: 'Responsible ministry', url: 'https://example.gov.in/programme', description: 'Read the ministry announcement for the programme dates and eligibility.' },
    { name: 'Fourth related source', url: 'https://example.gov.in/programme/faq', description: 'Compare the official programme terms with the claim.' },
  ];
  await openReport(page, fixture({ summary: {
    sources: [{ id: 1, publisher: 'Retrieved report', url: 'https://example.org/news', snippet: 'Context only.' }],
    recommended_sources: [suggestions[0], { ...suggestions[0], name: 'Duplicate portal', url: 'https://FACTCHECK.PIB.GOV.IN:443/#programme' }, suggestions[1], ...extra],
  } }));
  const card = page.locator('.report-card');
  await expect(card.locator('a.followup-link')).toHaveCount(3);
  await expect(card.locator('a.followup-link')).toHaveText([suggestions[0].name, suggestions[1].name, extra[0].name]);
  await expect(card.locator('a.followup-link').first()).toHaveAttribute('href', suggestions[0].url);
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card.locator('a[href="https://example.org/news"]')).toHaveCount(0);
  await expect(card).not.toContainText(/Retrieved report|Duplicate portal|Fourth related source/);
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
});

test('saved reports refresh the older recommendation policy', async ({ page }) => {
  const current = fixture();
  const saved = fixture({ summary: { recommendation_policy_version: 1 } });
  delete saved.result.report_summary.recommended_sources;
  const requests = await openReport(page, saved, current);
  await expect(page.locator('.report-card a.followup-link')).toHaveCount(2);
  await expect(page.locator('.report-card a.followup-link').first()).toHaveAttribute('href', suggestions[0].url);
  await expect(page.locator('.report-card details')).toHaveCount(0);
  expect(requests).toHaveLength(2);
});

test('unsafe recommendation and retrieved URLs are never clickable', async ({ page }) => {
  const urls = [
    'javascript:alert(1)', 'http://www.india.gov.in/', 'https://user:password@www.india.gov.in/',
    'https://user@www.india.gov.in/', 'https://www.india.gov.in:8443/',
    'https://not a host.example/record', 'https://example.gov.in/official notice', 'https://example.gov.in/record\u007f',
    'data:text/html,unsafe', 'file:///etc/passwd', '/relative', 'not a url',
  ];
  await openReport(page, fixture({ summary: {
    recommended_sources: [null, {}, ...urls.map((url, i) => ({ name: `Unsafe ${i}`, url, description: 'A justification cannot make an unsafe URL clickable.' })), suggestions[0]],
    sources: urls.map((url, i) => ({ id: i + 1, publisher: `Unsafe source ${i}`, url })),
  } }));
  const card = page.locator('.report-card');
  await expect(card.locator('a.followup-link')).toHaveCount(1);
  await expect(card.locator('a.followup-link')).toHaveAttribute('href', suggestions[0].url);
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card).not.toContainText('Unsafe');
});

test('a recommendation needs both a source name and a claim-specific justification', async ({ page }) => {
  await openReport(page, fixture({ summary: { recommended_sources: [
    { name: 'Unexplained source', url: 'https://example.gov.in/' },
    { name: 'Blank explanation', url: 'https://example.gov.in/blank', description: '  ' },
    { url: 'https://example.gov.in/unnamed', description: 'Check an announcement.' },
    { name: 42, url: 'https://example.gov.in/number', description: 'Check an announcement.' },
    { name: 'Non-string URL', url: ['https://example.gov.in/array'], description: 'Check an announcement.' },
  ] } }));
  await expect(page.locator('.report-card a.followup-link')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Check this claim yourself', exact: true })).toHaveCount(0);
  await expect(page.locator('.report-card')).not.toContainText(/Unexplained source|Blank explanation|Check an announcement/);
});

for (const verdict of ['true', 'false']) {
  test(`${verdict} result omits unverified follow-up links`, async ({ page }) => {
    await openReport(page, fixture({ summary: { verdict, summary_source_ids: [1],
      sources: [{ id: 1, publisher: 'Programme record', url: 'https://example.gov.in/programme', snippet: 'The original programme record.' }] } }));
    await expect(page.locator('.summary-verdict')).toHaveText(verdict.toUpperCase());
    await expect(page.locator('.report-card a.followup-link')).toHaveCount(0);
    await expect(page.locator('.summary-citations a')).toHaveAttribute('href', 'https://example.gov.in/programme');
    await expect(page.locator('.report-card details')).toHaveCount(0);
  });
}

for (const type of ['image', 'video']) {
  test(`${type} claim has justified manual-check links`, async ({ page }) => {
    await openReport(page, fixture({ type }));
    await expect(page.locator('.summary-verdict')).toHaveText('UNVERIFIED');
    await expect(page.locator('.report-card a.followup-link')).toHaveCount(2);
    await expect(page.locator('.report-card .metric')).toHaveCount(3);
  });

  test(`${type} without factual claim assessment omits portal suggestions`, async ({ page }) => {
    await openReport(page, fixture({ type, summary: { mode: 'media_only' } }));
    await expect(page.locator('.report-card')).toBeVisible();
    await expect(page.locator('.report-card a.followup-link')).toHaveCount(0);
    await expect(page.locator('.report-summary')).toHaveCount(0);
  });
}
