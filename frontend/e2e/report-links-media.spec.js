import { test, expect } from '@playwright/test';

const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAHgAAABQCAIAAABd+SbeAAAAyklEQVR4nO3QMQ0AIADAMEAS7nCETFSwqzWwZHOfO/hvBQ2M7hgdMTpidMToiNERoyNGR4yOGB0xOmJ0xOiI0RGjI0ZHjI4YHTE6YnTE6IjREaMjRkeMjhgdMTpidMToiNERoyNGR4yOGB0xOmJ0xOiI0RGjI0ZHjI4YHTE6YnTE6IjREaMjRkeMjhgdMTpidMToiNERoyNGR4yOGB0xOmJ0xOiI0RGjI0ZHjI4YHTE6YnTE6IjREaMjRkeMjhgdMTpidMToiNGj8QAc+wHzNfGH6gAAAABJRU5ErkJggg==', 'base64');
const clip = Buffer.from('GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAKBEU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHWTbuMU6uEElTDZ1OsggEZTbuMU6uEHFO7a1OsggJZ7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsCrXsYMPQkBNgIxMYXZmNjEuNy4xMDBXQYxMYXZmNjEuNy4xMDBEiYhAn0AAAAAAABZUrmu+rgEAAAAAAAA114EBc8WI3+xHh4nBgdecgQAitZyDdW5kiIEAhoVWX1ZQOIOBASPjg4QdzWUA4IawgUC6gTASVMNn13Nzn2PAgGfImUWjh0VOQ09ERVJEh4xMYXZmNjEuNy4xMDBzc7JjwItjxYjf7EeHicGB12fIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDIuMDAwMDAwMDAwAB9DtnVA3ueBAKPBgQAAgHADAJ0BKkAAMAAARwiFhYiFhIgBohATrVAfwBA4RwfvoTZ5cAD+/oqsf/q6L9XRfq6L/1dF/82BXNc+5oCjrIEB9AAxAgABEA0QAMAA0m9/oAG11iOY+P78WDf/s7v07v07v8DEEQzVjMeAo7yBA+iAMAMAnQEqQAAwAABHCIWFiIWEiAGiEBOtUBA3Mmcpr4gU3AD+8XIP/loX/loX/loX/LQvm+x7McCjqoEF3AAxAgABEA0QAMAA0n+gYACIYCqcAP7+H+L/zNH6TfSbvw4QS+EhABxTu2uju4+zgQC3iveBAfGCAXXwgQO7kLOCA+i3iveBAfGCAXXwgXQ=', 'base64');

function imageReport(id = 'image-linked') {
  const claim = 'The Riverford bridge opened in 1985.';
  const source = { id: 11, source_key: 'bridge-11', url: 'https://example.gov.in/bridge',
    title: 'Bridge opening record', publisher: 'Bridge archive', accepted: true,
    excerpt: 'The Riverford bridge opened to the public in 1985.' };
  return { id, type: 'image', title: 'Bridge picture.png', content: claim, status: 'complete', created_at: '2026-10-25T10:00:00Z',
    image_preview: { url: `/api/reports/${id}/preview`, mime_type: 'image/jpeg', width: 180, height: 90 },
    result: { claim_text: claim, ai_percentage: 51.9, real_percentage: 48.1, confidence: 51.9,
      media_origin: { verdict: 'inconclusive' }, media_inspection: { description: 'A bridge picture.' },
      report_summary: { schema_version: 5, verdict: 'true', headline: 'The bridge opening date is supported',
        summary: 'The archive supports this date [11].', reason: 'The original record states the date.',
        summary_source_ids: [11], sources: [{ ...source, snippet: source.excerpt }] },
      report_contract: { schema_version: 1, report_id: id, media_type: 'image', mode: 'claim_verification',
        claim: { checked_text: claim }, sources: [source], warnings: [],
        verification: { verdict: 'true', cited_source_ids: [11], unresolved_source_ids: [],
          claims: [{ id: 'claim-1', text: claim, verdict: 'true', reason: 'See the original record.', supporting_source_ids: [11] }] } }
    } };
}

function videoReport() {
  return { id: 'video-linked', type: 'video', title: 'My uploaded video.webm', content: '', status: 'complete', created_at: '2026-10-25T10:00:00Z',
    video_preview: { url: '/api/reports/video-linked/video', poster_url: '/api/reports/video-linked/poster', mime_type: 'video/webm' },
    result: { media_origin: { verdict: 'inconclusive' }, media_inspection: { description: 'A short uploaded clip.' },
      report_summary: { schema_version: 5, mode: 'media_only', verdict: 'unverified' } } };
}

async function setup(page, reports, delayedFirst) {
  await page.route('**/api/session', r => r.fulfill({ json: { user: { id: 'preview-user', name: 'Preview user', role: 'user' }, csrf_token: 'test' } }));
  await page.route('**/api/health', r => r.fulfill({ json: { status: 'ok' } }));
  await page.route('**/api/reports**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/reports') return route.fulfill({ json: { items: reports } });
    const report = reports.find(item => path.startsWith(`/api/reports/${item.id}`));
    if (!report) return route.fulfill({ status: 404 });
    if (path.endsWith('/preview') || path.endsWith('/poster')) return route.fulfill({ contentType: 'image/png', body: pixel });
    if (path.endsWith('/video')) return route.fulfill({ contentType: 'video/webm', body: clip });
    if (path.endsWith('/pdf')) return route.fulfill({ contentType: 'application/pdf', headers: { 'Content-Disposition': `attachment; filename="${report.id}.pdf"` }, body: '%PDF-1.4\n% Preview fixture\n%%EOF' });
    if (delayedFirst && report.id === reports[0].id) await delayedFirst;
    return route.fulfill({ json: report });
  });
}

test('history picture, direct source, report and PDF keep the same identity', async ({ page }) => {
  const report = imageReport();
  await setup(page, [report]);
  await page.goto('/#history');
  const row = page.locator('.history-row');
  await expect(row.locator('.report-preview-thumbnail img')).toHaveAttribute('src', report.image_preview.url);
  await row.getByRole('button', { name: 'Open media report: Bridge picture.png', exact: true }).click();
  const card = page.locator('.report-card');
  await expect(card).toHaveAttribute('data-report-id', report.id);
  const preview = page.locator('.input-card .report-media-preview img');
  await expect(preview).toHaveAttribute('src', report.image_preview.url);
  await expect(preview).toHaveJSProperty('complete', true);
  await expect(card.locator('.report-media-preview')).toHaveCount(0);
  await expect(card.locator('details')).toHaveCount(0);
  const source = card.locator('.summary-citations a[data-source-id="11"]');
  await expect(source).toHaveAttribute('href', 'https://example.gov.in/bridge');
  await expect(source).toHaveAttribute('target', '_blank');
  await expect(source).toHaveAttribute('rel', 'noopener noreferrer');
  await expect(page).toHaveURL(/#verify$/);
  await expect(card.locator('.report-bottom a')).toHaveAttribute('href', `/api/reports/${report.id}/pdf`);
  await page.screenshot({ path:'../artifacts/report-presentation-overview.png', fullPage:true });
  const download = page.waitForEvent('download');
  await card.locator('.report-bottom a').click();
  expect((await download).suggestedFilename()).toContain(report.id.slice(0, 10));
  await page.getByRole('button', { name: 'Reports & PDFs', exact: true }).first().click();
  await expect(page.locator('.report-library-row.selected img')).toHaveAttribute('src', report.image_preview.url);
  await expect(card.locator('.report-overview .report-media-preview img')).toHaveAttribute('src', report.image_preview.url);
  await expect(card.locator('.report-overview')).not.toContainText(report.title);
});

test('late image selection cannot replace the selected playable video', async ({ page }) => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const first = imageReport('late-image');
  const video = videoReport();
  await setup(page, [first, video], pending);
  await page.goto('/#history');
  await expect(page.locator('.history-row').last().locator('.report-preview-thumbnail img')).toHaveAttribute('src', video.video_preview.poster_url);
  await page.getByRole('button', { name: 'Open media report: Bridge picture.png', exact: true }).click();
  await page.getByRole('button', { name: 'Open media report: My uploaded video.webm', exact: true }).click();
  const card = page.locator('.report-card');
  await expect(card).toHaveAttribute('data-report-id', video.id);
  const oldResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/reports/${first.id}`);
  release(); await oldResponse;
  await expect(card).toHaveAttribute('data-report-id', video.id);
  const player = page.locator('.input-card .report-media-preview video');
  await expect(player).toHaveAttribute('src', video.video_preview.url);
  await expect(player).toHaveAttribute('poster', video.video_preview.poster_url);
  await expect(player).toHaveJSProperty('controls', true);
  await expect(player).toHaveJSProperty('autoplay', false);
  await expect.poll(() => player.evaluate(element => element.readyState)).toBeGreaterThan(0);
  await player.evaluate(element => element.play());
  await expect.poll(() => player.evaluate(element => element.currentTime)).toBeGreaterThan(0);
  await player.evaluate(element => element.pause());
  await expect(card.locator('.report-bottom a')).toHaveAttribute('href', `/api/reports/${video.id}/pdf`);
  await expect(card).not.toContainText('The Riverford bridge opened in 1985.');
});

test('ambiguous saved citations cannot link to a different source', async ({ page }) => {
  const report = imageReport();
  report.result.report_contract.sources.push({ ...report.result.report_contract.sources[0], source_key:'different-source', url:'https://example.gov.in/other' });
  report.result.report_contract.verification.unresolved_source_ids = [11];
  await setup(page,[report]);
  await page.goto('/#history');
  await page.getByRole('button',{name:'Open report',exact:true}).click();
  const card=page.locator('.report-card');
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card.locator('a[data-source-id="11"]')).toHaveCount(0);
  await expect(card.locator('a[href="https://example.gov.in/other"]')).toHaveCount(0);
  await expect(card.locator('details')).toHaveCount(0);
});

test('unreviewed and ambiguous context sources are absent from the concise result', async ({ page }) => {
  const report=imageReport();
  report.result.report_summary.verdict='unverified';
  report.result.report_contract.verification.verdict='unverified';
  report.result.report_contract.verification.claims[0].verdict='unverified';
  const original=report.result.report_contract.sources[0];
  report.result.report_contract.sources=[{...original,accepted:null},
    {...original,id:12,source_key:'duplicate-a',url:'https://example.org/first'},
    {...original,id:12,source_key:'duplicate-b',url:'https://example.org/second'},
    {...original,id:13,source_key:'related-only',url:'https://example.org/context'}];
  await setup(page,[report]);
  await page.goto('/#history');
  await page.getByRole('button',{name:'Open report',exact:true}).click();
  const card=page.locator('.report-card');
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card.locator('a[data-source-id]')).toHaveCount(0);
  await expect(card).not.toContainText(/duplicate-a|duplicate-b|Related directory/);
  await expect(card.locator('details')).toHaveCount(0);
});

test('opening a different saved report replaces its concise content and private preview', async ({ page }) => {
  const first=imageReport('first-image'),second=imageReport('second-image');
  const secondClaim='The Eastford bridge opened in 2001.';
  second.title='Second picture.png';second.content=secondClaim;second.result.claim_text=secondClaim;
  second.result.report_summary.summary='The dated record supports the Eastford opening date.';
  second.result.report_contract.claim.checked_text=secondClaim;
  second.result.report_contract.verification.claims[0].text=secondClaim;
  second.result.report_contract.sources[0].url='https://example.gov.in/eastford';
  await setup(page,[first,second]);
  await page.goto('/#reports');
  await page.locator('.report-library-row').first().locator('.report-library-open').click();
  const card=page.locator('.report-card');
  await expect(card).toHaveAttribute('data-report-id',first.id);
  await expect(card.locator('.claim-assessment-text')).toContainText(first.content);
  await page.locator('.report-library-row').last().locator('.report-library-open').click();
  await expect(card).toHaveAttribute('data-report-id',second.id);
  await expect(card.locator('.claim-assessment-text')).toContainText(secondClaim);
  await expect(card).not.toContainText('Riverford');
  await expect(card.locator('.report-media-preview img')).toHaveAttribute('src',second.image_preview.url);
  await expect(card.locator('.summary-citations a')).toHaveAttribute('href','https://example.gov.in/eastford');
  await expect(card.locator('.report-bottom a')).toHaveAttribute('href',`/api/reports/${second.id}/pdf`);
  await expect(card.locator('details')).toHaveCount(0);
});

for(const verdict of ['true','false']) {
  test(`${verdict} result links only direct verdict evidence and rejects malformed IDs`,async({page})=>{
    const report=imageReport(`direct-${verdict}`),direct=report.result.report_contract.sources[0];
    const related={...direct,id:3,source_key:'related-directory',title:'A directory mentioning the bridge',publisher:'Related directory',url:'https://example.org/directory',excerpt:'The bridge appears in a local directory.'};
    report.result.report_summary.verdict=verdict;
    report.result.report_summary.summary_source_ids=[3,11];
    report.result.report_contract.sources=[related,direct];
    report.result.report_contract.verification.verdict=verdict;
    report.result.report_contract.verification.cited_source_ids=[3,11];
    report.result.report_contract.verification.claims[0]={...report.result.report_contract.verification.claims[0],verdict,
      supporting_source_ids:verdict==='true'?[true,'3',11]:[],contradicting_source_ids:verdict==='false'?[true,'3',11]:[],related_source_ids:[3]};
    await setup(page,[report]);
    await page.goto('/#history');
    await page.getByRole('button',{name:'Open report',exact:true}).click();
    const card=page.locator('.report-card'),primary=card.locator('.summary-citations');
    await expect(card.locator('.summary-verdict')).toHaveText(verdict.toUpperCase());
    await expect(primary.getByRole('link')).toHaveCount(1);
    await expect(primary.locator('a[data-source-id="11"]')).toHaveAttribute('href',direct.url);
    await expect(card.locator('a[data-source-id="3"]')).toHaveCount(0);
    await expect(card.locator('a[href="https://example.org/directory"]')).toHaveCount(0);
    await expect(card).not.toContainText('Related directory');
    await expect(card.locator('.metric')).toHaveCount(3);
    await expect(card.locator('details')).toHaveCount(0);
  });
}
