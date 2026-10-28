import { test, expect } from '@playwright/test';

const claim = 'Elon Musk announced free Tesla cars for everyone in 2025.';
function fixture(reviewed = false) {
  return { id: 'tesla-review', type: 'image', status: 'complete', title: claim, content: claim,
    created_at: '2026-10-24T09:00:00Z', result: {
      ai_percentage: 51.9, confidence: 51.9, real_percentage: 48.1, claim_text: claim,
      media_origin: { verdict: 'inconclusive', reason: 'Experimental origin signal.' },
      media_inspection: { description: 'A news-style graphic with a vehicle.', ocr_performed: true,
        visible_text: 'apay free Tesla vehicles', ocr_uncertain: true,
        uncertain_spans: [{ text: 'apay', reason: 'The word is not clearly readable.' }] },
      ocr_text: reviewed ? claim : 'apay free Tesla vehicles',
      ocr_review: reviewed ? { user_reviewed: true, changed: true, original_text: 'apay free Tesla vehicles', corrected_text: claim } : undefined,
      media_text_analysis: { content: claim, text_model_assessment: { fake_percentage: 98.7, real_percentage: 1.3, confidence: 98.7 }, ai_percentage: 21 },
      report_summary: { schema_version: 5, verdict: 'unverified', sources: [{ id: 1, title: 'New Year greetings', url: 'https://www.thestatesman.com/greetings' }],
        recommendation_policy_version: 1, recommended_sources: [], verification_review: {
          policy_version: 1, claim, summary: 'Three returned pages were excluded. No passage settles the free-car offer.',
          providers: [{ name: 'Tavily web search', status: 'searched' }],
          searches: [{ claim, provider_status: 'searched', query_statuses: [{ query: claim, status: 'searched', requested_domains: ['reuters.com'] }] }],
          checked_sources: [
            { title: 'New Year greetings', url: 'https://www.thestatesman.com/greetings', accepted: false, reason: 'The page discusses greetings, not a vehicle giveaway.' },
            { title: 'Elon Musk robot haircut', url: 'https://factly.in/robot', accepted: false, reason: 'The page concerns a different event.' },
            { title: 'BBC programme page', url: 'https://www.bbc.com/programmes/podcast', accepted: false, reason: 'A programme directory does not establish the offer.' }],
          insufficiency_reasons: ['The retrieved pages do not address the claimed worldwide offer.'],
          additional_evidence_needed: ['An original Tesla announcement giving the offer dates, eligibility and terms.'],
          scope_note: 'Only recorded searches and returned pages are listed. Suggested portals are not checked evidence.'
        } }
    } };
}

async function open(page, report) {
  await page.route('**/api/session', r => r.fulfill({ json: { user: { id: 'qa', name: 'QA', role: 'user' }, csrf_token: 'qa' } }));
  await page.route('**/api/health', r => r.fulfill({ json: { status: 'ok' } }));
  await page.route('**/api/reports**', r => r.fulfill({ json: new URL(r.request().url()).pathname === '/api/reports' ? { items: [report] } : report }));
  await page.goto('/#history');
  await page.getByRole('button', { name: 'Open report', exact: true }).click();
}

for (const sample of [
  { name: 'health', claim: 'A dietary supplement makes people live forever.',
    needed: 'Clinical research or guidance from a recognised health authority directly assessing the claimed effect on human health or lifespan.',
    expected: /Clinical research.*health authority/ },
  { name: 'general', claim: 'A material has an unconfirmed physical property.',
    needed: 'A primary-source record or reliable fact-check directly supporting or contradicting this claim.',
    expected: /Reliable source evidence directly addressing this claim/ },
]) {
  test(`${sample.name} evidence guidance does not require an event announcement`, async ({ page }) => {
    const report = { id: 'guidance-' + sample.name, type: 'text', status: 'complete',
      title: sample.claim, content: sample.claim, created_at: '2026-10-28T03:00:00Z', result: {
        verification: { status: 'unverified', evidence: [] },
        report_summary: { schema_version: 5, verdict: 'unverified',
          summary: 'The available source evidence does not settle this claim.',
          verification_review: { policy_version: 2, claim: sample.claim, additional_evidence_needed: [sample.needed] } },
      } };
    await open(page, report);
    await expect(page.locator('.summary-verdict')).toHaveText('UNVERIFIED');
    await expect(page.locator('.report-next-step')).toContainText(sample.expected);
    await expect(page.locator('.report-next-step')).not.toContainText(/dated|announcement/i);
  });
}

test('unverified image keeps only the claim-specific reason and evidence needed', async ({ page }) => {
  await open(page, fixture());
  const card = page.locator('.report-card');
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.claim-assessment-text')).toContainText(claim);
  expect((await card.locator('.summary-body').innerText()).length).toBeLessThanOrEqual(220);
  await expect(card.locator('.report-next-step')).toContainText(/Tesla|announcement|offer/i);
  expect((await card.locator('.report-next-step').innerText()).replace(/^Needed:\s*/, '').length).toBeLessThanOrEqual(160);
  await expect(card.locator('details')).toHaveCount(0);
  await expect(card.locator('.summary-citations')).toHaveCount(0);
  await expect(card.locator('.metric')).toHaveCount(3);
  await expect(card.locator('.metric-label')).toHaveText(['Real image pattern score', 'AI image pattern score', 'Image model confidence']);
  await expect(card.locator('.metric strong')).toHaveText(['48.1%', '51.9%', '51.9%']);
  await expect(card).not.toContainText(/New Year greetings|robot haircut|BBC programme|Tavily|reuters.com|98.7%/);
  await expect(card.getByRole('link', {name:/audit/i})).toHaveCount(0);
  await page.screenshot({ path: '../artifacts/verification-review.png', fullPage: true });
});

test('reviewed wording is the claim without duplicate OCR or origin analyses', async ({ page }) => {
  await open(page, fixture(true));
  const card = page.locator('.report-card');
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.claim-assessment-text')).toContainText(claim);
  await expect(card.locator('.metric strong')).toHaveText(['48.1%', '51.9%', '51.9%']);
  await expect(card).not.toContainText(/apay|OCR|transcription|98.7%|AI-writing|More details/);
  await expect(card.getByRole('region', { name:'Model metrics', exact:true })).toHaveCount(1);
});

test('school closure report is one short factual assessment with one official next step', async ({ page }) => {
  const original = 'BREAKING NEWS: Authorities have announced the immediate closure of all schools across Andhra Pradesh following severe weather conditions. Parents are advised to keep children at home until further notice. The message is circulating widely, but it does not include an official government notification or the date on which the original announcement was issued.';
  const core = 'Authorities have announced the immediate closure of all schools across Andhra Pradesh following severe weather conditions.';
  const report = {id:'ap-school-report',type:'text',title:'School closure message',status:'complete',content:original,created_at:'2026-10-26T13:25:00Z',result:{
    ai_percentage:6.2,real_percentage:48.4,fake_percentage:51.6,confidence:51.6,
    text_model_assessment:{status:'assessed',reason:'The news-pattern score is 51.6% with 51.6% model confidence.'},
    evidence_metrics:{status:'assessed',supported_percentage:0,contradicted_percentage:0,unverified_percentage:100,supported_claim_count:0,contradicted_claim_count:0,unverified_claim_count:3,total_claim_count:3},
    report_summary:{schema_version:5,verdict:'unverified',summary:'The supplied source records contain no official confirmation of a statewide Andhra Pradesh school closure.',reason:'No dated official notice confirms a statewide school closure.',claims:[{text:core,verdict:'unverified'}],
      sources:[{id:1,publisher:'www.bbc.com',url:'https://www.bbc.com/child-maintenance',snippet:'An unrelated UK article.'}],
      recommended_sources:[{name:'Andhra Pradesh Government',url:'https://www.ap.gov.in/',description:'Check departmental announcements.'},{name:'National Portal of India',url:'https://www.india.gov.in/'}],
      verification_review:{policy_version:1,claim:original,summary:'The recorded search returned 31 candidate pages; 6 were relevant and 25 were excluded.',additional_evidence_needed:['A dated Andhra Pradesh government or district education notice confirming the affected schools and closure dates.'],checked_sources:[{url:'https://www.bbc.com/child-maintenance',title:'Child maintenance',accepted:true}]},recommendation_policy_version:1}
  }};
  await open(page,report);
  const card=page.locator('.report-card');
  await expect(card).toHaveCount(1);
  await expect(card.locator('.summary-verdict')).toHaveCount(1);
  await expect(card.locator('.summary-verdict')).toHaveText('UNVERIFIED');
  await expect(card.locator('.claim-assessment-text')).toContainText(/closure.*schools.*Andhra Pradesh/i);
  expect((await card.locator('.claim-assessment-text').innerText()).length).toBeLessThanOrEqual(240);
  expect((await card.locator('.summary-body').innerText()).length).toBeLessThanOrEqual(220);
  await expect(card.locator('.metric')).toHaveCount(3);
  await expect(card.locator('.metric strong')).toHaveText(['48.4%','51.6%','51.6%']);
  await expect(card.locator('a.followup-link')).toHaveCount(1);
  await expect(card.locator('a.followup-link')).toHaveAttribute('href','https://www.ap.gov.in/');
  await expect(card).not.toContainText(/bbc|child maintenance|31 candidate|25 were excluded|Tavily|AI-writing|Local text classifier|More details|98.5%/i);
  await expect(card.locator('details')).toHaveCount(0);
  await expect(card.getByRole('link',{name:'Download PDF',exact:true})).toHaveCount(1);
  expect((await card.innerText()).length).toBeLessThan(1000);
  await page.screenshot({path:'../artifacts/ap-school-concise-report.png',fullPage:true});
});
