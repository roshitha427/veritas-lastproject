import { test, expect } from '@playwright/test';

test('live keyless fact-check sources appear in the saved report @live',async({page})=>{
 test.skip(process.env.VERITAS_LIVE_EVIDENCE!=='1','Enable explicitly with the isolated public-provider QA server.');
 await page.goto('/#verify');
 await page.getByRole('button',{name:'Sign in',exact:true}).click();
 const dialog=page.getByRole('dialog');
 await dialog.getByLabel('Email address').fill('qa-admin@example.invalid');
 await dialog.getByLabel('Password',{exact:true}).fill('local-browser-qa-only-password');
 await dialog.getByRole('button',{name:'Sign in',exact:true}).click();
 await expect(dialog).not.toBeVisible();
 await page.getByLabel('Paste a claim or news article').fill('NASA moon landing is fake.');
 const queued = page.waitForResponse(response => response.url().endsWith('/api/analyze') && response.request().method() === 'POST');
 await page.getByRole('button',{name:'Analyze content',exact:true}).click();
 const submission = await queued;
 expect(submission.status()).toBe(202);
 const {id} = await submission.json();
 await expect(page.locator('.report-summary .summary-verdict')).toBeVisible({timeout:120000});
 const savedResponse = await page.request.get(`/api/reports/${id}`);
 expect(savedResponse.ok()).toBeTruthy();
 const saved = await savedResponse.json();
 expect(saved.status).toBe('complete');
 expect(saved.result.verification.provider_status).toBe('searched');
 expect(saved.result.verification.evidence.length).toBeGreaterThan(0);
 await page.screenshot({path:'../artifacts/live-free-verification.png',fullPage:true});
 const event=page.waitForEvent('download');
 await page.getByRole('link',{name:'Download PDF',exact:true}).click();
 await (await event).saveAs('../artifacts/live-free-verification.pdf');
});
