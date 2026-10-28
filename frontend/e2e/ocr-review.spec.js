import { test, expect } from '@playwright/test';

const image={name:'poster.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/0YAAAAASUVORK5CYII=','base64')};
const saved={id:'automatic-report',type:'image',status:'complete',title:'saved-poster.png',content:'A claim read automatically from an image.',created_at:'2026-10-25T09:00:00Z',
 image_preview:{url:'/api/reports/automatic-report/preview',mime_type:'image/jpeg'},
 result:{claim_text:'A claim read automatically from an image.',media_origin:{verdict:'inconclusive'},media_inspection:{ocr_performed:true,visible_text:'Automatically extracted words.',description:'An uploaded graphic.'},
  report_summary:{schema_version:5,verdict:'unverified',recommendation_policy_version:1,verification_review:{policy_version:1,claim:'A claim read automatically from an image.',summary:'No relevant evidence was retrieved.',checked_sources:[],insufficiency_reasons:['No matching source was found.'],additional_evidence_needed:['An original announcement.']}}}};

async function mockWorkspace(page){
 const submitted=[];let previewRequests=0;
 await page.route('**/api/session',route=>route.fulfill({json:{user:{id:'qa',name:'QA',role:'user'},csrf_token:'qa'}}));
 await page.route('**/api/health',route=>route.fulfill({json:{status:'ok'}}));
 await page.route('**/api/media/ocr-preview',route=>{previewRequests++;return route.fulfill({status:500,json:{error:'Manual OCR preview should not be requested.'}});});
 await page.route('**/api/analyze',route=>{submitted.push(route.request().postDataBuffer().toString('utf8'));return route.fulfill({status:202,json:{id:saved.id,status:'queued'}});});
 await page.route('**/api/reports**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/api/reports')return route.fulfill({json:{items:[saved]}});
  if(path.endsWith('/preview'))return route.fulfill({contentType:'image/png',body:image.buffer});
  return route.fulfill({json:saved});
 });
 return {submitted,get previewRequests(){return previewRequests;}};
}

test('image analysis starts with one click and automatic OCR without a review prompt',async({page})=>{
 const calls=await mockWorkspace(page);
 await page.goto('/#verify');
 await page.getByRole('tab',{name:'Image',exact:true}).click();
 await page.locator('input[type="file"]').setInputFiles(image);
 await expect(page.locator('.input-card textarea')).toHaveCount(0);
 await page.getByRole('button',{name:'Analyze content',exact:true}).click();
 await expect.poll(()=>calls.submitted.length).toBe(1);
 expect(calls.previewRequests).toBe(0);
 expect(calls.submitted[0]).toContain('name="file"; filename="poster.png"');
 expect(calls.submitted[0]).not.toContain('ocr_reviewed');
 expect(calls.submitted[0]).not.toContain('reviewed_ocr_text');
 await expect(page.getByRole('region',{name:'Review image text before verification'})).toHaveCount(0);
});

test('caption stays optional and is sent only when the user adds it',async({page})=>{
 const calls=await mockWorkspace(page);
 await page.goto('/#verify');
 await page.getByRole('tab',{name:'Image',exact:true}).click();
 await page.locator('input[type="file"]').setInputFiles(image);
 await page.getByRole('button',{name:'Add optional claim or caption'}).click();
 await page.getByLabel('Optional claim or caption',{exact:true}).fill('The optional claim supplied with this image.');
 await page.getByRole('button',{name:'Analyze content',exact:true}).click();
 await expect.poll(()=>calls.submitted.length).toBe(1);
 expect(calls.submitted[0]).toContain('name="text"\r\n\r\nThe optional claim supplied with this image.');
 expect(calls.submitted[0]).not.toContain('ocr_reviewed');
 expect(calls.previewRequests).toBe(0);
});

test('a reopened image appears in the left input area and choosing another file clears the old result',async({page})=>{
 const calls=await mockWorkspace(page);
 await page.goto('/#history');
 await page.getByRole('button',{name:'Open report',exact:true}).click();
 const input=page.locator('.input-card');
 await expect(input.getByAltText('Uploaded image preview: saved-poster.png',{exact:true})).toHaveAttribute('src','/api/reports/automatic-report/preview');
 await expect(input.getByRole('button',{name:'Choose another file',exact:true})).toBeVisible();
 await expect(input.getByRole('button',{name:'Analyze content',exact:true})).toBeDisabled();
 await expect(page.locator('.report-card .report-media-preview')).toHaveCount(0);
 await page.locator('input[type="file"]').setInputFiles({...image,name:'different.png'});
 await expect(page.locator('.report-card')).toHaveCount(0);
 await expect(input.getByRole('button',{name:'Analyze content',exact:true})).toBeEnabled();
 expect(calls.submitted).toHaveLength(0);
});
