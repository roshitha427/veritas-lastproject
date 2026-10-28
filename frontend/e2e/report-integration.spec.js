import { test, expect } from '@playwright/test';

function imageReport(id, title) {
 const claim=`${title} was announced in 2025.`;
 return {id,type:'image',status:'complete',title,content:claim,created_at:'2026-10-25T09:00:00Z',
  image_preview:{url:`/api/reports/${id}/preview`,mime_type:'image/jpeg',width:96,height:64},
  result:{claim_text:claim,ai_percentage:51.9,real_percentage:48.1,confidence:51.9,
   media_origin:{verdict:'inconclusive'},media_inspection:{ocr_performed:true,visible_text:claim,description:'A graphic containing a claim.'},
   ocr_review:{user_reviewed:true,corrected_text:claim,original_text:claim},
   report_summary:{schema_version:5,verdict:'true',headline:'Evidence supports the claim',summary:'The dated original announcement confirms this event.',reason:'The source names the same event and year.',
    summary_source_ids:[7],sources:[{id:7,title:`${title} announcement`,url:`https://example.org/${id}`,snippet:`Exact evidence for ${id}.`}],claims:[]},
   report_contract:{schema_version:1,report_id:id,media_type:'image',mode:'claim_verification',revision:`revision-${id}`,
    claim:{checked_text:claim,user_reviewed:true,reviewed_ocr_text:claim,original_ocr_text:claim},
    verification:{verdict:'true',cited_source_ids:[7],claims:[{id:'claim-1',text:claim,verdict:'true',reason:'The original source confirms the event.',supporting_source_ids:[7],contradicting_source_ids:[],related_source_ids:[]}]},
    sources:[{id:7,source_key:`source-${id}`,title:`${title} announcement`,publisher:'Original publisher',url:`https://example.org/${id}`,excerpt:`Exact evidence for ${id}.`,accepted:true,cited:true,retrieval_status:'original_text'}],warnings:[]}
  }};
}

async function mediaFixtures(page) {
 const jpeg=await page.evaluate(()=>{
  const canvas=document.createElement('canvas');canvas.width=96;canvas.height=64;
  const context=canvas.getContext('2d');context.fillStyle='#2cc0b7';context.fillRect(0,0,96,64);
  return canvas.toDataURL('image/jpeg').split(',')[1];
 });
 return {jpeg:Buffer.from(jpeg,'base64')};
}

async function session(page) {
 await page.route('**/api/session',route=>route.fulfill({json:{user:{id:'qa',name:'QA',role:'user'},csrf_token:'qa'}}));
 await page.route('**/api/health',route=>route.fulfill({json:{status:'ok'}}));
}

test('history selection keeps preview, source navigation and PDF tied to the latest report',async({page})=>{
 const {jpeg}=await mediaFixtures(page),a=imageReport('report-a','Earlier event'),b=imageReport('report-b','Selected event');
 await session(page);
 let releaseA,requestedA=false;
 const delayedA=new Promise(resolve=>{releaseA=resolve;});
 await page.route('**/api/reports**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/api/reports')return route.fulfill({json:{items:[a,b]}});
  if(path.endsWith('/preview'))return route.fulfill({contentType:'image/jpeg',body:jpeg});
  if(path.endsWith('/pdf'))return route.fulfill({contentType:'application/pdf',headers:{'Content-Disposition':'attachment; filename="report-b.pdf"'},body:'%PDF-1.4\n% report-b\n%%EOF'});
  if(path==='/api/reports/report-a'){requestedA=true;await delayedA;return route.fulfill({json:a});}
  return route.fulfill({json:b});
 });
 await page.goto('/#history');
 await expect(page.getByAltText('Uploaded image preview: Selected event',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Open media report: Earlier event',exact:true}).click();
 await expect.poll(()=>requestedA).toBe(true);
 await page.getByRole('button',{name:'Open media report: Selected event',exact:true}).click();
 const report=page.locator('.report-card[data-report-id="report-b"]');
 await expect(report).toBeVisible();
 releaseA();
 await expect(page.locator('.input-card').getByAltText('Uploaded image preview: Selected event',{exact:true})).toHaveAttribute('src','/api/reports/report-b/preview');
 const source=report.locator('.summary-citations a[data-source-id="7"]');
 await expect(source).toHaveAttribute('href','https://example.org/report-b');
 await expect(source).toHaveAttribute('target','_blank');
 await expect(source).toHaveAttribute('rel','noopener noreferrer');
 await expect(page).toHaveURL(/#verify$/);
 await expect(report.locator('[data-report-claim]')).toContainText('Selected event was announced in 2025.');
 await expect(report).not.toContainText('Earlier event');
 await expect(report.locator('details')).toHaveCount(0);
 await expect(report.locator('.metric')).toHaveCount(3);
 const downloadEvent=page.waitForEvent('download');
 await report.getByRole('link',{name:'Download PDF',exact:true}).click();
 const download=await downloadEvent;
 expect(download.suggestedFilename()).toContain('report-b');
 expect(new URL(download.url()).pathname).toBe('/api/reports/report-b/pdf');
 await expect(report).toBeVisible();
 await expect(page.locator('.report-card[data-report-id="report-a"]')).toHaveCount(0);
});

test('private video has a history poster, a controlled player and the same report PDF link',async({page})=>{
 const {jpeg}=await mediaFixtures(page);
 let releaseVideo;const mediaGate=new Promise(resolve=>{releaseVideo=resolve;});
 const report={id:'video-c',type:'video',status:'complete',title:'Selected clip',created_at:'2026-10-25T09:00:00Z',content:'',
  video_preview:{url:'/api/reports/video-c/video',poster_url:'/api/reports/video-c/poster',mime_type:'video/webm'},
  result:{media_origin:{verdict:'inconclusive'},media_inspection:{description:'A short clip.'},report_summary:{schema_version:5,mode:'media_only',verdict:'unverified'}}};
 await session(page);
 await page.route('**/api/reports**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/api/reports')return route.fulfill({json:{items:[report]}});
  if(path.endsWith('/poster'))return route.fulfill({contentType:'image/jpeg',body:jpeg});
  if(path.endsWith('/video')){await mediaGate;return route.fulfill({contentType:'video/webm',body:'unsupported media fixture'});}
  return route.fulfill({json:report});
 });
 await page.goto('/#history');
 const thumbnail=page.getByAltText('Uploaded video preview: Selected clip',{exact:true});
 await expect(thumbnail).toHaveAttribute('src','/api/reports/video-c/poster');
 await expect(page.locator('video[controls]')).toHaveCount(0);
 await page.getByRole('button',{name:'Open media report: Selected clip',exact:true}).click();
 const card=page.locator('.report-card[data-report-id="video-c"]');
 const player=page.locator('.input-card').getByLabel('Uploaded video preview: Selected clip',{exact:true});
 await expect(player).toBeVisible();
 await expect(player).toHaveAttribute('src','/api/reports/video-c/video');
 expect(await player.evaluate(video=>video.controls&&!video.autoplay&&video.paused)).toBe(true);
 releaseVideo();
 await expect(page.locator('.input-card').getByRole('link',{name:'Open or download uploaded video',exact:true})).toHaveAttribute('href','/api/reports/video-c/video');
 await expect(card.getByRole('link',{name:'Download PDF',exact:true})).toHaveAttribute('href','/api/reports/video-c/pdf');
 await page.evaluate(()=>{location.hash='reports';});
 await expect(page.locator('.report-library-row.selected')).toContainText('Selected clip');
 await expect(page.locator('.report-library-row.selected img')).toHaveAttribute('src','/api/reports/video-c/poster');
});
