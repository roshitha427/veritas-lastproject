import React, { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, FileText, ScanLine, X, UploadCloud, LockKeyhole, Globe2, ChevronDown, LoaderCircle, Plus, BookOpen, Leaf } from 'lucide-react';
import { api } from '../lib/api.js';
import { analysisErrorMessage } from '../lib/analysisErrors.js';
import { types } from '../lib/common.js';
import { IconBadge } from './ui.jsx';
import Report from './Report.jsx';
import ReportMediaPreview, { reportPreviewUrl } from './ReportMediaPreview.jsx';

export default function Workspace({t,lang,user,onLogin,kind,setKind,report,setReport,notify}) {
 const [text,setText]=useState(''),[file,setFile]=useState(null),[preview,setPreview]=useState(''),[drag,setDrag]=useState(false),[busy,setBusy]=useState(false);
 const [captionOpen,setCaptionOpen]=useState(false);
 const fileInput=useRef(null);
 useEffect(()=>{if(!file){setPreview('');return}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url)},[file]);
 const pending=busy||['queued','analyzing'].includes(report?.status);
 function choose(f){
  if(!f||pending)return;
  if(f.size>50*1024*1024){notify(t.maxFileSize);return}
  const allowed=kind==='image'?/\.(jpg|jpeg|png|webp)$/i:/\.(mp4|webm|mov)$/i;
  if(!allowed.test(f.name)){notify(kind==='image'?t.imageFormats:t.videoFormats);return}
  setFile(f);setReport(null);
 }
 async function submit(e){
  e.preventDefault();
  if(pending)return;
  if(kind==='text'&&text.trim().length<20){notify('Enter at least 20 characters of text.');return}
  if(kind!=='text'&&!file)return;
  if(!user){notify(t.signInFirst);onLogin();return}
  setBusy(true);
  try{
   const form=new FormData();
   form.append('type',kind);form.append('text',kind==='text'||captionOpen?text:'');form.append('language',lang);
   if(file)form.append('file',file);
   const response=await api('/analyze',{method:'POST',body:form});
   setReport({id:response.id,status:'queued',type:kind});
  }catch(e){notify(analysisErrorMessage(e))}finally{setBusy(false)}
 }
 function change(k){setKind(k);setText('');setCaptionOpen(false);setFile(null);setDrag(false);setReport(null);if(fileInput.current)fileInput.current.value=''}
 const savedMedia=!file&&kind!=='text'&&report?.type===kind&&report.status==='complete';
 const savedPreview=savedMedia&&reportPreviewUrl(report);
 const mediaPreviewOutside=kind!=='text'&&Boolean(file||savedPreview);

 return <section className="wrap section app-section">
  <div className="page-heading"><span className="eyebrow">{t.workspace}</span><h1>{t.verifyTitle}</h1><p>{t.verifyBody}</p></div>
  <div className="verify-guide" aria-label={t.quickGuide}>{[[FileText,t.guideChoose],[UploadCloud,t.guideAdd],[BookOpen,t.guideReview]].map(([Icon,label],i)=><div className="verify-guide-step" key={i}><span className="verify-guide-number">0{i+1}</span><Icon size={18}/><span>{label}</span></div>)}</div>
  <div className="workspace-grid"><div className="input-column"><form className="input-card" onSubmit={submit}>
   <div className="content-tabs" role="tablist" aria-label={t.verify}>{Object.entries(types).map(([key,Icon])=><button disabled={pending} key={key} type="button" id={`tab-${key}`} role="tab" aria-selected={kind===key} aria-controls="content-input" className={kind===key?'active':''} onClick={()=>change(key)}><Icon size={21}/>{t[key]}</button>)}</div>
   <div className="input-body" id="content-input" role="tabpanel" aria-labelledby={`tab-${kind}`}>
    {kind==='text'&&<p className="language-note"><Globe2 size={13}/>{t.englishOnly}</p>}
    {kind!=='text'&&<>
     <input ref={fileInput} className="file-input" type="file" disabled={pending} accept={kind==='image'?'.jpg,.jpeg,.png,.webp':'.mp4,.mov,.webm'} onChange={e=>choose(e.target.files?.[0])} tabIndex={-1} aria-hidden="true"/>
     {file?<div className="upload-preview">{kind==='image'?<img src={preview} alt={file.name}/>:<video src={preview} controls preload="metadata"/>}<div><span>{kind==='image'?(t.selectedImage||'Selected image'):(t.selectedVideo||'Selected video')}<small>{(file.size/1024/1024).toFixed(2)} MB</small></span><button disabled={pending} type="button" className="round-button" aria-label={t.remove} onClick={()=>{setFile(null);setReport(null);fileInput.current.value=''}}><X size={16}/></button></div></div>:savedMedia?<div className="saved-media-input">
      <ReportMediaPreview report={report} t={t} fallback={<p>{t.savedPreviewUnavailable||'This older report has no saved media preview.'}</p>}/>
      <p className="small muted">{t.savedMediaReadOnly||'This is a saved report. Choose another file to run a new analysis.'}</p>
      <button type="button" className="button subtle" onClick={()=>fileInput.current?.click()}>{t.chooseAnotherFile||'Choose another file'}</button>
     </div>:<button disabled={pending} type="button" className={`dropzone ${drag?'dragging':''}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);choose(e.dataTransfer.files?.[0])}} onClick={()=>fileInput.current.click()}><span className="upload-icon"><UploadCloud size={34} strokeWidth={1.4}/><Plus size={13}/></span><strong>{t.upload}</strong><span>{t.browse}</span><small>{kind==='image'?t.imageFormats:t.videoFormats}</small></button>}
    </>}
    {kind==='text'?<>
     <label className="field-label" htmlFor="news-content">{t.claim}</label><textarea id="news-content" disabled={pending} value={text} onChange={e=>setText(e.target.value)} placeholder={t.placeholder} maxLength={12000} rows={6}/><div className="text-count">{text.length.toLocaleString()} / 12,000 {t.chars}</div>
    </>:!savedMedia&&<div className="optional-caption"><button type="button" className="optional-caption-toggle" aria-expanded={captionOpen} disabled={pending} onClick={()=>{if(captionOpen)setText('');setCaptionOpen(!captionOpen)}}>{captionOpen?'Remove optional claim or caption':'Add optional claim or caption'} <ChevronDown size={16}/></button>{captionOpen&&<><label className="field-label" htmlFor="news-content">Optional claim or caption</label><textarea id="news-content" disabled={pending} value={text} onChange={e=>setText(e.target.value)} placeholder="Optional: add a factual claim to check; leave blank for media-only analysis." maxLength={12000} rows={3}/><div className="text-count">{text.length.toLocaleString()} / 12,000 {t.chars}</div></>}</div>}
    <button className="button primary analyze-button" disabled={pending||(!file&&kind!=='text')} type="submit">{pending?<LoaderCircle size={19} className="spin"/>:<ScanLine size={20}/>} {pending?t.analyzing:t.analyze}{!pending&&<ArrowUpRight size={18}/>}</button>
    <p className="privacy-note"><LockKeyhole size={13}/>{t.privacyNote}</p>
   </div></form><div className="context-tip"><IconBadge icon={Leaf}/><div><strong>{t.responsible}</strong><p>{t.learnBody}</p></div></div></div>
   <Report report={report} t={t} lang={lang} mediaPreviewOutside={mediaPreviewOutside} onNew={()=>setReport(null)}/>
  </div>
 </section>
}
