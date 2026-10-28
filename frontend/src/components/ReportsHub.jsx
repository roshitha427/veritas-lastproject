import React, { useEffect, useState } from 'react';
import { ArrowUpRight, BookOpenCheck, ChartNoAxesCombined, Download, FileText, SearchCheck } from 'lucide-react';
import { api } from '../lib/api.js';
import { date, types } from '../lib/common.js';
import Report from './Report.jsx';
import ContentTypeMenu from './ContentTypeMenu.jsx';
import ReportMediaPreview, { mediaReportTitle } from './ReportMediaPreview.jsx';

export default function ReportsHub({ t, lang, user, report, onOpen, start }) {
 const [items,setItems]=useState([]),[busy,setBusy]=useState(false);
 useEffect(()=>{
  if(!user){setItems([]);return}
  let active=true;
  setBusy(true);
  api('/reports').then(data=>{
   if(active)setItems((data.items||[]).map(item=>item.id===report?.id?{...item,...report}:item));
  }).catch(()=>{}).finally(()=>{if(active)setBusy(false)});
  return()=>{active=false};
 },[user?.id,report?.id,report?.status]);
 return <section className="view active"><div className="page-head"><div><div className="eyebrow">{t.reportsEyebrow}</div><h1>{t.reportsTitle}</h1><p>{t.reportsDescription}</p></div>{user&&<ContentTypeMenu t={t} start={start} variant="dashboard"/>}</div>
  {!user?<div className="card"><h3>{t.noAccountHistory}</h3><p className="subtle">{t.signInFirst}</p></div>:<div className="grid two-col reports-hub-grid">
   {report ? <Report report={report} t={t} lang={lang} onNew={() => start(report.type)}/> : <div className="card reports-preview-card"><div className="empty-illustration"><FileText size={52} strokeWidth={1.2}/></div><h3>{items.length ? t.reportPreviewTitle : t.reportsEmpty}</h3><p className="subtle">{items.length ? t.reportPreviewBody : t.noAnalysesBody}</p>{items.length===0&&!busy&&<ContentTypeMenu t={t} start={start} variant="dashboard"/>}</div>}
   <div className="card reports-library-card"><div className="mini-title"><h3>{items.length ? t.savedReports : t.reportIncludes}</h3><span className="subtle">{items.length ? `${items.length} ${t.reportCount}` : t.reportFromAnalysis}</span></div>
    {items.length>0 ? <div className="report-library">{items.map(item=>{const Icon=types[item.type]||FileText;return <article className={`report-library-row ${report?.id===item.id?'selected':''}`} key={item.id}>
      <button className="report-library-open" onClick={()=>onOpen(item.id)} aria-label={`${t.open} · ${item.title}`}><ReportMediaPreview report={item} t={t} thumbnail large fallback={<span className="pipe-icon"><Icon size={17}/></span>}/><span><strong>{mediaReportTitle(item,t)}</strong><small>{t[item.type]} · {date(item.created_at,lang)}</small></span><ArrowUpRight size={15}/></button>
      <span className={`status-pill ${item.status}`}>{t[item.status]||item.status}</span>
      {item.status==='complete'&&<a className="report-library-download" href={`/api/reports/${item.id}/pdf`} download={`verification-report-${item.id.slice(0,10)}.pdf`} aria-label={`${t.downloadPdf} · ${item.title}`} title={t.downloadPdf}><Download size={16}/></a>}
    </article>})}</div> : <div className="signal-list"><div className="signal"><div className="signal-icon"><SearchCheck size={17}/></div><div><strong>{t.modelFindings}</strong><small>{t.modelFindingsBody}</small></div></div><div className="signal"><div className="signal-icon"><ChartNoAxesCombined size={17}/></div><div><strong>{t.contentSignals}</strong><small>{t.contentSignalsBody}</small></div></div><div className="signal"><div className="signal-icon"><BookOpenCheck size={17}/></div><div><strong>{t.evidenceReferences}</strong><small>{t.evidenceReferencesBody}</small></div></div><div className="signal"><div className="signal-icon">!</div><div><strong>{t.limitsAndContext}</strong><small>{t.limitsAndContextBody}</small></div></div></div>}
   </div>
  </div>}
 </section>;
}
