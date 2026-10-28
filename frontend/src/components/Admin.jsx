import React, { useEffect, useRef, useState } from 'react';
import { ShieldCheck, ArrowUpRight, ArrowRight, ArrowLeft, FileText, Image as ImageIcon, Video, Search, ScanLine, Check, X, UploadCloud, LockKeyhole, Globe2, Volume2, History, LogOut, Menu, ChevronRight, ChevronDown, CircleHelp, Sparkles, Fingerprint, Link2, Download, LoaderCircle, AlertTriangle, ExternalLink, Trash2, LayoutDashboard, Users, Database, Settings2, CircleCheck, Plus, RefreshCw, Play, Eye, BookOpen, Pause, Leaf } from 'lucide-react';
import { api } from '../lib/api.js';
import { types, date, verificationMessage, scopeMessage, limitMessage, observationLabel } from '../lib/common.js';
import { Brand, IconBadge } from './ui.jsx';

function datasetStatus(dataset, t) {
 const labels={available:t.datasetAvailable,sampled:t.datasetSampled,metadata_only:t.datasetMetadataOnly,pending_media:t.datasetPendingMedia,missing_media:t.datasetMissingMedia};
 return labels[dataset.availability]||t.datasetMetadataRegistered;
}

function datasetCounts(counts, full=false) {
 const entries=Object.entries(counts||{}).filter(([,value])=>Number.isInteger(value)&&value>=0);
 if(full||!entries.some(([key])=>key.includes('/')))return entries;
 const totals={};
 for(const [key,value] of entries){const split=key.split('/')[0];totals[split]=(totals[split]||0)+value;}
 return Object.entries(totals);
}

function DatasetCard({dataset,t}) {
 const roles={training_and_evaluation:t.datasetTrainingEvaluation,experimental_training:t.datasetExperimentalTraining,evaluation_only:t.datasetEvaluationOnly,retrieval_only:t.datasetRetrievalOnly,metadata_only:t.datasetMetadataOnly,training_references:t.datasetTrainingReferences,training_pool:t.datasetTrainingPool};
 const links=(dataset.provenance?.source_urls||[]).filter(url=>{try{const parsed=new URL(url);return parsed.protocol==='https:'&&!parsed.username&&!parsed.password;}catch{return false;}});
 const manifest=/^(?:data|models|docs)\/[a-zA-Z0-9_./-]+$/.test(dataset.source_manifest||'')?dataset.source_manifest:null;
 return <section className="admin-model-card" aria-label={dataset.name}>
  <div><IconBadge icon={Database}/><span><strong>{dataset.name}</strong><small>{t[dataset.modality]||dataset.modality} · v{dataset.version}</small></span><span className={`status-pill ${dataset.availability==='available'?'complete':'queued'}`}>{datasetStatus(dataset,t)}</span></div>
  {dataset.role&&<p><strong>{t.datasetUse}:</strong> {roles[dataset.role]||dataset.role.replaceAll('_',' ')}</p>}
  {dataset.status_note&&<p>{dataset.status_note}</p>}
  {datasetCounts(dataset.counts).length>0&&<dl>{datasetCounts(dataset.counts).map(([key,value])=><div key={key}><dt>{t[key]||key.replaceAll('_',' ')}</dt><dd>{value.toLocaleString()}</dd></div>)}</dl>}
  {dataset.composition&&dataset.composition!=='standalone'&&<p><strong>{dataset.composition==='subset'?t.datasetSubset:t.datasetCombined}</strong>{dataset.overlaps_with?.length>0&&` · ${t.datasetOverlaps}: ${dataset.overlaps_with.join(', ')}`}</p>}
  <details className="report-evidence-details"><summary>{t.datasetDetails}</summary>
   {dataset.task&&<p><strong>{t.datasetTask}:</strong> {dataset.task.replaceAll('_',' ')}</p>}
   {dataset.label_meaning&&<p><strong>{t.datasetLabels}:</strong> {dataset.label_meaning}</p>}
   <p><strong>{t.datasetLicense}:</strong> {typeof dataset.license==='string'?dataset.license:t.notAssessed}</p>
   {manifest&&<p><strong>{t.datasetManifest}:</strong> <code>{manifest}</code></p>}
   {datasetCounts(dataset.counts,true).length>datasetCounts(dataset.counts).length&&<p>{datasetCounts(dataset.counts,true).map(([key,value])=>`${key}: ${value.toLocaleString()}`).join(' · ')}</p>}
   {links.length>0&&<ul>{links.map(url=><li key={url}><a href={url} target="_blank" rel="noopener noreferrer">{new URL(url).hostname}</a></li>)}</ul>}
   {dataset.limitations?.length>0&&<ul>{dataset.limitations.map((item,index)=><li key={index}>{item}</li>)}</ul>}
  </details>
 </section>;
}

function ModelCard({kind,card,t}) {
 const unvalidated=kind==='ai_origin'&&card.binary_classification_enabled===false;
 const evaluation=card.evaluation;
 const measured=Number.isFinite(evaluation?.accuracy)&&evaluation?.test_count>0;
 const diagnostics=Object.entries(card.evaluation_by_dataset||{}).filter(([,value])=>Number.isFinite(value?.accuracy)||Number.isFinite(value?.real_recall));
 return <section className="admin-model-card">
  <div><IconBadge icon={['image','video','ai_origin'].includes(kind)?ImageIcon:FileText}/><span><strong>{card.name}</strong><small>{card.version}</small></span>{unvalidated&&<span className="status-pill queued">{t.modelExperimentalUnvalidated}</span>}</div>
  {unvalidated&&<p>{t.originUnvalidatedNote}</p>}
  {measured?<dl><div><dt>{t.accuracy}</dt><dd>{(evaluation.accuracy*100).toFixed(1)}%</dd></div><div><dt>{t.testExamples}</dt><dd>{evaluation.test_count.toLocaleString()}</dd></div><div><dt>F1</dt><dd>{Number.isFinite(evaluation.f1)?evaluation.f1.toFixed(3):t.notAssessed}</dd></div></dl>:<p>{diagnostics.length>0&&`${t.datasetOverallAccuracy}: `}{t.notAssessed}</p>}
  {diagnostics.length>0&&<details className="report-evidence-details"><summary>{t.datasetDiagnostics}</summary><div className="table-scroll"><table><thead><tr><th>{t.datasetName}</th><th>{t.datasetMetric}</th><th>{t.datasetResult}</th><th>{t.testExamples}</th></tr></thead><tbody>{diagnostics.map(([name,value])=><tr key={name}><td>{name}</td><td>{Number.isFinite(value.accuracy)?t.datasetDiagnosticAccuracy:t.datasetRealRecall}</td><td>{((Number.isFinite(value.accuracy)?value.accuracy:value.real_recall)*100).toFixed(1)}%</td><td>{Number.isInteger(value.test_count??value.count)?(value.test_count??value.count).toLocaleString():t.notAssessed}</td></tr>)}</tbody></table></div><p>{t.datasetDiagnosticsNote}</p></details>}
  <p>{t.systemNote}</p>
 </section>;
}

export default function Admin({t,user,notify}){
 const [data,setData]=useState(null),[tab,setTab]=useState('settings'),[saving,setSaving]=useState(false);
 const videoChecks = data?.video_detection_status;
 async function load(){try{setData(await api('/admin/overview'))}catch(e){notify(e.message)}}useEffect(()=>{if(user?.role==='admin')load()},[user]);
 async function mutate(path,method,body){setSaving(true);try{await api(path,{method,body:JSON.stringify(body)});await load();return true}catch(e){notify(e.message);return false}finally{setSaving(false)}}
 if(user?.role!=='admin')return <div className="wrap section empty-state"><LockKeyhole/><h2>{t.admin}</h2><p>{t.administratorRequired}</p></div>;
 return <section className="wrap section app-section"><div className="page-heading"><span className="eyebrow">{t.footerNote}</span><h1>{t.adminTitle}</h1><p>{t.adminBody}</p></div>{!data?<LoaderCircle className="spin"/>:<><div className="admin-stats">{Object.entries(data.counts).map(([k,v])=><button className="admin-stat" onClick={()=>setTab(k)} key={k}><span>{t[k]}</span><strong>{v}</strong><ArrowUpRight size={20}/></button>)}</div><div className="admin-tabs">{[['settings',Settings2],['users',Users],['reports',FileText],['sources',Globe2],['datasets',Database]].map(([key,Icon])=><button className={tab===key?'active':''} onClick={()=>setTab(key)} key={key}><Icon size={17}/>{t[key]}</button>)}</div><div className="admin-panel">
 {tab==='settings'&&<><h3>{t.checkHealth}</h3><p className="muted">{t.systemNote}</p><div className="system-grid">{Object.entries(data.models).map(([key,ready])=><div key={key}><span><CircleCheck size={18}/>{t[key]||key}</span><span className={`status-pill ${ready?'complete':'queued'}`}>{ready?t.readyModel:key==='video_synthetic_detection'?t.videoSyntheticDisabled:t.notReady}</span></div>)}<div><span><Database size={18}/>{t.databaseLabel}</span><strong>{t.mysqlDatabase || 'MySQL'}</strong></div><div><span><ShieldCheck size={18}/>{t.passwordReset}</span><strong>{data.password_reset_email?t.readyModel:t.notReady}</strong></div></div>{data.models.video_synthetic_detection===false&&<p className="small muted">{t.videoSyntheticUnavailable}</p>}{Object.entries(data.model_cards||{}).map(([key,card])=><ModelCard key={key} kind={key} card={card} t={t}/>)}</>}
 {tab==='settings'&&videoChecks?.candidates?.length>0&&<details className="report-evidence-details video-validation-details"><summary>{t.videoValidationDetails}</summary><p>{videoChecks.reason}</p>{videoChecks.candidates.map(candidate=><section key={candidate.version||candidate.name}><strong>{candidate.name}</strong><small>{candidate.version} · {t.videoCandidateInactive}</small><p>{candidate.reason}</p>{Object.entries(candidate.dataset_diagnostics||{}).map(([name,metric])=><p key={name}>{name}: {t.datasetDiagnosticAccuracy} {Number.isFinite(metric.accuracy)?`${(metric.accuracy*100).toFixed(1)}%`:t.notAssessed}{Number.isInteger(metric.test_count)?` (${metric.test_count} ${t.testExamples})`:''}</p>)}</section>)}<p>{t.datasetDiagnosticsNote}</p></details>}
 {tab==='users'&&<><h3>{t.userAccess}</h3><div className="table-scroll"><table><thead><tr><th>{t.name}</th><th>{t.email}</th><th>{t.status}</th><th/></tr></thead><tbody>{data.users.map(u=><tr key={u.id}><td>{u.name}<small>{u.role}</small></td><td>{u.email}</td><td>{u.disabled?t.disabled:t.enabled}</td><td>{u.id!==user.id&&<button className="button subtle" disabled={saving} onClick={()=>mutate(`/admin/users/${u.id}`,'PATCH',{disabled:!u.disabled})}>{u.disabled?t.enable:t.disable}</button>}</td></tr>)}</tbody></table></div></>}
 {tab==='reports'&&<><h3>{t.reports}</h3><div className="table-scroll"><table><thead><tr><th>{t.claim}</th><th>{t.modality}</th><th>{t.status}</th><th>{t.date}</th></tr></thead><tbody>{data.reports.map(r=><tr key={r.id}><td>{r.title}</td><td>{t[r.type]}</td><td>{r.status}</td><td>{new Date(r.created_at).toLocaleDateString()}</td></tr>)}</tbody></table></div></>}
 {tab==='sources'&&<><form className="admin-form" onSubmit={async e=>{e.preventDefault();const form=e.currentTarget;if(await mutate('/admin/sources','POST',Object.fromEntries(new FormData(form))))form.reset()}}><h3>{t.addSource}</h3><input required name="name" placeholder={t.sourceName} aria-label={t.sourceName}/><input required name="url" type="url" placeholder={t.sourceUrl} aria-label={t.sourceUrl}/><button disabled={saving} className="button primary"><Plus size={17}/>{t.addSource}</button></form>{data.sources.map(s=><div className="admin-source" key={s.id}><div><strong>{s.name}</strong><span>{s.domain}</span></div><button disabled={saving} className={`toggle ${s.enabled?'on':''}`} aria-label={`${s.enabled?t.disable:t.enable} ${s.name}`} aria-pressed={s.enabled} onClick={()=>mutate(`/admin/sources/${s.id}`,'PATCH',{enabled:!s.enabled})}><i/></button></div>)}</>}
 {tab==='datasets'&&<><form className="admin-form" onSubmit={async e=>{e.preventDefault();const form=e.currentTarget;if(await mutate('/admin/datasets','POST',{...Object.fromEntries(new FormData(form)),counts:{}}))form.reset()}}><h3>{t.addDataset}</h3><p>{t.datasetNote}</p><input name="name" placeholder={t.datasetName} aria-label={t.datasetName} required/><input name="version" placeholder={t.version} aria-label={t.version} required/><select name="modality" aria-label={t.modality}>{Object.keys(types).map(k=><option key={k}>{k}</option>)}</select><button disabled={saving} className="button primary"><Plus size={16}/>{t.save}</button></form><p className="muted">{t.datasetInventoryNote}</p>{data.datasets.map(d=><DatasetCard key={d.id} dataset={d} t={t}/>)}</>}
 </div></>}</section>
}

