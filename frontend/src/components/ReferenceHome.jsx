import React, { useEffect, useState } from 'react';
import { ArrowRight, BookOpenCheck, FileText, Image as ImageIcon, Play, ScanEye, ShieldCheck, Upload, Video } from 'lucide-react';
import { api } from '../lib/api.js';
import ContentTypeMenu from './ContentTypeMenu.jsx';

const modes = [
  { key: 'text', Icon: FileText, tone: 'text', description: 'modeTextDescription' },
  { key: 'image', Icon: ImageIcon, tone: 'image', description: 'modeImageDescription' },
  { key: 'video', Icon: Video, tone: 'video', description: 'modeVideoDescription' },
];

export default function ReferenceHome({ t, user, health, start, go }) {
  const [items, setItems] = useState([]);
  useEffect(() => {
    if (!user) { setItems([]); return; }
    let active = true;
    api('/reports').then(data => { if (active) setItems(data.items || []); }).catch(() => {});
    return () => { active = false; };
  }, [user]);
  const evidenceChecked = items.filter(item => item.result?.verification?.status && item.result.verification.status !== 'unverified').length;
  return <section className="view active" aria-label={t.home}>
    <div className="grid hero-grid">
      <div className="card hero">
        <div className="hero-copy">
          <div className="eyebrow">{t.homeHeroEyebrow}</div>
          <h2>{t.homeHeroLine1}<br/>{t.homeHeroLine2}</h2>
          <p>{t.homeHeroBody}</p>
          <ContentTypeMenu t={t} start={start} variant="home"/>
        </div>
        <div className="hero-art" aria-hidden="true"><div className="orbit"/><div className="shield"><ShieldCheck size={40}/></div><div className="float-tag tag-a">✦ {t.modelSignals}</div><div className="float-tag tag-b">⌕ {t.evidenceReview}</div><div className="float-tag tag-c">✓ {t.checkBeforeSharing}</div></div>
      </div>
      <div className="right-stack">
        <div className="card">
          <div className="mini-title"><h3>{t.howVerificationWorks}</h3><span className="subtle">{t.yourWorkflow}</span></div>
          <div className="pipeline">
            {[[Upload,t.submitContent,t.content],[ScanEye,t.inspectSignals,t.signals],[BookOpenCheck,t.reviewEvidence,t.evidence],[FileText,t.saveReport,t.report]].map(([Icon,title,detail],i)=><React.Fragment key={title}>{i>0&&<span className="pipe-arrow">›</span>}<div className="pipe-step"><div className="pipe-icon"><Icon size={18}/></div><b>{title}</b><small>{detail}</small></div></React.Fragment>)}
          </div>
        </div>
        <div className="card">
          <div className="mini-title"><h3>{t.yourWorkspace}</h3><span className="subtle">{t.liveAccountData}</span></div>
          <div className="metric-row"><div className="metric"><strong>{user ? items.length : '—'}</strong><span>{t.savedAnalyses}</span></div><div className="metric"><strong>{user ? evidenceChecked : '—'}</strong><span>{t.evidenceReviews}</span></div><div className="metric"><strong>{health?.status === 'ok' ? t.on : '—'}</strong><span>{t.apiConnection}</span></div></div>
        </div>
      </div>
    </div>
    <div className="section-row"><h2>{t.chooseWhatToVerify}</h2><button onClick={() => go('dashboard')}>{t.openDashboard} <ArrowRight size={15}/></button></div>
    <div className="grid mode-grid">{modes.map(({key,Icon,tone,description})=><button className="card mode-card" key={key} onClick={() => start(key)}><div className={`mode-visual ${tone}`}>{key==='text'?<div className="visual-sheet"><i/><i/><i/><i/></div>:key==='image'?<div className="visual-lens"/>:<span className="play"><Play size={23} fill="currentColor"/></span>}</div><div className="mode-content"><h3>{t[key]}</h3><p>{t[description]}</p><span className="text-link">{t.startCheck} <ArrowRight size={15}/></span></div></button>)}</div>
  </section>;
}
