import React, { useEffect, useMemo, useState } from 'react';
import { ArrowUpRight, FileText, Image as ImageIcon, Video } from 'lucide-react';
import { api } from '../lib/api.js';
import { date, types } from '../lib/common.js';
import ContentTypeMenu from './ContentTypeMenu.jsx';

const icons = { text: FileText, image: ImageIcon, video: Video };

export default function Dashboard({ t, lang = 'en', user, start, onOpen }) {
  const [items, setItems] = useState([]);
  useEffect(() => {
    if (!user) { setItems([]); return; }
    let active = true;
    api('/reports').then(data => { if (active) setItems(data.items || []); }).catch(() => {});
    return () => { active = false; };
  }, [user]);
  const summary = useMemo(() => ({
    byType: Object.fromEntries(Object.keys(types).map(key => [key, items.filter(row => row.type === key).length])),
    weekdays: Array.from({length: 7}, (_, offset) => {
      const day = new Date();
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - (6 - offset));
      return {
        label: day.toLocaleDateString({en:'en-IN',hi:'hi-IN',te:'te-IN'}[lang], {weekday:'short'}),
        count: items.filter(row => new Date(row.created_at).toDateString() === day.toDateString()).length,
      };
    }),
    completed: items.filter(row => row.status === 'complete').length,
    active: items.filter(row => ['queued','analyzing'].includes(row.status)).length,
  }), [items, lang]);
  const total = items.length;
  const colors = { text: 'var(--violet)', image: 'var(--aqua)', video: 'var(--amber)' };
  const recent = items.slice(0, 4);
  return <section className="view active">
    <div className="page-head"><div><div className="eyebrow">{t.dashboardEyebrow}</div><h1>{t.dashboardTitle}</h1><p>{t.dashboardBody}</p></div><ContentTypeMenu t={t} start={start} variant="dashboard"/></div>
    {!user ? <div className="card"><h3>{t.dashboardPrivateTitle}</h3><p className="subtle">{t.dashboardPrivateBody}</p></div> : <>
      <div className="grid two-col">
        <div className="card"><div className="mini-title"><h3>{t.analysisBreakdown}</h3><span className="subtle">{t.savedReports}</span></div><div className="chart"><div className="donut" style={{background: total ? `conic-gradient(var(--violet) 0 ${(summary.byType.text/total)*100}%,var(--aqua) ${(summary.byType.text/total)*100}% ${((summary.byType.text+summary.byType.image)/total)*100}%,var(--amber) ${((summary.byType.text+summary.byType.image)/total)*100}% 100%)` : 'conic-gradient(#e9e8f3 0 100%)'}}><div className="donut-center"><b>{total}</b><small>{t.analyses}</small></div></div><div className="legend">{Object.entries(summary.byType).map(([key,count])=>{const Icon=icons[key];return <div className="legend-row" key={key}><span><i className="dot" style={{background:colors[key]}}/><Icon size={15}/>&nbsp;{t[key]}</span><b>{count}</b></div>})}<div className="legend-row"><span>{t.complete}</span><b>{summary.completed}</b></div></div></div></div>
        <div className="card"><div className="mini-title"><h3>{t.weeklyActivity}</h3><span className="subtle">{t.pastSevenDays}</span></div><div className="bars">{summary.weekdays.map(({label,count},i)=><div className="bar-col" key={i}><div className="bar" style={{height:`${count ? Math.max(12,Math.min(100,count*22)) : 3}%`,opacity:count?1:.24}}/><small>{label}</small></div>)}</div><div className="metric-row"><div className="metric"><strong>{summary.active}</strong><span>{t.inProgress}</span></div><div className="metric"><strong>{summary.completed}</strong><span>{t.completed}</span></div><div className="metric"><strong>{items.filter(row=>row.status==='failed').length}</strong><span>{t.needsAttention}</span></div></div></div>
      </div>
      <div className="section-row"><h2>{t.recentAnalyses}</h2><button onClick={() => location.hash='history'}>{t.viewHistory} <ArrowUpRight size={15}/></button></div>
      {recent.length ? <div className="card recent-list">{recent.map(item=>{const Icon=icons[item.type]||FileText;return <button key={item.id} className="recent-row" onClick={() => onOpen(item.id)}><span className="recent-icon"><Icon size={17}/></span><span className="recent-title"><small>{t[item.type]} · {date(item.created_at,lang)}</small><strong>{item.title}</strong></span><span className={`status-pill ${item.status}`}><i className="tiny-dot"/>{t[item.status]||item.status}</span><ArrowUpRight size={16}/></button>})}</div> : <div className="card empty-data"><h3>{t.noAnalysesYet}</h3><p>{t.noAnalysesBody}</p><ContentTypeMenu t={t} start={start} variant="dashboard"/></div>}
    </>}
  </section>;
}
