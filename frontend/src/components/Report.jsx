import React, { useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, CircleCheck, CircleHelp, Download, ExternalLink, FileText,
  Fingerprint, ScanLine, Search, ShieldCheck, Sparkles } from 'lucide-react';
import { date } from '../lib/common.js';
import { api } from '../lib/api.js';
import { analysisErrorMessage } from '../lib/analysisErrors.js';
import { IconBadge } from './ui.jsx';
import ReportMediaPreview from './ReportMediaPreview.jsx';

function brief(text, limit) {
  const value = String(text || '').replace(/\s+/g, ' ').trim();
  if (value.length <= limit) return value;
  const end = value.lastIndexOf(' ', limit - 1);
  return value.slice(0, end > 0 ? end : limit - 1).trimEnd() + '\u2026';
}

function sourceLink(value) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

function sourceRows(summary, contract) {
  const rows = Array.isArray(contract?.sources) ? contract.sources :
    (Array.isArray(summary?.sources) ? summary.sources : []).map((row, index) => ({ ...row, id: row.id ?? index + 1 }));
  const counts = new Map();
  rows.forEach(row => counts.set(String(row.id), (counts.get(String(row.id)) || 0) + 1));
  const rejected = new Set((summary?.verification_review?.checked_sources || [])
    .filter(row => row.accepted === false).map(row => sourceLink(row.url)).filter(Boolean));
  return rows.filter(row => Number.isInteger(row.id) && row.id > 0 && counts.get(String(row.id)) === 1 &&
    row.citation_usable !== false && (contract ? row.accepted === true : row.accepted !== false) &&
    sourceLink(row.url) && !rejected.has(sourceLink(row.url)));
}

function directSources(summary, contract, verdict) {
  if (!['true', 'false'].includes(verdict)) return [];
  const claims = contract?.verification?.claims || summary.claims || [];
  const ids = claims.filter(row => row.verdict === verdict).flatMap(row => verdict === 'true'
    ? row.supporting_source_ids ?? row.supporting_evidence ?? []
    : row.contradicting_source_ids ?? row.contradicting_evidence ?? []);
  const wanted = new Set((ids.length ? ids : summary.summary_source_ids || [])
    .filter(id => Number.isInteger(id) && id > 0));
  const seen = new Set();
  return sourceRows(summary, contract).filter(row => {
    const url = sourceLink(row.url);
    if (!wanted.has(row.id) || seen.has(url)) return false;
    seen.add(url); return true;
  }).slice(0, 2);
}

function manualCheckSources(summary) {
  const recommendations = Array.isArray(summary.recommended_sources) ? summary.recommended_sources : [];
  const seen = new Set();
  return recommendations.filter(row => {
    if (typeof row?.name !== 'string' || !row.name.trim() || typeof row.url !== 'string' ||
        typeof row.description !== 'string' || !row.description.trim()) return false;
    if (/[\s\u0000-\u001f\u007f]/.test(row.url.trim())) return false;
    const href = sourceLink(row.url);
    if (!href) return false;
    const normalized = new URL(href);
    if (normalized.port) return false;
    normalized.hash = '';
    normalized.pathname = normalized.pathname.replace(/\/+$/, '') || '/';
    normalized.searchParams.sort();
    const key = normalized.href;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, 3);
}

function narrative(value, limit) {
  const text = String(value || '').trim();
  const technical = /internal (?:error|failure|model)|provider (?:error|failure)|failed[- ]model|model[- ]only|appears? below|(?:Gemini|Mistral) (?:generation|service|API|network|failed|error)/i.test(text);
  return technical ? '' : brief(text, limit);
}

function explanation(summary, verdict) {
  const value = narrative(summary.summary, 1200) || narrative(summary.reason, 1200);
  return value || (verdict === 'unverified'
    ? 'Reliable evidence is insufficient to confirm or refute this claim. The available records do not establish every material part of the statement, so it remains unverified.'
    : 'The cited evidence was compared with the submitted claim. Review the explanation and original sources to understand which part of the statement is supported or contradicted.');
}

function distinctReason(summary, overall) {
  const reason = narrative(summary.reason, 900);
  const normalize = value => value.replace(/\[\d+\]/g, '').replace(/[^\p{L}\p{N}]+/gu, ' ').trim().toLowerCase();
  if (!reason || normalize(overall).includes(normalize(reason))) return '';
  return reason;
}

function generalExplanation(summary, overall) {
  const assessment = summary.gemini_model_assessment;
  if (summary.verdict !== 'unverified' || assessment?.status !== 'completed' || typeof assessment.answer !== 'string') return '';
  const answer = narrative(assessment.answer.replace(/\[\d+\]/g, ''), 500);
  if (!answer || overall.toLowerCase().includes(answer.toLowerCase())) return '';
  return answer;
}

function evidencePoints(summary, contract, overall, reason) {
  const available = new Set(sourceRows(summary, contract).map(source => source.id));
  const seen = new Set();
  return (Array.isArray(summary.key_points) ? summary.key_points : []).filter(point => {
    const text = narrative(point?.text, 300);
    const ids = point?.source_ids;
    if (!text || !Array.isArray(ids) || !ids.length || !ids.every(id => available.has(id))) return false;
    const key = text.toLowerCase().replace(/\s+/g, ' ');
    if (seen.has(key) || overall.toLowerCase().includes(key) || reason.toLowerCase().includes(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, 3);
}

function evidenceNeeded(summary, claim) {
  if (/\bschools?\b/i.test(claim) && /\b(?:clos\w*|shut\w*|holiday)\b/i.test(claim))
    return 'A dated official notice naming the affected schools and closure dates is needed.';
  const items = summary.verification_review?.additional_evidence_needed || summary.missing_evidence || [];
  const first = Array.isArray(items) ? items.find(value => typeof value === 'string' && value.trim()) : '';
  if (!first) return '';
  return brief(/^A primary-source record or reliable fact-check directly/i.test(first)
    ? 'Reliable source evidence directly addressing this claim is needed.' : first, 160);
}

function Metric({ title, value, icon: Icon, tone, t }) {
  const available = typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100;
  return <div className={`metric ${tone} ${available ? '' : 'metric-unavailable'}`}>
    <div className="metric-label"><Icon size={16}/><span>{title}</span></div>
    <strong>{available ? <>{value.toFixed(1)}<small>%</small></> : <span className="not-assessed">{t.notAssessed || 'Not assessed'}</span>}</strong>
    <div className="metric-track" aria-hidden="true">{available && <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }}/>}</div>
  </div>;
}

const MODEL_METRIC_LABELS = {
  text: ['Real news pattern score', 'Fake news pattern score', 'Text model confidence'],
  image: ['Real image pattern score', 'AI image pattern score', 'Image model confidence'],
  video: ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence'],
};

function ModelMetrics({ result, kind, t }) {
  const isMedia = kind === 'image' || kind === 'video';
  const labels = MODEL_METRIC_LABELS[kind] || MODEL_METRIC_LABELS.text;
  const fake = result.fake_percentage ?? (isMedia ? result.ai_percentage : null);
  return <section className="model-metrics" aria-label="Model metrics">
    <div className="metric-grid">
      <Metric title={labels[0]} value={result.real_percentage} icon={ShieldCheck} tone="green" t={t}/>
      <Metric title={labels[1]} value={fake} icon={ScanLine} tone="coral" t={t}/>
      <Metric title={labels[2]} value={result.confidence} icon={Fingerprint} tone="blue" t={t}/>
    </div>
    {kind === 'text' && <p className="media-metric-note">Real/Fake describe news-writing patterns; they do not verify this claim.</p>}
  </section>;
}

function MediaResult({ result, kind, t }) {
  const origin = result.media_origin || {}, labels = result.media_labels || {};
  const verdicts = { ai_generated: 'Likely AI-generated', likely_real: kind === 'image' ? 'Real image' : 'Not flagged as AI-generated',
    likely_not_ai_generated: kind === 'image' ? 'Real image' : 'Not flagged as AI-generated', ai_content_declared: 'AI-generated content declared' };
  const styles = { photograph: 'Photograph', cartoon_or_illustration: 'Cartoon or illustration', screenshot_or_document: 'Screenshot or document' };
  const headline = labels.origin_label || origin.headline || verdicts[origin.verdict] || 'Origin inconclusive';
  const description = origin.basis === 'content_credentials' ? origin.reason : result.media_inspection?.description;
  const context = narrative(origin.reason, 600);
  return <>
    <section className="media-inspection-card" aria-label="Media result">
      <div className="media-inspection-copy"><div className="media-inspection-heading">
        <strong title={labels.confirmed_origin ? 'Supplied reference label; separate from model scores.' : undefined}>{headline}</strong>
        <span>{labels.style || styles[result.media_inspection?.visual_type] || ''}</span>
      </div>{description && <p>{narrative(description, 900)}</p>}
      {context && context !== narrative(description, 900) && <div className="report-verdict-reason"><h4>Assessment context</h4><p>{context}</p></div>}
      {labels.confirmed_origin && <small>Supplied reference label.</small>}</div>
    </section>
  </>;
}

function savedReportSummary() {
  return { verdict: 'unverified', summary: 'This saved report has no completed evidence assessment.' };
}

export default function Report({ report, t, lang, onNew, mediaPreviewOutside = false }) {
  const [loadedReport, setDisplayReport] = useState(report);
  const displayReport = loadedReport?.id === report?.id ? loadedReport : report;
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    setDisplayReport(report);
    const needsMetricRefresh = report?.type === 'text' && (
      report.result?.text_model_assessment?.status === 'claim_out_of_domain' ||
      report.result?.ai_assessment?.status === 'too_short');
    const isMediaReport = report?.type === 'image' || report?.type === 'video';
    const needsKnowledgeRefresh = report?.type === 'text' &&
      report.result?.report_summary?.gemini_model_assessment?.status === 'completed' &&
      (report.result?.report_summary?.knowledge_policy_version || 0) < 2;
    const needsEvidenceMetricRefresh = report?.type === 'text' && !report.result?.evidence_metrics;
    const needsRecommendationRefresh = report?.result?.report_summary?.verdict === 'unverified' &&
      report.result.report_summary.mode !== 'media_only' &&
      (report.result.report_summary.recommendation_policy_version || 0) < 2;
    const needsVerificationReview = report?.result?.report_summary?.verdict === 'unverified' &&
      report.result.report_summary.mode !== 'media_only' &&
      (report.result.report_summary.verification_review?.policy_version || 0) < 2;
    if (!report?.id || report.status !== 'complete' ||
        (report.result?.report_summary?.schema_version >= 4 && !needsMetricRefresh && !isMediaReport && !needsKnowledgeRefresh && !needsEvidenceMetricRefresh && !needsRecommendationRefresh && !needsVerificationReview)) {
      setRefreshing(false);
      return undefined;
    }
    let active = true;
    setRefreshing(true);
    api(`/reports/${report.id}`).then(updated => {
      if (active) setDisplayReport(updated);
    }).catch(() => {
      // Keep the saved report visible and show a useful fallback below.
    }).finally(() => {
      if (active) setRefreshing(false);
    });
    return () => { active = false; };
  }, [report?.id, report?.status, report?.result?.report_summary]);

  if (!displayReport) return <div className="empty-result">
    <div className="empty-orbit"><Search size={45} strokeWidth={1.2}/><span><Sparkles size={18}/></span></div>
    <div className="eyebrow">{t.result}</div><h3>{t.ready}</h3><p>{t.readyBody}</p>
    <div className="empty-steps"><FileText/><span>›</span><ScanLine/><span>›</span><ShieldCheck/></div>
  </div>;
  if (['queued', 'analyzing'].includes(displayReport.status)) return <div className="empty-result loading-result" role="status" aria-live="polite">
    <div className="analysis-loader"><ScanLine size={45}/><i/></div><div className="eyebrow">{t.result}</div>
    <h3>{displayReport.status === 'queued' ? t.queued : t.processing}</h3><p>{t.pendingBody}</p>
    <div className="loader-track"><span/></div><div className="small muted">{t.reportIdLabel} · {report.id.slice(0, 10)}</div>
  </div>;
  if (displayReport.status === 'failed') return <div className="empty-result">
    <IconBadge icon={AlertTriangle} tone="amber"/><h3>{t.failed}</h3><p>{analysisErrorMessage(displayReport.error)}</p>
    <button className="button primary" onClick={onNew}>{t.retry}<ArrowRight size={18}/></button>
  </div>;

  const result = displayReport.result || {};
  const summary = result.report_summary?.schema_version >= 4 ? result.report_summary : savedReportSummary();
  const contract = result.report_contract?.schema_version >= 1 && result.report_contract.report_id === displayReport.id ? result.report_contract : null;
  const claims = contract?.verification?.claims || summary.claims || [];
  const original = contract?.claim?.checked_text || summary.verification_review?.claim || result.claim_text || displayReport.content || result.media_text_analysis?.content || '';
  const claimText = claims.find(row => row.text?.trim())?.text || original;
  const isMedia = ['image', 'video'].includes(displayReport.type);
  const hasClaim = summary.mode !== 'media_only' && (!isMedia || Boolean(claimText.trim()) || ['true', 'false'].includes(summary.verdict));
  const verdict = ['true', 'false'].includes(summary.verdict) ? summary.verdict : 'unverified';
  const sources = directSources(summary, contract, verdict);
  const followups = verdict === 'unverified' ? manualCheckSources(summary) : [];
  const needed = verdict === 'unverified' ? evidenceNeeded(summary, claimText) : '';
  const overall = explanation(summary, verdict);
  const general = !isMedia ? generalExplanation(summary, overall) : '';
  const reason = distinctReason(summary, overall);
  const points = evidencePoints(summary, contract, overall, reason);
  const VerdictIcon = verdict === 'true' ? CircleCheck : verdict === 'false' ? AlertTriangle : CircleHelp;
  return <article className="report-card concise-report single-report" data-report-id={displayReport.id} aria-live="polite">
    <div className="report-heading"><span className="eyebrow">{t.verificationReport || 'Verification report'}</span><span className="saved-note"><CircleCheck size={13}/>{t.saved}</span></div>
    <div className="report-title-block report-overview"><div>
      <h3 className="report-title">{hasClaim ? (t.claimVerificationTitle || 'Claim verification') : (t.mediaInspectionTitle || 'Media inspection')}</h3>
      {isMedia && !mediaPreviewOutside && <ReportMediaPreview report={displayReport} t={t}/>}
      <p className="report-overview-meta"><span>{t[displayReport.type] || displayReport.type}</span><span>{date(displayReport.created_at, lang)}</span><span className="report-id">#{displayReport.id.slice(0, 6)}</span></p>
    </div></div>
    {hasClaim ? <>
      <section className={`report-summary summary-${verdict}`} aria-label="Concise evidence summary">
        <div className="summary-heading"><span className={`summary-verdict ${verdict}`}><VerdictIcon size={16}/>{verdict.toUpperCase()}</span></div>
        {claimText && <p className="claim-assessment-text" data-report-claim><strong>Claim:</strong> {brief(claimText, 220)}</p>}
        <div className="report-overall-explanation"><h4>Overall explanation</h4><p className="summary-body">{overall}</p>
          {general && <><p className="report-general-explanation">{general}</p><p className="report-knowledge-note">This explanation uses general knowledge; the retrieved sources have not independently confirmed it.</p></>}
        </div>
        {reason && <div className="report-verdict-reason"><h4>Why this verdict</h4><p>{reason}</p></div>}
        {points.length > 0 && <div className="report-key-evidence"><h4>Key evidence</h4><ul>{points.map((point, index) => <li key={index}>{narrative(point.text, 300)}</li>)}</ul></div>}
        {needed && <p className="report-next-step"><strong>Needed:</strong> {needed}</p>}
        {sources.length > 0 && <div className="summary-citations" aria-label="Evidence sources">{sources.map(source =>
          <a key={source.id} data-source-id={source.id} href={sourceLink(source.url)} target="_blank" rel="noopener noreferrer">[{source.id}] {brief(source.publisher || source.title || new URL(source.url).hostname, 70)}</a>)}</div>}
        {followups.length > 0 && <section className="report-followups" aria-label="Check this claim yourself">
          <h4>Check this claim yourself</h4>
          <ul className="followup-sources">{followups.map(source => <li key={sourceLink(source.url)}>
            <a className="followup-link" href={sourceLink(source.url)} target="_blank" rel="noopener noreferrer">
              <span>{brief(source.name, 80)}</span><ExternalLink size={16} aria-hidden="true"/>
            </a>
            <p>{brief(source.description, 170)}</p>
          </li>)}</ul>
        </section>}
      </section>
    </> : <MediaResult result={result} kind={displayReport.type} t={t}/>}
    <ModelMetrics result={result} kind={displayReport.type} t={t}/>
    <div className="report-bottom"><a className="button primary" href={`/api/reports/${encodeURIComponent(displayReport.id)}/pdf`} download={`verification-report-${displayReport.id.slice(0,10)}.pdf`}><Download size={17}/>{t.download}</a></div>
  </article>;
}
