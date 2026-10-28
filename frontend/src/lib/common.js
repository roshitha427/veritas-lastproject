import { FileText, Image as ImageIcon, Video } from 'lucide-react';
export const types = { text: FileText, image: ImageIcon, video: Video };
export const date = (value, lang) => new Date(value).toLocaleString({ en: 'en-IN', hi: 'hi-IN', te: 'te-IN' }[lang], { dateStyle: 'medium', timeStyle: 'short' });

export function verificationMessage(v, t) {
  if (v.provider === 'Local evidence model') {
    return t[v.provider_status === 'no_claim' ? 'verifyNoClaim' : v.provider_status === 'not_configured' ? 'verifyLocalMissing' :
      v.provider_status === 'searched' ? (v.evidence.length ? 'verifyLocalRelated' : 'verifyLocalNone') : 'verifyLocalUnavailable'];
  }
  if (v.provider === 'Public fact-check search' || v.provider === 'Public publisher feeds') {
    return t[v.provider_status === 'no_claim' ? 'verifyNoClaim' : v.provider_status === 'query_needed' ? 'verifyPublicQuery' :
      v.provider_status === 'no_sources' ? 'verifyPublicSources' : v.provider_status === 'busy' ? 'verifyBusy' :
      v.provider_status === 'searched' ? (v.evidence.length ? 'verifyPublicRelated' : 'verifyPublicNone') : 'verifyPublicUnavailable'];
  }
  if (v.provider === 'GDELT') {
    return t[v.provider_status === 'no_claim' ? 'verifyNoClaim' : v.provider_status === 'query_needed' ? 'verifyGdeltQuery' :
      v.provider_status === 'busy' ? 'verifyBusy' : v.provider_status === 'searched' ?
        (v.evidence.length ? 'verifyGdeltRelated' : 'verifyGdeltNone') : 'verifyGdeltUnavailable'];
  }
  return t[v.provider_status === 'not_configured' ? 'verifyNoKey' : v.provider_status === 'no_claim' ? 'verifyNoClaim' :
    v.provider_status === 'unavailable' ? 'verifyUnavailable' : v.evidence.length ? 'verifyRelated' : 'verifyNone'];
}

export function scopeMessage(r, t) {
  if (r.temporal_assessment?.status === 'ready') return t.scopeTemporal;
  return t[r.models.some(m => m.status === 'unsupported_language') ? 'scopeLanguage' :
    r.fake_percentage == null ? 'scopeMissing' : r.models.some(m => m.name.includes('BERT')) ? 'scopeText' : 'scopeFaceVisual'];
}

export function limitMessage(text, t) {
  const prefixes = [['AI generation', 'limitAISeparation'], ['No validated AI-writing', 'limitAIWriting'],
    ['No dedicated AI-image', 'limitAIImageMissing'], ['The pretrained AI-image', 'aiScoreNote'],
    ['The model publisher', 'limitAIGeneralization'], ['Video AI scores', 'aiFrameNote'],
    ['Speech is an automatic', 'transcriptNote'], ['The visual classifier was trained', 'scopeFaceVisual'],
    ['Model confidence', 'limitConfidence'], ['Text model not trained', 'limitTextMissing'],
    ['No trained visual', 'limitVisualMissing'], ['Video scores', 'limitVideo'], ['This text model', 'limitLanguage']];
  const key = prefixes.find(([prefix]) => text.startsWith(prefix))?.[1];
  return key ? (t[key] || text) : text;
}

export function observationLabel(label, t) {
  return t[{ Dimensions: 'dimensions', Duration: 'duration', 'Sampled frames': 'sampledFrames',
    'Faces detected (maximum per frame)': 'faceCount', 'Frame score variation': 'frameVariation' }[label]] || label;
}
