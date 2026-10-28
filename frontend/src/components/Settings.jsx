import React, { useState } from 'react';
import { Languages, Type } from 'lucide-react';

export default function Settings({ t, lang, setLang }) {
 const [large,setLarge]=useState(()=>localStorage.getItem('veritas-large-text')==='true');
 function toggle(){setLarge(value=>{const next=!value;localStorage.setItem('veritas-large-text',String(next));document.body.classList.toggle('large-text',next);return next})}
 return <section className="card settings-card profile-settings" aria-labelledby="settings-title"><div className="settings-heading"><div><div className="eyebrow">{t.settingsEyebrow}</div><h2 id="settings-title">{t.settingsTitle}</h2><p>{t.settingsBody}</p></div></div><div className="mini-title"><h3>{t.accessibilityLanguage}</h3><span className="subtle">{t.savedOnDevice}</span></div><div className="signal settings-row"><div className="signal-icon"><Type size={17}/></div><div><strong>{t.largerText}</strong><small>{t.largerTextBody}</small></div><button className="secondary small" onClick={toggle}>{large?t.turnOff:t.turnOn}</button></div><div className="signal settings-row"><div className="signal-icon"><Languages size={17}/></div><div><strong>{t.language}</strong><small>{t.languageHelp}</small></div><select className="settings-language" value={lang} onChange={e=>setLang(e.target.value)} aria-label={t.language}><option value="en">English</option><option value="te">తెలుగు</option><option value="hi">हिन्दी</option></select></div></section>;
}
