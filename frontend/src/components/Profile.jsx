import React, { useState } from 'react';
import { ArrowRight, LoaderCircle, LockKeyhole } from 'lucide-react';
import { api } from '../lib/api.js';
import Settings from './Settings.jsx';

export default function Profile({ t, user, onUpdate, lang, setLang }) {
 const [name,setName]=useState(user.name),[email,setEmail]=useState(user.email);
 const [currentPassword,setCurrentPassword]=useState(''),[newPassword,setNewPassword]=useState('');
 const [confirmPassword,setConfirmPassword]=useState(''),[busy,setBusy]=useState(false);
 const [error,setError]=useState(''),[success,setSuccess]=useState('');

 async function submit(event) {
  event.preventDefault();
  setError('');
  setSuccess('');
  if(newPassword!==confirmPassword){setError(t.passwordMismatch);return}
  const changedEmail=email.trim().toLowerCase()!==user.email.toLowerCase();
  if((changedEmail||newPassword)&&!currentPassword){setError(t.currentPasswordRequired);return}
  if(name.trim()===user.name&& !changedEmail&&!newPassword){setError(t.noProfileChanges);return}

  const body={name:name.trim(),email:email.trim()};
  if(changedEmail||newPassword)body.current_password=currentPassword;
  if(newPassword){body.password=newPassword;body.confirm_password=confirmPassword}
  setBusy(true);
  try{
   const result=await api('/profile',{method:'PATCH',body:JSON.stringify(body)});
   onUpdate(result.user);
   setName(result.user.name);setEmail(result.user.email);
   setCurrentPassword('');setNewPassword('');setConfirmPassword('');
   setSuccess(t.profileSaved);
  }catch(err){setError(err.message||t.error)}
  finally{setBusy(false)}
 }

 return <section className="wrap section app-section profile-section">
  <div className="page-heading"><span className="eyebrow">{t.profile}</span><h1>{t.profileTitle}</h1><p>{t.profileBody}</p></div>
  <form className="profile-card" onSubmit={submit}>
   <div className="profile-fields">
    <label>{t.name}<input autoComplete="name" value={name} onChange={event=>setName(event.target.value)} minLength={2} maxLength={80} required/></label>
    <label>{t.email}<input type="email" autoComplete="email" autoCapitalize="none" spellCheck={false} value={email} onChange={event=>setEmail(event.target.value)} maxLength={254} required/></label>
   </div>
   <div className="profile-password-heading"><LockKeyhole size={18}/><strong>{t.changePassword}</strong><span>{t.passwordHelp}</span></div>
   <div className="profile-fields profile-password-fields">
    <label>{t.currentPassword}<input type="password" autoComplete="current-password" value={currentPassword} onChange={event=>setCurrentPassword(event.target.value)} maxLength={128}/></label>
    <label>{t.newPassword}<input type="password" autoComplete="new-password" value={newPassword} onChange={event=>setNewPassword(event.target.value)} minLength={12} maxLength={128}/></label>
    <label>{t.confirmNewPassword}<input type="password" autoComplete="new-password" value={confirmPassword} onChange={event=>setConfirmPassword(event.target.value)} minLength={12} maxLength={128}/></label>
   </div>
   <p className="profile-security-note">{t.profileSecurityNote}</p>
   {error&&<div className="inline-error" role="alert">{error}</div>}
   {success&&<div className="auth-success" role="status">{success}</div>}
   <button disabled={busy} className="button primary profile-submit">{busy?<LoaderCircle size={18} className="spin"/>:t.save}<ArrowRight size={17}/></button>
  </form>
  <Settings t={t} lang={lang} setLang={setLang}/>
 </section>;
}
