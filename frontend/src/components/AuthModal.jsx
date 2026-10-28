import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowRight, LoaderCircle, Sparkles, X } from 'lucide-react';
import { api } from '../lib/api.js';
import { Brand } from './ui.jsx';

export default function AuthModal({t,onClose,onSuccess,notify,resetToken='',onResetComplete,initialMode='login'}){
 const [mode,setMode]=useState(resetToken?'reset':initialMode);
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[success,setSuccess]=useState('');
 const ref=useRef(null);

 useEffect(()=>{ref.current?.showModal();return()=>ref.current?.close()},[]);
 useEffect(()=>{setMode(resetToken?'reset':initialMode);setError('');setSuccess('')},[resetToken,initialMode]);

 async function submit(e){
  e.preventDefault();
  const values=Object.fromEntries(new FormData(e.currentTarget));
  if(mode==='reset'&&values.password!==values.confirmPassword){setError(t.authPasswordMismatch);return}
  setBusy(true);setError('');setSuccess('');
  try{
   if(mode==='forgot'){
    await api('/auth/forgot-password',{method:'POST',body:JSON.stringify({email:values.email})});
    setSuccess(t.authResetSent);
   }else if(mode==='reset'){
    await api('/auth/reset-password',{method:'POST',body:JSON.stringify({token:resetToken,password:values.password})});
    notify(t.authResetDone);
    onResetComplete?.();
   }else{
    const result=await api(`/auth/${mode==='register'?'register':'login'}`,{method:'POST',body:JSON.stringify(values)});
    onSuccess(result.user);onClose();
   }
  }catch(err){
   if(mode==='reset'&&[400,401,410].includes(err.status))setError(t.authResetInvalid);
   else if((mode==='forgot'||mode==='reset')&&[404,500,502,503].includes(err.status))setError(t.authResetUnavailable);
   else setError(err.message||t.error);
  }finally{setBusy(false)}
 }

 const title=mode==='register'?t.welcomeNew:mode==='forgot'?t.authForgotTitle:mode==='reset'?t.authResetTitle:t.welcome;
 const body=mode==='forgot'?t.authForgotBody:mode==='reset'?t.authResetBody:t.authBody;
 const submitText=mode==='register'?t.signup:mode==='forgot'?t.authSendLink:mode==='reset'?t.authUpdatePassword:t.login;

 return <dialog ref={ref} className="auth-modal" onCancel={onClose} onClick={e=>{if(e.target===ref.current)onClose()}}>
  <div className="auth-inner">
   <button type="button" className="round-button modal-close" aria-label={t.close} onClick={onClose}><X size={19}/></button>
   <Brand small/><span className="auth-spark" aria-hidden="true">✳</span>
   <h2>{title}</h2><p>{body}</p>
   <form id={`${mode}-form`} onSubmit={submit} autoComplete="on">
    {mode==='register'&&<label>{t.name}<input autoComplete="name" name="name" required minLength={2} maxLength={80}/></label>}
    {mode!=='reset'&&<label>{t.email}<input autoComplete={mode==='login'?'username':'email'} autoCapitalize="none" spellCheck={false} type="email" name="email" required maxLength={254} autoFocus={mode!=='reset'}/></label>}
    {(mode==='login'||mode==='register'||mode==='reset')&&<label>{mode==='reset'?t.authNewPassword:t.password}<input autoComplete={mode==='login'?'current-password':'new-password'} type="password" name="password" required minLength={mode==='register'||mode==='reset'?12:1} maxLength={128} autoFocus={mode==='reset'}/></label>}
    {mode==='reset'&&<label>{t.authConfirmPassword}<input autoComplete="new-password" type="password" name="confirmPassword" required minLength={12} maxLength={128}/></label>}
    {mode==='register'&&<span className="small muted">{t.passwordHelp}</span>}
    {error&&<div className="inline-error" role="alert"><AlertTriangle size={16}/>{error}</div>}
    {success&&<div className="auth-success" role="status">{success}</div>}
    <button disabled={busy} className="button primary auth-submit">{busy?<LoaderCircle size={18} className="spin"/>:submitText}<ArrowRight size={17}/></button>
   </form>
   <div className="auth-switch">
    {mode==='register'?<>{t.haveAccount}<button type="button" onClick={()=>{setMode('login');setError('');setSuccess('')}}>{t.login}</button></>:mode==='login'?<>{t.noAccount}<button type="button" onClick={()=>{setMode('register');setError('');setSuccess('')}}>{t.signup}</button></>:<button type="button" onClick={()=>{setMode('login');setError('');setSuccess('')}}>{t.authBackLogin}</button>}
   </div>
  </div>
 </dialog>
}

