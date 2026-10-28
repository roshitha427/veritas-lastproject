import React, { useEffect, useRef, useState } from 'react';
import { CircleHelp, Database, FileText, Fingerprint, History, LayoutDashboard, LogOut, Menu, ShieldCheck, Volume2, X } from 'lucide-react';
import { translations } from './i18n.js';
import { api } from './lib/api.js';
import { verificationMessage } from './lib/common.js';
import { Brand, IconBadge } from './components/ui.jsx';
import ReferenceHome from './components/ReferenceHome.jsx';
import Dashboard from './components/Dashboard.jsx';
import Workspace from './components/Workspace.jsx';
import Reports from './components/HistoryPage.jsx';
import ReportsHub from './components/ReportsHub.jsx';
import Admin from './components/Admin.jsx';
import AuthModal from './components/AuthModal.jsx';
import Profile from './components/Profile.jsx';
import ContentTypeMenu from './components/ContentTypeMenu.jsx';
import LandingPage from './components/LandingPage.jsx';

function resetTokenFromLocation() {
  const match = location.hash.match(/^#reset(?:\/|=)([A-Za-z0-9_-]{20,})$/);
  return match?.[1] || '';
}

function initialPageFromLocation() {
  if (resetTokenFromLocation()) return 'home';
  const route = location.hash.slice(1);
  if (['about', 'how-it-works', 'features', 'top'].includes(route)) return 'home';
  if (route === 'sources') return 'reports';
  if (route === 'settings') return 'profile';
  return route || 'home';
}

function SignInPrompt({ t, onLogin }) {
  return <section className="view active"><div className="card access-card"><div className="empty-illustration"><ShieldCheck size={48}/></div><h2>{t.accountRequired || 'Your private workspace'}</h2><p>{t.signInFirst}</p><button className="primary" onClick={onLogin}>{t.login}</button></div></section>;
}

function WorkspaceFooter({ t, health, go }) {
  return <footer className="workspace-footer">
    <div className="workspace-footer-main">
      <div className="workspace-footer-purpose"><span className="workspace-footer-mark"><ShieldCheck size={19}/></span><div><strong>{t.responsibleTitle}</strong><p>{t.responsibleBody}</p></div></div>
      <nav aria-label={t.workspaceLinks}><button onClick={() => go('home')}>{t.home}</button><button onClick={() => go('dashboard')}>{t.dashboard}</button><button onClick={() => go('history')}>{t.history}</button><button onClick={() => go('reports')}>{t.reports}</button></nav>
      <span className={`workspace-footer-status ${health?.status === 'offline' || health?.status === 'degraded' ? 'offline' : ''}`}><i/>{health?.status === 'offline' ? t.apiOffline : health?.status === 'degraded' ? t.serviceAttention : health?.status === 'ok' ? t.workspaceAvailable : t.checkingService}</span>
    </div>
    <div className="workspace-footer-bottom"><span>© {new Date().getFullYear()} AI-Powered Fake News Detection and Verification System</span><span>{t.checkEvidenceBeforeSharing}</span></div>
  </footer>;
}

export default function ReferenceApp() {
  const [lang, setLang] = useState(localStorage.getItem('news-verification-language') || 'en');
  const [page, setPage] = useState(initialPageFromLocation);
  const [user, setUser] = useState(null);
  const [sessionReady, setSessionReady] = useState(false);
  const [health, setHealth] = useState(null);
  const [auth, setAuth] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [resetToken, setResetToken] = useState(() => resetTokenFromLocation());
  const [mobile, setMobile] = useState(false);
  const [toast, setToast] = useState('');
  const [kind, setKind] = useState('text');
  const [report, setReport] = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const [pendingStart, setPendingStart] = useState('');
  const toastTimer = useRef(null);
  const authRevision = useRef(0);
  const reportRequestRevision = useRef(0);
  const t = translations[lang] || translations.en;

  function notify(message) {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(''), 6500);
  }

  useEffect(() => {
    document.title = 'AI-Powered Fake News Detection and Verification System';
    document.body.classList.toggle('large-text', localStorage.getItem('veritas-large-text') === 'true');
    const sessionRevision = authRevision.current;
    api('/session').then(data => {
      if (authRevision.current !== sessionRevision) return;
      const account = data.user || null;
      setUser(account);
      if (account && (!location.hash || location.hash === '#home')) setPage('home');
    }).catch(() => {
      if (authRevision.current === sessionRevision) notify(t.offline);
    }).finally(() => setSessionReady(true));
    let healthStopped = false;
    let healthTimer;
    async function refreshHealth() {
      try {
        const data = await api('/health');
        if (!healthStopped) setHealth(data);
      } catch {
        if (!healthStopped) setHealth({ status: 'offline' });
      }
      if (!healthStopped) healthTimer = setTimeout(refreshHealth, 15000);
    }
    refreshHealth();
    const change = () => {
      reportRequestRevision.current += 1;
      const token = resetTokenFromLocation();
      setResetToken(token);
      setAuth(false);
      setPage(token ? 'home' : initialPageFromLocation());
      setMobile(false);
      document.querySelector('main')?.scrollTo({ top: 0, behavior: 'instant' });
    };
    window.addEventListener('hashchange', change);
    return () => {
      window.removeEventListener('hashchange', change);
      healthStopped = true;
      clearTimeout(healthTimer);
      clearTimeout(toastTimer.current);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem('news-verification-language', lang);
    document.documentElement.lang = lang;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [lang]);

  useEffect(() => {
    if (!report?.id || !['queued', 'analyzing'].includes(report.status)) return;
    let stopped = false;
    let timer;
    async function poll() {
      try {
        const result = await api(`/reports/${report.id}`);
        if (!stopped) {
          setReport(current => current?.id === report.id ? result : current);
          if (['queued', 'analyzing'].includes(result.status)) timer = setTimeout(poll, 1400);
        }
      } catch (error) {
        if (!stopped) {
          notify(error.message);
          timer = setTimeout(poll, 5000);
        }
      }
    }
    timer = setTimeout(poll, 700);
    return () => { stopped = true; clearTimeout(timer); };
  }, [report?.id, report?.status]);

  function go(destination) {
    reportRequestRevision.current += 1;
    setMobile(false);
    setPage(destination);
    if (location.hash !== `#${destination}`) location.hash = destination;
    document.querySelector('main')?.scrollTo({ top: 0, behavior: 'instant' });
  }

  function start(type) {
    reportRequestRevision.current += 1;
    setKind(type);
    setReport(null);
    if (!user) {
      setPendingStart(type);
      setAuthMode('login');
      setAuth(true);
      return;
    }
    go('verify');
  }

  function enterWorkspace(account) {
    reportRequestRevision.current += 1;
    authRevision.current += 1;
    setSessionReady(true);
    setUser(account);
    if (pendingStart) {
      setKind(pendingStart);
      setPendingStart('');
      go('verify');
    } else go('home');
  }

  function afterLogin(account) {
    const signedInAccount = account?.user || account;
    if (signedInAccount?.id) {
      enterWorkspace(signedInAccount);
      return;
    }
    const requestRevision = ++authRevision.current;
    api('/session').then(data => {
      if (authRevision.current === requestRevision && data.user) enterWorkspace(data.user);
      else if (!data.user) notify(t.signInFirst);
    }).catch(error => notify(error.message || t.offline));
  }

  async function openReport(id, destination = 'verify') {
    const requestRevision = ++reportRequestRevision.current;
    const accountRevision = authRevision.current;
    try {
      const result = await api(`/reports/${encodeURIComponent(id)}`);
      if (requestRevision !== reportRequestRevision.current || accountRevision !== authRevision.current) return;
      if (result.id !== id) { notify(t.error); return; }
      setReport(result);
      setKind(result.type);
      go(destination);
    } catch (error) {
      if (requestRevision === reportRequestRevision.current && accountRevision === authRevision.current) notify(error.message);
    }
  }

  async function logout() {
    try {
      await api('/auth/logout', { method: 'POST' });
      authRevision.current += 1;
      setUser(null);
      setReport(null);
      go('home');
    } catch (error) {
      notify(error.message);
    }
  }

  function closeAuth() {
    setAuth(false);
    setPendingStart('');
    if (resetToken) {
      history.replaceState(null, '', `${location.pathname}${location.search}#home`);
      setResetToken('');
    }
  }

  function openAuth(mode = 'login') {
    setAuthMode(mode);
    setAuth(true);
  }

  function completePasswordReset() {
    history.replaceState(null, '', `${location.pathname}${location.search}#home`);
    setResetToken('');
    setAuth(false);
  }

  function read() {
    if (!('speechSynthesis' in window)) {
      notify(t.readAloudUnsupported);
      return;
    }
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const verification = report?.result?.verification;
    const content = page === 'verify'
      ? `${t.verifyTitle} ${t.verifyBody} ${t.learnBody} ${verification ? `${t[verification.status]} ${verificationMessage(verification, t)}` : ''}`
      : `${t[page] || page} ${page === 'home' ? t.heroBody : ''}`;
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = { en: 'en-IN', hi: 'hi-IN', te: 'te-IN' }[lang];
    utterance.rate = 0.9;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => { setSpeaking(false); notify(t.voiceUnavailable); };
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  }

  const workspaceNav = [
    ['home', t.home, ShieldCheck],
    ['dashboard', t.dashboard, LayoutDashboard],
    ['verify', t.checkContent || t.verify, null],
    ['history', t.history, History],
    ['reports', t.reports || 'Reports', FileText],
  ];
  const accountNav = [
    ['profile', t.profile || 'Profile', Fingerprint],
    ...(user?.role === 'admin' ? [['admin', t.admin, Database]] : []),
  ];
  const privatePage = ['dashboard', 'verify', 'history', 'reports', 'profile', 'admin'].includes(page) && !user;
  let content;
  if (page === 'home') content = <ReferenceHome t={t} user={user} health={health} start={start} go={go}/>;
  else if (page === 'dashboard') content = <Dashboard t={t} lang={lang} user={user} start={start} onOpen={id => openReport(id, 'reports')}/>;
  else if (page === 'verify') content = user
    ? <div className="view active"><Workspace t={t} lang={lang} user={user} onLogin={() => openAuth('login')} kind={kind} setKind={setKind} report={report} setReport={setReport} notify={notify}/></div>
    : <SignInPrompt t={t} onLogin={() => openAuth('login')}/>;
  else if (page === 'history') content = user
    ? <div className="view active"><Reports t={t} lang={lang} user={user} onLogin={() => openAuth('login')} onOpen={id => openReport(id, 'verify')} onDelete={id => setReport(current => current?.id === id ? null : current)} notify={notify}/></div>
    : <SignInPrompt t={t} onLogin={() => openAuth('login')}/>;
  else if (page === 'reports') content = user
    ? <ReportsHub t={t} lang={lang} user={user} report={report} start={start} onOpen={id => openReport(id, 'reports')}/> 
    : <SignInPrompt t={t} onLogin={() => openAuth('login')}/>;
  else if (page === 'profile') content = user
    ? <div className="view active"><Profile t={t} user={user} onUpdate={setUser} lang={lang} setLang={setLang}/></div>
    : <SignInPrompt t={t} onLogin={() => openAuth('login')}/>;
  else if (page === 'admin') content = <div className="view active"><Admin t={t} user={user} notify={notify}/></div>;
  else content = <ReferenceHome t={t} user={user} health={health} start={start} go={go}/>;

  if (!sessionReady && !user && page === 'home' && !resetToken) return <div className="session-loader"><span className="session-loader-icon"><ShieldCheck size={25}/></span><span>{t.sessionOpening}</span></div>;
  if (!user && page === 'home') return <>
    <LandingPage onLogin={openAuth}/>
    {(auth || resetToken) && <AuthModal key={resetToken ? `reset-${resetToken}` : 'auth'} t={t} initialMode={authMode} onClose={closeAuth} onSuccess={afterLogin} notify={notify} resetToken={resetToken} onResetComplete={completePasswordReset}/>}
    <div className={`toast ${toast ? 'show' : ''}`} role="status" aria-live="polite">{toast}<button aria-label={t.close} onClick={() => setToast('')}><X size={16}/></button></div>
  </>;

  return <div className="app">
    <a className="skip-link" href="#main">{t.skipContent}</a>
    <header className="topbar">
      <button className="mobile-menu" onClick={() => setMobile(value => !value)} aria-label={t.toggleNavigation} aria-expanded={mobile}>{mobile ? <X size={19}/> : <Menu size={19}/>}</button>
      <Brand/>
      <div className="top-actions">
        <span className="top-caption">{health?.status === 'offline' ? t.apiOffline : t.verificationWorkspace}</span>
        <select className="language" value={lang} onChange={event => setLang(event.target.value)} aria-label={t.language}><option value="en">English</option><option value="te">తెలుగు</option><option value="hi">हिन्दी</option></select>
        {user ? <><button className="avatar" onClick={() => go('profile')} aria-label={t.profile}>{user.name.slice(0, 1).toUpperCase()}</button><button className="secondary small" onClick={logout}>{t.logout}</button></> : <button className="secondary small" onClick={() => openAuth('login')}>{t.login}</button>}
      </div>
    </header>
    <div className="shell">
      <aside id="sidebar" className={`sidebar ${mobile ? 'open' : ''}`}>
        <div className="nav-label">{t.workspaceGroup}</div>
        {workspaceNav.map(([key, label, Icon]) => key === 'verify'
          ? <ContentTypeMenu key={key} t={t} start={start} variant="nav" active={page === 'verify'} onChoose={type => { start(type); setMobile(false); }}/>
          : <button key={key} className={`nav-btn ${page === key ? 'active' : ''}`} onClick={() => go(key)} aria-current={page === key ? 'page' : undefined}><Icon className="ico" size={18}/><span>{label}</span></button>)}
        <div className="nav-label">{t.accountGroup}</div>
        {accountNav.map(([key, label, Icon]) => <button key={key} className={`nav-btn ${page === key ? 'active' : ''}`} onClick={() => go(key)} aria-current={page === key ? 'page' : undefined}><Icon className="ico" size={18}/><span>{label}</span></button>)}
        <div className="sidebar-bottom"><strong>{t.responsibleTitle}</strong><p>{t.responsibleBody}</p><span className="tiny-pill">{health?.status === 'offline' ? t.apiOffline : t.connectedWorkspace}</span></div>
      </aside>
      <main id="main" className={document.body.classList.contains('large-text') ? 'large-text' : ''}>{privatePage ? <SignInPrompt t={t} onLogin={() => openAuth('login')}/> : content}<WorkspaceFooter t={t} health={health} go={go}/></main>
    </div>
    {(auth || resetToken) && <AuthModal key={resetToken ? `reset-${resetToken}` : 'auth'} t={t} initialMode={authMode} onClose={closeAuth} onSuccess={afterLogin} notify={notify} resetToken={resetToken} onResetComplete={completePasswordReset}/>}
    <div className={`toast ${toast ? 'show' : ''}`} role="status" aria-live="polite">{toast}<button aria-label={t.close} onClick={() => setToast('')}><X size={16}/></button></div>
  </div>;
}
