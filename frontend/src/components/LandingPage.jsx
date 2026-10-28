import React, { useEffect, useState } from 'react';
import {
  ArrowRight, ArrowUpRight, BookOpenCheck, Check, ChevronLeft, ChevronRight,
  FileText, Fingerprint, Image as ImageIcon, LockKeyhole, Play, ScanLine,
  ShieldCheck, Sparkles, Video,
} from 'lucide-react';
import editorialPhoto from '../assets/news-verification-editorial.png';
import textReviewPhoto from '../assets/feature-text-review.png';
import imageReviewPhoto from '../assets/feature-image-review.png';
import videoReviewPhoto from '../assets/feature-video-review.png';

const featureItems = [
  { icon: FileText, image: textReviewPhoto, alt: 'Hands reviewing a newspaper article beside a laptop', title: 'Check a claim or article', body: 'Review text signals and see where an automated assessment has limits.' },
  { icon: ImageIcon, image: imageReviewPhoto, alt: 'A person comparing a photo on a phone and computer screen', title: 'Look closer at images', body: 'Inspect image signals alongside the claim and context you provide.' },
  { icon: Video, image: videoReviewPhoto, alt: 'A laptop displaying a video timeline during a review', title: 'Review video frames', body: 'Examine sampled frames and available speech observations in one report.' },
  { icon: BookOpenCheck, title: 'Keep evidence in view', body: 'Compare model observations with related sources when evidence is available.' },
];

const heroSlides = [
  { image: editorialPhoto, alt: 'A person comparing a newspaper, phone, and laptop while reviewing a story', badge: 'See the whole story' },
  { image: textReviewPhoto, alt: 'Hands reviewing a newspaper article beside a laptop', badge: 'Read beyond the headline' },
  { image: imageReviewPhoto, alt: 'A person comparing an image on a phone and computer screen', badge: 'Look closer at the image' },
  { image: videoReviewPhoto, alt: 'A laptop displaying a video timeline during a review', badge: 'Review the video context' },
];

function scrollToLandingSection(id) {
  return event => {
    event.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
}

function LandingMark() {
  return <a className="landing-brand" href="#top" onClick={scrollToLandingSection('top')} aria-label="AI-Powered Fake News Detection and Verification System home">
    <span className="landing-brand-icon"><ShieldCheck size={23} strokeWidth={1.8}/></span>
    <span><strong>AI-Powered Fake News</strong><small>Detection and Verification System</small></span>
  </a>;
}

function LandingFooter({ onLogin }) {
  return <footer className="landing-footer">
    <div className="landing-container landing-footer-main">
      <div className="landing-footer-about"><LandingMark/><p>Explore information carefully with AI signals, source context, and a clear view of each report’s limits.</p></div>
      <div className="landing-footer-column"><strong>Explore</strong><a href="#about">About</a><a href="#how-it-works">How it works</a><a href="#features">Features</a></div>
      <div className="landing-footer-column"><strong>Workspace</strong><button onClick={() => onLogin('login')}>Sign in</button><button onClick={() => onLogin('register')}>Create an account</button></div>
      <div className="landing-footer-note"><span className="landing-footer-icon"><Fingerprint size={20}/></span><div><strong>Use AI thoughtfully</strong><p>Model signals are clues. Review the evidence and context before making a decision or sharing a claim.</p></div></div>
    </div>
    <div className="landing-container landing-footer-bottom"><span>© {new Date().getFullYear()} AI-Powered Fake News Detection and Verification System</span><a href="#top" onClick={scrollToLandingSection('top')}>Back to top <ArrowUpRight size={14}/></a></div>
  </footer>;
}

export default function LandingPage({ onLogin }) {
  const [activeSlide, setActiveSlide] = useState(0);
  const [pauseSlides, setPauseSlides] = useState(false);
  useEffect(() => {
    if (pauseSlides || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined;
    const timer = window.setInterval(() => setActiveSlide(current => (current + 1) % heroSlides.length), 4300);
    return () => window.clearInterval(timer);
  }, [pauseSlides]);
  const changeSlide = offset => setActiveSlide(current => (current + offset + heroSlides.length) % heroSlides.length);
  const slide = heroSlides[activeSlide];

  return <div className="landing-app" id="top">
    <header className="landing-header">
      <div className="landing-container landing-header-inner">
        <LandingMark/>
        <nav className="landing-nav" aria-label="Main navigation"><a href="#about" onClick={scrollToLandingSection('about')}>About</a><a href="#how-it-works" onClick={scrollToLandingSection('how-it-works')}>How it works</a><a href="#features" onClick={scrollToLandingSection('features')}>Features</a></nav>
        <div className="landing-header-actions"><button className="landing-signin" onClick={() => onLogin('login')}>Sign in</button></div>
      </div>
    </header>

    <main className="landing-main">
      <section className="landing-container landing-hero" aria-labelledby="landing-title">
        <div className="landing-hero-copy">
          <span className="landing-kicker"><span/>A calmer way to check what you see</span>
          <h1 id="landing-title">Pause before sharing.<br/><em>See more of the story.</em></h1>
          <p>Explore text, image, and video signals with source context in one private verification workspace. Review the evidence and limitations before you decide what to share.</p>
          <div className="landing-hero-actions"><button className="landing-start landing-start-large" onClick={() => onLogin('register')}>Get started <ArrowRight size={17}/></button><a className="landing-explore" href="#how-it-works" onClick={scrollToLandingSection('how-it-works')}><span><Play size={12} fill="currentColor"/></span>See how it works</a></div>
          <div className="landing-assurances"><span><LockKeyhole size={15}/>Private saved reports</span><span><ShieldCheck size={15}/>Evidence stays visible</span></div>
        </div>
        <div className="landing-visual" role="group" aria-roledescription="carousel" aria-label="Verification examples" onPointerEnter={() => setPauseSlides(true)} onPointerLeave={() => setPauseSlides(false)} onFocus={() => setPauseSlides(true)} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setPauseSlides(false); }}>
          <div className="landing-visual-glow"/>
          <img key={slide.image} className="landing-editorial-photo" src={slide.image} alt={slide.alt} fetchPriority={activeSlide === 0 ? 'high' : undefined}/>
          <span className="landing-photo-badge"><ShieldCheck size={16}/>{slide.badge}</span>
          <div className="landing-signal-card"><span className="landing-signal-icon"><Sparkles size={17}/></span><span><small>REVIEW TOGETHER</small><strong>Signals + evidence</strong></span><Check size={17}/></div>
          <span className="landing-float landing-float-one"><FileText size={15}/> Text</span><span className="landing-float landing-float-two"><ImageIcon size={15}/> Image</span><span className="landing-float landing-float-three"><Video size={15}/> Video</span>
          <div className="landing-carousel-controls" aria-label="Choose a verification example"><button type="button" aria-label="Previous image" onClick={() => changeSlide(-1)}><ChevronLeft size={16}/></button><span>{heroSlides.map((item,index)=><button type="button" key={item.badge} aria-label={`Show image ${index + 1}: ${item.badge}`} aria-pressed={activeSlide === index} className={activeSlide === index ? 'active' : ''} onClick={() => setActiveSlide(index)}/>)}</span><button type="button" aria-label="Next image" onClick={() => changeSlide(1)}><ChevronRight size={16}/></button></div>
        </div>
      </section>

      <div className="landing-proof-strip"><div className="landing-container"><span>One workspace</span><i/><span>Three content types</span><i/><span>Evidence and limits together</span></div></div>

      <section className="landing-container landing-about" id="about">
        <div className="landing-section-label">01 <span>ABOUT THE PROJECT</span></div>
        <div className="landing-about-layout">
          <div className="landing-about-copy"><h2>Technology can help us ask <em>better questions.</em></h2><p>News travels quickly, while the context behind a claim can take longer to find. This workspace brings together machine learning signals, related reporting, and the original material so you can make a more informed review.</p>
            <ul className="landing-about-points"><li><span><FileText size={16}/></span><div><strong>Keep the original in view</strong><small>Review the submitted claim, image, or video alongside its results.</small></div></li><li><span><ScanLine size={16}/></span><div><strong>Understand what the model sees</strong><small>Explore signals and limits without treating a prediction as proof.</small></div></li><li><span><BookOpenCheck size={16}/></span><div><strong>Follow the available context</strong><small>Open related sources and articles to check dates, details, and framing.</small></div></li></ul>
            <button className="landing-text-link" onClick={() => onLogin('register')}>Open your workspace <ArrowRight size={16}/></button>
          </div>
          <div className="landing-about-gallery" aria-label="Examples of content reviewed in the workspace">
            <figure className="landing-about-photo landing-about-photo-main"><img src={textReviewPhoto} alt="A reader comparing an article with information on a laptop" loading="lazy"/><figcaption><small>TEXT & CLAIMS</small><strong>Read beyond the headline</strong></figcaption></figure>
            <figure className="landing-about-photo"><img src={imageReviewPhoto} alt="A person reviewing an image on a phone and computer" loading="lazy"/><figcaption><small>IMAGE CONTEXT</small><strong>Check the visual details</strong></figcaption></figure>
            <figure className="landing-about-photo"><img src={videoReviewPhoto} alt="Video footage open for review on a laptop" loading="lazy"/><figcaption><small>VIDEO FRAMES</small><strong>See more of the clip</strong></figcaption></figure>
          </div>
        </div>
        <div className="landing-about-note"><ShieldCheck size={20}/><span><strong>Built for careful review</strong><small>A model prediction does not establish whether a claim is true. Check the source, date, and context.</small></span><ArrowUpRight size={17}/></div>
      </section>

      <section className="landing-how" id="how-it-works">
        <div className="landing-container">
          <div className="landing-section-heading"><div><div className="landing-section-label">02 <span>HOW IT WORKS</span></div><h2>From a question to a clearer view.</h2></div><p>Each step keeps your original content, model observations, and available evidence easy to review.</p></div>
          <div className="landing-steps">
            {[[FileText,'Choose what to check','Paste a claim or select an image or video.',textReviewPhoto,'Example · a claim and article'],[ScanLine,'Review the signals','See the model assessment and what it can evaluate.',imageReviewPhoto,'Example · image details'],[BookOpenCheck,'Check the context','Read available evidence and return to your saved report.',videoReviewPhoto,'Example · frame and source review']].map(([Icon,title,body,image,example],index)=><article className="landing-step" key={title}><div className="landing-step-top"><span>0{index+1}</span><Icon size={22}/></div><img className="landing-step-image" src={image} alt={example} loading="lazy"/><span className="landing-step-example">{example}</span><h3>{title}</h3><p>{body}</p>{index<2&&<ArrowRight className="landing-step-arrow" size={18}/>}</article>)}
          </div>
        </div>
      </section>

      <section className="landing-container landing-features" id="features">
        <div className="landing-section-heading"><div><div className="landing-section-label">03 <span>FEATURES</span></div><h2>Useful detail, all in one place.</h2></div><p>Designed to make each check understandable, not just produce a label.</p></div>
        <div className="landing-feature-grid">{featureItems.map(({icon:Icon,image,alt,title,body},index)=><article className={`landing-feature landing-feature-${index+1}`} key={title}>
          {image ? <img className="landing-feature-image" src={image} alt={alt} loading="lazy"/> : <div className="landing-feature-evidence-art" aria-hidden="true"><span className="landing-feature-evidence-mark"><BookOpenCheck size={21}/></span><span className="landing-feature-evidence-lines"><i/><i/><i/></span><span className="landing-feature-evidence-tag">SOURCE NOTES</span></div>}
          <span className="landing-feature-icon"><Icon size={20}/></span><span className="landing-feature-index">0{index+1}</span><h3>{title}</h3><p>{body}</p>
        </article>)}</div>
      </section>

      <section className="landing-container landing-cta"><div className="landing-cta-art"><span/><ShieldCheck size={35}/><i/></div><div><span className="landing-section-label">YOUR NEXT CHECK STARTS HERE</span><h2>Make room for a closer look.</h2><p>Create an account to start checking content and keep your reports together.</p></div><button className="landing-start landing-start-large" onClick={() => onLogin('register')}>Get started <ArrowRight size={17}/></button></section>
    </main>
    <LandingFooter onLogin={onLogin}/>
  </div>;
}
