import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, FileText, Image as ImageIcon, ScanLine, Video } from 'lucide-react';

const contentTypes = [
  { key: 'text', Icon: FileText, description: 'chooseTextBody' },
  { key: 'image', Icon: ImageIcon, description: 'chooseImageBody' },
  { key: 'video', Icon: Video, description: 'chooseVideoBody' },
];

export default function ContentTypeMenu({ t, start, variant = 'button', active = false, onChoose }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    function onPointerDown(event) {
      if (!root.current?.contains(event.target)) setOpen(false);
    }
    function onKeyDown(event) {
      if (event.key === 'Escape') {
        setOpen(false);
        root.current?.querySelector('button')?.focus();
      }
    }
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  return <div className={`content-picker content-picker-${variant}`} ref={root}>
    <button type="button" className={`${variant === 'nav' ? `nav-btn ${active ? 'active' : ''}` : 'primary'} content-picker-trigger`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(value => !value)}>
      <ScanLine className="ico" size={18}/><span>{t.checkContent || t.verify}</span><ChevronDown className={`content-picker-chevron ${open ? 'open' : ''}`} size={15}/>
    </button>
    {open && <div className="content-picker-menu" role="menu" aria-label={t.chooseContentType}>
      <span className="content-picker-heading">{t.chooseContentType}</span>
      {contentTypes.map(({ key, Icon, description }) => <button type="button" role="menuitem" key={key} onClick={() => { setOpen(false); (onChoose || start)(key); }}>
        <span className={`content-picker-icon content-picker-icon-${key}`}><Icon size={17}/></span>
        <span className="content-picker-option-copy"><strong>{t[key]}</strong><small>{t[description]}</small></span>
      </button>)}
    </div>}
  </div>;
}
