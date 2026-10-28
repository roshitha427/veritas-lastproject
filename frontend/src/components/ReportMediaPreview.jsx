import React, { useState } from 'react';

function ownedMediaUrl(report, value, suffix) {
 if(typeof value!=='string'||!report?.id)return null;
 try{
  const url=new URL(value,location.origin);
  const expected=`/api/reports/${encodeURIComponent(report.id)}/${suffix}`;
  return url.origin===location.origin&&url.pathname===expected&&!url.search&&!url.hash?expected:null;
 }catch{return null;}
}

export function reportPreviewUrl(report) {
 if(report?.type==='image')return ownedMediaUrl(report,report.image_preview?.url,'preview');
 if(report?.type==='video')return ownedMediaUrl(report,report.video_preview?.url,'video');
 return null;
}

export function mediaReportTitle(report,t) {
 const title=String(report?.title||'');
 const fileName=/\.(?:jpe?g|png|webp|mp4|mov|webm)$/i.test(title)||/^[a-z]:[\\/]/i.test(title);
 if(fileName&&report?.type==='image')return t.savedImageAnalysis||'Image analysis';
 if(fileName&&report?.type==='video')return t.savedVideoAnalysis||'Video analysis';
 return title||t[report?.type]||t.open;
}

export default function ReportMediaPreview({report,t,thumbnail=false,large=false,fallback=null}) {
 const [failed,setFailed]=useState(null);
 const url=reportPreviewUrl(report);
 if(!url)return fallback;
 const video=report.type==='video';
 const poster=video?ownedMediaUrl(report,report.video_preview?.poster_url,'poster'):null;
 if(failed===url)return video&&!thumbnail?<p className="media-preview-unavailable"><a href={url} download>{t.downloadUploadedVideo||'Open or download uploaded video'}</a></p>:fallback;
 const Container=thumbnail?'span':'figure';
 const label=`${video?(t.uploadedVideoPreview||'Uploaded video preview'):(t.uploadedImagePreview||'Uploaded image preview')}: ${report.title||t[report.type]}`;
 return <Container className={thumbnail?`report-preview-thumbnail${large?' report-preview-large':''}`:'report-media-preview'}>
  {video&&(!thumbnail||!poster||failed===poster)
   ? <video src={url} poster={poster&&failed!==poster?poster:undefined} aria-label={label} controls={!thumbnail} muted={thumbnail} preload="metadata" playsInline onError={()=>setFailed(url)}/>
   : <img src={video?poster:url} alt={label} loading={thumbnail?'lazy':'eager'} decoding="async" onError={()=>setFailed(video?poster:url)}/>}
  {thumbnail&&video&&<span className="report-preview-play" aria-hidden="true">▶</span>}
  {!thumbnail&&<figcaption>{video?(t.privateVideoPreview||'Private uploaded video'):(t.privateImagePreview||'Private image preview')}</figcaption>}
 </Container>;
}
