import re


def clean_text(text):
    text=re.sub(r'<[^>]+>',' ',str(text))
    text=re.sub(r'^.{0,160}?\(Reuters\)\s*[-–—]\s*','',text,flags=re.I|re.S)
    text=re.sub(r'https?://\S+',' ',text)
    text=re.sub(r'\bReuters\b','',text,flags=re.I)
    return re.sub(r'\s+',' ',text).strip()


def content_language(text):
    if re.search(r'[\u0c00-\u0c7f]',text):
        return 'te'
    if re.search(r'[\u0900-\u097f]',text):
        return 'hi'
    if re.search(r'[\u0370-\u052f\u0590-\u06ff\u0980-\u0bff\u0c80-\u0fff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]',text):
        return 'unsupported'
    return 'en'
