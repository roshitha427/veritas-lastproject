"""Shared query normalization and rules for claims that need live information."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re


def normalize_places(text):
    text = str(text or '')
    # Correct only this unambiguous office-title typo for search and matching.
    # Never rewrite the alleged holder's name or supply an office-holder answer.
    text = re.sub(r'\bcheif\s+minister\b', 'chief minister', text, flags=re.I)
    if re.search(r'\b(government|chief minister|cm|assembly|governor)\b', text, re.I):
        text = re.sub(r'\bAP\b', 'andhra pradesh', text, flags=re.I)
    for compact, spaced in [('tamilnadu', 'tamil nadu'), ('andhrapradesh', 'andhra pradesh'),
                            ('uttarpradesh', 'uttar pradesh'), ('madhyapradesh', 'madhya pradesh')]:
        text = re.sub(r'\b' + compact + r'\b', spaced, text, flags=re.I)
    return text


def is_death_claim(text):
    return bool(re.search(r'\b(dead|died|death|alive|passed away|killed|assassinated)\b', str(text), re.I))


def needs_current_sources(text):
    text = normalize_places(text)
    if re.search(r'\b(current|currently|today|yesterday|tomorrow|now|latest|recently|this week|this month)\b', text, re.I):
        return True
    historical = bool(re.search(r'\b(?:in|on|during|before|since)\b[^.!?]{0,25}\b(?:18|19|20)\d{2}\b|\b(former|was|were)\b', text, re.I))
    if historical:
        return False
    if re.search(r'\b(?:school|schools|college|colleges)\b', text, re.I) and re.search(
            r'\b(?:immediate(?:ly)?|until further notice|with immediate effect)\b', text, re.I) and re.search(
            r'\b(?:clos(?:e|ed|ing|ure|ures)|shut|suspended|suspension)\b', text, re.I):
        return True
    return is_death_claim(text) or bool(re.search(
        r'\b(chief minister|prime minister|president|governor|ceo|cm|pm)\b', text, re.I))


def _age_seconds(value):
    if not value:
        return None
    try:
        try:
            stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            stamp = parsedate_to_datetime(str(value))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    except (ValueError, TypeError, OverflowError):
        return None


def source_can_settle_current_claim(claim, source):
    if source.get('retrieval_status') == 'metadata_only':
        return False
    if not needs_current_sources(claim):
        return True
    if is_death_claim(claim):
        # A directory listing or an old interview cannot settle a new death rumour.
        if source.get('source_role') == 'office_directory':
            return False
        text = ' '.join(str(source.get(k) or '') for k in ('claim', 'snippet', 'matched_excerpt'))
        if not is_death_claim(text):
            return False
        age = _age_seconds(source.get('reviewed_at') or source.get('published_at'))
        return age is not None and 0 <= age <= 7 * 86400
    age = _age_seconds(source.get('reviewed_at') or source.get('published_at'))
    if age is not None:
        return 0 <= age <= 90 * 86400
    # A fresh fetch of an original page is different from an imported snapshot.
    age = _age_seconds(source.get('retrieved_at'))
    return (source.get('retrieval_status') == 'article_excerpt_retrieved'
            and age is not None and 0 <= age <= 86400)


def current_information_notice():
    return {'status': 'needs_current_sources', 'verdict': 'unverified', 'answer': '',
        'reason': 'Stored model knowledge cannot confirm current office holders, deaths, or other recent changes.',
        'basis': 'Current source verification is required.',
        'user_message': 'This claim needs up-to-date evidence. We could not confirm it from the retrieved sources. '
                        'Please check a current official announcement or reliable recent report; older model knowledge may be outdated.'}
