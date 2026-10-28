"""Keyless retrieval from fixed public publisher feeds; never auto-verifies truth."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from email.utils import parsedate_to_datetime
from html import unescape
import re
import threading
import time
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import requests
from backend.store import now

FEEDS = [
    ('BBC World', 'bbc.com', 'https://feeds.bbci.co.uk/news/world/rss.xml'),
    ('Alt News', 'altnews.in', 'https://www.altnews.in/feed/'),
    ('Factly', 'factly.in', 'https://factly.in/feed/'),
]
_cache = {}
_lock = threading.Lock()


def tokens(text):
    from backend.verification import STOPWORDS
    return {x for x in re.findall(r'[a-zA-Z][a-zA-Z0-9]{2,}', text.lower()) if x not in STOPWORDS}


def read_feed(feed):
    name, domain, url = feed
    try:
        with requests.get(url, timeout=(4, 8), stream=True, headers={'User-Agent':'Veritas/1.0 (public RSS reader)'}) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > 2 * 1024 * 1024:
                    raise ValueError('Feed exceeds supported size')
                chunks.append(chunk)
        root = ET.fromstring(b''.join(chunks))
        records = []
        for node in root.findall('./channel/item')[:60]:
            title = unescape(re.sub(r'<[^>]+>', '', node.findtext('title', '')))[:500]
            summary = unescape(re.sub(r'<[^>]+>', '', node.findtext('description', '')))[:1200]
            published = None
            try:
                published = parsedate_to_datetime(node.findtext('pubDate', '')).astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError, AttributeError):
                pass
            records.append({'kind':'related_reporting', 'title':title, 'url':node.findtext('link', '').strip(),
                'publisher':name, 'claim':'', 'rating':'', 'reviewed_at':published,
                'snippet':re.sub(r'\s+', ' ', summary).strip()[:700],
                'relationship':'Related publisher content, not an automatic verification verdict.',
                '_keywords':tokens(title + ' ' + summary), '_title_keywords':tokens(title)})
        return records
    except (requests.RequestException, ValueError, ET.ParseError):
        return None


def verify_rss(text, sources):
    from backend.verification import trusted_url
    result = {'status':'unverified', 'provider':'Public publisher feeds', 'provider_status':'unavailable',
        'checked_at':now(), 'evidence':[], 'evidence_type':'related_reporting',
        'explanation':'Public publisher feeds could not be reached. The claim remains unverified.'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search recent publisher content.')
        return result
    query = tokens(text)
    if len(query) < 2:
        result.update(provider_status='query_needed', explanation='Add at least two English keywords to search the currently connected English publisher feeds.')
        return result
    domains = {s['domain'] for s in sources if s.get('enabled', True)}
    feeds = [feed for feed in FEEDS if feed[1] in domains]
    if not feeds:
        result.update(provider_status='no_sources', explanation='Enable a supported public-feed source such as BBC or Alt News.')
        return result
    key = tuple(feed[1] for feed in feeds)
    # One refresher per process; ten-minute cache avoids unnecessary publisher requests.
    if not _lock.acquire(blocking=False):
        result.update(provider_status='busy', explanation='Public feeds are refreshing. Please try again shortly.')
        return result
    try:
        cached = _cache.get(key)
        hit = bool(cached and cached['expires'] > time.monotonic())
        if not hit:
            with ThreadPoolExecutor(max_workers=3) as executor:
                responses = list(executor.map(read_feed, feeds))
            available = [r for r in responses if r is not None]
            if not available:
                return result
            cached = {'records':[record for group in available for record in group], 'feed_count':len(available),
                      'fetched_at':now(), 'expires':time.monotonic() + 600}
            _cache[key] = cached
        result.update(provider_status='searched', cached=hit, fetched_at=cached['fetched_at'],
                      feed_count=cached['feed_count'], indexed_articles=len(cached['records']))
        ranked = []
        for record in cached['records']:
            if not trusted_url(record['url'], sources):
                continue
            overlap = query & record['_keywords']
            title_overlap = query & record['_title_keywords']
            if len(overlap) >= 2 and title_overlap:
                score = 2 * len(title_overlap) + len(overlap) / max(1, len(query))
                ranked.append((score, record))
        urls = set()
        for _, record in sorted(ranked, key=lambda pair:pair[0], reverse=True):
            if record['url'] not in urls:
                urls.add(record['url'])
                result['evidence'].append({k:v for k,v in record.items() if not k.startswith('_')})
            if len(result['evidence']) >= 8:
                break
        result.update(status='needs_review' if result['evidence'] else 'unverified',
            explanation='Related content was found in recent publisher feeds. Read the original article and compare its exact claim, date, and context. No automatic fact-check verdict is supplied.'
            if result['evidence'] else 'No matching content was found in the recent publisher feeds. Feed coverage is limited; this does not establish whether the claim is true or false.')
        return result
    finally:
        _lock.release()
