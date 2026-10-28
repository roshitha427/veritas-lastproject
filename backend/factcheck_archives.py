"""Public, read-only fact-check publisher archive search. No API keys or billing."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
import re
import threading
import time
from backend.public_evidence import verify_rss, tokens
from backend.store import now
import requests

ARCHIVES = [
    ('Alt News', 'altnews.in', 'https://www.altnews.in/wp-json/wp/v2/posts'),
    ('Factly', 'factly.in', 'https://factly.in/wp-json/wp/v2/posts'),
]
_cache = {}
_lock = threading.Lock()


def query_keywords(text):
    from collections import Counter
    from backend.verification import STOPWORDS
    words = re.findall(r'[A-Za-z][A-Za-z0-9]{2,}', text[:2500].lower())
    if not words:
        # Public WordPress search also supports original-script strings.
        return ' '.join(text.split()[:5])[:150]
    ignored = STOPWORDS | set('please breaking update watch video image photo report reports reuters bbc monday tuesday wednesday thursday friday saturday sunday january february march april june july august september october november december'.split())
    counts = Counter(word for word in words if word not in ignored)
    names = list(dict.fromkeys(word.lower() for word in re.findall(r'\b[A-Z][A-Za-z0-9]{2,}\b', text[:2500]) if word.lower() not in ignored))
    terms = sorted(names, key=lambda term:-counts[term])[:2]
    for term, _ in counts.most_common():
        if term not in terms:
            terms.append(term)
        if len(terms) == 3:
            break
    return ' '.join(terms)


def search_archive(archive, query):
    name, domain, endpoint = archive
    try:
        response = requests.get(endpoint, params={'search':query, 'per_page':6,
            '_fields':'link,title,date_gmt,excerpt', 'orderby':'relevance'}, timeout=(4, 8))
        response.raise_for_status()
        if len(response.content) > 2 * 1024 * 1024:
            raise ValueError('Oversized publisher response')
        articles = response.json()
        if not isinstance(articles, list):
            raise ValueError('Unexpected publisher response')
        result = []
        for article in articles[:6]:
            title = unescape(re.sub(r'<[^>]+>', '', article.get('title', {}).get('rendered', '')))[:500]
            excerpt = unescape(re.sub(r'<[^>]+>', '', article.get('excerpt', {}).get('rendered', '')))[:1500]
            published = None
            try:
                published = datetime.fromisoformat(article.get('date_gmt', '')).replace(tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                pass
            result.append({'kind':'fact_check_article', 'title':title, 'url':article.get('link', ''),
                'publisher':name, 'claim':'', 'rating':'', 'reviewed_at':published,
                'snippet':excerpt[:700],
                'relationship':'Publisher fact-check search match. Compare the exact claim and read its original verdict.',
                '_keywords':tokens(title + ' ' + excerpt), '_title_keywords':tokens(title)})
        return result
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None


def verify_public(text, sources):
    from backend.verification import trusted_url
    result = {'status':'unverified', 'provider':'Public fact-check search', 'provider_status':'unavailable',
        'checked_at':now(), 'evidence':[], 'providers':[], 'evidence_type':'publisher_search',
        'explanation':'Public verification sources could not be reached. The claim remains unverified.'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search the public fact-check archives.')
        return result
    query = query_keywords(text)
    if not query.strip():
        result.update(provider_status='query_needed', explanation='Add a claim or a few meaningful search terms.')
        return result
    domains = {s['domain'] for s in sources if s.get('enabled', True)}
    archives = [archive for archive in ARCHIVES if archive[1] in domains]
    cache_key = (query, tuple(archive[1] for archive in archives))
    records = []
    if archives and _lock.acquire(blocking=False):
        try:
            cached = _cache.get(cache_key)
            if cached and cached['expires'] > time.monotonic():
                records, providers = cached['records'], cached['providers']
                result['cached'] = True
            else:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    responses = list(executor.map(lambda archive:search_archive(archive, query), archives))
                records = [record for response in responses if response is not None for record in response]
                providers = [{'name':archive[0], 'status':'searched' if response is not None else 'unavailable'}
                             for archive, response in zip(archives, responses)]
                if len(_cache) > 128:
                    _cache.clear()
                _cache[cache_key] = {'records':records, 'providers':providers, 'expires':time.monotonic()+600}
            result['providers'] = providers
        finally:
            _lock.release()
    elif archives:
        result['providers'] = [{'name':archive[0], 'status':'busy'} for archive in archives]
    terms = tokens(query)
    for record in records:
        # Publisher search may return broad matches; retain lexical relevance and trusted domains.
        english_match = len(terms & record['_keywords']) >= min(2, max(1, len(terms)))
        original_script = not terms and any(term in record['title'] for term in query.split())
        if trusted_url(record['url'], sources) and (english_match or original_script):
            result['evidence'].append({key:value for key,value in record.items() if not key.startswith('_')})
    searched = any(provider['status']=='searched' for provider in result['providers'])
    if not result['evidence']:
        fallback = verify_rss(text, sources)
        result['providers'].append({'name':'Recent publisher feeds', 'status':fallback['provider_status']})
        if fallback['provider_status']=='searched':
            searched = True
            result['evidence'] = fallback['evidence']
            result['feed_fetched_at'] = fallback.get('fetched_at')
    result['evidence'] = list({item['url']:item for item in result['evidence']}.values())[:8]
    if searched:
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified',
            explanation='Related publisher fact-checks or reporting were found. Read each original article, compare the claim and context, and review its verdict. A search match is not automatic verification.'
            if result['evidence'] else 'No matching content was found in the connected public archives and recent feeds. Coverage is limited; this does not establish whether the claim is true or false.')
    return result
