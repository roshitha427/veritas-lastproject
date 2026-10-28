"""Bounded web discovery: only enabled publishers' source text is evidence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import ipaddress
import re
import threading
import time
from urllib.parse import urlparse

import requests

from backend.store import now

_cache = {}
_blocked = {}
_lock = threading.Lock()


def _search_query(claim, sources, api_key=None, request_domains=None, discover_institution=False):
    from backend.verification import trusted_url, source_reliability
    result = {'provider': 'Tavily web search', 'provider_status': 'not_configured',
              'checked_at': now(), 'status': 'unverified', 'evidence': [],
              'search_query': claim[:2500], 'requested_domains': [],
              'request_attempted': False, 'domain_scope': 'not_requested'}
    if not api_key or not claim.strip():
        return result
    domains = sorted({s.get('domain', '').lower() for s in sources
                      if s.get('enabled', True) and re.fullmatch(r'[a-zA-Z0-9.-]+', s.get('domain', ''))})
    if not domains:
        return dict(result, provider_status='no_sources')
    domains = [] if discover_institution else request_domains or domains
    account = hashlib.sha256(api_key.encode()).hexdigest()
    cache_key = (account, claim[:2500], tuple(domains), discover_institution,
                 tuple(sorted((s.get('domain', ''), s.get('enabled', True)) for s in sources)))
    with _lock:
        if _blocked.get(account, 0) > time.monotonic():
            return dict(result, provider_status='quota_or_auth_unavailable')
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 300:
            return dict(deepcopy(cached[1]), cached=True)
    try:
        result.update(requested_domains=domains[:300], request_attempted=True,
                      domain_scope='public_discovery' if discover_institution else 'enabled_source_domains')
        response = requests.post('https://api.tavily.com/search',
            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
            json={'query': claim[:2500], 'search_depth': 'basic', 'topic': 'general',
                  'max_results': 6, 'include_domains': domains[:300],
                  'include_published_date': True,
                  'include_answer': False, 'include_raw_content': True,
                  'auto_parameters': False, 'include_usage': True}, timeout=(4, 22))
        result['http_status'] = response.status_code
        if response.status_code in (401, 403, 429, 432, 433):
            with _lock:
                _blocked[account] = time.monotonic() + 300
            return dict(result, provider_status='quota_or_auth_unavailable')
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('results', []), list):
            raise ValueError('Invalid search envelope')
        seen = set()
        for item in payload.get('results', [])[:6]:
            if not isinstance(item, dict):
                continue
            url = item.get('url')
            trusted = trusted_url(url, sources)
            if not trusted and not (discover_institution and _public_discovery_url(url, sources)):
                continue
            if url in seen:
                continue
            seen.add(url)
            raw = item.get('raw_content')
            # Search summaries and Tavily's generated answer cannot settle claims.
            excerpt = raw.strip()[:14000] if isinstance(raw, str) else ''
            record = {'kind': 'source_article', 'url': url, 'title': str(item.get('title') or '')[:300],
                'publisher': urlparse(url).hostname, 'snippet': excerpt,
                'published_at': item.get('published_date'), 'retrieved_at': now(),
                'retrieval_status': 'article_excerpt_retrieved' if excerpt else 'metadata_only',
                'retrieval_method': 'tavily_original_page_extraction',
                'relationship': 'Publisher page text retrieved by Tavily; discovery is not a truth verdict.'}
            record['reliability'] = source_reliability(record)
            if not trusted:
                record.update(kind='institution_discovery_reference',
                    reliability='institution_website_not_independently_confirmed',
                    relationship='A related website discovered through search. Its official institutional status has not been independently confirmed; treat it as a lead, not proof.')
            result['evidence'].append(record)
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified')
        with _lock:
            if len(_cache) >= 128:
                _cache.clear()
            _cache[cache_key] = (time.monotonic(), deepcopy(result))
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        result['provider_status'] = 'unavailable'
    return result


def _public_discovery_url(url, sources):
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or '').casefold()
        if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
                or '.' not in host or host.endswith(('.local', '.localhost', '.internal', '.test', '.invalid'))):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        return not any(not s.get('enabled', True) and
                       (host == s.get('domain') or host.endswith('.' + str(s.get('domain')))) for s in sources)
    except (ValueError, TypeError, AttributeError):
        return False


def search_tavily(claim, sources, api_key=None):
    from backend.source_routing import route_sources, directory_urls
    from backend.verification import _article_excerpt, trusted_url, source_reliability, merge_source_record, source_record_key
    plan = route_sources(claim, sources)
    if not api_key:
        result = _search_query(plan['queries'][0], sources, api_key)
        result['query_statuses'] = [{'query': result['search_query'], 'status': result['provider_status'],
                                    'requested_domains': [], 'request_attempted': False, 'domain_scope': 'not_requested'}]
        return result
    queries = plan['queries']
    def search(index):
        return _search_query(queries[index], sources, api_key, plan['domains'] if index else None,
                             discover_institution=plan['topic'] == 'institution' and index == 0)
    with ThreadPoolExecutor(max_workers=2) as executor:
        found = list(executor.map(search, range(len(queries))))
    result = dict(found[0], evidence=[], routing=plan,
                  query_statuses=[{'query': q, 'status': r['provider_status'],
                                   'requested_domains': r.get('requested_domains', []),
                                   'request_attempted': r.get('request_attempted', False) and not r.get('cached'),
                                   'cached': bool(r.get('cached')), 'domain_scope': r.get('domain_scope', 'not_requested')}
                                  for q, r in zip(queries, found)])
    if any(r['provider_status'] == 'searched' for r in found):
        result['provider_status'] = 'searched'
    merged = {}
    for row in [row for response in found for row in response.get('evidence', [])]:
        key = source_record_key(row)
        merged[key] = merge_source_record(merged.get(key), row)
    seeds = directory_urls(claim)
    targets = list(dict.fromkeys(seeds + [r['url'] for r in merged.values()
        if r.get('reliability') in ('official_primary_source', 'academic_source')]))[:3]
    targets = [url for url in targets if trusted_url(url, sources)]
    from backend.current_affairs import _office_context
    office = _office_context(claim)
    neutral = ' '.join([*office[1], office[0]]) if office else claim
    with ThreadPoolExecutor(max_workers=3) as executor:
        fresh = list(executor.map(lambda url: _article_excerpt(url, neutral, sources), targets))
    for url, (excerpt, published) in zip(targets, fresh):
        key = source_record_key({'url': url})
        record = merged.get(key)
        if not excerpt:
            if record:
                record['direct_fetch_status'] = 'unavailable_provider_text_retained'
            else:
                result.setdefault('checked_sources', []).append({
                    'url': url, 'title': 'Official office directory', 'publisher': urlparse(url).hostname,
                    'retrieval_status': 'unavailable', 'accepted': False, 'reason_code': 'retrieval_failed',
                    'reason': 'The official directory could not be retrieved, so its contents could not be checked.',
                    'claim_indices': [], 'excerpt': ''})
            continue
        if record is None:
            record = {'kind': 'source_article', 'url': url, 'title': 'Official office directory',
                      'publisher': urlparse(url).hostname}
            merged[key] = record
        record.update(snippet=excerpt, retrieved_at=now(), retrieval_status='article_excerpt_retrieved',
                      published_at=published, retrieval_method='direct_official_page_fetch', direct_fetch_status='completed')
        if url in seeds:
            record['source_role'] = 'office_directory'
        record['reliability'] = source_reliability(record)
    result['evidence'] = list(merged.values())
    result['status'] = 'needs_review' if result['evidence'] else 'unverified'
    return result
