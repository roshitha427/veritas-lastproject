"""Retrieve fact checks; a search match alone never verifies an arbitrary claim."""
from urllib.parse import urlparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
from html import unescape
from concurrent.futures import ThreadPoolExecutor
import re
import threading
import time
import requests
from backend.store import now

DEFAULT_SOURCES = [
    ('Indian academic institutions', 'https://ac.in', 'Academic institution pages; relevance and claim context still require checking'),
    ('Indian educational institutions', 'https://edu.in', 'Educational institution pages and published notices'),
    ('NASA', 'https://www.nasa.gov', 'Space and Earth science primary sources'),
    ('ISRO', 'https://www.isro.gov.in', 'Indian space research primary sources'),
    ('NOAA', 'https://www.noaa.gov', 'Weather, oceans and Earth science'),
    ('NIST', 'https://www.nist.gov', 'Measurement and physical science research'),
    ('WHO', 'https://www.who.int', 'Public health evidence and guidance'),
    ('CDC', 'https://www.cdc.gov', 'Public health evidence and guidance'),
    ('NIH and PubMed', 'https://www.nih.gov', 'Biomedical research and published literature'),
    ('European Space Agency', 'https://www.esa.int', 'Space science primary sources'),
    ('IPCC', 'https://www.ipcc.ch', 'Published climate assessment reports'),
    ('Reuters', 'https://www.reuters.com', 'International reporting'),
    ('Tesla', 'https://www.tesla.com', 'Original manufacturer announcements; a primary company source, not an independent fact-check verdict'),
    ('Associated Press', 'https://apnews.com', 'Independent reporting'),
    ('BBC', 'https://www.bbc.com', 'International reporting'),
    ('AFP Fact Check', 'https://factcheck.afp.com', 'Fact checking'),
    ('Full Fact', 'https://fullfact.org', 'Fact checking'),
    ('BOOM', 'https://www.boomlive.in', 'India · Fact checking and source investigations'),
    ('Vishvas News', 'https://www.vishvasnews.com', 'India · Multilingual fact checking'),
    ('NewsMeter', 'https://newsmeter.in', 'India · Fact checking and reporting'),
    ('PolitiFact', 'https://www.politifact.com', 'Published fact checks'),
    ('Snopes', 'https://www.snopes.com', 'Published fact checks and investigations'),
    ('Fact Crescendo', 'https://www.factcrescendo.com', 'Multilingual fact checking'),
    ('Newschecker', 'https://www.newschecker.in', 'India · Fact checking'),
    ('The Statesman', 'https://www.thestatesman.com', 'Original Indian reporting'),
    ('The Tribune', 'https://www.tribuneindia.com', 'Original Indian reporting'),
    ('The Hindu', 'https://www.thehindu.com', 'Original Indian reporting'),
    ('The Indian Express', 'https://indianexpress.com', 'Original Indian reporting'),
    ('ABP Live', 'https://www.abplive.com', 'Original reporting in multiple Indian languages'),
    ('Alt News', 'https://www.altnews.in', 'India · Fact checking'),
    ('Factly', 'https://factly.in', 'India · Fact checking'),
    ('PIB Fact Check', 'https://pib.gov.in', 'India · Official information'),
    ('National Portal of India', 'https://www.india.gov.in', 'India · Official information'),
    ('Legislative Department, Government of India', 'https://legislative.gov.in', 'India · Primary legal source'),
    ('India Code', 'https://www.indiacode.nic.in', 'India · Primary legal source'),
    ('Indian government source corpus', 'https://gov.in', 'Imported HTTPS government documents; source snapshots, not live news'),
    ('Indian NIC source corpus', 'https://nic.in', 'Imported HTTPS NIC documents; source snapshots, not live news'),
]


def trusted_url(url, sources):
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or '').lower()
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
        matched = [source for source in sources if source.get('domain') and
                   (host == source['domain'].lower() or host.endswith('.' + source['domain'].lower()))]
        return bool(matched) and not any(not source.get('enabled', True) for source in matched)
    except (TypeError, ValueError, AttributeError):
        return False


_gdelt_lock = threading.Lock()
_gdelt_cache = {}
_gdelt_last_request = 0.
STOPWORDS = set('this that there their these those which while where when what would could should about after before being because through against between during under over into with from have has had were was are been not will said says also news claim says just more some than then them they your our you its and the for but any all who how can may'.split())
_SEARCH_STOPWORDS = STOPWORDS - {'not', 'no', 'never'}


def gdelt_query(text, sources):
    """Build exact-phrase and entity/value searches without passing query syntax."""
    raw_words = re.findall(r"\d{1,4}|[A-Za-z][A-Za-z0-9'-]{1,}", text[:2500])
    counts = Counter(word.casefold() for word in raw_words
                     if word.casefold() not in _SEARCH_STOPWORDS and len(word) > 1)
    # Keep the user's order so names, negation, dates, and quantities survive.
    ordered = list(dict.fromkeys(word for word in raw_words
                                 if word.casefold() in counts))
    priority = [word for word in ordered if word.isdigit() or
                (word[:1].isupper() and word.casefold() not in _SEARCH_STOPWORDS)]
    terms = list(dict.fromkeys(priority + ordered))[:10]
    domains = [s['domain'] for s in sources if s.get('enabled', True)
               and not s.get('domain', '').endswith('wikipedia.org')
               and re.fullmatch(r'[a-z0-9.-]+', s.get('domain', ''))][:12]
    if len(terms) < 2 or not domains:
        return None
    phrase_terms = ordered[:14]
    compact = ' '.join(terms)
    phrase = '"' + ' '.join(phrase_terms).replace('"', '') + '"' if len(phrase_terms) >= 4 else compact
    domain_clause = '(' + ' OR '.join('domain:' + domain for domain in domains) + ')'
    return f'({phrase} OR ({compact})) {domain_clause}'


def source_reliability(item):
    """A transparent source class; this is not a guarantee of correctness."""
    url = str((item or {}).get('url') or '')
    host = (urlparse(url).hostname or '').lower()
    kind = str((item or {}).get('kind') or '').lower()
    if kind == 'fact_check_review' and (item or {}).get('rating'):
        return 'published_fact_check_with_rating'
    if kind == 'reference_article':
        return 'encyclopedia_background'
    if any(host == domain or host.endswith('.' + domain) for domain in (
            'boomlive.in', 'vishvasnews.com', 'newsmeter.in', 'factly.in', 'altnews.in',
            'factcrescendo.com', 'newschecker.in', 'politifact.com', 'snopes.com', 'factcheck.afp.com', 'fullfact.org')):
        return 'fact_check_publisher_page_unrated'
    if host.endswith(('.gov', '.gov.in', '.gov.uk', '.gov.au', '.nic.in')) or any(
            host == d or host.endswith('.' + d) for d in ('who.int', 'esa.int', 'ipcc.ch', 'tesla.com')):
        return 'official_primary_source'
    if host.endswith(('.edu', '.ac.in', '.edu.in', '.ac.uk')):
        return 'academic_source'
    if kind == 'fact_check_article':
        return 'fact_check_publisher_page_unrated'
    if any(host == domain or host.endswith('.' + domain) for domain in ('reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'afp.com',
            'thestatesman.com', 'tribuneindia.com', 'thehindu.com', 'indianexpress.com', 'abplive.com')):
        return 'established_news_reporting'
    return 'publisher_not_independently_classified'


def merge_source_record(existing, incoming):
    """Choose a coherent page snapshot and merge only its claim associations.

    Replacing text alone leaves stale retrieval dates/status on new text. Prefer
    live original-page passages over snapshots, then date/length within a class.
    """
    from datetime import datetime, timezone
    def quality(record):
        text = str(record.get('snippet') or '')
        live = bool(text and record.get('retrieval_status') == 'article_excerpt_retrieved')
        value = record.get('retrieved_at') if live else record.get('fetched_at') or record.get('reviewed_at')
        try:
            stamp = datetime.fromisoformat(str(value or '').replace('Z', '+00:00'))
            stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
            timestamp = stamp.timestamp()
        except (ValueError, TypeError, OverflowError):
            timestamp = 0
        return (bool(text), live, timestamp, len(text))
    if not existing:
        return dict(incoming)
    winner = incoming if quality(incoming) > quality(existing) else existing
    result = dict(winner)
    indexes = sorted({index for record in (existing, incoming)
                      for index in record.get('candidate_claim_indices', []) if type(index) is int})
    if indexes:
        result['candidate_claim_indices'] = indexes
    return result


def source_record_key(record, fallback=''):
    """Deduplicate fragments and host case without folding case-sensitive paths."""
    url = str(record.get('url') or '').strip()
    if url:
        try:
            parsed = urlparse(url)
            return parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment='').geturl()
        except ValueError:
            return url
    return str(record.get('title') or fallback)


class _ArticleParser(HTMLParser):
    """Extract page metadata and short paragraph text without storing HTML."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs = []
        self.visible_text = []
        self.headings = []
        self.page_title = []
        self._heading = None
        self.meta = {}
        self.published_at = None
        self._paragraph = None
        self._title = False
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'noscript', 'svg', 'nav', 'header', 'footer', 'aside'):
            self._ignored += 1
        if tag == 'meta':
            key = (attrs.get('property') or attrs.get('name') or '').lower()
            value = attrs.get('content') or ''
            if key in ('description', 'og:description', 'twitter:description') and value:
                self.meta.setdefault(key, value)
            elif key in ('article:published_time', 'datepublished', 'date') and value:
                self.published_at = value
        elif tag == 'time' and not self.published_at:
            self.published_at = attrs.get('datetime')
        elif tag == 'title':
            self._title = True
        elif tag == 'p' and self._paragraph is None:
            self._paragraph = []
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') and not self._ignored:
            self._heading = []

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'svg', 'nav', 'header', 'footer', 'aside') and self._ignored:
            self._ignored -= 1
        if tag == 'title':
            self._title = False
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') and self._heading is not None:
            value = re.sub(r'\s+', ' ', ' '.join(self._heading)).strip()
            if value:
                self.headings.append(value)
            self._heading = None
        elif tag == 'p' and self._paragraph is not None:
            value = re.sub(r'\s+', ' ', ' '.join(self._paragraph)).strip()
            if len(value) >= 50:
                self.paragraphs.append(value)
            self._paragraph = None

    def handle_data(self, data):
        if self._ignored:
            return
        if data.strip():
            self.visible_text.append(data.strip())
        if self._paragraph is not None:
            self._paragraph.append(data)
        if self._heading is not None:
            self._heading.append(data)
        if self._title:
            self.page_title.append(data.strip())


def _office_heading_excerpt(parser, claim):
    from backend.current_affairs import _office_context
    office = _office_context(claim)
    if not office:
        return None
    for index, heading in enumerate(parser.headings):
        match = re.search(r'\b' + re.escape(office[0]) + r'\b', heading.casefold())
        if not match or re.search(r'\b(?:deputy|vice|former|ex)[\s-]*$', heading[:match.start()], re.I):
            continue
        # Preserve the page's own role/name sequence, including adjacent role
        # labels so a deputy's name is never detached from its original title.
        nearby = parser.headings[max(0, index - 3):index + 4]
        return ' '.join(parser.page_title + nearby)[:1800]
    return None


def _article_excerpt(url, claim, sources):
    """Fetch a bounded excerpt only from an enabled HTTPS source host."""
    if not trusted_url(url, sources):
        return None, None
    try:
        response = requests.get(url, timeout=(3, 5), stream=True, allow_redirects=False,
                                headers={'User-Agent': 'NewsVerificationResearchProject/1.0 (source excerpt reader)'})
        if response.status_code >= 300:
            response.close()
            return None, None
        if 'html' not in response.headers.get('Content-Type', '').lower():
            response.close()
            return None, None
        chunks, total = [], 0
        for chunk in response.iter_content(chunk_size=32768):
            total += len(chunk)
            if total > 512 * 1024:
                break
            chunks.append(chunk)
        encoding = response.encoding or 'utf-8'
        response.close()
        parser = _ArticleParser()
        parser.feed(b''.join(chunks).decode(encoding, errors='replace'))
        from backend.evidence_assessment import claim_tokens
        query_terms = claim_tokens(claim)
        ranked = sorted(parser.paragraphs,
                        key=lambda para: len(query_terms & claim_tokens(para)), reverse=True)
        selected = [para for para in ranked[:4]
                    if len(query_terms & claim_tokens(para)) >= min(2, len(query_terms))]
        # A generic paragraph mentioning the location must not hide the page's
        # explicitly labeled office holder. Keep the original heading block
        # first, followed by any matching source paragraphs within the bound.
        heading_excerpt = _office_heading_excerpt(parser, claim)
        if heading_excerpt:
            selected.insert(0, heading_excerpt)
        if not selected:
            # Official leadership directories often put the holder's name in
            # headings or table cells rather than paragraph elements.
            visible = re.sub(r'\s+', ' ', ' '.join(parser.visible_text))
            windows = [visible[index:index + 1400] for index in range(0, len(visible), 900)]
            windows.sort(key=lambda window: len(query_terms & claim_tokens(window)), reverse=True)
            selected = [window for window in windows[:1]
                        if len(query_terms & claim_tokens(window)) >= min(2, len(query_terms))]
        # Search/page descriptions are discovery metadata, not original article
        # evidence. A failed body extraction must stay explicit.
        excerpt = ' '.join(selected) if selected else ''
        excerpt = unescape(re.sub(r'\s+', ' ', excerpt)).strip()
        return excerpt[:1800] or None, parser.published_at
    except (requests.RequestException, ValueError, UnicodeError, TypeError):
        return None, None


def enrich_source_records(verification, claim, sources):
    """Add source classes and, where possible, short text from each original URL."""
    verification = dict(verification or {})
    records = [dict(item) for item in verification.get('evidence') or [] if isinstance(item, dict)]
    for record in records:
        record['reliability'] = source_reliability(record)
    # Public page retrieval is bounded and concurrent so a list of slow or
    # inaccessible publishers cannot make report generation wait serially.
    can_fetch_original = verification.get('provider') != 'Local evidence model'
    targets = ([index for index, record in enumerate(records[:8])
                if record.get('url') and len(str(record.get('snippet') or '')) < 350]
               if can_fetch_original else [])
    if targets:
        with ThreadPoolExecutor(max_workers=4) as executor:
            retrieved = list(executor.map(
                lambda index: _article_excerpt(records[index].get('url', ''), claim, sources), targets))
        for index, (excerpt, published_at) in zip(targets, retrieved):
            record = records[index]
            if excerpt:
                if record.get('snippet'):
                    record['matched_excerpt'] = record['snippet']
                record['snippet'] = excerpt
                record['retrieval_status'] = 'article_excerpt_retrieved'
            else:
                record['retrieval_status'] = 'metadata_only'
            if published_at and not record.get('published_at'):
                record['published_at'] = published_at[:80]
    for record in records:
        if not record.get('retrieval_status'):
            record['retrieval_status'] = 'matched_excerpt_only' if record.get('snippet') else 'metadata_only'
        if record.get('seen_at') and not record.get('published_at'):
            # GDELT's seen date is an observation date, not the article's
            # publication date. Keep it separate so the UI cannot mislabel it.
            record['seen_at'] = record['seen_at']
    enriched = records
    verification['evidence'] = enriched
    return verification


def verify_gdelt(text, sources):
    global _gdelt_last_request
    result = {'status':'unverified', 'checked_at':now(), 'evidence':[], 'provider':'GDELT DOC 2.0',
        'provider_status':'unavailable', 'evidence_type':'related_reporting',
        'explanation':'The public news search could not be reached. The claim remains unverified.',
        'attribution_url':'https://www.gdeltproject.org/'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search for related reporting.')
        return result
    query = gdelt_query(text, sources)
    if query is None:
        result.update(provider_status='query_needed', explanation='For GDELT search, add at least two English keywords about the claim and enable trusted sources. No translation model is installed.')
        return result
    result['search_query'] = query
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    if not _gdelt_lock.acquire(blocking=False):
        result.update(provider_status='busy', explanation='The public news search is busy. Please try again shortly.')
        return result
    try:
        cached = _gdelt_cache.get(cache_key)
        if cached and cached[0] > time.monotonic() - 600:
            return dict(cached[1], cached=True)
        delay = max(0., 5 - (time.monotonic() - _gdelt_last_request))
        if delay:
            time.sleep(delay)
        _gdelt_last_request = time.monotonic()
        # DOC 2.0 searches a rolling three-month window. Context 2.0 is limited
        # to 72 hours and therefore misses many historical claims.
        response = requests.get('https://api.gdeltproject.org/api/v2/doc/doc', params={
            'query':query, 'mode':'ArtList', 'format':'json', 'maxrecords':20,
            'timespan':'3months', 'sort':'HybridRel'}, timeout=(5, 18))
        response.raise_for_status()
        from datetime import datetime, timezone
        seen_urls = set()
        payload = response.json()
        for article in payload.get('articles', []) if isinstance(payload, dict) else []:
            url = article.get('url', '')
            if url in seen_urls or not trusted_url(url, sources):
                continue
            seen_urls.add(url)
            published = None
            seen_date = article.get('seendate') or article.get('seen_at') or ''
            if seen_date:
                try:
                    published = datetime.strptime(seen_date, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    try:
                        published = datetime.fromisoformat(seen_date.replace('Z', '+00:00')).isoformat()
                    except ValueError:
                        pass
            snippet = re.sub(r'\s+', ' ', str(article.get('context') or article.get('sentence') or article.get('snippet') or '')).strip()
            result['evidence'].append({'kind':'related_reporting', 'title':article.get('title', '')[:500],
                'url':url, 'publisher':urlparse(url).hostname, 'claim':'', 'rating':'',
                'reviewed_at':None, 'seen_at':published, 'snippet':snippet[:1200],
                'retrieval_status':'metadata_only',
                'relationship':'GDELT found this article by search terms. Open the original page; a search match is not evidence that it supports or contradicts the claim.'})
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified',
            explanation='Articles from enabled trusted publishers were found in GDELT DOC search. A title or search match alone is not a fact-check verdict.'
                if result['evidence'] else 'No matching reporting from enabled trusted publishers was found in the search window. This does not establish truth or falsehood.')
        if len(_gdelt_cache) > 128:
            _gdelt_cache.clear()
        _gdelt_cache[cache_key] = (time.monotonic(), dict(result))
    except (requests.RequestException, ValueError, TypeError):
        pass
    finally:
        _gdelt_lock.release()
    if result['provider_status'] == 'unavailable':
        # Keep GDELT as the selected provider, but let the project's free
        # fact-check archives and publisher feeds provide a fallback when its
        # endpoint is down or unreachable. A fallback search still never
        # turns a match into an automatic truth verdict.
        try:
            from backend.factcheck_archives import verify_public
            fallback = verify_public(text, sources)
        except Exception:
            fallback = None
        if fallback is not None:
            provider_statuses = [{'name': 'GDELT DOC 2.0', 'status': 'unavailable'}]
            provider_statuses.extend(fallback.get('providers') or [{
                'name': fallback.get('provider', 'Public fact-check search'),
                'status': fallback.get('provider_status', 'unavailable'),
            }])
            if fallback.get('provider_status') == 'searched':
                fallback['provider'] = 'GDELT with public archive/RSS fallback'
                fallback['primary_provider_status'] = 'unavailable'
                fallback['providers'] = provider_statuses
                fallback['explanation'] = 'GDELT could not be reached. ' + fallback.get('explanation', '')
                return fallback
            result['providers'] = provider_statuses
            result['fallback_provider_status'] = fallback.get('provider_status', 'unavailable')
            result['explanation'] = (
                'GDELT and the free fact-check archives/RSS fallback could not be reached. '
                'The claim remains unverified.'
            )
    return result


def _verify_core(text, language, sources, provider='auto', evidence_path=None):
    if provider == 'local':
        from backend.local_evidence import verify_local
        return verify_local(text, sources, evidence_path)
    if provider in ('public', 'auto'):
        from backend.factcheck_archives import verify_public
        return verify_public(text, sources)
    if provider == 'rss':
        from backend.public_evidence import verify_rss
        return verify_rss(text, sources)
    if provider == 'gdelt':
        return verify_gdelt(text, sources)
    result = {'status': 'unverified', 'checked_at': now(), 'evidence': [],
              'provider': 'Evidence search', 'provider_status': 'not_configured',
              'explanation': 'No live fact-check service is configured. Model predictions are not evidence of truth.'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search for supporting fact checks.')
        return result
    return result


def verify_wikipedia(text):
    """Retrieve short encyclopedia excerpts as labeled background references."""
    result = {'name':'Wikipedia reference search', 'status':'unavailable'}
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text[:2500])
    ignored = STOPWORDS | {'current', 'currently', 'today', 'true', 'false', 'fake', 'real', 'verify', 'verified'}
    query = ' '.join(dict.fromkeys(word for word in words if word.lower() not in ignored))[:180]
    if len(query) < 4:
        result['status'] = 'query_needed'
        return result
    try:
        response = requests.get('https://en.wikipedia.org/w/api.php', params={
            'action':'query', 'list':'search', 'srsearch':query, 'srlimit':3,
            'format':'json', 'formatversion':2, 'utf8':1},
            headers={'User-Agent':'NewsVerificationResearchProject/1.0 (evidence retrieval)'}, timeout=(4, 8))
        response.raise_for_status()
        pages = response.json().get('query', {}).get('search', [])
        for page in pages[:3]:
            page_id = page.get('pageid')
            if not isinstance(page_id, int):
                continue
            snippet = re.sub(r'<[^>]+>', '', str(page.get('snippet', '')))
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            result.setdefault('evidence', []).append({
                'kind':'reference_article', 'title':page.get('title', '')[:300],
                'url':f'https://en.wikipedia.org/?curid={page_id}', 'publisher':'Wikipedia',
                'claim':'', 'rating':'', 'reviewed_at':None, 'snippet':snippet[:700],
                'relationship':'Encyclopedia background retrieved by keyword search. It is not a fact-check verdict; confirm current facts with official or independent reporting.'})
        result['status'] = 'searched'
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass
    return result


def _refresh_current_office_sources(text, corpus, sources):
    """Refresh bounded official-source snapshots without treating import dates as current."""
    from backend.current_affairs import _office_context
    office = _office_context(text)
    if not office or not re.search(r'\b(current|currently|today|now|latest)\b', text, re.I):
        return corpus
    neutral_query = ' '.join([*office[1], office[0]])
    targets = [record for record in corpus.get('evidence') or []
               if office[0] in str(record.get('snippet', '')).casefold()][:2]
    if not targets:
        return corpus
    with ThreadPoolExecutor(max_workers=2) as executor:
        fresh = list(executor.map(lambda record: _article_excerpt(record['url'], neutral_query, sources), targets))
    for record, (excerpt, published_at) in zip(targets, fresh):
        if not excerpt:
            record['live_refresh_status'] = 'unavailable_snapshot_retained'
            continue
        record.update(snapshot_excerpt=record['snippet'], snippet=excerpt, kind='official_source_live',
                      retrieved_at=now(), retrieval_status='article_excerpt_retrieved',
                      published_at=published_at, live_refresh_status='completed',
                      relationship='Live excerpt from an enabled official source. Retrieval time is not publication time.')
    return corpus


def verify(text, language, sources, provider='auto', evidence_path=None):
    # An imported source corpus is searched before potentially slow public APIs.
    # Keep explicit offline/disabled provider choices intact.
    if provider not in ('none', 'local'):
        from backend.current_affairs import search_current_affairs
        corpus = search_current_affairs(text, sources)
        if (corpus.get('evidence') and
                corpus['evidence'][0].get('relevance_score', 0) >= .7):
            corpus['providers'] = [{'name': corpus['provider'], 'status': corpus['provider_status']}]
            return _refresh_current_office_sources(text, corpus, sources)
    result = _verify_core(text, language, sources, provider, evidence_path)
    # If GDELT DOC 2.0 is reachable but returns no matches, also search the
    # free fact-check archives and publisher RSS feeds so claims are not left
    # with only encyclopedia references.
    if (text.strip() and provider == 'gdelt' and not result.get('evidence')
            and result.get('provider_status') != 'not_configured'
            and not result.get('fallback_provider_status')
            and not result.get('primary_provider_status')):
        try:
            from backend.factcheck_archives import verify_public
            archive_result = verify_public(text, sources)
        except Exception:
            archive_result = None
        if archive_result:
            result.setdefault('providers', []).append({'name':'GDELT DOC 2.0',
                                                       'status':result.get('provider_status', 'unavailable')})
            archive_evidence = archive_result.get('evidence') or []
            result.setdefault('providers', []).extend(archive_result.get('providers') or [{
                'name': archive_result.get('provider', 'Public fact-check search'),
                'status': archive_result.get('provider_status', 'unavailable'),
            }])
            if archive_evidence:
                result['evidence'] = (result.get('evidence') or []) + archive_evidence
                result['evidence'] = result['evidence'][:12]
                result['provider'] = 'GDELT with public archive/RSS fallback'
                result['archive_provider_status'] = archive_result.get('provider_status', 'searched')
                result['status'] = 'needs_review'
                result['explanation'] = (
                    'No matching recent GDELT reporting was found. Related fact-check archive or publisher-feed '
                    'results were found; compare the claim, publication date, and context before drawing a conclusion.'
                )
    # The no-key Wikimedia reference search is a secondary path when the
    # selected evidence source has no passages. GDELT results stay focused on
    # current reporting when it has relevant matches.
    if text.strip() and provider not in ('none', 'local') and (provider != 'gdelt' or not result.get('evidence')):
        reference = verify_wikipedia(text)
        result.setdefault('providers', []).append({'name':reference['name'], 'status':reference['status']})
        if reference.get('evidence'):
            result.setdefault('evidence', []).extend(reference['evidence'])
            result['evidence'] = result['evidence'][:12]
            result['reference_scope'] = 'Encyclopedia excerpts are background references, not current fact-check verdicts.'
    return enrich_source_records(result, text, sources)


def _provider_verify_submission(text, language, sources, provider='auto', evidence_path=None):
    """Search each independent claim separately, then merge the source records."""
    from backend.evidence_assessment import split_candidate_claims, contextual_candidate_claims

    claims = split_candidate_claims(text)
    queries = contextual_candidate_claims(text)
    if len(claims) <= 1:
        result = verify(claims[0] if claims else text, language, sources, provider, evidence_path)
        from backend.source_routing import normalize_search_submission
        normalized = normalize_search_submission(text)
        if normalized != str(text or '').strip():
            result['normalized_claim'] = normalized
        for item in result.get('evidence') or []:
            item.setdefault('candidate_claim_indices', [0] if claims else [])
        result['claim_searches'] = ([{'claim_index': 0, 'claim': claims[0],
            'provider_status': result.get('provider_status', 'unknown'),
            'evidence_count': len(result.get('evidence') or []),
            'search_query': result.get('search_query')}] if claims else [])
        return result

    searches = []
    merged = {}
    provider_rows = {}
    for claim_index, candidate in enumerate(claims):
        found = verify(queries[claim_index], language, sources, provider, evidence_path)
        claim_ids = []
        for row in found.get('evidence') or []:
            if not isinstance(row, dict):
                continue
            key = source_record_key(row, f"{claim_index}:{row.get('title', '')}:{row.get('snippet', '')}")
            merged[key] = merge_source_record(merged.get(key), dict(row, candidate_claim_indices=[claim_index]))
            claim_ids.append(key)
        searches.append({'claim_index': claim_index, 'claim': candidate,
            'provider_status': found.get('provider_status', 'unknown'),
            'evidence_keys': claim_ids, 'search_query': found.get('search_query')})
        for provider_row in found.get('providers') or [{'name': found.get('provider', 'Unknown'),
                                                         'status': found.get('provider_status', 'unknown')}]:
            name = provider_row.get('name', 'Unknown')
            row = provider_rows.setdefault(name, {'name': name, 'status': provider_row.get('status', 'unknown'),
                                                   'searched_claim_indices': []})
            if provider_row.get('status') == 'searched':
                row['status'] = 'searched'
                row['searched_claim_indices'].append(claim_index)
            elif row['status'] not in ('searched', 'unavailable'):
                row['status'] = provider_row.get('status', 'unknown')

    evidence = list(merged.values())
    statuses = [item['provider_status'] for item in searches]
    overall_status = ('searched' if 'searched' in statuses else
                      'unavailable' if statuses and all(status == 'unavailable' for status in statuses) else
                      statuses[0] if statuses else 'no_claim')
    # Resolve temporary URL keys into the stable post-deduplication IDs used by
    # the report and by Gemini's citation validation.
    for search in searches:
        search['evidence_ids'] = [index for index, row in enumerate(evidence, start=1)
                                  if search['claim_index'] in row.get('candidate_claim_indices', [])]
        search.pop('evidence_keys', None)
    return {'status': 'needs_review' if evidence else 'unverified',
        'checked_at': now(), 'evidence': evidence,
        'provider': 'Claim-specific evidence search', 'provider_status': overall_status,
        'providers': list(provider_rows.values()), 'claim_searches': searches,
        'search_queries': [item['search_query'] for item in searches if item.get('search_query')],
        'evidence_type': 'claim_specific_reporting',
        'explanation': ('Each factual statement was searched separately. Retrieved reporting is context for review, not a verdict by itself.'
                        if evidence else 'Each factual statement was searched separately, but the configured sources returned no matching excerpts. This does not establish truth or falsehood.'),
        'attribution_url': 'https://www.gdeltproject.org/' if provider == 'gdelt' else None}


def verify_submission(text, language, sources, provider='auto', evidence_path=None,
                      dataset_path=None, gemini_key=None, gemini_model='gemini-3.5-flash-lite', grounded_search=False,
                      tavily_key=None):
    """Combine supplied source articles, configured providers and grounded discovery."""
    if provider == 'none' or (not dataset_path and not grounded_search and provider != 'tavily'):
        return _provider_verify_submission(text, language, sources, provider, evidence_path)
    from backend.current_affairs import search_dataset_articles, search_current_affairs
    from backend.evidence_assessment import split_candidate_claims, contextual_candidate_claims
    claims = split_candidate_claims(text)
    queries = contextual_candidate_claims(text)
    additions, providers, searches = [], [], []
    checked_sources = []
    if dataset_path:
        for index, claim in enumerate(claims):
            found = search_dataset_articles(queries[index], sources, dataset_path)
            providers.append({'name': found['provider'], 'status': found['provider_status']})
            additions.extend(dict(record, candidate_claim_indices=[index]) for record in found.get('evidence', []))
        for index, claim in enumerate(claims):
            found = search_current_affairs(queries[index], sources)
            if provider != 'local':
                found = _refresh_current_office_sources(queries[index], found, sources)
            providers.append({'name': found['provider'], 'status': found['provider_status']})
            additions.extend(dict(record, candidate_claim_indices=[index]) for record in found.get('evidence', []))
    # Strong local article coverage avoids a slow GDELT round-trip. Gemini still
    # compares the retrieved text later; sources are never replaced by labels.
    strong_local = bool(claims) and all(any(index in row.get('candidate_claim_indices', [])
        and row.get('relevance_score', 0) >= .7 for row in additions) for index in range(len(claims)))
    from backend.report_summary import filter_relevant_evidence
    local_filtered = filter_relevant_evidence(text, {'evidence': additions})
    additions = local_filtered['evidence']
    checked_sources.extend(local_filtered.get('checked_sources') or [])
    strong_local = strong_local and all(any(index in row.get('relevant_claim_indices', [])
        for row in additions) for index in range(len(claims)))
    # Archived articles cannot replace a fresh search for a current claim.
    current = bool(re.search(r'\b(current|currently|today|now|latest)\b', text, re.I))
    if provider == 'tavily':
        from backend.tavily_search import search_tavily
        # split_candidate_claims already bounds this to five. Do not silently
        # leave candidates four and five without their own source search.
        for index, claim in enumerate(claims):
            found = search_tavily(queries[index], sources, tavily_key)
            providers.append({'name': found['provider'], 'status': found['provider_status'],
                              'searched_claim_indices': [index]})
            additions.extend(dict(row, candidate_claim_indices=[index]) for row in found.get('evidence', []))
            searches.append({'claim_index': index, 'claim': claim,
                             'provider_status': found.get('provider_status', 'unknown'),
                             'search_query': found.get('search_query') or claim,
                             'query_statuses': found.get('query_statuses') or [],
                             'evidence_count': len(found.get('evidence') or [])})
            checked_sources.extend(dict(row, claim_indices=[index]) for row in found.get('checked_sources') or [])
    grounded = None
    if provider not in ('none', 'local', 'tavily') and grounded_search and gemini_key and (
            not strong_local or re.search(r'\b(current|currently|today|now|latest)\b', text, re.I)):
        from backend.grounded_search import search_grounded
        grounded = search_grounded(text, sources, gemini_key, gemini_model)
        providers.append({'name': grounded['provider'], 'status': grounded['provider_status']})
        additions.extend(grounded.get('evidence', []))
    if provider == 'tavily':
        baseline = {'provider': 'Local articles and Tavily', 'checked_at': now(), 'evidence': [],
                    'claim_searches': searches,
                    'provider_status': ('searched' if strong_local or any(p['status'] == 'searched' for p in providers)
                                        else providers[-1]['status'] if providers else 'not_configured')}
    elif strong_local:
        baseline = {'provider': 'Supplied source-article corpus', 'provider_status': 'searched',
                    'checked_at': now(), 'evidence': [], 'claim_searches': []}
    elif grounded and grounded.get('evidence'):
        baseline = grounded
    else:
        baseline = _provider_verify_submission(text, language, sources, provider, evidence_path)
    if grounded:
        baseline['grounded_search_queries'] = grounded.get('search_queries', [])
    merged = {}
    for record in (baseline.get('evidence') or []) + additions:
        key = source_record_key(record)
        merged[key] = merge_source_record(merged.get(key), record)
    baseline['evidence'] = list(merged.values())
    baseline['checked_sources'] = (baseline.get('checked_sources') or []) + checked_sources
    baseline['status'] = 'needs_review' if baseline['evidence'] else 'unverified'
    baseline['providers'] = (baseline.get('providers') or [{'name': baseline.get('provider'),
                              'status': baseline.get('provider_status')}]) + providers
    from backend.source_routing import normalize_search_submission
    normalized = normalize_search_submission(text)
    if normalized != str(text or '').strip():
        baseline['normalized_claim'] = normalized
    return baseline
