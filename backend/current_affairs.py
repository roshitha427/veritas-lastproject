"""Search locally imported source documents; source snapshots are not live news."""
from pathlib import Path
from contextlib import closing
import re
import sqlite3
from urllib.parse import urlparse

from backend.store import now
from backend.claim_context import normalize_places


DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'models/current-affairs/evidence.sqlite'


def official_url(url):
    parsed = urlparse(str(url or ''))
    host = (parsed.hostname or '').lower()
    return (parsed.scheme == 'https' and not parsed.username and not parsed.password
            and parsed.port in (None, 443)
            and any(host.endswith('.' + root) or host == root for root in ('gov.in', 'nic.in')))


def _terms(text):
    from backend.verification import STOPWORDS
    text = normalize_places(text)
    text = re.sub(r'\bcm\b', 'chief minister', text, flags=re.I)
    text = re.sub(r'\bpm\b', 'prime minister', text, flags=re.I)
    ignored = STOPWORDS | {'is', 'of', 'to', 'an', 'as', 'at', 'on', 'be', 'it', 'do', 'does',
                          'current', 'currently', 'today', 'latest', 'please', 'tell', 'true', 'false', 'fake', 'real'}
    return list(dict.fromkeys(word.casefold() for word in re.findall(r'[^\W_]+', text[:2500])
                             if len(word) > 1 and word.casefold() not in ignored))[:20]


def _office_context(text):
    """Search the office and jurisdiction independently of the alleged holder."""
    text = re.sub(r'\s+', ' ', normalize_places(text).casefold()).strip(' .?!')
    text = re.sub(r'\bcm\b', 'chief minister', text)
    text = re.sub(r'\bpm\b', 'prime minister', text)
    role = r'(chief minister|prime minister|president|governor)'
    location = r'([a-z]+(?: [a-z]+){0,4}?)'
    match = re.search(r'\b' + role + r' (?:of|in) ' + location +
                      r'(?= (?:is|was|are|has|will)\b|$)', text)
    if match:
        return match.group(1), _terms(match.group(2))
    match = re.match(r'(?:(?:the|current|currently) )*' + location + r' ' + role + r'\b', text)
    if match:
        return match.group(2), _terms(match.group(1))
    return None


def _excerpt(text, terms, limit=1800, office=None):
    # Search the full document, not just its opening navigation or title.
    chunks = [text[i:i + 1300] for i in range(0, len(text), 1000)]
    scored = []
    for index, chunk in enumerate(chunks):
        words = set(re.findall(r'[^\W_]+', chunk.casefold()))
        overlap = len(set(terms) & words)
        pairs = sum(' '.join(terms[i:i + 2]) in re.sub(r'\s+', ' ', chunk.casefold())
                    for i in range(len(terms) - 1))
        context = bool(office and office[0] in re.sub(r'\s+', ' ', chunk.casefold())
                       and all(term in words for term in office[1]))
        scored.append((overlap + pairs * 0.5 + (20 if context else 0), -index, chunk))
    selected = sorted(scored, reverse=True)[:2]
    return '\n[...]\n'.join(item[2].strip() for item in selected if item[0] > 0)[:limit]


def search_current_affairs(text, sources, path=None, official_only=True):
    from backend.verification import trusted_url, source_reliability
    result = {'provider': 'Local Indian official-source corpus', 'provider_status': 'not_configured',
              'checked_at': now(), 'status': 'unverified', 'evidence': []}
    if not official_only:
        result['provider'] = 'Supplied source-article corpus'
    path = Path(path or DEFAULT_PATH)
    if not path.is_file():
        return result
    terms = _terms(text)
    if len(terms) < 2:
        result['provider_status'] = 'query_needed'
        return result
    # All tokens are quoted; user input cannot inject FTS operators or SQL.
    query = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
    office = _office_context(text)
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            metadata = dict(db.execute('SELECT key, value FROM metadata'))
            rows = db.execute('''SELECT d.*, bm25(documents_fts, 5.0, 1.0) AS rank
                FROM documents_fts JOIN documents d ON d.id=documents_fts.rowid
                WHERE documents_fts MATCH ? ORDER BY rank LIMIT 80''', (query,)).fetchall()
            if office and office[1]:
                context_query = ' AND '.join('"' + term + '"' for term in [office[0], *office[1]])
                context_rows = db.execute('''SELECT d.*, bm25(documents_fts, 5.0, 1.0) AS rank
                    FROM documents_fts JOIN documents d ON d.id=documents_fts.rowid
                    WHERE documents_fts MATCH ? ORDER BY rank LIMIT 80''', (context_query,)).fetchall()
                rows = list({row['id']: row for row in [*rows, *context_rows]}.values())
        result.update(provider_status='searched', corpus=metadata)
        ranked = []
        for row in rows:
            if (official_only and not official_url(row['url'])) or not trusted_url(row['url'], sources):
                continue
            host = urlparse(row['url']).hostname
            if any(not source.get('enabled', True) and
                   (host == source.get('domain') or host.endswith('.' + str(source.get('domain'))))
                   for source in sources):
                continue
            passage = _excerpt(row['text'], terms, office=office)
            words = set(re.findall(r'[^\W_]+', (row['title'] + ' ' + passage).casefold()))
            overlap = set(terms) & words
            if len(overlap) < 2 or len(overlap) / len(terms) < .4:
                continue
            context = bool(office and office[0] in re.sub(r'\s+', ' ', passage.casefold())
                           and all(term in words for term in office[1]))
            context_rank = int(context)
            if context and office[1]:
                normalized = re.sub(r'\s+', ' ', passage.casefold())
                jurisdiction = ' '.join(office[1])
                if (office[0] + ' of ' + jurisdiction in normalized
                        or jurisdiction in row['title'].casefold()):
                    context_rank += 1
            coverage = max(len(overlap) / len(terms), 1.0 if context else 0)
            ranked.append((context_rank, coverage, -row['rank'], row, passage))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        seen = set()
        for _, coverage, _, row, passage in ranked:
            if row['url'] in seen:
                continue
            seen.add(row['url'])
            reliability = source_reliability({'url': row['url'], 'kind': 'source_article'})
            result['evidence'].append({'kind': 'official_source_snapshot' if official_only else
                'fact_check_article' if reliability == 'fact_check_publisher_page_unrated' else 'source_article', 'title': row['title'],
                'url': row['url'], 'publisher': row['publisher'], 'snippet': passage,
                'claim': '', 'rating': '', 'reviewed_at': None,
                'fetched_at': row['fetched_at'], 'retrieval_status': 'imported_source_excerpt',
                'reliability': 'official_primary_source' if official_only else reliability,
                'relevance_score': round(coverage, 3),
                'relationship': ('Archived source text. Import/snapshot date: ' + row['fetched_at'] +
                    '. This is not a publication date or a guarantee that the fact is still current.')})
            if 'published_at' in row.keys():
                result['evidence'][-1]['published_at'] = row['published_at']
            if len(result['evidence']) == 5:
                break
        result['status'] = 'needs_review' if result['evidence'] else 'unverified'
        result['explanation'] = ('Relevant official-source snapshots were retrieved locally. Compare dates and cited '
                                 'passages; an archived document is not automatically a verified current fact.')
    except (sqlite3.Error, OSError, ValueError):
        result['provider_status'] = 'unavailable'
    return result


def search_dataset_articles(text, sources, path=None):
    path = path or DEFAULT_PATH.parent.parent / 'dataset-articles-v2' / 'evidence.sqlite'
    return search_current_affairs(text, sources, path, official_only=False)
