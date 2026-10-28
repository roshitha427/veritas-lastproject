"""Display only well-formed manual-check links with a recorded justification."""
import re
from urllib.parse import urlparse, urlunparse


def manual_check_links(summary, limit=3):
    if summary.get('verdict') != 'unverified' or summary.get('mode') == 'media_only':
        return []
    links, seen = [], set()
    rows = summary.get('recommended_sources')
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        name, description, url = (row.get(key) for key in ('name', 'description', 'url'))
        if not all(isinstance(value, str) and value.strip() for value in (name, description, url)):
            continue
        url = url.strip()
        if re.search(r'[\s\x00-\x1f\x7f]', url):
            continue
        try:
            parsed = urlparse(url)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                    or parsed.port not in (None, 443)):
                continue
            key = urlunparse(parsed._replace(netloc=parsed.hostname.lower(), fragment='')).rstrip('/')
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        links.append({'name': name.strip(), 'url': url.strip(), 'description': description.strip()})
        if len(links) >= limit:
            break
    return links
