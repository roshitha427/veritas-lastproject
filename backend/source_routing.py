"""Topic/jurisdiction search routes contain institutions and URLs, never answers."""
import re
from backend.claim_context import normalize_places, needs_current_sources
from backend.store import now


def normalize_search_submission(claim):
    """Join OCR line wrapping only when explicit news-card banners identify it.

    Ordinary newline-separated statements retain their independent candidates.
    Raw OCR remains available in the media result; this is search normalization.
    """
    text = str(claim or '').strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    banners = {'breaking news', 'fake news', 'global news', 'latest news', 'news update'}
    has_banner = any(line.casefold() in {'breaking news', 'fake news'} for line in lines)
    # Require wrapped prose as well as a banner. A normal list with a heading
    # must not hide independent short claims such as 'The Earth is flat.'
    wrapped = sum(not re.search(r'[.!?]$', line) for line in lines if line.casefold() not in banners)
    if len(lines) >= 4 and has_banner and wrapped >= 3:
        return ' '.join(line for line in lines if line.casefold() not in banners)
    return text


def core_claim_query(claim):
    """Remove request/visual framing while preserving names, actions and values."""
    text = re.sub(r'\s+', ' ', normalize_places(normalize_search_submission(claim))).strip()
    text = re.sub(r'^(?:please\s+)?(?:fact[- ]?check|verify|check)\s+(?:this\s+)?(?:claim\s*:\s*)?', '', text, flags=re.I)
    text = re.sub(r'^(?:is it true that|the (?:image|screenshot|post) (?:claims|says|reads) that)\s+', '', text, flags=re.I)
    return text[:2000]


def is_health_claim(claim):
    """Recognize health effects without confusing device life with human life."""
    text = str(claim or '')
    if re.search(r'\b(health|dengue|vaccines?|vaccination|diseases?|medicine|cancer|virus|medical|'
                 r'treatments?|infection|cures?|clinical|doctors?)\b', text, re.I):
        return True
    biological = bool(re.search(
        r'\b(humans?|people|person|body|bodies|cells?|organisms?|animals?|plants?|'
        r'eat|eats|eating|diet|dietary|nutrition|nutrients?|foods?|fruits?|vegetables?|supplements?|vitamins?)\b',
        text, re.I))
    if re.search(r'\b(batteries|battery|iphone|ipad|macbook|laptops?|smartphones?|devices?|software)\b', text, re.I) and not biological:
        return False
    nutrition = re.search(r'\b(eat|eats|eating|diet|dietary|nutrition|nutritional|nutrients?|'
                          r'foods?|fruits?|vegetables?|supplements?|vitamins?)\b', text, re.I)
    longevity = re.search(r'\b(longevity|lifespan|life span|ageing|aging|immortal|immortality|live forever)\b', text, re.I)
    return bool(nutrition or (longevity and biological))


def route_sources(claim, sources):
    from backend.current_affairs import _office_context
    from backend.verification import trusted_url
    text = core_claim_query(claim)
    office = _office_context(text)
    regions = {'andhra pradesh': ['ap.gov.in', 'apa.neva.gov.in'], 'telangana': ['telangana.gov.in'],
               'tamil nadu': ['tn.gov.in', 'tnla.neva.gov.in']}
    topic, domains = 'general', []
    region = next((name for name in regions if name in text.casefold()), None)
    if re.search(r'\b(college|university|institute|school|hospital|clinic|campus)\b', text, re.I):
        topic = 'institution'
        domains = ['ac.in', 'edu.in', 'gov.in', 'nic.in']
    elif office or re.search(r'\b(government|minister|governor|assembly|election|parliament|constitution)\b', text, re.I):
        topic = 'government'
        domains = regions.get(region, ['gov.in', 'nic.in'])
    elif is_health_claim(text):
        topic, domains = 'health', ['who.int', 'cdc.gov', 'nih.gov', 'icmr.gov.in', 'mohfw.gov.in']
    elif re.search(r'\b(earth|space|moon|planet|sun|nasa|isro|satellite|astronomy|orbit|gravity|rocket)\b', text, re.I):
        topic, domains = 'space_and_earth_science', ['nasa.gov', 'isro.gov.in', 'noaa.gov', 'esa.int']
    elif re.search(r'\b(weather|climate|temperature|rainfall|cyclone|ocean)\b', text, re.I):
        topic, domains = 'climate_and_weather', ['noaa.gov', 'imd.gov.in', 'ipcc.ch', 'nasa.gov']
    elif re.search(r'\b(science|scientific|physics|chemical|chemistry|atom|electron|proton|dna|biology|freezes|boils)\b', text, re.I):
        topic, domains = 'science', ['nist.gov', 'nih.gov', 'nasa.gov', 'noaa.gov']
    elif re.search(r'\b(giveaway|giveaways|giving away|free (?:tesla|cars?|vehicles?|money)|promotion|sweepstakes)\b', text, re.I):
        topic, domains = 'promotion', ['tesla.com', 'reuters.com', 'apnews.com', 'factcheck.afp.com', 'fullfact.org', 'snopes.com']
    domains = [d for d in domains if trusted_url('https://' + d, sources)]
    queries = [text]
    if office and needs_current_sources(claim):
        queries.append(' '.join([*office[1], office[0], 'current official', now()[:10]]))
    elif topic == 'promotion':
        queries.append(text + ' fact check official announcement')
    elif topic == 'health':
        queries.append(text + ' scientific evidence clinical research')
    elif topic != 'general':
        queries.append(text + ' official source' + (' ' + now()[:10] if needs_current_sources(claim) else ''))
    return {'topic': topic, 'jurisdiction': region or (' '.join(office[1]) if office else None),
            'domains': domains, 'queries': list(dict.fromkeys(queries))[:2]}


def directory_urls(claim):
    from backend.current_affairs import _office_context
    office = _office_context(claim)
    if not office or office[0] != 'chief minister' or not needs_current_sources(claim):
        return []
    # Discovery seeds only. Names and verdicts must come from the fetched page.
    return {'tamil nadu': ['https://tnla.neva.gov.in/Home/Index/TN?UnitTypeId=2'],
            'andhra pradesh': ['https://apa.neva.gov.in/Home/'],
            'telangana': ['https://www.telangana.gov.in/Government/Chief-Minister/']}.get(' '.join(office[1]), [])
