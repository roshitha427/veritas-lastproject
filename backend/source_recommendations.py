"""Direct source portals for manual follow-up, separate from retrieved evidence."""
import re

from backend.source_routing import core_claim_query, is_health_claim, route_sources
from backend.verification import trusted_url


RECOMMENDATION_POLICY_VERSION = 2

# Reviewed public landing pages; no user-provided or model-generated URLs.
PORTALS = {
    'india': ('National Portal of India', 'https://www.india.gov.in/',
              'Find official government departments, schemes, services, and announcements.', 'official_portal'),
    'pib': ('PIB Fact Check', 'https://www.pib.gov.in/factcheck.aspx',
            'Check published clarifications about Government of India claims. Its remit is central government information.', 'official_portal'),
    'andhra pradesh': ('Andhra Pradesh Government', 'https://www.ap.gov.in/',
                       'Check Andhra Pradesh government information and departmental announcements.', 'official_portal'),
    'telangana': ('Telangana Government', 'https://www.telangana.gov.in/',
                  'Check Telangana government information and official announcements.', 'official_portal'),
    'tamil nadu': ('Tamil Nadu Government', 'https://www.tn.gov.in/',
                   'Check Tamil Nadu government information and official announcements.', 'official_portal'),
    'ugc': ('University Grants Commission (UGC)', 'https://www.ugc.gov.in/',
            'Check higher-education notices and links to recognized institutions and autonomous colleges.', 'official_portal'),
    'gist': ('GIST official website', 'https://gist.edu.in/gist/gist-home/',
             'Check information published by Geethanjali Institute of Science and Technology in Nellore.', 'institution_source'),
    'who': ('World Health Organization (WHO)', 'https://www.who.int/',
            'Review health-topic guidance, disease information, and public-health updates.', 'primary_source'),
    'nasa': ('NASA', 'https://www.nasa.gov/',
             'Check original space-mission news and Earth and space science information.', 'official_portal'),
    'isro': ('Indian Space Research Organisation (ISRO)', 'https://www.isro.gov.in/',
             'Check Indian space-mission information and official announcements.', 'official_portal'),
    'nist': ('National Institute of Standards and Technology (NIST)', 'https://www.nist.gov/',
             'Consult physical-science research, measurement data, and standards information.', 'official_portal'),
    'noaa': ('National Oceanic and Atmospheric Administration (NOAA)', 'https://www.noaa.gov/',
             'Check weather, ocean, and climate science information.', 'official_portal'),
    'imd': ('India Meteorological Department (IMD)', 'https://mausam.imd.gov.in/',
            'Check official Indian weather forecasts, district warnings, and cyclone updates.', 'official_portal'),
    'ipcc': ('Intergovernmental Panel on Climate Change (IPCC)', 'https://www.ipcc.ch/',
             'Consult published climate assessment reports.', 'primary_source'),
    'afp': ('AFP Fact Check', 'https://factcheck.afp.com/',
            'Look for a published investigation of the same claim and compare its date and context.', 'fact_check_service'),
    'tesla': ('Tesla official news', 'https://www.tesla.com/blog',
              'Look for an original company announcement with the same offer, dates, eligibility, and terms. This link does not confirm that an offer exists.', 'company_source'),
}


def _matches(pattern, text):
    return bool(re.search(pattern, text, re.I))


def _description(key, claim):
    """Explain the record to look for, tied to the actual submitted claim."""
    subject = ' '.join(claim.split())
    if len(subject) > 50:
        subject = subject[:47].rsplit(' ', 1)[0] + '...'
    records = {
        'india': 'government-department notices, scheme details and dated announcements',
        'pib': 'published central government clarifications addressing the same announcement',
        'andhra pradesh': 'dated Andhra Pradesh departmental notices with matching places, dates and policy',
        'telangana': 'dated Telangana departmental notices with matching places, dates and policy',
        'tamil nadu': 'dated Tamil Nadu departmental notices with matching places, dates and policy',
        'ugc': 'UGC records of recognized higher-education institutions and autonomous-college status for the named institution',
        'gist': 'GIST institutional notices and published campus information for the named location, event or status',
        'who': 'WHO health or nutrition guidance addressing the same claimed health effect',
        'nasa': 'NASA scientific explanations or mission records addressing the same Earth or space-science statement',
        'isro': 'dated ISRO mission updates or launch records for the named Indian mission',
        'nist': 'NIST physical-science and measurement references for the named substance, quantity and conditions',
        'noaa': 'NOAA climate or ocean records addressing the same phenomenon',
        'imd': 'dated IMD district forecasts and warnings for the named Indian location and event',
        'ipcc': 'IPCC assessment findings addressing the same climate-change statement',
        'tesla': 'an original Tesla announcement with matching offer dates, eligibility and terms',
    }
    if key == 'noaa' and _matches(r'\b(weather|forecast|storm|hurricane|rainfall|snow|cyclone|temperature)\b', claim):
        record = 'dated NOAA forecasts or weather warnings for the named United States location and event'
    else:
        record = records[key]
    return f'For "{subject}", check {record}.'


def source_recommendations(claim, summary, sources):
    """Offer only enabled primary portals whose remit fits this claim."""
    if (summary.get('verdict') != 'unverified' or summary.get('mode') == 'media_only'
            or not str(claim or '').strip()):
        return []
    plan = route_sources(claim, sources)
    text = core_claim_query(claim)
    topic = plan['topic']
    gist = bool(re.search(r'geethanjali\s+institute\s+of\s+science', claim, re.I)
                or re.search(r'\bGIST\b', claim)
                or (re.search(r'\bgist\b', claim, re.I)
                    and re.search(r'\b(college|institute|nellore|gangavaram)\b', claim, re.I)))
    region = plan.get('jurisdiction')
    state = region if region in ('andhra pradesh', 'telangana', 'tamil nadu') else None
    indian = bool(state) or _matches(
        r'\b(india|indian|nellore|gangavaram|hyderabad|delhi|mumbai|chennai|bengaluru|kolkata|vijayawada|visakhapatnam|tirupati)\b', text)
    national_india = _matches(r'\b(india|indian|narendra modi|pmo|lok sabha|rajya sabha)\b', text)
    central = national_india and _matches(
        r'\b(central government|union government|government of india|indian government|prime minister|narendra modi|pmo|lok sabha|rajya sabha|parliament)\b', text)
    education = _matches(r'\b(schools?|colleges?|universit(?:y|ies)|institutes?|campus)\b', text)
    higher_education = _matches(r'\b(colleges?|universit(?:y|ies)|institutes?|higher education|ugc)\b', text) or gist
    recognition = _matches(r'\b(ugc|recogni[sz]ed|recognition|autonomous|autonomy|degree[- ]awarding)\b', text)
    weather = _matches(r'\b(weather|forecast|cyclone|hurricane|rain|rainfall|storm|flood|heatwave|snow|snowfall|temperature)\b', text)
    health = is_health_claim(text)
    if gist:
        keys = ['gist'] + (['ugc'] if recognition else [])
        if weather and indian:
            keys.append('imd')
    elif (topic == 'institution' or education) and higher_education and recognition:
        keys = ['ugc'] if indian or _matches(r'\bugc\b', text) else []
    elif education:
        keys = [state] if state and _matches(r'\b(holiday|clos\w*|shut\w*|suspend\w*|notice|government)\b', text) else []
        if weather and indian:
            keys.append('imd')
    elif topic == 'health' or health:
        keys = ['who']
    elif topic == 'space_and_earth_science' and not _matches(r'\b(climate|global warming|greenhouse|carbon emissions|sea level)\b', text):
        indian_mission = _matches(r'\b(india|indian|isro|chandrayaan|mangalyaan|gaganyaan)\b', text)
        keys = ['isro'] if indian_mission else ['nasa']
    elif (topic == 'science' and _matches(r'\b(physical|physics|chemical|chemistry|atom|electron|proton|measurement|freezes|boils)\b', text)) or (topic == 'climate_and_weather'
            and re.search(r'\b(chemical|chemistry|atom|electron|proton|freezes|boils)\b', claim, re.I)
            and not re.search(r'\b(weather|climate|ocean|cyclone|rainfall)\b', claim, re.I)):
        keys = ['nist']
    elif topic != 'government' and (topic == 'climate_and_weather' or weather or _matches(r'\b(climate|global warming|greenhouse|carbon emissions|sea level)\b', text)):
        if _matches(r'\b(climate|global warming|greenhouse|carbon emissions|sea level)\b', text):
            keys = ['ipcc', 'noaa']
        elif weather:
            united_states = _matches(r'\b(united states|usa|florida|california|texas|new york|noaa)\b', text)
            keys = ['imd'] if indian else ['noaa'] if united_states else []
        elif _matches(r'\b(ocean|noaa)\b', text):
            keys = ['noaa']
        else:
            keys = []
    elif topic == 'government':
        if state:
            keys = [state]
        elif national_india:
            keys = ['india'] + (['pib'] if central else [])
        else:
            keys = []
        if weather and indian:
            keys.append('imd')
    elif topic == 'promotion':
        keys = ['tesla'] if _matches(r'\btesla\b', text) else []
    elif _matches(r'\bugc\b', text):
        keys = ['ugc']
    elif _matches(r'\b(isro|chandrayaan|mangalyaan|gaganyaan)\b', text):
        keys = ['isro']
    else:
        keys = []
    recommendations = []
    for key in keys:
        name, url, _, kind = PORTALS[key]
        if trusted_url(url, sources):
            recommendations.append({'name': name, 'url': url, 'description': _description(key, text), 'kind': kind})
        if len(recommendations) == 4:
            break
    return recommendations


def with_source_recommendations(summary, claim, sources):
    summary = dict(summary or {})
    summary['recommended_sources'] = source_recommendations(claim, summary, sources)
    summary['recommendation_policy_version'] = RECOMMENDATION_POLICY_VERSION
    return summary
