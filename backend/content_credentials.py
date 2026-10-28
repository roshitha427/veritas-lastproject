"""Read embedded C2PA declarations locally; absence never means an image is real.

Uses the official contentauth/c2pa-python Reader. Remote manifest and OCSP
fetching are disabled so uploaded files cannot cause arbitrary network requests.
"""
import json
from pathlib import Path

AI_SOURCE = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'


def summarize_credentials(data):
    manifests = data.get('manifests') or {}
    active = data.get('active_manifest')
    if not active or active not in manifests:
        return {'status': 'not_found', 'ai_declared': False}
    validation = data.get('validation_results') or {}
    active_results = validation.get('activeManifest') or {}
    successes = {item.get('code') for item in active_results.get('success') or []}
    failures = list(data.get('validation_status') or []) + list(active_results.get('failure') or [])
    codes = sorted({str(item.get('code', 'unknown')) for item in failures})
    binding_valid = ('claimSignature.validated' in successes and
                     bool(successes & {'assertion.dataHash.match', 'assertion.bmffHash.match', 'assertion.boxesHash.match'}) and
                     not any(any(word in code.lower() for word in ('mismatch', 'missing', 'invalid', 'revoked')) for code in codes))
    # Only inspect the active manifest and explicitly linked ingredients.
    pending = [(active, 'image')]
    seen = set()
    declarations = []
    while pending and len(seen) < 16:
        key, scope = pending.pop(0)
        if key in seen or key not in manifests:
            continue
        seen.add(key)
        manifest = manifests[key]
        if not isinstance(manifest, dict):
            continue
        for assertion in manifest.get('assertions') or []:
            if not isinstance(assertion, dict) or not str(assertion.get('label', '')).startswith('c2pa.actions'):
                continue
            if not isinstance(assertion.get('data'), dict):
                continue
            for action in assertion['data'].get('actions') or []:
                if isinstance(action, dict) and action.get('digitalSourceType') == AI_SOURCE:
                    declarations.append({'scope': scope, 'digital_source_type': AI_SOURCE})
        for ingredient in manifest.get('ingredients') or []:
            if not isinstance(ingredient, dict):
                continue
            linked = ingredient.get('active_manifest')
            if linked:
                pending.append((linked, 'source_component'))
    fully_valid = data.get('validation_state') == 'Trusted' and not codes and binding_valid
    declared = bool(declarations) and binding_valid
    return {'status': 'verified' if fully_valid else 'unverified',
            'ai_declared': declared, 'binding_valid': binding_valid,
            'validation_state': data.get('validation_state', 'Unknown'),
            'validation_codes': codes, 'declarations': declarations if declared else [],
            'scope': ('image' if any(d['scope'] == 'image' for d in declarations) else 'source_component') if declared else None,
            'message': ('Embedded Content Credentials declare AI-generated content. '
                        + ('Credential validation passed.' if fully_valid else
                           'Credential validation has warnings or failures; this declaration is not independently verified.')
                        if declared else 'Content Credentials did not establish an AI-generation declaration.')}


def inspect_content_credentials(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 50*1024*1024:
        return {'status': 'unavailable', 'ai_declared': False}
    try:
        from c2pa import Context, Reader, C2paError
    except ImportError:
        return {'status': 'not_configured', 'ai_declared': False}
    try:
        with Context.from_dict({'verify': {'remote_manifest_fetch': False, 'ocsp_fetch': False},
                                'core': {'allowed_network_hosts': []}}) as context:
            with Reader(str(path), context=context) as reader:
                return summarize_credentials(json.loads(reader.json()))
    except (C2paError, ValueError, OSError, KeyError, TypeError):
        return {'status': 'unavailable_or_absent', 'ai_declared': False}
