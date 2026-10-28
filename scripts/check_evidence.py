"""Check the free keyless evidence provider with a public example query."""
import json
import argparse
from urllib.parse import urlparse
from backend.verification import DEFAULT_SOURCES, verify

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--provider',choices=['public','rss','gdelt'],default='public')
    parser.add_argument('--query',default='NASA moon landing fake')
    args=parser.parse_args()
    sources=[{'domain':urlparse(url).hostname.removeprefix('www.'),'enabled':True} for _,url,_ in DEFAULT_SOURCES]
    result=verify(args.query,'en',sources,provider=args.provider)
    print(json.dumps({'provider':result['provider'],'provider_status':result['provider_status'],
        'articles':len(result['evidence']),'verification_status':result['status'],
        'providers':result.get('providers',[]),
        'example_sources':[item['url'] for item in result['evidence'][:3]]},indent=2))
    raise SystemExit(0 if result['provider_status']=='searched' else 1)
