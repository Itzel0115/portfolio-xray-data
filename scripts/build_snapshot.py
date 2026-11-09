#!/usr/bin/env python3
"""Fetch real, dated ETF holdings from TickerInside and publish one audited snapshot.

No fictional fallback. The Gradio app never calls this script: its only data connection
is a single GitHub Raw JSON URL serving the output of this script.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

TICKERS = ('VOO','QQQ','IVV','VTI','VUG','VTV','SCHD','IWM')
PROVIDER = 'https://tickerinside.com/api/v1/fund/{}.json'


def fetch_json(url: str, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={'User-Agent':'Portfolio-XRay/3.0 (educational, source-attributed)', 'Accept':'application/json'})
            with urlopen(req, timeout=25) as resp:
                if resp.status != 200:
                    raise ValueError(f'HTTP {resp.status}')
                payload = resp.read(8_000_001)
                if len(payload) > 8_000_000:
                    raise ValueError('ETF response exceeds 8 MB')
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError('ETF response is not JSON object')
            return result
        except (URLError, HTTPError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries-1: time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'Failed: {url}: {type(last).__name__}: {last}')


def normalize_fund(ticker: str, obj: dict) -> dict:
    if str(obj.get('ticker','')).upper() != ticker:
        raise ValueError(f'{ticker}: wrong ticker in response: {obj.get("ticker")}')
    as_of = obj.get('holdings_as_of')
    try:
        dt.date.fromisoformat(as_of)
    except (ValueError, TypeError):
        raise ValueError(f'{ticker}: missing/invalid holdings_as_of') from None
    layer = str(obj.get('source_layer') or '').strip()
    if layer not in ('issuer', 'sec_nport'):
        raise ValueError(f'{ticker}: source_layer {layer!r} is not an approved equity holdings source')
    raw = obj.get('holdings')
    if not isinstance(raw, dict) or len(raw) < 30:
        raise ValueError(f'{ticker}: missing/insufficient holdings map (found {len(raw) if isinstance(raw,dict) else 0})')
    rows=[]
    for symbol, weight in raw.items():
        if not isinstance(symbol,str) or not symbol.strip():
            raise ValueError(f'{ticker}: missing security symbol')
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight):
            raise ValueError(f'{ticker}/{symbol}: invalid weight {weight!r}')
        if weight < -1e-8:
            raise ValueError(f'{ticker}/{symbol}: negative position; not supported in equity-only model')
        if weight > 0:
            rows.append({'ticker':symbol.strip().upper(), 'weight_pct':round(float(weight),8)})
    if len({r['ticker'] for r in rows}) != len(rows):
        raise ValueError(f'{ticker}: duplicated tickers after normalization')
    total=sum(r['weight_pct'] for r in rows)
    if not 85 <= total <= 100.75:
        raise ValueError(f'{ticker}: holdings coverage {total:.3f}% outside safe 85–100.75% range')
    rows.sort(key=lambda x:(-x['weight_pct'],x['ticker']))
    return {
        'name':str(obj.get('name') or ticker),
        'holdings_as_of':as_of,
        'source_layer':layer,
        'upstream_url':PROVIDER.format(ticker.lower()),
        'original_filing_url':obj.get('filing_url') or obj.get('source_url') or None,
        'positions_reported':obj.get('positions'),
        'holdings_count':len(rows),
        'loaded_equity_weight_pct':round(total,6),
        'unmodeled_weight_pct':round(max(0,100-total),6),
        'holdings':rows,
    }


def build_snapshot() -> dict:
    funds={}
    for sym in TICKERS:
        print(f'Fetching {sym} ...', flush=True)
        obj=fetch_json(PROVIDER.format(sym.lower()))
        funds[sym]=normalize_fund(sym,obj)
        print(f'   {funds[sym]["holdings_count"]} positions, {funds[sym]["loaded_equity_weight_pct"]:.2f}%, '
              f'{funds[sym]["holdings_as_of"]}, {funds[sym]["source_layer"]}', flush=True)
    dates=[dt.date.fromisoformat(v['holdings_as_of']) for v in funds.values()]
    return {
        'schema_version':'3.0',
        'dataset_name':'Portfolio X-Ray Real ETF Holdings',
        'data_provider':'TickerInside',
        'provider_url':'https://tickerinside.com/api/',
        'license':'CC BY 4.0 (TickerInside; check source terms for upstream filings)',
        'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'currency':'USD',
        'ticker_order':list(TICKERS),
        'earliest_holdings_as_of':min(dates).isoformat(),
        'latest_holdings_as_of':max(dates).isoformat(),
        'date_range_days':(max(dates)-min(dates)).days,
        'classification_note':'This snapshot has no security-level sector labels. Sector stress is disabled.',
        'funds':funds,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/portfolio_xray_holdings.json')
    args=parser.parse_args()
    snap=build_snapshot()    # fail atomically; do not replace an existing snapshot on partial failure
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(snap,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp=out.with_suffix('.tmp')
    temp.write_text(content,encoding='utf-8')
    temp.replace(out)
    print(f'PASS: {len(snap["funds"])} funds; snapshot at {out}; SHA256 '
          f'{hashlib.sha256(content.encode()).hexdigest()}')
    if snap['date_range_days'] > 30:
        print(f'IMPORTANT: holdings as-of dates differ by {snap["date_range_days"]} days. '
              'Analysis must be labeled as cross-date, not synchronized.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'BUILD FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
