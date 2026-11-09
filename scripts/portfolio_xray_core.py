"""Portfolio X-Ray 3.0 – real holdings only; no data generation and no fictional fallback."""
from __future__ import annotations
import datetime as dt
import io, json, math, os, re, tempfile, time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import requests

UNIVERSE=('VOO','QQQ','IVV','VTI','VUG','VTV','SCHD','IWM')
MAX_SNAPSHOT_BYTES=12_000_000


def read_github_snapshot(raw_url):
    """ONE application data connection. Only HTTPS raw.githubusercontent.com JSON."""
    u=urlparse(str(raw_url).strip())
    if (u.scheme!='https' or u.hostname!='raw.githubusercontent.com' or u.username or u.password
        or u.query or u.fragment or not u.path.lower().endswith('.json') or '..' in u.path
        or len([p for p in u.path.split('/') if p]) < 4):
        raise ValueError('請設定公開的 GitHub Raw JSON 網址（https://raw.githubusercontent.com/帳號/儲存庫/main/data/portfolio_xray_holdings.json）。')
    response=requests.get(raw_url,timeout=(7,25),allow_redirects=False,
                          headers={'Accept':'application/json','User-Agent':'Portfolio-XRay/3.0 educational'})
    if response.status_code!=200:
        raise ValueError(f'GitHub Raw HTTP {response.status_code}；請確認已執行 GitHub Actions、儲存庫公開且 URL 可開啟。')
    if len(response.content)>MAX_SNAPSHOT_BYTES:
        raise ValueError('JSON 超過 12 MB。')
    try: document=response.json()
    except ValueError:
        raise ValueError('GitHub 回傳的內容不是 JSON；請檢查是否貼到一般 GitHub 檔案頁面。') from None
    validate_snapshot(document)
    return document


def validate_snapshot(dataset):
    if not isinstance(dataset,dict) or dataset.get('schema_version')!='3.0':
        raise ValueError('不支援的資料結構，必須由 Portfolio X-Ray 3.0 Snapshot Builder 產生。')
    if set(dataset.get('funds',{}))!=set(UNIVERSE):
        raise ValueError('資料庫須包含完整八檔 ETF：'+', '.join(UNIVERSE))
    if dataset.get('data_provider')!='TickerInside':
        raise ValueError('來源識別不符合預期：TickerInside。')
    if dataset.get('currency')!='USD':
        raise ValueError('本版只接受 USD 名目分析。')
    quality=[]
    for etf in UNIVERSE:
        fund=dataset['funds'][etf]
        if not isinstance(fund,dict): raise ValueError(f'{etf}: fund 不是 object')
        try: dt.date.fromisoformat(fund['holdings_as_of'])
        except (ValueError,KeyError,TypeError): raise ValueError(f'{etf}: 無效持股日期') from None
        if fund.get('source_layer') not in ('issuer','sec_nport'):
            raise ValueError(f'{etf}: 不支援的持股來源層級')
        holdings=fund.get('holdings')
        if not isinstance(holdings,list) or len(holdings)<30:
            raise ValueError(f'{etf}: 缺少完整的持股清單（至少 30 列）。')
        seen=set();total=0
        for row in holdings:
            if not isinstance(row,dict):raise ValueError(f'{etf}: 持股列型別錯誤')
            ticker=row.get('ticker');wt=row.get('weight_pct')
            if not isinstance(ticker,str) or not ticker.strip() or ticker in seen:
                raise ValueError(f'{etf}: 缺少或重複股票代碼 {ticker!r}')
            seen.add(ticker)
            if type(wt) not in (int,float) or not math.isfinite(wt) or wt<=0:
                raise ValueError(f'{etf}/{ticker}: 權重不合法')
            total+=wt
        if not 85<=total<=100.75:
            raise ValueError(f'{etf}: 已載入權重 {total:.3f}%，超出安全範圍')
        quality.append({'ETF':etf,'As of':fund['holdings_as_of'],
                        'Source layer':fund['source_layer'],'Positions':len(holdings),
                        'Equity coverage (%)':round(total,3),
                        'Unmodeled (%)':round(max(0,100-total),3),
                        'Upstream URL':fund.get('upstream_url','')})
    return pd.DataFrame(quality)


def normalize_allocation(table, capital):
    """Expect Gradio editable Dataframe or list-of-rows; all 8 ETF rows fixed."""
    if isinstance(table,pd.DataFrame): rows=table.to_dict('records')
    elif isinstance(table,list): rows=[dict(zip(['ETF','Weight (%)'],v)) for v in table]
    else: raise ValueError('ETF 配置表格式錯誤。')
    values={}
    for row in rows:
        sym=str(row.get('ETF','')).strip().upper()
        if not sym and str(row.get('Weight (%)','')).strip() in ('','nan','None'): continue
        if sym not in UNIVERSE or sym in values:
            raise ValueError(f'未知或重複的 ETF：{sym}。')
        wt=row.get('Weight (%)')
        try: val=float(wt)
        except (ValueError,TypeError): raise ValueError(f'{sym} 權重必須是數字。') from None
        if not math.isfinite(val) or not 0<=val<=100: raise ValueError(f'{sym} 權重必須在 0～100 之間。')
        values[sym]=val
    # Optional zero holdings permitted, but the table must represent all eight tickers.
    if set(values)!=set(UNIVERSE):raise ValueError('配置表須保留八檔 ETF（不持有者填 0）。')
    if not math.isclose(sum(values.values()),100,rel_tol=0,abs_tol=0.001):
        raise ValueError(f'權重合計 {sum(values.values()):.3f}%，必須等於 100%。')
    if sum(1 for v in values.values() if v>0)<2:
        raise ValueError('請至少配置兩檔 ETF。')
    if isinstance(capital,bool):raise ValueError('本金必須是有效數值。')
    try:capital=float(capital)
    except (TypeError,ValueError):raise ValueError('本金必須是有效數值。') from None
    if not math.isfinite(capital) or not 0<capital<=1e12:raise ValueError('本金需介於 0～1 兆 USD。')
    return values,capital


def _maps(dataset):
    # only instrument IDs; no assumed industry labels
    return {etf:{r['ticker']:float(r['weight_pct']) for r in dataset['funds'][etf]['holdings']}
            for etf in UNIVERSE}


def inspect_portfolio(dataset, weights_pct, capital):
    maps=_maps(dataset);expo=defaultdict(float);sources=defaultdict(dict)
    for etf,port_w in weights_pct.items():
        if port_w<=0:continue
        for stock,stock_w in maps[etf].items():
            contrib=port_w*stock_w/100.0  # percentage points of total portfolio
            expo[stock]+=contrib
            sources[stock][etf]=round(contrib,7)
    holdings=[{'ticker':sym,'exposure_pct':round(w,7),'amount_usd':round(capital*w/100,2),
               'sources_by_etf_pp':sources[sym]}
              for sym,w in expo.items()]
    holdings.sort(key=lambda r:(-r['exposure_pct'],r['ticker']))
    cover=sum(h['exposure_pct'] for h in holdings)
    hhi=sum((r['exposure_pct']/100)**2 for r in holdings)
    latest=max(dataset['funds'][etf]['holdings_as_of'] for etf in weights_pct if weights_pct[etf]>0)
    earliest=min(dataset['funds'][etf]['holdings_as_of'] for etf in weights_pct if weights_pct[etf]>0)
    return {'ok':True,'tool':'inspect_portfolio','dataset':dataset['dataset_name'],
            'weights_pct':weights_pct,'capital_usd':capital,
            'holdings':holdings,'holdings_count':len(holdings),
            'largest_ticker':holdings[0]['ticker'],
            'largest_exposure_pct':holdings[0]['exposure_pct'],
            'top_two_pct':round(sum(r['exposure_pct'] for r in holdings[:2]),6),
            'modeled_coverage_pct':round(cover,6),
            'unmodeled_pct':round(max(0,100-cover),6),
            'hhi_known_positions':round(hhi,8),
            'holdings_date_range':{'earliest':earliest,'latest':latest,
                 'days':(dt.date.fromisoformat(latest)-dt.date.fromisoformat(earliest)).days},
            'source_note':'ETF 持股日期可能不一致；數值是不同快照的穿透分析。'}


def analyze_overlap(dataset, etf_a, etf_b):
    a=str(etf_a).upper().strip();b=str(etf_b).upper().strip()
    if a not in UNIVERSE or b not in UNIVERSE or a==b:
        raise ValueError('請輸入兩檔不同、且列於八檔資料庫中的 ETF。')
    maps=_maps(dataset);aa=maps[a];bb=maps[b]
    rows=[{'ticker':s,f'{a}_pct':aa[s],f'{b}_pct':bb[s],
           'matched_pct':round(min(aa[s],bb[s]),6)} for s in (aa.keys() & bb.keys())]
    rows.sort(key=lambda x:(-x['matched_pct'],x['ticker']))
    obs=sum(r['matched_pct'] for r in rows)
    cov={a:sum(aa.values()),b:sum(bb.values())}
    dates={a:dataset['funds'][a]['holdings_as_of'],b:dataset['funds'][b]['holdings_as_of']}
    delta=abs((dt.date.fromisoformat(dates[a])-dt.date.fromisoformat(dates[b])).days)
    return {'ok':True,'tool':'analyze_overlap','etf_a':a,'etf_b':b,
            'observed_weighted_overlap_pct':round(obs,6),'common_tickers_count':len(rows),
            'common_holdings':rows,'fund_coverage_pct':cov,'holdings_dates':dates,
            'date_gap_days':delta,'metric_type':('snapshot_based_approximately_full'
             if all(v>=99.5 for v in cov.values()) else 'observed_lower_bound'),
            'note':'相異日期持股不等於同步歷史配置；未涵蓋部分可能影響精確重疊率。'}


def run_stress_test(dataset,weights_pct,capital,shocks):
    if type(shocks) is not list or not (1<=len(shocks)<=5):
        raise ValueError('shocks 必須是 1～5 筆清單。')
    maps=_maps(dataset); available=set().union(*[set(x) for x in maps.values()]);seen=set()
    specs=[]
    for sh in shocks:
        if type(sh) is not dict or set(sh)!={'level','target','change_pct'}:
            raise ValueError('衝擊須包含且僅包含 level、target、change_pct。')
        kind=sh['level']; target=str(sh['target']).strip().upper(); amount=sh['change_pct']
        if type(amount) not in (int,float) or not math.isfinite(amount) or not (-100<=amount<=500):
            raise ValueError('衝擊百分比須介於 −100 至 +500。')
        if kind=='market' and target=='ALL': affected=set(available)
        elif kind=='stock' and target in available:affected={target}
        elif kind=='sector':raise ValueError('資料集未提供完整逐檔產業分類，暫不支援 sector shock；請指定股票代碼。')
        else:raise ValueError(f'未知的衝擊目標：{kind}/{target}')
        if seen & affected:raise ValueError('衝擊範圍重複，同一證券不可重複套用衝擊。')
        seen.update(affected)
        specs.append({'level':kind,'target':target,'change_pct':amount,'affected_count':len(affected),'_set':affected})
    xray=inspect_portfolio(dataset,weights_pct,capital)
    shock_map={stock:float(sh['change_pct']) for sh in specs for stock in sh['_set']}
    rows=[]
    for row in xray['holdings']:
        shock=shock_map.get(row['ticker'],0)
        pp=row['exposure_pct']*shock/100
        rows.append({'ticker':row['ticker'],'exposure_pct':row['exposure_pct'],
                     'shock_pct':shock,'pnl_contribution_pp':round(pp,8),
                     'pnl_usd':round(capital*pp/100,2)})
    rows.sort(key=lambda r:(r['pnl_contribution_pp'],r['ticker']))
    total=sum(r['pnl_contribution_pp'] for r in rows)
    result=[{k:v for k,v in sh.items() if k!='_set'} for sh in specs]
    return {'ok':True,'tool':'run_stress_test','weights_pct':weights_pct,'capital_usd':capital,
            'applied_shocks':result,'holdings':rows,
            'modeled_pnl_usd':round(capital*total/100,2),
            'modeled_return_contribution_pct':round(total,7),
            'modeled_coverage_pct':xray['modeled_coverage_pct'],
            'unmodeled_pct':xray['unmodeled_pct'],
            'holdings_date_range':xray['holdings_date_range'],
            'note':'靜態個股／市場衝擊，未建模部位與追蹤誤差未計入。此為情境貢獻，不是價格預測。'}


def dispatch(dataset,weights,capital,request):
    if type(request) is not dict or set(request)!={'tool','arguments'}:
        raise ValueError('工具請求必須且只能包含 tool、arguments。')
    tool=request['tool'];args=request['arguments']
    if type(tool) is not str or type(args) is not dict:raise ValueError('工具名稱及 arguments 格式錯誤。')
    if tool=='clarify':
        if set(args)!={'question'} or not isinstance(args['question'],str) or not args['question'].strip():
            raise ValueError('clarify 必須提供 question。')
        return {'ok':True,'tool':'clarify','question':args['question']}
    if tool=='inspect_portfolio':
        if args:raise ValueError('inspect_portfolio 不接受額外參數。')
        return inspect_portfolio(dataset,weights,capital)
    if tool=='analyze_overlap':
        if set(args)!={'etf_a','etf_b'}:raise ValueError('analyze_overlap 須提供 etf_a、etf_b。')
        return analyze_overlap(dataset,args['etf_a'],args['etf_b'])
    if tool=='run_stress_test':
        if set(args)!={'shocks'}:raise ValueError('run_stress_test 只接受 shocks。')
        return run_stress_test(dataset,weights,capital,args['shocks'])
    raise ValueError(f'未授權的工具：{tool}')


def overlap_matrix(dataset):
    names=list(UNIVERSE);matrix=[]
    for a in names:
        row=[]
        for b in names:
            if a==b:row.append(100.0)  # diagonal signifies self, with no cross-date issue
            else:row.append(analyze_overlap(dataset,a,b)['observed_weighted_overlap_pct'])
        matrix.append(row)
    return names,matrix


def plot_exposure(xray):
    top=xray['holdings'][:12][::-1]
    fig,ax=plt.subplots(figsize=(9,5.2));ax.barh([r['ticker'] for r in top],[r['exposure_pct'] for r in top],color='#2E618B')
    ax.set_xlabel('Look-through portfolio exposure (%)');ax.set_title('Top 12 underlying securities',loc='left',pad=16)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();return fig


def plot_source(xray):
    top=xray['holdings'][:10][::-1]
    fig,ax=plt.subplots(figsize=(9,5.0));left=[0.0]*len(top)
    colors=plt.get_cmap('tab10')
    for i,etf in enumerate(UNIVERSE):
        vals=[r['sources_by_etf_pp'].get(etf,0) for r in top]
        if not any(v>0 for v in vals):continue
        ax.barh([r['ticker'] for r in top],vals,left=left,label=etf,color=colors(i))
        left=[l+v for l,v in zip(left,vals)]
    ax.set_title('Where does the exposure come from?',loc='left',pad=15)
    ax.set_xlabel('Portfolio exposure, percentage points');ax.legend(loc='lower right',ncol=4,fontsize=8)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);fig.tight_layout();return fig


def plot_matrix(dataset):
    names,mat=overlap_matrix(dataset)
    fig,ax=plt.subplots(figsize=(8.2,6.1));im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues')
    ax.set_xticks(range(len(names)),labels=names);ax.set_yticks(range(len(names)),labels=names)
    for i in range(len(names)):
        for j in range(len(names)):
            v=mat[i][j];ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=8,color='white' if v>65 else 'black')
    ax.set_title('Pairwise ETF holdings overlap (%)',loc='left',pad=15)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04);fig.tight_layout();return fig


def plot_stress(stress):
    rows=[r for r in stress['holdings'] if r['shock_pct']!=0]
    rows=sorted(rows,key=lambda r:abs(r['pnl_contribution_pp']),reverse=True)[:12]
    rows=rows[::-1]
    fig,ax=plt.subplots(figsize=(9,4.7))
    vals=[r['pnl_contribution_pp'] for r in rows]
    ax.barh([r['ticker'] for r in rows],vals,color=['#B85057' if x<0 else '#3C8672' for x in vals]);ax.axvline(0,color='grey',lw=.7)
    ax.set_title('Scenario return attribution',loc='left',pad=15)
    ax.set_xlabel('Contribution to portfolio return (percentage points)');fig.tight_layout();return fig


def table_from_xray(xray):
    rows=[]
    for r in xray['holdings'][:80]:
        out={'Ticker':r['ticker'],'Exposure (%)':r['exposure_pct'],'Value (USD)':r['amount_usd']}
        for etf in UNIVERSE:out[f'{etf} (pp)']=r['sources_by_etf_pp'].get(etf,0)
        rows.append(out)
    return pd.DataFrame(rows)


def human_summary(xray):
    dates=xray['holdings_date_range']
    return (f"### Look-through Analytics\n"
            f"**最大證券曝險：{xray['largest_ticker']} {xray['largest_exposure_pct']:.2f}%** · "
            f"**前兩大：{xray['top_two_pct']:.2f}%** · "
            f"**已辨識股票權重：{xray['modeled_coverage_pct']:.2f}%**\n\n"
            f"已辨識證券數：**{xray['holdings_count']:,}** · 未建模部分：**{xray['unmodeled_pct']:.2f}%**。"
            f"持股日期：**{dates['earliest']}～{dates['latest']}**"
            f"（差距 {dates['days']} 天）。跨日期資料不能視為同一日的持股。")

ROUTER_V1 = """你是金融工具選擇器。根據使用者問題，回傳 JSON：tool 與 arguments。
工具有 inspect_portfolio、analyze_overlap、run_stress_test、clarify。
只回傳 JSON，不要寫文字。"""

ROUTER_V3 = """你是 Portfolio X-Ray 的金融分析 Tool Router。只輸出一個合法 JSON object，
精確包含 tool 和 arguments 兩個鍵，不可輸出 Markdown 或其他文字。
工具規格：
1. {"tool":"inspect_portfolio","arguments":{}}：查整體個股穿透、最大持股、來源拆解、集中度。
2. {"tool":"analyze_overlap","arguments":{"etf_a":"VOO","etf_b":"QQQ"}}：比較兩檔不同 ETF 的持股重疊。
   基金只能選 VOO, QQQ, IVV, VTI, VUG, VTV, SCHD, IWM。
3. {"tool":"run_stress_test","arguments":{"shocks":[{"level":"stock","target":"NVDA","change_pct":-20}]}}：個股衝擊。
   可同時有 1～5 筆獨立股票衝擊；股票代碼請採大寫；跌幅 -20 代表跌 20%。
   若整體所有已建模股票同幅變動，使用 {"level":"market","target":"ALL","change_pct":-10}。
4. {"tool":"clarify","arguments":{"question":"具體追問"}}：條件不足或超出資料範圍時使用。
注意：資料沒有逐檔股票產業分類，不支援 sector shock。如果使用者只說「科技股跌 20%」，請追問股票代碼。
壓力測試必須有明確數值。若只選了兩檔 ETF 且問題問兩檔重疊，可依已選 ETF 自動帶入。
ETF 權重與本金已由 UI 傳入 Python，不需放入 arguments。工具計算由 Python 執行。
範例：
輸入「VOO 與 QQQ 持股重疊多少？」→ {"tool":"analyze_overlap","arguments":{"etf_a":"VOO","etf_b":"QQQ"}}
輸入「NVDA 下跌 20% 且 AAPL 下跌 30%？」→ {"tool":"run_stress_test","arguments":{"shocks":[{"level":"stock","target":"NVDA","change_pct":-20},{"level":"stock","target":"AAPL","change_pct":-30}]}}
輸入「全部股票一起跌 10%」→ {"tool":"run_stress_test","arguments":{"shocks":[{"level":"market","target":"ALL","change_pct":-10}]}}
輸入「科技類股跌 20%」→ {"tool":"clarify","arguments":{"question":"目前沒有完整逐股產業分類。請指定想測試的股票代號，例如 NVDA、MSFT。"}}
"""


def get_groq_api_key():
    val=os.environ.get('GROQ_API_KEY','').strip()
    if val:return val
    try:
        from google.colab import userdata
        return (userdata.get('Groq') or '').strip()
    except Exception: return ''


def create_llm_client():
    key=get_groq_api_key()
    if not key:raise RuntimeError('請先於 Colab Secrets 設定 Groq API 金鑰（名稱：Groq）。')
    from openai import OpenAI
    return OpenAI(api_key=key,base_url='https://api.groq.com/openai/v1',
                  timeout=30,max_retries=1)


def _raw_call(client,messages):
    res=client.chat.completions.create(model='openai/gpt-oss-120b',temperature=0,
                                        messages=messages)
    return res.choices[0].message.content.strip()


def parse_tool_json(raw):
    text=str(raw).strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text)
        text=re.sub(r'\s*```$','',text)
    obj=json.loads(text)
    if not isinstance(obj,dict):raise ValueError('Tool Router 輸出必須是 JSON object。')
    return obj


def compact_financial_result(result):
    tool=result.get('tool')
    if tool=='inspect_portfolio':
        return ({k:result[k] for k in ('tool','capital_usd','largest_ticker','largest_exposure_pct',
                                     'top_two_pct','modeled_coverage_pct','unmodeled_pct','holdings_date_range')}
               | {'top_holdings':result['holdings'][:12]})
    if tool=='analyze_overlap':
        return {k:v for k,v in result.items() if k!='common_holdings'} | {'top_common_holdings':result['common_holdings'][:12]}
    if tool=='run_stress_test':
        return {k:v for k,v in result.items() if k!='holdings'} | {'largest_contributions':
                  sorted(result['holdings'],key=lambda x:abs(x['pnl_contribution_pp']),reverse=True)[:12]}
    return result


def run_agent(dataset,weights,capital,user_question,version='V3'):
    weights={k:float(v) for k,v in weights.items()}
    record={'status':'STARTED','question':user_question,'weights_pct':weights,
            'capital_usd':capital,'prompt_version':version,
            'dataset_generated_at_utc':dataset['generated_at_utc'],
            'data_provider':dataset['data_provider'],'raw_tool_json':None,
            'request':None,'tool_result':None,'answer':None,'error':None}
    if not str(user_question).strip():
        record.update(status='INPUT_ERROR',error='請先輸入分析問題。')
        return record
    system_prompt=ROUTER_V1 if version=='V1' else ROUTER_V3
    active=', '.join(k for k,v in weights.items() if v>0)
    try:
        client=create_llm_client()
        raw=_raw_call(client,[{'role':'system','content':system_prompt},
            {'role':'user','content':f'目前持有 ETF：{active}；問題：{user_question}'}])
        record['raw_tool_json']=raw
    except Exception as e:
        record.update(status='API_ERROR',error=f'{type(e).__name__}: {str(e)[:280]}')
        return record
    try:
        req=parse_tool_json(raw)
        record['request']=req
    except Exception as e:
        record.update(status='FORMAT_ERROR',error=f'{type(e).__name__}: {e}')
        return record
    try:
        result=dispatch(dataset,weights,capital,req)
        record['tool_result']=result
    except Exception as e:
        record.update(status='ARGUMENT_ERROR',error=f'{type(e).__name__}: {e}')
        return record
    if req['tool']=='clarify':
        record.update(status='SUCCESS',answer=result['question'])
        return record
    explain_prompt=("你是嚴謹的財務分析師。使用繁體中文簡要回答使用者的問題，引用以下 Python 工具實際計算結果。"
        "所有百分比、持股、損益數字必須取自結果；不得自行補數字或猜測未提供的持股。"
        "ETF 的持股日期可能不同；說明資料日期與覆蓋率，不要將情境損益寫成預測。"
        "僅有股票代碼，不代表已知公司名稱或產業。回答優先控制在 130～200 字。")
    try:
        answer=_raw_call(client,[{'role':'system','content':explain_prompt},
              {'role':'user','content':f'使用者問題：{user_question}\n分析工具回傳：\n'
               +json.dumps(compact_financial_result(result),ensure_ascii=False,allow_nan=False)}])
        record.update(status='SUCCESS',answer=answer)
    except Exception as e:
        # Keep verified numerical result and display it; do not fabricate a model answer.
        record.update(status='ANSWER_API_ERROR',error=f'{type(e).__name__}: {str(e)[:280]}')
    return record


def save_trace(trace):
    fd,name=tempfile.mkstemp(prefix='portfolio_xray_v3_trace_',suffix='.json')
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        json.dump(trace,f,ensure_ascii=False,indent=2,allow_nan=False)
    return name


def run_financial_tests(dataset):
    """All tests use the actual 8-fund snapshot; no fictional ETFs in the app or test data."""
    quality=validate_snapshot(dataset)
    rows=[]
    def one(label,fn):
        try:fn(); rows.append({'Test':label,'Status':'PASS','Details':''})
        except Exception as e:rows.append({'Test':label,'Status':'FAIL','Details':str(e)[:180]})
    def near(x,y,tol=.05):
        assert abs(x-y)<=tol,(x,y)
    base={k:(60.0 if k=='VOO' else 40.0 if k=='QQQ' else 0.0) for k in UNIVERSE}
    x=inspect_portfolio(dataset,base,100000)
    one('T01 / Eight real fund rows',lambda: (len(quality)==8) or (_ for _ in ()).throw(AssertionError('missing funds')))
    one('T02 / Equity exposure & unmodeled add to approximately 100%',lambda:near(x['modeled_coverage_pct']+x['unmodeled_pct'],100,0.8))
    one('T03 / Exposures equal source contributions',lambda: [near(sum(r['sources_by_etf_pp'].values()),r['exposure_pct'],1e-4) for r in x['holdings']])
    one('T04 / VOO-QQQ overlap symmetric',lambda:near(analyze_overlap(dataset,'VOO','QQQ')['observed_weighted_overlap_pct'],analyze_overlap(dataset,'QQQ','VOO')['observed_weighted_overlap_pct']))
    one('T05 / Overlap within [0,100]',lambda: 0<=analyze_overlap(dataset,'VOO','QQQ')['observed_weighted_overlap_pct']<=100 or (_ for _ in ()).throw(AssertionError('out of range')))
    m=run_stress_test(dataset,base,100000,[{'level':'market','target':'ALL','change_pct':-10}])
    one('T06 / Market shock conservation',lambda:near(m['modeled_return_contribution_pct'],-.1*x['modeled_coverage_pct']))
    one('T07 / Stress contributions add up',lambda:near(sum(r['pnl_contribution_pp'] for r in m['holdings']),m['modeled_return_contribution_pct'],1e-4))
    a=run_stress_test(dataset,base,100000,[{'level':'stock','target':'NVDA','change_pct':-20}])
    one('T08 / NVDA stress = exposure × shock',lambda:near(a['modeled_return_contribution_pct'],-.2*next(z['exposure_pct'] for z in x['holdings'] if z['ticker']=='NVDA')))
    def fails(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Expected rejection')
    one('T09 / Reject invalid ETF weight total',lambda:fails(lambda:normalize_allocation(pd.DataFrame({'ETF':UNIVERSE,'Weight (%)':[60,50,0,0,0,0,0,0]}),100000)))
    one('T10 / Reject invented tool name',lambda:fails(lambda:dispatch(dataset,base,100000,{'tool':'execute_shell','arguments':{}})))
    one('T11 / Reject unsupported sector shock',lambda:fails(lambda:dispatch(dataset,base,100000,{'tool':'run_stress_test','arguments':{'shocks':[{'level':'sector','target':'Technology','change_pct':-20}]}})))
    one('T12 / Reject duplicated shock target',lambda:fails(lambda:run_stress_test(dataset,base,100000,[{'level':'stock','target':'NVDA','change_pct':-20},{'level':'stock','target':'NVDA','change_pct':-30}])))
    one('T13 / Reject impossible shock (<-100%)',lambda:fails(lambda:run_stress_test(dataset,base,100000,[{'level':'stock','target':'NVDA','change_pct':-110}])))
    return pd.DataFrame(rows)


ROUTING_CASES = [
 ('A01','我目前持有最多的是哪檔股票？','inspect_portfolio'),
 ('A02','VOO 和 QQQ 哪些底層持股重複？','analyze_overlap'),
 ('A03','假設 NVDA 下跌 20%，我會虧多少？','run_stress_test'),
 ('A04','如果 NVDA 跌 20%、AAPL 跌 30%，我損失多少？','run_stress_test'),
 ('A05','全部股票都下跌 10% 的情境呢？','run_stress_test'),
 ('A06','科技股全部下跌 20%，我會損失多少？','clarify'),
 ('A07','金融股大跌，我會怎麼樣？','clarify'),
 ('A08','IVV 與 VOO 的重疊率有多少？','analyze_overlap'),
]


def run_routing_comparison(dataset,weights,capital,pause_sec=1.5):
    """Offline tool evaluation can't establish actual LLM accuracy. Requires real Groq API."""
    client=create_llm_client()
    results=[]
    for i,(case,query,expected) in enumerate(ROUTING_CASES):
        for version in (('V1','V3') if i%2==0 else ('V3','V1')):
            output={'Case':case,'Version':version,'Expected':expected,'Selected':None,
                    'Status':'API_ERROR','Correct':None,'Runnable':None,'Error':''}
            try:
                sys=ROUTER_V1 if version=='V1' else ROUTER_V3
                raw=_raw_call(client,[{'role':'system','content':sys},
                        {'role':'user','content':f'持有 VOO、QQQ。問題：{query}'}])
                req=parse_tool_json(raw)
                output['Selected']=req.get('tool')
                output['Correct']=req.get('tool')==expected
                try:
                    dispatch(dataset,weights,capital,req)
                    output['Runnable']=True
                    output['Status']='SUCCESS' if output['Correct'] else 'ROUTING_ERROR'
                except Exception as e:
                    output['Runnable']=False
                    output['Status']='ARGUMENT_ERROR' if output['Correct'] else 'ROUTING_ERROR'
                    output['Error']=str(e)[:180]
            except (json.JSONDecodeError,ValueError) as e:
                output['Status']='FORMAT_ERROR';output['Error']=str(e)[:180]
            except Exception as e:
                output['Status']='API_ERROR';output['Error']=f'{type(e).__name__}: {str(e)[:180]}'
            results.append(output)
            time.sleep(pause_sec)
    return pd.DataFrame(results)


def build_gradio(dataset):
    import gradio as gr
    validate_snapshot(dataset)
    default_alloc=pd.DataFrame({'ETF':list(UNIVERSE),'Weight (%)':[60,40,0,0,0,0,0,0]})
    default_w={t:(60.0 if t=='VOO' else 40.0 if t=='QQQ' else 0.0) for t in UNIVERSE}
    base=inspect_portfolio(dataset,default_w,100000)
    metadata=(f"**資料供應：** {dataset['data_provider']} · **快照建置：** {dataset['generated_at_utc'][:19]} UTC · "
             f"**持股日期範圍：** {dataset['earliest_holdings_as_of']} ～ {dataset['latest_holdings_as_of']} "
             f"（差距 {dataset['date_range_days']} 天）")
    css='''
    .gradio-container {max-width: 1600px !important;}
    #banner {padding: 14px 20px; border-radius: 12px; background: #12213a; color:white;}
    #banner h1,#banner p {color:white !important;}
    '''
    with gr.Blocks(title='PORTFOLIO X-RAY 3.0') as demo:
        gr.Markdown('''# PORTFOLIO X-RAY 3.0
**Look Through Your ETF · Stress Test What You Actually Own**

Real ETF Holdings · Tool Use Agent · Eight-Fund Portfolio Analytics''',elem_id='banner')
        gr.Markdown(metadata)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4,min_width=320):
                gr.Markdown('### 01｜Portfolio Setup')
                alloc=gr.Dataframe(value=default_alloc,headers=['ETF','Weight (%)'],row_count=8,
                                   column_count=2,datatype=['str','number'],type='pandas',
                                   interactive=True,label='ETF Portfolio Allocation (%)',
                                   max_height=360)
                gr.Markdown('**權重總和需為 100%；不持有者填 0。** 保留八個 ETF 代碼；請勿修改第一欄。')
                capital=gr.Number(label='Investment Capital (USD, nominal)',value=100000,
                                  minimum=1,maximum=1e12,precision=2)
                gr.Markdown('### 02｜Ask the AI Analyst')
                query=gr.Textbox(label='自然語言分析問題',lines=3,value='VOO 和 QQQ 的底層持股重疊多少？')
                prompt=gr.Dropdown(label='Agent Prompt Version',choices=['V3','V1'],value='V3')
                analyze_btn=gr.Button('Analyze Portfolio',variant='primary',size='lg')
                status=gr.Markdown('等待分析；目前圖表來自已驗證的真實快照金融計算。')
                gr.Markdown('''**問題範例**
- 我真正持有最多的股票是誰？
- VOO 與 IVV 的共同持股重疊多少？
- NVDA 跌 20%、AAPL 跌 30% 的情境損益如何？
- 如果所有已建模股票都下跌 10% 呢？

**情境限制：** 目前只有股票代碼及權重，不支援產業分類衝擊。''')
                gr.Markdown('### 03｜Data Provenance')
                gr.Dataframe(value=validate_snapshot(dataset),interactive=False,
                             label='ETF data quality and source dates',max_height=340)
            with gr.Column(scale=7,min_width=450):
                with gr.Tabs():
                    with gr.Tab('Financial Results'):
                        summary=gr.Markdown(value=human_summary(base))
                        main_chart=gr.Plot(value=plot_exposure(base),label='Look-through Exposure',show_label=False)
                        second_chart=gr.Plot(value=plot_source(base),label='ETF Contribution',show_label=False)
                        gr.Markdown('**Pairwise Overlap Matrix**｜全部 8 檔 ETF 兩兩比較；非同日持股比較需閱讀日期差異。')
                        matrix_chart=gr.Plot(value=plot_matrix(dataset),label='ETF Overlap Heatmap',show_label=False)
                        breakdown=gr.Dataframe(value=table_from_xray(base),interactive=False,
                                              label='Underlying Exposure by ETF (top 80)',max_height=390)
                    with gr.Tab('AI Conversation'):
                        conversation=gr.Chatbot(label='AI Financial Analyst',value=[],height=470)
                        gr.Markdown('模型回答應依據 Python 計算數字。缺少 Groq Key 時不會產生假回答。')
                    with gr.Tab('Agent Execution Trace'):
                        trace_view=gr.JSON(label='LLM request → Python dispatcher → verified results')
                        trace_file=gr.File(label='Download full JSON trace',interactive=False)
        def action(alloc_table,amount,question,prompt_ver):
            try:
                weights,cap=normalize_allocation(alloc_table,amount)
                x=inspect_portfolio(dataset,weights,cap)
                record=run_agent(dataset,weights,cap,question,prompt_ver)
                main=plot_exposure(x)
                if record['status']=='SUCCESS' and record['request']['tool']=='run_stress_test':
                    main=plot_stress(record['tool_result'])
                stat=record['status']
                answer=record.get('answer') or (record.get('error') or '未取得回答')
                if stat=='ANSWER_API_ERROR':
                    answer='已完成 Python 金融工具運算，但第二階段模型解讀失敗；請查看 Agent Trace。'
                trace_path=save_trace(record)
                note=human_summary(x)
                if record['status']=='SUCCESS':
                    tool=record['request']['tool']
                    if tool=='analyze_overlap':
                        r=record['tool_result'];note+=(f"\n\n**ETF Overlap：{r['etf_a']} × {r['etf_b']}** · "
                            f"已觀察重疊 **{r['observed_weighted_overlap_pct']:.2f}%** · "
                            f"共同代碼 {r['common_tickers_count']} 檔 · "
                            f"持股日期相差 {r['date_gap_days']} 天。")
                    elif tool=='run_stress_test':
                        r=record['tool_result'];note+=(f"\n\n**Scenario P/L：{r['modeled_return_contribution_pct']:+.3f}%** "
                            f"（{r['modeled_pnl_usd']:+,.2f} USD）；未建模曝險 {r['unmodeled_pct']:.2f}%。")
                logs=f'**執行狀態：{stat}**。' + (f'錯誤：{record["error"]}' if record.get('error') else '完整請求及回覆已保存在 Trace。')
                return note,main,plot_source(x),table_from_xray(x),[
                    {'role':'user','content':question},{'role':'assistant','content':answer}],record,trace_path,logs
            except Exception as e:
                msg=f'**INPUT_ERROR｜{type(e).__name__}：{e}**'
                return msg,None,None,pd.DataFrame(),[],{'status':'INPUT_ERROR','error':str(e)},None,msg
        analyze_btn.click(fn=action,inputs=[alloc,capital,query,prompt],
                          outputs=[summary,main_chart,second_chart,breakdown,
                                   conversation,trace_view,trace_file,status],show_progress='full')
    return demo
