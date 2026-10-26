"""Map-only Colab application for historical accidents and baseline predictions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from IPython.display import IFrame, display


def show_risk_map(project_dir, max_roads=6000):
    """Display an HTML map app and save it to road_safety_explorer.html.

    Uses trusted local prediction exports. Internet access is needed for Plotly
    and map tiles. Road shapes are approximated by endpoint connections.
    """
    root = Path(project_dir)
    paths = sorted((root / "results").glob("*/seed*_test_predictions.pt"))
    states = {}
    available = {p.parent.name.split("_")[0] for p in paths}
    available.update(s for s in ("DE", "MD")
                     if (root / "data" / s / "accidents_monthly.csv").exists())
    if not available:
        raise FileNotFoundError("No state data or prediction exports found")
    for state in sorted(available):
        state_paths = [p for p in paths if p.parent.name.startswith(state + "_")]
        if state_paths:
            first = torch.load(state_paths[0], map_location="cpu", weights_only=False)
        else:
            adj = torch.load(root / "data" / state / "adj_matrix.pt",
                             map_location="cpu", weights_only=False).coalesce()
            uv = adj.indices()[:, adj.values() != 0].numpy()
            keys = np.minimum(*uv) * adj.shape[0] + np.maximum(*uv)
            _, representative = np.unique(keys, return_index=True)
            first = {"road_index": uv[:, representative]}
        endpoints = np.asarray(first["road_index"])
        coords = first.get("node_coordinates")
        if coords is None:
            nodes = pd.read_csv(next((root / "data" / state / "Nodes").glob("*.csv")))
            coords = nodes[["lat", "lon"]].to_numpy()  # Released columns are reversed.
        coords = np.asarray(coords)
        # Some training exports preserve the reversed source-column ordering.
        # These states have negative longitudes (~-79 to -75) and positive
        # latitudes (~38 to 40); broad global bounds cannot detect the swap.
        center = np.nanmedian(coords, axis=0)
        if 35 < center[0] < 45 and -85 < center[1] < -70:
            coords = coords[:, ::-1].copy()
        center = np.nanmedian(coords, axis=0)
        if not (-85 < center[0] < -70 and 35 < center[1] < 45):
            raise ValueError(f"{state}: unexpected longitude/latitude centre {center.tolist()}")
        print(f"{state}: longitude {center[0]:.3f}, latitude {center[1]:.3f}")
        n = len(coords)
        keys = np.minimum(*endpoints) * n + np.maximum(*endpoints)
        frame = pd.read_csv(root / "data" / state / "accidents_monthly.csv")
        if state == "DE":
            frame = frame.loc[~((frame.year == 2022) & (frame.month == 10))].copy()
        a, b = frame.node_1_idx.to_numpy(), frame.node_2_idx.to_numpy()
        frame["road"] = np.minimum(a, b) * n + np.maximum(a, b)
        frame["date"] = pd.to_datetime(frame[["year", "month"]].assign(day=1)).dt.strftime("%Y-%m")
        counts = frame.groupby(["date", "road"]).acc_count.sum()
        totals = counts.groupby("road").sum().reindex(keys, fill_value=0).to_numpy()
        budget = min(max_roads, len(keys))
        background = np.random.default_rng(42).choice(len(keys), budget // 3, replace=False)
        ids = np.unique(np.r_[background, np.argsort(totals)[-(budget-len(background)):]])
        dates = sorted(frame.date.unique())
        actual, summary = {}, {}
        for date in dates:
            values = counts.loc[date].reindex(keys, fill_value=0).to_numpy()
            actual[date] = values[ids].astype(int).tolist()
            summary[date] = {"total": int(values.sum()), "positive": int((values > 0).sum())}
        period_summary = {}
        dated = counts.reset_index()
        for frequency in ("Q", "Y"):
            dated["period"] = pd.to_datetime(dated["date"]).dt.to_period(frequency).astype(str)
            for period, group in dated.groupby("period"):
                per_road = group.groupby("road").acc_count.sum().reindex(keys, fill_value=0)
                period_summary[period] = {"total": int(per_road.sum()), "positive": int(per_road.gt(0).sum())}
        runs = {}
        for path in state_paths:
            p = torch.load(path, map_location="cpu", weights_only=False)
            if not np.array_equal(np.asarray(p["road_index"]), endpoints):
                raise ValueError(f"Road ordering differs in {path}")
            task = "classification" if "classification" in path.parent.name else "regression"
            predictions, statistics, aggregates = {}, {}, {}
            for i, (year, month) in enumerate(p["months"]):
                date = f"{int(year)}-{int(month):02d}"
                if date not in actual:
                    continue
                pred = np.asarray(p["pred"][i]).reshape(-1)
                observed = np.asarray(p["y"][i]).reshape(-1)
                predictions[date] = np.round(pred[ids], 5).tolist()
                statistics[date] = {"total": float(pred.sum()), "mae": float(np.abs(pred-observed).mean())}
                if task == "regression":
                    for period in (str(year), f"{int(year)}Q{(int(month)-1)//3+1}"):
                        if period not in aggregates:
                            aggregates[period] = [np.zeros_like(pred), np.zeros_like(observed)]
                        aggregates[period][0] += pred
                        aggregates[period][1] += observed
            for period, (pred, observed) in aggregates.items():
                statistics[period] = {"total": float(pred.sum()), "mae": float(np.abs(pred-observed).mean()),
                                      "observed": float(observed.sum()), "positive": int((observed > 0).sum())}
            runs[f"{path.parent.name} · {path.stem.split('_')[0]}"] = {
                "task": task, "pred": predictions, "stats": statistics}
        xy = coords[endpoints[:, ids]].transpose(1, 0, 2)
        states[state] = {"dates": dates, "actual": actual, "summary": summary, "periodSummary": period_summary,
                         "runs": runs, "roads": endpoints[:, ids].T.tolist(),
                         "xy": xy.tolist(), "roadCount": len(keys),
                         "center": center.tolist()}

    document = _APP.replace("__DATA__", json.dumps(states, separators=(",", ":"), allow_nan=False))
    destination = root / "road_safety_explorer.html"
    destination.write_text(document, encoding="utf-8")
    # Serve the document rather than embedding a large srcdoc frame. Colab's
    # output proxy gives scripts and map tiles a normal browser document context.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    previous = getattr(show_risk_map, "_server", None)
    if previous is not None:
        previous.shutdown()
        previous.server_close()
    content = document.encode("utf-8")

    class AppHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), AppHandler)
    show_risk_map._server = server
    Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        from google.colab import output
    except ImportError:
        display(IFrame(f"http://127.0.0.1:{port}", width="100%", height=860))
    else:
        output.serve_kernel_port_as_iframe(port, height=860)
    print(f"Saved standalone application: {destination}")
    return destination


_APP = r'''<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#152238;font:14px system-ui,sans-serif}
header{height:86px;background:#101e34;color:white;padding:20px 26px;display:flex;justify-content:space-between;align-items:center}
h1{font-size:23px;margin:0;letter-spacing:-.5px}header p{margin:5px 0;color:#aebed3;font-size:12px}.badge{color:#85e0ce;border:1px solid #355469;padding:7px 12px;border-radius:20px;font-size:11px}
main{display:grid;grid-template-columns:250px 1fr;gap:16px;padding:18px}aside,.card{background:white;border:1px solid #e1e7ef;border-radius:12px}
aside{padding:20px;height:700px}label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin:22px 0 8px}label:first-child{margin-top:0}
select{width:100%;padding:10px 8px;border:1px solid #d8e0ec;border-radius:7px;background:white;color:#1e293b;font-size:12px}button{border:1px solid #d8e0ec;background:white;border-radius:7px;padding:9px;cursor:pointer;color:#334155}
.nav{display:flex;gap:8px;margin-top:9px}.nav button{flex:1}.note{font-size:11px;color:#64748b;line-height:1.7;margin-top:24px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}.stat{padding:15px 18px}.stat span{display:block;color:#64748b;font-size:11px}.stat strong{display:block;font-size:24px;margin-top:5px;font-weight:650}.maphead{display:flex;justify-content:space-between;padding:16px 18px;border-bottom:1px solid #edf1f6}.maphead b{font-size:15px}.maphead span{font-size:11px;color:#64748b}#map{height:520px}.foot{padding:10px 18px;font-size:11px;color:#64748b;line-height:1.6}#status{color:#b45309;padding-top:8px} @media(max-width:750px){main{grid-template-columns:1fr}aside{height:auto}.stats{gap:5px}.stat strong{font-size:18px}}
</style></head><body>
<header><div><h1>Road Safety Explorer</h1><p>Historical accidents and baseline model predictions</p></div><div class="badge">CS5284 · GROUP 01</div></header>
<main><aside><label>State</label><select id="state"></select><label>Basemap</label><select id="basemap"><option value="carto-darkmatter">Dark · high contrast</option><option value="carto-positron">Light · minimal</option><option value="open-street-map">Street detail</option></select><label>Map layer</label><select id="view"><option value="observed">Recorded accidents</option><option value="predicted">Model prediction</option><option value="error">Prediction error</option></select><label>Model / seed</label><select id="model"></select><label>Aggregation</label><select id="aggregation"><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><label>Period</label><select id="month"></select><div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div><label>Road details</label><div id="details" class="note">Hover over a road to inspect its values.</div><div class="note">Historical layers cover the retained data period. Predictions cover exported test months only. Opposite directions are combined into physical-road proxies.</div></aside>
<section><div class="stats"><div class="card stat"><span>RECORDED ACCIDENTS</span><strong id="total">—</strong></div><div class="card stat"><span>ROADS WITH ACCIDENTS</span><strong id="positive">—</strong></div><div class="card stat"><span id="thirdLabel">MODEL</span><strong id="third">—</strong></div></div><div class="card"><div class="maphead"><b id="title">Accident map</b><span id="period"></span></div><div id="map"></div><div class="foot" id="foot"></div></div><div id="status"></div></section></main>
<script>
const DATA=__DATA__, $=id=>document.getElementById(id), fmt=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:2});
let PERIODS={};
function periodKey(date){const mode=$('aggregation').value;return mode==='yearly'?date.slice(0,4):mode==='quarterly'?date.slice(0,4)+'Q'+Math.ceil(Number(date.slice(5))/3):date}
function options(el,values){el.replaceChildren(...values.map(x=>new Option(x,x)))}
function stateChanged(){const s=DATA[$('state').value],names=Object.keys(s.runs);options($('model'),names.length?names:['No prediction exports']);
for(const v of ['predicted','error'])$('view').querySelector(`[value=${v}]`).disabled=!names.length;
if(!names.length)$('view').value='observed';datesChanged();}
function datesChanged(){const s=DATA[$('state').value],r=s.runs[$('model').value]||{task:'none',pred:{}},old=$('month').value;
const classification=r.task==='classification'&&$('view').value!=='observed';$('aggregation').disabled=classification;if(classification)$('aggregation').value='monthly';
const dates=$('view').value==='observed'?s.dates:Object.keys(r.pred).filter(d=>s.actual[d]);PERIODS={};dates.sort().forEach(d=>{const k=periodKey(d);(PERIODS[k]||=[]).push(d)});
options($('month'),Object.keys(PERIODS));for(const option of $('month').options){const ds=PERIODS[option.value],expected=$('aggregation').value==='yearly'?12:$('aggregation').value==='quarterly'?3:1;option.text=option.value+(expected>1?' · '+ds[0]+' to '+ds[ds.length-1]+(ds.length<expected?' (partial)':''):'')}
if([...$('month').options].some(x=>x.value===old))$('month').value=old;
$('model').disabled=$('view').value==='observed';$('view').querySelector('[value=error]').disabled=r.task!=='regression';if(r.task!=='regression'&&$('view').value==='error'){$('view').value='predicted';datesChanged();return}draw();}
function draw(){try{if(!window.Plotly)throw Error('Plotly could not load. Check internet access or open the saved HTML in a browser.');
const s=DATA[$('state').value],r=s.runs[$('model').value]||{task:'none',pred:{},stats:{}},date=$('month').value,view=$('view').value,ds=PERIODS[date]||[],reg=r.task==='regression';
const sumArrays=arrays=>arrays.reduce((acc,row)=>acc.map((x,i)=>x+row[i]),Array(s.roads.length).fill(0));
const a=ds.length?sumArrays(ds.map(d=>s.actual[d])):null;
const p=ds.length&&ds.every(d=>r.pred[d])&&(reg||ds.length===1)?sumArrays(ds.map(d=>r.pred[d])):null;
if(!a)throw Error('No month is available for this selection.');
const vals=a.map((x,i)=>view==='observed'?x:view==='error'?p[i]-x:p[i]),error=view==='error';
const reference=(reg&&p)?a.concat(p):vals,positive=reference.filter(x=>x>0).sort((x,y)=>x-y);
let limit=positive.length?positive[Math.floor((positive.length-1)*.98)]:1;
if(error){const sorted=vals.map(Math.abs).sort((x,y)=>x-y);limit=Math.max(.1,sorted[Math.floor((sorted.length-1)*.98)]);}limit=Math.max(limit,.01);
const colors=error?['#2166ac','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#ef8a62','#b2182b']:(basemap.value==='carto-darkmatter'?['#475569','#22d3ee','#2dd4bf','#a3e635','#fde047','#fb923c','#fb4664']:['#cbd5e1','#7fcdbb','#41b6c4','#1d91c0','#fdae61','#f46d43','#a50026']);
const bins=colors.map(()=>({lon:[],lat:[]}));vals.forEach((v,i)=>{let f=error?(v/limit+1)/2:Math.log1p(Math.max(0,v))/Math.log1p(limit);let k=Math.min(colors.length-1,Math.max(0,Math.floor(f*colors.length)));let b=bins[k],xy=s.xy[i];b.lon.push(xy[0][0],xy[1][0],null);b.lat.push(xy[0][1],xy[1][1],null)});
const traces=bins.map((b,i)=>({type:'scattermap',mode:'lines',lon:b.lon,lat:b.lat,line:{color:colors[i],width:i===0&&!error?1.2:4},hoverinfo:'skip',showlegend:false}));
const transformed=vals.map(x=>error?x:Math.log1p(Math.max(0,x))),scale=colors.map((c,i)=>[i/(colors.length-1),c]);
traces.push({type:'scattermap',mode:'markers',lon:s.xy.map(x=>(x[0][0]+x[1][0])/2),lat:s.xy.map(x=>(x[0][1]+x[1][1])/2),marker:{size:4,color:transformed,colorscale:scale,cmin:error?-limit:0,cmax:error?limit:Math.log1p(limit),opacity:.35,colorbar:{thickness:12,title:{text:error?'Predicted − actual':'Count / score'},tickvals:error?[-limit,0,limit]:[0,Math.log1p(limit/2),Math.log1p(limit)],ticktext:error?[-limit,0,limit].map(fmt):[0,limit/2,limit].map(fmt)}},customdata:s.roads.map((road,i)=>[...road,a[i],p?p[i]:null]),hovertemplate:'<b>Road %{customdata[0]} ↔ %{customdata[1]}</b><br>Recorded: %{customdata[2]}<br>Prediction: %{customdata[3]:.3f}<extra></extra>',showlegend:false});
Plotly.react('map',traces,{map:{style:basemap.value,center:{lon:s.center[0],lat:s.center[1]},zoom:$('state').value==='DE'?8:6.7},margin:{l:0,r:0,t:0,b:0},uirevision:$('state').value,paper_bgcolor:'white',font:{family:'system-ui'}},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}).catch(e=>{$('status').textContent='Map rendering failed: '+e.message});
$('map').removeAllListeners('plotly_hover');$('map').on('plotly_hover',event=>{const x=event.points[0].customdata;if(x)$('details').textContent=`Road ${x[0]} ↔ ${x[1]} · Recorded: ${fmt(x[2])}${x[3]!==null?' · '+(reg?'Predicted count':'Risk score')+': '+fmt(x[3]):''}`});
const stats=r.stats[date],overview=s.summary[date]||s.periodSummary[date];
const predictedPeriod=view!=='observed'&&ds.length>1;
$('total').textContent=fmt(predictedPeriod?stats.observed:overview.total);$('positive').textContent=fmt(predictedPeriod?stats.positive:overview.positive);
$('thirdLabel').textContent=reg&&p?'PREDICTED TOTAL':'PREDICTION AVAILABILITY';$('third').textContent=reg&&p?fmt(r.stats[date].total):p?'Available':'Not exported';
$('title').textContent=view==='observed'?'Recorded accidents':error?'Prediction error':reg?'Predicted accident counts':'Classification risk scores';$('period').textContent=$('month').selectedOptions[0].text+' · '+$('state').value;
$('foot').textContent=`Showing ${fmt(s.roads.length)} of ${fmt(s.roadCount)} roads: a fixed background sample plus roads with the highest historical totals. Summary cards use all roads. Colours saturate at ${fmt(limit)}. Straight lines approximate road geometry.`;
$('status').textContent=r.task==='none'?'Historical observations only: no prediction exports found for this state.':reg?(p?'All-road MAE: '+fmt(r.stats[date].mae):'Historical observations only; predictions were not exported for this month.'):'Classification scores are not calibrated accident probabilities.';
}catch(e){$('status').textContent=e.message}}
options($('state'),Object.keys(DATA));$('state').onchange=stateChanged;$('model').onchange=datesChanged;$('view').onchange=datesChanged;$('month').onchange=draw;$('basemap').onchange=draw;$('aggregation').onchange=datesChanged;
for(const [id,step]of [['prev',-1],['next',1]])$(id).onclick=()=>{const m=$('month');m.selectedIndex=Math.max(0,Math.min(m.options.length-1,m.selectedIndex+step));draw()};stateChanged();
</script></body></html>'''



