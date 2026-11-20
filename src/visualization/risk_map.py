"""Notebook/Colab maps for full-period EDA and exported test predictions."""
from pathlib import Path
import html
import json
import re

import numpy as np
import pandas as pd
import torch
from IPython.display import IFrame, display


TRAINING_YEARS = {"DE": (2009, 2012), "MD": (2015, 2017)}


def list_prediction_exports(results_dir):
    """Find genuine exports, including report_ and report_history_ run names."""
    exports = []
    for path in sorted(Path(results_dir).rglob("seed*_test_predictions.pt")):
        config = {}
        for filename in ("metrics.json", "run_metadata.json"):
            metadata = path.parent / filename
            if metadata.exists():
                config = json.loads(metadata.read_text(encoding="utf-8")).get("config", {})
                break
        match = re.search(r"(?:^|_)(DE|MD)(?:_|$)", path.parent.name)
        state = config.get("state") or (match.group(1) if match else None)
        task = config.get("task") or ("classification" if "classification" in path.parent.name else
                                      "regression" if "regression" in path.parent.name else None)
        seed = re.fullmatch(r"seed(\d+)_test_predictions", path.stem)
        if state not in TRAINING_YEARS or task not in ("classification", "regression") or seed is None:
            raise ValueError(f"Cannot determine state, task and seed for {path}")
        exports.append({"path": path, "state": state, "task": task,
                        "run": path.parent.name, "seed": int(seed.group(1))})
    return exports


def build_risk_map_data(project_dir, max_roads=6000, *, mode="all", data_dir=None,
                        results_dir=None, state_codes=None, run_names=None, seeds=None):
    """Build map payloads without displaying HTML or opening a server.

    EDA uses all available accident months and includes compatible prediction
    exports when present. Results use only exported evaluation months and select
    roads from training counts.
    A fixed random background supplements high-count roads; cards cover all roads.
    ``all`` retains the original all-period historical explorer.
    """
    if mode not in ("all", "eda", "results"):
        raise ValueError("mode must be all, eda or results")
    if max_roads is not None and (not isinstance(max_roads, int) or max_roads < 1):
        raise ValueError("max_roads must be a positive integer or None")
    root = Path(project_dir)
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    results_dir = Path(results_dir) if results_dir is not None else root / "results"
    exports = list_prediction_exports(results_dir)
    exports = [e for e in exports if (run_names is None or e["run"] in run_names)
               and (seeds is None or e["seed"] in seeds)]
    metadata_by_path = {e["path"]: e for e in exports}
    states = {}
    available = {e["state"] for e in exports}
    if mode != "results":
        available.update(s for s in TRAINING_YEARS if (data_dir / s / "accidents_monthly.csv").exists())
    if state_codes is not None:
        available &= set(state_codes)
    if not available:
        raise FileNotFoundError("No prediction exports found" if mode == "results" else "No state data found")
    for state in sorted(available):
        state_paths = [e["path"] for e in exports if e["state"] == state]
        if state_paths:
            first = torch.load(state_paths[0], map_location="cpu", weights_only=False)
        else:
            adj = torch.load(data_dir / state / "adj_matrix.pt",
                             map_location="cpu", weights_only=False).coalesce()
            uv = adj.indices()[:, adj.values() != 0].numpy()
            keys = np.minimum(*uv) * adj.shape[0] + np.maximum(*uv)
            _, representative = np.unique(keys, return_index=True)
            first = {"road_index": uv[:, representative]}
        endpoints = np.asarray(first["road_index"])
        coords = first.get("node_coordinates")
        if coords is None:
            nodes = pd.read_csv(next((data_dir / state / "Nodes").glob("*.csv")))
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
        n = len(coords)
        keys = np.minimum(*endpoints) * n + np.maximum(*endpoints)
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"{state}: prediction/map roads must be unique physical roads")
        adjacency_path = data_dir / state / "adj_matrix.pt"
        if adjacency_path.exists():
            adj = torch.load(adjacency_path, map_location="cpu", weights_only=False).coalesce()
            uv = adj.indices()[:, adj.values() != 0].numpy()
            graph_keys, representative = np.unique(np.minimum(*uv) * n + np.maximum(*uv), return_index=True)
            if not np.array_equal(np.sort(keys), graph_keys):
                raise ValueError(f"{state}: exports do not cover the benchmark's physical-road population")
            # Canonical graph order makes the EDA/results subset identical even
            # when a valid prediction export stores roads in a different order.
            endpoints, keys = uv[:, representative], graph_keys
        frame = pd.read_csv(data_dir / state / "accidents_monthly.csv")
        if state == "DE" and mode != "eda":
            frame = frame.loc[~((frame.year == 2022) & (frame.month == 10))].copy()
        a, b = frame.node_1_idx.to_numpy(), frame.node_2_idx.to_numpy()
        frame["road"] = np.minimum(a, b) * n + np.maximum(a, b)
        if not frame["road"].isin(keys).all():
            raise ValueError(f"{state}: accident records contain roads outside the map population")
        frame["date"] = pd.to_datetime(frame[["year", "month"]].assign(day=1)).dt.strftime("%Y-%m")
        first_year, last_year = TRAINING_YEARS[state]
        training = frame[frame.year.between(first_year, last_year)]
        selection_frame = frame if mode == "eda" else training
        selection_totals = selection_frame.groupby("road").acc_count.sum().reindex(keys, fill_value=0).to_numpy()
        counts = frame.groupby(["date", "road"]).acc_count.sum()
        if max_roads is None or max_roads >= len(keys):
            ids = np.arange(len(keys))
        else:
            budget = min(max_roads, len(keys))
            background = np.random.default_rng(42).choice(len(keys), budget // 3, replace=False)
            ids = np.unique(np.r_[background, np.argsort(selection_totals)[-(budget-len(background)):]])
        dates = set(frame.date.unique())
        for path in (data_dir / state / "Nodes").glob("node_features_*.csv"):
            match = re.fullmatch(r"node_features_(\d{4})_(\d{1,2})", path.stem)
            if match:
                year, month = map(int, match.groups())
                if not (state == "DE" and (year, month) == (2022, 10) and mode != "eda"):
                    dates.add(f"{year}-{month:02d}")
        if mode == "results":
            exported_dates = set()
            for path in state_paths:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                exported_dates.update(f"{int(y)}-{int(m):02d}" for y, m in payload["months"])
            dates &= exported_dates
        dates = sorted(dates)
        frame = frame[frame.date.isin(dates)]
        counts = frame.groupby(["date", "road"]).acc_count.sum()
        actual, summary = {}, {}
        recorded_dates = set(frame.date)
        for date in dates:
            observed = counts.loc[date] if date in recorded_dates else pd.Series(dtype=float)
            values = observed.reindex(keys, fill_value=0).to_numpy()
            selected = values[ids].astype(int)
            nonzero = np.flatnonzero(selected)
            actual[date] = [nonzero.tolist(), selected[nonzero].tolist()]
            summary[date] = {"total": int(values.sum()), "positive": int((values > 0).sum())}
        period_summary = {}
        dated = counts.reset_index()
        for frequency in ("Q", "Y"):
            for period in pd.PeriodIndex(dates, freq="M").asfreq(frequency).astype(str):
                period_summary.setdefault(period, {"total": 0, "positive": 0})
            dated["period"] = pd.to_datetime(dated["date"]).dt.to_period(frequency).astype(str)
            for period, group in dated.groupby("period"):
                per_road = group.groupby("road").acc_count.sum().reindex(keys, fill_value=0)
                period_summary[period] = {"total": int(per_road.sum()), "positive": int(per_road.gt(0).sum())}
        runs = {}
        for path in state_paths:
            p = torch.load(path, map_location="cpu", weights_only=False)
            if p.get("target_unit", "physical") != "physical":
                raise ValueError(f"{path}: map requires physical-road prediction targets")
            uv = np.asarray(p["road_index"])
            prediction_keys = np.minimum(*uv) * n + np.maximum(*uv)
            if len(np.unique(prediction_keys)) != len(keys) or len(prediction_keys) != len(keys):
                raise ValueError(f"{path}: prediction road population differs from map roads")
            positions = pd.Index(prediction_keys).get_indexer(keys)
            if (positions < 0).any():
                raise ValueError(f"{path}: prediction roads do not match map roads")
            task = metadata_by_path[path]["task"]
            if np.shape(p["pred"]) != (len(p["months"]), len(keys)) or np.shape(p["y"]) != np.shape(p["pred"]):
                raise ValueError(f"{path}: prediction/target shapes do not match months and roads")
            if len(set(map(tuple, p["months"]))) != len(p["months"]):
                raise ValueError(f"{path}: duplicate prediction months")
            predictions, statistics, aggregates = {}, {}, {}
            for i, (year, month) in enumerate(p["months"]):
                date = f"{int(year)}-{int(month):02d}"
                if date not in actual:
                    continue
                pred = np.asarray(p["pred"][i]).reshape(-1)[positions]
                observed = np.asarray(p["y"][i]).reshape(-1)[positions]
                recorded = counts.loc[date] if date in recorded_dates else pd.Series(dtype=float)
                if not np.array_equal(observed, recorded.reindex(keys, fill_value=0).to_numpy()):
                    raise ValueError(f"{path}: exported targets disagree with recorded physical-road counts for {date}")
                if not np.isfinite(pred).all() or (pred < 0).any() or (task == "classification" and (pred > 1).any()):
                    raise ValueError(f"{path}: invalid prediction values")
                predictions[date] = np.round(pred[ids], 5).tolist()
                statistics[date] = {"observed": float(observed.sum()), "positive": int((observed > 0).sum())}
                if task == "regression":
                    statistics[date].update(total=float(pred.sum()), mae=float(np.abs(pred-observed).mean()))
                    for period in (str(year), f"{int(year)}Q{(int(month)-1)//3+1}"):
                        if period not in aggregates:
                            aggregates[period] = [np.zeros_like(pred), np.zeros_like(observed)]
                        aggregates[period][0] += pred
                        aggregates[period][1] += observed
            for period, (pred, observed) in aggregates.items():
                statistics[period] = {"total": float(pred.sum()), "mae": float(np.abs(pred-observed).mean()),
                                      "observed": float(observed.sum()), "positive": int((observed > 0).sum())}
            if predictions:
                runs[f"{path.parent.name} · {path.stem.split('_')[0]}"] = {
                    "task": task, "pred": predictions, "stats": statistics}
        if mode == "results" and not runs:
            continue
        xy = coords[endpoints[:, ids]].transpose(1, 0, 2)
        states[state] = {"dates": dates, "actual": actual, "summary": summary, "periodSummary": period_summary,
                         "runs": runs, "roads": endpoints[:, ids].T.tolist(),
                         "xy": xy.tolist(), "roadCount": len(keys),
                         "center": center.tolist(),
                         "sampleDescription": ("all roads" if len(ids) == len(keys) else
                             "fixed random background plus roads with highest full-period accident totals" if mode == "eda" else
                             f"fixed random background plus roads with highest {first_year}–{last_year} training accident totals")}
    if not states:
        raise FileNotFoundError("No prediction exports align with retained observation months")
    return states


def show_risk_map(project_dir, max_roads=6000, *, mode="all", data_dir=None,
                  results_dir=None, state_codes=None, run_names=None, seeds=None,
                  output_path=None, display_mode="auto"):
    """Save/display one map with a separate EDA or results purpose.

    A local server gives the map a normal HTTP origin for OpenStreetMap tiles;
    it runs until the notebook kernel stops. Colab uses its output proxy.
    The map library and basemap tiles need internet. ``display_mode='none'`` only saves.
    Multiple Colab maps keep independent servers, so a results map does not
    disconnect the earlier EDA map. Prediction files must be trusted local exports.
    """
    if display_mode not in ("auto", "inline", "server", "none"):
        raise ValueError("display_mode must be auto, inline, server or none")
    root = Path(project_dir)
    states = build_risk_map_data(root, max_roads, mode=mode, data_dir=data_dir,
        results_dir=results_dir, state_codes=state_codes, run_names=run_names, seeds=seeds)
    config = {"mode": mode}
    document = _APP.replace("__DATA__", json.dumps(states, separators=(",", ":"), allow_nan=False).replace("<", "\u003c"))
    document = document.replace("__CONFIG__", json.dumps(config))
    destination = Path(output_path) if output_path is not None else root / "road_safety_explorer.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    if display_mode == "none":
        return destination
    try:
        from google.colab import output
    except ImportError:
        output = None
    if display_mode == "inline":
        frame = ('<iframe title="Road Safety Explorer" width="100%" height="860" '
                 'style="border:0" srcdoc="' + html.escape(document, quote=True) + '"></iframe>')
        display({"text/html": frame}, raw=True)
        print(f"Saved standalone application: {destination}")
        return destination

    # Serve the document rather than embedding a large srcdoc frame. Colab's
    # output proxy gives scripts and map tiles a normal browser document context.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    servers = getattr(show_risk_map, "_servers", {})
    previous = servers.get(str(destination.resolve()))
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
    servers[str(destination.resolve())] = server
    show_risk_map._servers = servers
    Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    if output is None:
        display(IFrame(f"http://127.0.0.1:{port}", width="100%", height=860))
    else:
        output.serve_kernel_port_as_iframe(port, height=860)
    print(f"Saved standalone application: {destination}")
    return destination


_APP = r'''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#152238;font:14px system-ui,sans-serif}
header{height:86px;background:#101e34;color:white;padding:20px 26px;display:flex;justify-content:space-between;align-items:center}
h1{font-size:23px;margin:0;letter-spacing:-.5px}header p{margin:5px 0;color:#aebed3;font-size:12px}.badge{color:#85e0ce;border:1px solid #355469;padding:7px 12px;border-radius:20px;font-size:11px}
main{display:grid;grid-template-columns:250px 1fr;gap:16px;padding:18px}aside,.card{background:white;border:1px solid #e1e7ef;border-radius:12px}
aside{padding:20px;height:700px}label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin:22px 0 8px}label:first-child{margin-top:0}
select{width:100%;padding:10px 8px;border:1px solid #d8e0ec;border-radius:7px;background:white;color:#1e293b;font-size:12px}button{border:1px solid #d8e0ec;background:white;border-radius:7px;padding:9px;cursor:pointer;color:#334155}
.nav{display:flex;gap:8px;margin-top:9px}.nav button{flex:1}.note{font-size:11px;color:#64748b;line-height:1.7;margin-top:24px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}.stat{padding:15px 18px}.stat span{display:block;color:#64748b;font-size:11px}.stat strong{display:block;font-size:24px;margin-top:5px;font-weight:650}.maphead{display:flex;justify-content:space-between;padding:16px 18px;border-bottom:1px solid #edf1f6}.maphead b{font-size:15px}.maphead span{font-size:11px;color:#64748b}#map{height:520px}.foot{padding:10px 18px;font-size:11px;color:#64748b;line-height:1.6}#status{color:#b45309;padding-top:8px} @media(max-width:750px){main{grid-template-columns:1fr}aside{height:auto}.stats{gap:5px}.stat strong{font-size:18px}}
</style></head><body>
<header><div><h1 id="appTitle">Road Safety Explorer</h1><p id="appSubtitle">Historical accidents and baseline model predictions</p></div><div class="badge">CS5284 · GROUP 01</div></header>
<main><aside><label>State</label><select id="state"></select><label>Map layer</label><select id="view"><option value="observed">Recorded accidents</option><option value="predicted">Model prediction</option><option value="error">Prediction error</option></select><label>Model / seed</label><select id="model"></select><label>Aggregation</label><select id="aggregation"><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><label>Period</label><select id="month"></select><div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div></aside>
<section><div class="stats"><div class="card stat"><span>RECORDED ACCIDENTS</span><strong id="total">—</strong></div><div class="card stat"><span>ROADS WITH ACCIDENTS</span><strong id="positive">—</strong></div><div class="card stat"><span id="thirdLabel">MODEL</span><strong id="third">—</strong></div></div><div class="card"><div class="maphead"><b id="title">Accident map</b><span id="period"></span></div><div id="map"></div></div><div id="tileStatus" role="status"></div><div id="status"></div></section></main>
<script>
const DATA=__DATA__, CONFIG=__CONFIG__, $=id=>document.getElementById(id), fmt=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:2});
let PERIODS={}, leafletMap=null, roadLayer=null, tileLayer=null, mapState=null, tileStyle=null, legend=null;
function renderLeaflet(s,vals,a,p,colors,limit,error,reg,view){
 if(!window.L)throw Error('Map library could not load. Check internet access.');
 const element=$('map'), state=$('state').value, style='open-street-map';
 if(!leafletMap){element.innerHTML='';leafletMap=L.map(element,{preferCanvas:true,zoomControl:true});roadLayer=L.layerGroup().addTo(leafletMap);}
 if(mapState!==state){leafletMap.setView([s.center[1],s.center[0]],state==='DE'?8:7);mapState=state;}
 const tileUrl='https://tile.openstreetmap.org/{z}/{x}/{y}.png';
 if(tileStyle!==style){if(tileLayer)leafletMap.removeLayer(tileLayer);$('tileStatus').textContent='';let loaded=0,failed=0;
  tileLayer=L.tileLayer(tileUrl,{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'});
  tileLayer.on('loading',()=>{loaded=0;failed=0});tileLayer.on('tileload',()=>{loaded++;$('tileStatus').textContent=''});tileLayer.on('tileerror',()=>{failed++});
  tileLayer.on('load',()=>{if(failed&&!loaded)$('tileStatus').textContent='OpenStreetMap tiles could not load. Check internet access and whether the notebook viewer allows requests to tile.openstreetmap.org.'});
  tileLayer.addTo(leafletMap);tileStyle=style;}
 roadLayer.clearLayers();
 const groups=colors.map(()=>[]);
 vals.forEach((v,i)=>{const f=error?(v/limit+1)/2:Math.log1p(Math.max(0,v))/Math.log1p(limit),k=Math.min(colors.length-1,Math.max(0,Math.floor(f*colors.length))),xy=s.xy[i];
  groups[k].push([[xy[0][1],xy[0][0]],[xy[1][1],xy[1][0]]]);
 });
 groups.forEach((lines,k)=>{if(lines.length)L.polyline(lines,{color:colors[k],weight:k===0&&!error?1.2:4,opacity:.9,interactive:false}).addTo(roadLayer);});
 if(legend)leafletMap.removeControl(legend);
 legend=L.control({position:'bottomright'});legend.onAdd=()=>{const div=L.DomUtil.create('div');div.style.cssText='background:white;padding:8px 10px;border-radius:5px;box-shadow:0 1px 5px #888;font:12px system-ui';const title=error?'Predicted − recorded':view==='observed'?'Recorded count':reg?'Predicted count':'Risk score';div.innerHTML=`<b>${title}</b><div style="width:150px;height:9px;margin:5px 0;background:linear-gradient(to right,${colors.join(',')})"></div><span>${fmt(error?-limit:0)}</span><span style="float:right">${fmt(limit)}+</span>`;return div;};legend.addTo(leafletMap);
 setTimeout(()=>leafletMap&&leafletMap.invalidateSize(),0);
}
function periodKey(date){const mode=$('aggregation').value;return mode==='yearly'?date.slice(0,4):mode==='quarterly'?date.slice(0,4)+'Q'+Math.ceil(Number(date.slice(5))/3):date}
function options(el,values){el.replaceChildren(...values.map(x=>new Option(x,x)))}
function stateChanged(){const s=DATA[$('state').value],names=Object.keys(s.runs);options($('model'),names.length?names:[CONFIG.mode==='eda'?'Observed-data mode':'No prediction exports']);
for(const v of ['predicted','error'])$('view').querySelector(`[value=${v}]`).disabled=!names.length;
if(!names.length)$('view').value='observed';datesChanged();}
function datesChanged(){const s=DATA[$('state').value],r=s.runs[$('model').value]||{task:'none',pred:{}},old=$('month').value;
const classification=r.task==='classification'&&$('view').value!=='observed';$('aggregation').disabled=classification;if(classification)$('aggregation').value='monthly';
const dates=$('view').value==='observed'?s.dates:Object.keys(r.pred).filter(d=>s.actual[d]);PERIODS={};dates.sort().forEach(d=>{const k=periodKey(d);(PERIODS[k]||=[]).push(d)});
options($('month'),Object.keys(PERIODS));for(const option of $('month').options){const ds=PERIODS[option.value],expected=$('aggregation').value==='yearly'?12:$('aggregation').value==='quarterly'?3:1;option.text=option.value+(expected>1?' · '+ds[0]+' to '+ds[ds.length-1]+(ds.length<expected?' (partial)':''):'')}
if([...$('month').options].some(x=>x.value===old))$('month').value=old;
$('model').disabled=$('view').value==='observed';$('view').querySelector('[value=error]').disabled=r.task!=='regression';if(r.task!=='regression'&&$('view').value==='error'){$('view').value='predicted';datesChanged();return}draw();}
function draw(){try{
const s=DATA[$('state').value],r=s.runs[$('model').value]||{task:'none',pred:{},stats:{}},date=$('month').value,view=$('view').value,ds=PERIODS[date]||[],reg=r.task==='regression';
const sumArrays=arrays=>arrays.reduce((acc,row)=>acc.map((x,i)=>x+row[i]),Array(s.roads.length).fill(0));
const a=ds.length?Array(s.roads.length).fill(0):null;
if(a)ds.forEach(d=>{const [indices,values]=s.actual[d];indices.forEach((index,i)=>{a[index]+=values[i]})});
const p=ds.length&&ds.every(d=>r.pred[d])&&(reg||ds.length===1)?sumArrays(ds.map(d=>r.pred[d])):null;
if(!a)throw Error('No month is available for this selection.');
const vals=a.map((x,i)=>view==='observed'?x:view==='error'?p[i]-x:p[i]),error=view==='error';
const reference=(reg&&p)?a.concat(p):vals,positive=reference.filter(x=>x>0).sort((x,y)=>x-y);
let limit=positive.length?positive[Math.floor((positive.length-1)*.98)]:1;
if(error){const sorted=vals.map(Math.abs).sort((x,y)=>x-y);limit=Math.max(.1,sorted[Math.floor((sorted.length-1)*.98)]);}limit=Math.max(limit,.01);
const colors=error?['#2166ac','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#ef8a62','#b2182b']:['#cbd5e1','#7fcdbb','#41b6c4','#1d91c0','#fdae61','#f46d43','#a50026'];
renderLeaflet(s,vals,a,p,colors,limit,error,reg,view);
const stats=r.stats[date],overview=s.summary[date]||s.periodSummary[date];
const predictedPeriod=view!=='observed'&&ds.length>1;
$('total').textContent=fmt(predictedPeriod?stats.observed:overview.total);$('positive').textContent=fmt(predictedPeriod?stats.positive:overview.positive);
$('thirdLabel').textContent=view==='observed'?'NETWORK ROADS':reg&&p?'PREDICTED TOTAL':'PREDICTION AVAILABILITY';$('third').textContent=view==='observed'?fmt(s.roadCount):reg&&p?fmt(r.stats[date].total):p?'Available':'Not exported';
$('title').textContent=view==='observed'?'Recorded accidents':error?'Prediction error':reg?'Predicted accident counts':'Classification risk scores';$('period').textContent=$('month').selectedOptions[0].text+' · '+$('state').value;
$('status').textContent=view==='observed'?(ds.includes('2022-10')&&$('state').value==='DE'?'Delaware October 2022 is incomplete and excluded from the main EDA.':''):r.task==='none'?'No prediction exports found for this state.':reg?(p?'All-road '+(ds.length>1?'MAE of period totals: ':'monthly MAE: ')+fmt(r.stats[date].mae):'Predictions were not exported for this period.'):'Classification scores are not calibrated accident probabilities.';
}catch(e){$('status').textContent=e.message}}
options($('state'),Object.keys(DATA));$('view').value=CONFIG.mode==='results'?'predicted':'observed';
if(CONFIG.mode==='eda'){$('appTitle').textContent='Road Safety Explorer · Full-period EDA';$('appSubtitle').textContent='Observed accident patterns across all available months';}
if(CONFIG.mode==='results'){$('appTitle').textContent='Road Safety Explorer · Test Results';$('appSubtitle').textContent='Exported predictions, recorded counts and regression errors';}
$('state').onchange=stateChanged;$('model').onchange=datesChanged;$('view').onchange=datesChanged;$('month').onchange=draw;$('aggregation').onchange=datesChanged;
for(const [id,step]of [['prev',-1],['next',1]])$(id).onclick=()=>{const m=$('month');m.selectedIndex=Math.max(0,Math.min(m.options.length-1,m.selectedIndex+step));draw()};stateChanged();
</script></body></html>'''



