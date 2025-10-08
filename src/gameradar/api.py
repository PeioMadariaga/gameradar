from __future__ import annotations
import os, json, io, base64
from typing import List, Dict

# Silenciar logs informativos de TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from tensorflow import keras as K

# -----------------------------
# Metadatos Swagger
# -----------------------------
TAGS_METADATA = [
    {"name": "predict", "description": "Predicción de éxito mundial y por país. Devuelve también un gráfico (PNG base64) listo para incrustar."},
    {"name": "whatif", "description": "Simulación de escenarios (precio, plataformas, marketing) con deltas respecto al caso base."},
    {"name": "demo", "description": "Página HTML mínima para enseñar en clase (no requiere front externo)."},
]

APP_DESCRIPTION = (
    "GameRadar — microservicio de IA (Python + Keras) para predecir el éxito de videojuegos.\n\n"
    "- POST /predict → probabilidad mundial, por país, horas medias y gráfico (PNG base64).\n"
    "- POST /whatif → compara variantes vs. un caso base (bajar precio, añadir plataformas, etc.).\n"
    "- GET /demo → página visual para la demo."
)

# -----------------------------
# Crear app (ANTES de rutas)
# -----------------------------
app = FastAPI(
    title="GameRadar API",
    version="0.1.0",
    description=APP_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    contact={"name": "Equipo Python", "email": "demo@example.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Carga perezosa de modelo/meta
# -----------------------------
ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL = None
META = None
GENRES: list[str] | None = None
PLATFORMS: list[str] | None = None
COUNTRIES: list[str] | None = None

def _ensure_loaded():
    global MODEL, META, GENRES, PLATFORMS, COUNTRIES
    if MODEL is None:
        MODEL = K.models.load_model(os.path.join(ARTIFACTS, "model.keras"))
        with open(os.path.join(ARTIFACTS, "meta.json")) as f:
            META = json.load(f)
        GENRES = META["GENRES"]; PLATFORMS = META["PLATFORMS"]; COUNTRIES = META["COUNTRIES"]

# -----------------------------
# Seguridad (API Key opcional)
# -----------------------------
def check_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -----------------------------
# Schemas
# -----------------------------
class PredictIn(BaseModel):
    title: str | None = None
    genres: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    pegi: int = 12
    price_eur: float = 39.99
    marketing_budget_k: int = 120
    is_sequel: bool = False
    has_crossplay: bool = True
    coop: bool = False
    studio_tier: str | None = None
    art_style: str | None = None
    release_quarter: str | None = None
    pegi_age: int = 12  # admite 3,7,12,16,18
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "genres":["RPG","Adventure"],
                "platforms":["PC","Switch"],
                "price_eur":39.99,
                "marketing_budget_k":120,
                "is_sequel": True,
                "has_crossplay": True,
                "coop": True
            }]
        }
    }

class PredictOut(BaseModel):
    success_worldwide: float
    success_by_country: Dict[str,float]
    avg_playtime_by_country: Dict[str,float]
    heatmap_base64: str
    units_by_country: Dict[str, int]
    revenue_global_eur: float
    pegi_age: int

class WhatIfIn(BaseModel):
    base_payload: PredictIn
    variants: list[dict]

class WhatIfOutItem(BaseModel):
    change: str
    success_worldwide: float
    delta: float

class WhatIfOut(BaseModel):
    base: float
    variants: list[WhatIfOutItem]

# -----------------------------
# Utilidades
# -----------------------------
def vectorize_one(payload: dict) -> np.ndarray:
    assert GENRES is not None and PLATFORMS is not None
    def multi_hot(items, vocab):
        v = np.zeros((len(vocab),), dtype=np.float32)
        idx = {w:i for i,w in enumerate(vocab)}
        for it in items:
            if it in idx: v[idx[it]] = 1.0
        return v
    Xg = multi_hot(payload.get("genres",[]), GENRES)
    Xp = multi_hot(payload.get("platforms",[]), PLATFORMS)
    nums = np.array([payload.get("price_eur",39.99), payload.get("marketing_budget_k",120)], dtype=np.float32)
    bools = np.array([
        float(payload.get("is_sequel", False)),
        float(payload.get("has_crossplay", True)),
        float(payload.get("coop", False))
    ], dtype=np.float32)
    return np.concatenate([Xg, Xp, nums, bools], axis=0)[None, :]

def bars_png(d: Dict[str,float]) -> str:
    items = sorted(d.items(), key=lambda x:x[1], reverse=True)
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(labels, vals)
    ax.invert_yaxis()
    ax.set_xlabel("Success prob")
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# Afinidades para derivar probabilidades por país desde el score mundial (demo)
AFF = {
    "JP": {"RPG": 1.20, "Shooter": 0.90, "Adventure": 1.08, "Strategy": 1.05},
    "US": {"Shooter": 1.20, "Sports": 1.10, "Action": 1.05, "Racing": 1.05},
    "ES": {"Sports": 1.15, "RPG": 1.05, "Adventure": 1.05},
    "BR": {"Sports": 1.20, "Action": 1.05, "Family": 1.05},
    "FR": {"Adventure": 1.05, "Indie": 1.05},
    "DE": {"Simulation": 1.10, "Strategy": 1.08, "Racing": 1.05},
    "GB": {"Sports": 1.10, "Action": 1.05},
    "IT": {"Sports": 1.12, "Racing": 1.08}
}
PEGI_GLOBAL_MULT = {3: 1.00, 7: 0.98, 12: 0.95, 16: 0.90, 18: 0.85}

# -----------------------------
# Rutas
# -----------------------------
@app.get("/", tags=["demo"])
def root():
    return {"status": "ok", "docs": "/docs", "demo": "/demo"}

@app.post("/predict", response_model=PredictOut, tags=["predict"])
def predict(payload: PredictIn):
    _ensure_loaded()
    X = vectorize_one(payload.model_dump())
    world = float(MODEL.predict(X, verbose=0)[0,0])

    # Derivar países con multiplicadores por género (demo vistosa)
    assert COUNTRIES is not None
    by_country = {}
    for c in COUNTRIES:
        mult = 1.0
        for g in payload.genres:
            mult *= AFF.get(c, {}).get(g, 1.0)
        by_country[c] = float(np.clip(world * mult, 0.01, 0.99))
    
    # aplicar multiplicador PEGI al score global
    world *= PEGI_GLOBAL_MULT.get(payload.pegi_age, 1.0)

    # aplicar multiplicador PEGI por país
    for c in list(by_country.keys()):
        by_country[c] = float(
            np.clip(by_country[c] * PEGI_GLOBAL_MULT.get(payload.pegi_age, 1.0), 0.01, 0.99)
        )

    # Playtime sencillo (demo)
    play = {}
    base_play = 12.0 \
        + (8.0 if "RPG" in payload.genres else 0.0) \
        + (4.0 if "Strategy" in payload.genres else 0.0) \
        - 0.04*(payload.price_eur - 39.0)
    for c in COUNTRIES:
        play[c] = float(np.clip(base_play, 3.0, 60.0))
    
    # --- NUEVO: unidades y revenue ---
    base_units = 80000  # unidades “base” por país cuando p=1.0 (ajustable para demo)
    units_by_country = {c: int(base_units * p) for c, p in by_country.items()}
    revenue_global_eur = float(sum(units_by_country.values()) * payload.price_eur)

    heat_b64 = bars_png(by_country)
    return PredictOut(
        success_worldwide=world,
        success_by_country=by_country,
        avg_playtime_by_country=play,
        heatmap_base64=heat_b64,
        units_by_country=units_by_country,
        revenue_global_eur=revenue_global_eur,
        pegi_age=payload.pegi_age
    )

@app.post("/whatif", response_model=WhatIfOut, tags=["whatif"])
def whatif(req: WhatIfIn):
    _ensure_loaded()
    base_payload = req.base_payload.model_dump()

    # 1) Construimos todos los payloads (base + variantes)
    payloads = [base_payload]
    Xs = [vectorize_one(base_payload)]
    for change in req.variants:
        p = base_payload.copy()
        p.update(change)
        payloads.append(p)
        Xs.append(vectorize_one(p))

    # 2) Predicción en LOTE (una sola llamada a Keras)
    Xmat = np.vstack(Xs)                 # (N+1, D)
    preds = MODEL.predict(Xmat, verbose=0).reshape(-1)  # array de probabilities

    # 3) Aplicamos PEGI global a cada payload para coherencia con /predict
    pegi_mults = [PEGI_GLOBAL_MULT.get(p.get("pegi_age", 12), 1.0) for p in payloads]
    preds = preds * np.array(pegi_mults, dtype=np.float32)

    base_world = float(preds[0])
    items: list[WhatIfOutItem] = []
    for i, change in enumerate(req.variants, start=1):
        w = float(preds[i])
        items.append(WhatIfOutItem(change=str(change), success_worldwide=w, delta=w - base_world))

    return WhatIfOut(base=base_world, variants=items)
class PriceCurveIn(BaseModel):
    payload: PredictIn
    min_price: float = 19.0
    max_price: float = 69.0
    steps: int = 21

class PriceCurveOut(BaseModel):
    points: list[tuple[float, float]]   # [(precio, prob)]
    best_price: float
    best_prob: float

@app.post("/curve/price", response_model=PriceCurveOut, tags=["predict"])
def curve_price(req: PriceCurveIn):
    _ensure_loaded()
    ps = np.linspace(req.min_price, req.max_price, req.steps)
    pts: list[tuple[float,float]] = []
    best_p, best_prob = None, -1.0
    base = req.payload.model_dump()
    for p in ps:
        base["price_eur"] = float(p)
        X = vectorize_one(base)
        prob = float(MODEL.predict(X, verbose=0)[0,0])
        pts.append((float(p), prob))
        if prob > best_prob:
            best_prob, best_p = prob, float(p)
    return PriceCurveOut(points=pts, best_price=best_p, best_prob=best_prob)

@app.get("/demo", response_class=HTMLResponse, tags=["demo"])
def demo():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GameRadar — Demo WOW</title>
<style>
:root{
  --bg:#0b1220; --card:#111a2a; --muted:#8aa0c8; --txt:#e8eefc; --accent:#5ce1e6; --ok:#22c55e; --bad:#f87171; --line:#233;
}
html[data-theme='light']{
  --bg:#f5f7fb; --card:#ffffff; --muted:#5a6b86; --txt:#0b1220; --accent:#0066ff22; --ok:#16a34a; --bad:#dc2626; --line:#e6ecf5;
}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,Segoe UI,Roboto,Arial;background:linear-gradient(120deg,var(--bg),#0e1430);color:var(--txt)}
.wrap{max-width:1100px;margin:24px auto;padding:0 16px}
h1{margin:8px 0 16px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}
.tab{padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#0e1628;cursor:pointer}
html[data-theme='light'] .tab{background:#f2f6ff}
.tab.active{outline:2px solid var(--accent)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.20);border:1px solid var(--line)}
.section{display:none}
.section.active{display:block}
label{font-size:14px;color:var(--muted)}
.input-row{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.pill{padding:6px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;display:inline-block}
.pill input{vertical-align:middle;margin-right:6px}
input,button{font-size:14px;border-radius:10px;border:1px solid var(--line);padding:10px 12px;background:#0e1628;color:var(--txt)}
html[data-theme='light'] input, html[data-theme='light'] button{background:#fff}
button.primary{background:var(--accent);border:none;font-weight:700;color:#002}
.kpis{display:flex;gap:16px;flex-wrap:wrap}
.kpi{flex:1;min-width:220px;display:flex;gap:16px;align-items:center}
.gauge{--p:0; width:96px;height:96px;border-radius:50%;background:
 conic-gradient(#29b5ff var(--p), #2a3347 0);
 display:grid;place-items:center}
.gauge span{background:var(--card);width:74px;height:74px;border-radius:50%;display:grid;place-items:center;font-weight:800}
.small{font-size:12px;color:var(--muted)}
.table{width:100%;border-collapse:collapse;margin-top:8px}
.table th,.table td{border-bottom:1px solid var(--line);padding:8px;text-align:left}
.badge{padding:4px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.tools{display:flex;gap:8px;flex-wrap:wrap}
canvas{width:100%;height:220px;border:1px solid var(--line);border-radius:12px;background:#0e1628}
.theme{position:fixed;right:16px;top:16px}
img{max-width:100%}
/* Barra mini dentro de la celda */
.bar { position: relative; height: 10px; background: #223; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg,#ef4444,#22c55e); transition: width .5s; }
.flag { font-size: 18px; margin-right: 6px }
.win { color: #22c55e; font-weight: 700 }
.lose { color: #f87171; font-weight: 700 }

/* Overlay de carga y confetti */
#overlay { position: fixed; inset: 0; display:none; place-items: center; backdrop-filter: blur(2px); }
.spinner { width: 36px; height: 36px; border-radius: 50%; border: 3px solid #3a4a6a; border-top-color: #5ce1e6; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg) } }
#confetti { position: fixed; inset: 0; pointer-events: none; display:none; }

.grid-genres{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
@media (min-width:900px){.grid-genres{grid-template-columns:repeat(4,1fr)}}
</style>
</head>
<body>
<div class="theme">
  <label class="row"><input id="theme" type="checkbox"> Tema claro</label>
</div>

<div class="wrap">
  <h1>🎮 GameRadar — Demo</h1>

  <!-- TABS -->
  <div class="tabs">
    <div class="tab active" data-tab="resumen">Resumen</div>
    <div class="tab" data-tab="paises">Países</div>
    <div class="tab" data-tab="whatif">What-if</div>
    <div class="tab" data-tab="precio">Precio óptimo</div>
  </div>

  <div class="grid">
    <!-- Panel izquierdo: parámetros -->
    <div class="card">
      <h2>Parámetros</h2>

     <div class="input-row">
         <div>
            <label>Géneros</label><br/>
            <div class="grid-genres">
              <label class="pill"><input type="checkbox" name="genre" value="RPG">RPG</label>
              <label class="pill"><input type="checkbox" name="genre" value="Adventure">Adventure</label>
              <label class="pill"><input type="checkbox" name="genre" value="Action">Action</label>
              <label class="pill"><input type="checkbox" name="genre" value="Shooter">Shooter</label>
              <label class="pill"><input type="checkbox" name="genre" value="Sports">Sports</label>
              <label class="pill"><input type="checkbox" name="genre" value="Simulation">Simulation</label>
              <label class="pill"><input type="checkbox" name="genre" value="Strategy">Strategy</label>
              <label class="pill"><input type="checkbox" name="genre" value="Horror">Horror</label>
              <label class="pill"><input type="checkbox" name="genre" value="Racing">Racing</label>
              <label class="pill"><input type="checkbox" name="genre" value="Family">Family</label>
              <label class="pill"><input type="checkbox" name="genre" value="Indie">Indie</label>
              <label class="pill"><input type="checkbox" name="genre" value="Platformer">Platformer</label>
            </div>
          </div>
      </div>


      <div class="input-row">
        <div>
          <label>Plataformas</label><br/>
          <label class="pill"><input type="checkbox" name="plat" value="PC">PC</label>
          <label class="pill"><input type="checkbox" name="plat" value="PS5">PS5</label>
          <label class="pill"><input type="checkbox" name="plat" value="Xbox">Xbox</label>
          <label class="pill"><input type="checkbox" name="plat" value="Switch">Switch</label>
        </div>
      </div>

      <div class="input-row row">
        <div><label>Precio (€)</label><br/><input id="price" type="number" step="0.01" value="39.99"></div>
        <div><label>Marketing (K€)</label><br/><input id="mk" type="number" step="1" value="120"></div>
        <div><label>PEGI</label><br/>
            <select id="pegi">
              <option>3</option>
              <option>7</option>
              <option selected>12</option>
              <option>16</option>
              <option>18</option>
            </select>
        </div>
        <label class="pill"><input id="seq" type="checkbox">Secuela</label>
        <label class="pill"><input id="cross" type="checkbox" checked>Crossplay</label>
        <label class="pill"><input id="coop" type="checkbox">Coop</label>
    </div>

     <!-- <div class="input-row row"> -->
     <!--   <div><label>API Key (opcional)</label><br/><input id="apikey" placeholder="X-API-Key si la definiste"></div> -->
     <!-- </div> -->

      <div class="row">
        <button class="primary" id="btnPred">⚡ Predecir</button>
        <button id="btnWhat">🧪 What-if</button>
      </div>
    </div>

    <!-- Panel derecho: contenido por pestañas -->
    <div class="card">

      <!-- RESUMEN -->
      <div class="section active" id="tab-resumen">
        <h2>Resultados</h2>
        <div class="kpis">
          <div class="kpi">
            <div class="gauge" id="gauge" style="--p:0deg"><span id="pct">—</span></div>
            <div>
              <div style="font-weight:800;font-size:28px" id="pctTxt">—</div>
              <div class="small">Probabilidad de éxito mundial</div>
              <div class="small" id="rev">Ingresos estimados: —</div>
              <div class="small" id="badgePegi">PEGI: 12</div>
            </div>
          </div>
        </div>

        <div class="tools" style="margin:10px 0">
          <span class="badge" id="badgeGen">—</span>
          <span class="badge" id="badgePlat">—</span>
          <button id="btnCopy">Copiar JSON</button>
          <button id="btnDownload">Descargar gráfico</button>
        </div>

        <img id="img" alt="chart" style="display:none;margin-top:8px"/>
      </div>

      <!-- PAISES -->
      <div class="section" id="tab-paises">
        <h2>Top países</h2>
        <table class="table"><thead><tr><th>País</th><th>Prob.</th><th>Unidades</th></tr></thead><tbody id="tbodyCountries"></tbody></table>
      </div>

      <!-- WHATIF -->
      <div class="section" id="tab-whatif">
        <h2>What-if</h2>
        <div id="whatCards" class="row" style="gap:12px;flex-wrap:wrap"></div>
      </div>

      <!-- PRECIO -->
      <div class="section" id="tab-precio">
        <h2>Curva de precio</h2>
        <canvas id="cv"></canvas>
        <div class="small" id="best"></div>
      </div>

    </div>
  </div>
</div>
<div id="overlay"><div class="spinner"></div></div>
<canvas id="confetti"></canvas>
<script>
const API = location.origin;
const $ = s => document.querySelector(s);
const $all = s => [...document.querySelectorAll(s)];
// Banderas y nombre "bonito" de país
const FLAGS = { ES:"🇪🇸", US:"🇺🇸", JP:"🇯🇵", BR:"🇧🇷", FR:"🇫🇷", DE:"🇩🇪", GB:"🇬🇧", IT:"🇮🇹" };
const NAMES = { ES:"España", US:"Estados Unidos", JP:"Japón", BR:"Brasil", FR:"Francia", DE:"Alemania", GB:"Reino Unido", IT:"Italia" };

// Color del gauge según porcentaje
function gaugeColor(p){ // p in [0..1]
  // rojo (0) -> amarillo (0.5) -> verde (1)
  const r = Math.round(255 * (1-p));
  const g = Math.round(255 * p);
  return `rgb(${r},${g},80)`;
}
// Formatos
const fmtPct = x => (x*100).toFixed(1) + "%";
const fmtMoney = x => new Intl.NumberFormat('es-ES',{style:'currency',currency:'EUR'}).format(x);
const fmtInt = x => Number(x).toLocaleString('es-ES');

function launchConfetti(ms=1500){
  const cv = document.getElementById('confetti'); const ctx = cv.getContext('2d');
  const DPR = window.devicePixelRatio || 1; cv.style.display='block';
  const W = cv.width = innerWidth * DPR, H = cv.height = innerHeight * DPR;
  const parts = Array.from({length: 180}, ()=>({
    x: Math.random()*W, y: -20*DPR, vy: (2+Math.random()*3)*DPR,
    vx: (Math.random()*2-1)*DPR, s: (4+Math.random()*6)*DPR,
    c: `hsl(${Math.random()*360},90%,60%)`, r: Math.random()*Math.PI
  }));
  let alive=true; const tEnd=performance.now()+ms;
  (function anim(){
    ctx.clearRect(0,0,W,H);
    parts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.vy+=0.04*DPR; p.r+=0.1;
      ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.r);
      ctx.fillStyle=p.c; ctx.fillRect(-p.s/2,-p.s/2,p.s,p.s); ctx.restore();
    });
    parts.forEach((p,i)=>{ if(p.y>H+50*DPR) parts.splice(i,1); });
    if(alive && performance.now()<tEnd) requestAnimationFrame(anim);
    else { cv.style.display='none'; }
  })();
}
const overlay = {
  show(){ document.getElementById('overlay').style.display='grid'; },
  hide(){ document.getElementById('overlay').style.display='none'; }
};

function renderCountries(data){
  const body = document.getElementById('tbodyCountries'); if(!body) return;
  body.innerHTML = "";
  // construimos array [pais, prob, units]
  const rows = Object.entries(data.success_by_country).map(([c,p])=>[c,p, data.units_by_country ? data.units_by_country[c] : null]);
  rows.sort((a,b)=>b[1]-a[1]);
  rows.forEach(([c,p,u])=>{
    const tr = document.createElement('tr');
    const flag = FLAGS[c] || "🏳️";
    const name = NAMES[c] || c;
    tr.innerHTML = `
      <td><span class="flag">${flag}</span>${name}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="bar" style="flex:1"><div class="bar-fill" style="width:${(p*100).toFixed(0)}%"></div></div>
          <div ${p>=0.5?'class="win"':'class="lose"'}>${fmtPct(p)}</div>
        </div>
      </td>
      <td>${u!=null ? fmtInt(u) : "—"}</td>
    `;
    body.appendChild(tr);
  });
}


function headers(){
  const h={"Content-Type":"application/json"};
  const el = $("#apikey");
  const k = el ? el.value.trim() : "";
  if(k) h["X-API-Key"]=k;
  return h;
}
function checked(name){ return $all('input[name="'+name+'"]:checked').map(x=>x.value); }
function bodyBase(){
  return {
    genres: checked("genre"),
    platforms: checked("plat"),
    price_eur: parseFloat($("#price").value),
    marketing_budget_k: parseFloat($("#mk").value),
    is_sequel: $("#seq").checked,
    has_crossplay: $("#cross").checked,
    coop: $("#coop").checked,
    pegi_age: parseInt(document.getElementById('pegi').value, 10)
  };
}

let lastPredict = null; // guardamos último JSON para tabs

// Tabs
$all(".tab").forEach(t=>t.onclick=()=>{
  $all(".tab").forEach(x=>x.classList.remove("active"));
  t.classList.add("active");
  $all(".section").forEach(x=>x.classList.remove("active"));
  $("#tab-"+t.dataset.tab).classList.add("active");
});

// Tema
const themeEl = $("#theme");
function applyTheme(){ document.documentElement.setAttribute("data-theme", themeEl.checked ? "light":"dark"); }
themeEl.onchange = applyTheme; applyTheme();

// Animación gauge
function animateGauge(p){ // p en [0,1]
  const deg = Math.round(p*360);
  $("#gauge").style.setProperty("--p", deg+"deg");
  // contador
  const target = Math.round(p*1000)/10;
  let cur = 0;
  const step = () => { cur += Math.max(0.3,(target-cur)/10); $("#pct").textContent = cur.toFixed(1)+"%"; if(cur<target-0.1) requestAnimationFrame(step); };
  $("#pctTxt").textContent = target.toFixed(1)+"%";
  requestAnimationFrame(step);
}

async function predict(){
  try{
    overlay.show();

    const body = bodyBase();
    const r = await fetch(API + "/predict", {method:"POST", headers: headers(), body: JSON.stringify(body)});
    if(!r.ok){ alert("Error "+r.status); return; }
    const data = await r.json(); lastPredict = data;

    // Badge PEGI (mantén esto si quieres mostrarlo)
    const pegi = (data.pegi_age ?? parseInt(document.getElementById('pegi').value, 10));
    const b = document.getElementById('badgePegi');
    if (b) {
      b.textContent = "PEGI: " + pegi;
      b.style.padding = "4px 8px";
      b.style.borderRadius = "999px";
      b.style.border = "1px solid var(--line)";
      b.style.background = (pegi >= 18 ? "#4a0f0f" : pegi >= 16 ? "#4a2f0f" : "transparent");
    }

    // KPI: gauge con color + contador
    const p = data.success_worldwide;
    document.getElementById('gauge').style.background =
      `conic-gradient(${gaugeColor(p)} ${Math.round(p*360)}deg, #2a3347 0)`;
    animateGauge(p);

    // Ingresos
    document.getElementById('rev').textContent =
      "Ingresos estimados: " + new Intl.NumberFormat('es-ES',{style:'currency',currency:'EUR'}).format(data.revenue_global_eur);

    // Badges de selección
    document.getElementById('badgeGen').textContent =
      "Géneros: " + (body.genres.join(", ") || "—");
    document.getElementById('badgePlat').textContent =
      "Plataformas: " + (body.platforms.join(", ") || "—");

    // Imagen (barras por país generadas en el backend)
    const img = document.getElementById('img');
    img.src = "data:image/png;base64," + data.heatmap_base64;
    img.style.display="block";

    // Tabla de países con banderas + barras + % + unidades
    renderCountries(data);

    // Confetti si muy alto
    if (p >= 0.85) launchConfetti();

  } finally {
    overlay.hide();
  }
}


$("#btnPred").onclick = predict;

// Descargar gráfico / Copiar JSON
$("#btnDownload").onclick = ()=>{
  if(!lastPredict) return;
  const a = document.createElement('a');
  a.href = "data:image/png;base64," + lastPredict.heatmap_base64;
  a.download = "gameradar_paises.png";
  a.click();
};
$("#btnCopy").onclick = ()=>{
  if(!lastPredict) return;
  navigator.clipboard.writeText(JSON.stringify(lastPredict, null, 2));
};

// WHAT-IF
$("#btnWhat").onclick = async ()=>{
  try {
    overlay.show(); // 🔹 aparece el spinner de carga

    const base_payload = bodyBase();
    // Si quieres asegurar que el PEGI se envía correctamente:
    base_payload.pegi_age = parseInt(document.getElementById('pegi').value, 10);

    const variants = [
      {"price_eur": base_payload.price_eur - 10},
      {"price_eur": base_payload.price_eur + 10},
      {"marketing_budget_k": base_payload.marketing_budget_k + 80},
      {"platforms": Array.from(new Set([...base_payload.platforms, "PS5","Xbox"]))},
      {"coop": !base_payload.coop}
    ];

    const r = await fetch(API + "/whatif", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ base_payload, variants })
    });

    if (!r.ok) {
      alert("Error " + r.status);
      return;
    }

    const data = await r.json();

    const wrap = document.getElementById("whatCards");
    wrap.innerHTML = "";

    data.variants.forEach(v => {
      const up = v.delta >= 0;
      const card = document.createElement("div");
      card.className = "card";
      card.style.width = "220px";
      card.innerHTML = `
        <div class="small" style="opacity:.8">${v.change}</div>
        <div style="font-weight:800;font-size:22px">${(v.success_worldwide * 100).toFixed(1)}%</div>
        <div style="color:${up ? 'var(--ok)' : 'var(--bad)'};font-weight:700">
          ${up ? '▲' : '▼'} ${(v.delta * 100).toFixed(1)} pp
        </div>
      `;
      wrap.appendChild(card);
    });

    // Cambia automáticamente a la pestaña What-if
    document.querySelector('.tab[data-tab="whatif"]').click();

  } finally {
    overlay.hide(); // oculta el spinner, pase lo que pase
  }
};

// CURVA DE PRECIO
async function drawPriceCurve(){
  const payload = bodyBase();
  const r = await fetch(API + "/curve/price", {method:"POST", headers: headers(), body: JSON.stringify({payload, min_price:19, max_price:69, steps:21})});
  if(!r.ok){ return; }
  const data = await r.json();
  const cv = $("#cv"); const ctx=cv.getContext("2d"); const W=cv.width= cv.clientWidth*2; const H=cv.height= 240;
  ctx.clearRect(0,0,W,H);
  // ejes
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--line');
  ctx.beginPath(); ctx.moveTo(40,H-30); ctx.lineTo(W-10,H-30); ctx.moveTo(40,H-30); ctx.lineTo(40,10); ctx.stroke();
  // línea
  const ps = data.points; const xs=ps.map(p=>p[0]), ys=ps.map(p=>p[1]);
  const minx=Math.min(...xs), maxx=Math.max(...xs), miny=0, maxy=Math.max(1, ...ys);
  const sx=x=> 40 + (x-minx)/(maxx-minx) * (W-60);
  const sy=y=> (H-30) - (y-miny)/(maxy-miny) * (H-50);
  ctx.beginPath(); ctx.strokeStyle = "#29b5ff"; ctx.lineWidth=3;
  ps.forEach((p,i)=>{ const x=sx(p[0]), y=sy(p[1]); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
  // mejor punto
  const bx = sx(data.best_price), by = sy(data.best_prob);
  ctx.fillStyle="#22c55e"; ctx.beginPath(); ctx.arc(bx,by,6,0,Math.PI*2); ctx.fill();
  $("#best").textContent = "Mejor precio ≈ " + data.best_price.toFixed(2) + " €  → " + (data.best_prob*100).toFixed(1) + "%";
}
document.querySelector('.tab[data-tab="precio"]').addEventListener('click', drawPriceCurve);

// predicción inicial para que entre con algo
// predict();
function clearUI(){
  // KPI
  document.getElementById('gauge').style.background = 'conic-gradient(#2a3347 0deg, #2a3347 0)';
  document.getElementById('pct').textContent = '—';
  document.getElementById('pctTxt').textContent = '—';
  document.getElementById('rev').textContent = 'Ingresos estimados: —';
  const b = document.getElementById('badgePegi');
  if (b) { b.textContent = "PEGI: —"; b.style.background = "transparent"; }

  // Gráfico
  const img = document.getElementById('img');
  img.style.display = 'none';
  img.src = '';

  // Tabla países
  const body = document.getElementById('tbodyCountries');
  if (body) body.innerHTML = '';

  // What-if
  document.getElementById("whatCards").innerHTML = '';
  document.getElementById("btnWhat").disabled = true;

  // Limpia último resultado
  lastPredict = null;
}
// Llama al cargar
clearUI();
</script>
</body>
</html>
    """

# -----------------------------
# Swagger: documentar X-API-Key
# -----------------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=TAGS_METADATA,
    )
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["ApiKeyHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Clave opcional para la API (si está configurada la variable de entorno API_KEY)."
    }
    openapi_schema["security"] = [{"ApiKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
