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
    "JP": {"RPG":1.2, "Shooter":0.9},
    "US": {"Shooter":1.2, "Sports":1.1},
    "ES": {"Sports":1.15, "RPG":1.05},
    "BR": {"Sports":1.2, "Action":1.05},
    "FR": {}
}

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

    # Playtime sencillo (demo)
    play = {}
    base_play = 12.0 \
        + (8.0 if "RPG" in payload.genres else 0.0) \
        + (4.0 if "Strategy" in payload.genres else 0.0) \
        - 0.04*(payload.price_eur - 39.0)
    for c in COUNTRIES:
        play[c] = float(np.clip(base_play, 3.0, 60.0))

    heat_b64 = bars_png(by_country)
    return PredictOut(
        success_worldwide=world,
        success_by_country=by_country,
        avg_playtime_by_country=play,
        heatmap_base64=heat_b64
    )

@app.post("/whatif", response_model=WhatIfOut, tags=["whatif"])
def whatif(req: WhatIfIn):
    _ensure_loaded()
    Xbase = vectorize_one(req.base_payload.model_dump())
    base_world = float(MODEL.predict(Xbase, verbose=0)[0,0])
    items: list[WhatIfOutItem] = []
    for v in req.variants:
        p = req.base_payload.model_dump()
        p.update(v)
        Xv = vectorize_one(p)
        w = float(MODEL.predict(Xv, verbose=0)[0,0])
        items.append(WhatIfOutItem(change=str(v), success_worldwide=w, delta=w-base_world))
    return WhatIfOut(base=base_world, variants=items)

@app.get("/demo", response_class=HTMLResponse, tags=["demo"])
def demo():
    return """
<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GameRadar — Demo</title>
<style>
:root{--bg:#0b1220;--card:#121a2a;--accent:#5ce1e6;--txt:#e8eefc}
body{margin:0;font-family:system-ui,Segoe UI,Roboto,Arial;background:linear-gradient(120deg,#0b1220,#0e1430);color:var(--txt)}
.wrap{max-width:1000px;margin:32px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
h1,h2{margin:0 0 12px}
label{font-size:14px;opacity:.9}
input,button{font-size:14px;border-radius:10px;border:1px solid #334;padding:10px 12px;background:#0e1628;color:var(--txt)}
.row{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}
.pill{padding:6px 10px;border:1px solid #334;border-radius:999px;cursor:pointer}
.pill input{vertical-align:middle;margin-right:6px}
button.primary{background:var(--accent);color:#002;padding:10px 16px;border:none;font-weight:700;cursor:pointer}
.big{font-size:32px;font-weight:800}
table{width:100%;border-collapse:collapse}
th,td{padding:8px;border-bottom:1px solid #233}
img{max-width:100%}
.muted{opacity:.8}
</style></head><body>
<div class="wrap">
  <h1>🎮 GameRadar — Demo</h1>
  <p class="muted">Predice el éxito de un videojuego y muestra un gráfico listo para incrustar en cualquier web.</p>

  <div class="grid">
    <div class="card">
      <h2>1) Parámetros</h2>

      <div class="row"><div>
        <label>Géneros</label><br/>
        <label class="pill"><input type="checkbox" name="genre" value="RPG">RPG</label>
        <label class="pill"><input type="checkbox" name="genre" value="Adventure">Adventure</label>
        <label class="pill"><input type="checkbox" name="genre" value="Action">Action</label>
        <label class="pill"><input type="checkbox" name="genre" value="Shooter">Shooter</label>
        <label class="pill"><input type="checkbox" name="genre" value="Sports">Sports</label>
      </div></div>

      <div class="row"><div>
        <label>Plataformas</label><br/>
        <label class="pill"><input type="checkbox" name="plat" value="PC">PC</label>
        <label class="pill"><input type="checkbox" name="plat" value="PS5">PS5</label>
        <label class="pill"><input type="checkbox" name="plat" value="Xbox">Xbox</label>
        <label class="pill"><input type="checkbox" name="plat" value="Switch">Switch</label>
      </div></div>

      <div class="row">
        <div><label>Precio (€)</label><br/><input id="price" type="text" value="39,99"></div>
        <div><label>Marketing (K€)</label><br/><input id="mk" type="text" value="120"></div>
      </div>
      <div class="row">
        <label class="pill"><input id="seq" type="checkbox">Secuela</label>
        <label class="pill"><input id="cross" type="checkbox" checked>Crossplay</label>
        <label class="pill"><input id="coop" type="checkbox">Coop</label>
      </div>
      <div class="row">
        <div><label>API Key (opcional)</label><br/><input id="apikey" placeholder="X-API-Key si la definiste"></div>
      </div>
      <div class="row">
        <button class="primary" onclick="pred()">⚡ Predecir</button>
        <button onclick="wif()">🧪 What-if</button>
      </div>
    </div>

    <div class="card">
      <h2>2) Resultados</h2>
      <div class="big" id="world">—</div>
      <p class="muted">Probabilidad de éxito mundial</p>
      <img id="img" alt="chart" style="display:none;margin-top:8px"/>
      <h3 style="margin-top:16px">Top países</h3>
      <table id="table"><thead><tr><th>País</th><th>Prob.</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>3) What-if</h2>
    <table id="what"><thead><tr><th>Cambio</th><th>Nuevo %</th><th>Δ respecto base</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<script>
const API = location.origin;
function n(v){ return parseFloat(String(v).replace(",", ".")); }
function headers(){ const h={"Content-Type":"application/json"}; const k=document.getElementById('apikey').value.trim(); if(k) h["X-API-Key"]=k; return h; }
function checked(name){ return [...document.querySelectorAll('input[name="'+name+'"]:checked')].map(x=>x.value); }

async function pred(){
  const body = {
    genres: checked("genre"),
    platforms: checked("plat"),
    price_eur: n(document.getElementById('price').value),
    marketing_budget_k: n(document.getElementById('mk').value),
    is_sequel: document.getElementById('seq').checked,
    has_crossplay: document.getElementById('cross').checked,
    coop: document.getElementById('coop').checked
  };
  const r = await fetch(API + "/predict", {method:"POST", headers: headers(), body: JSON.stringify(body)});
  if(!r.ok){ let p; try{p=await r.json()}catch{p=await r.text()} alert("Error "+r.status+"\\n"+(typeof p==="string"?p:JSON.stringify(p))); return; }
  const data = await r.json();
  document.getElementById('world').textContent = (data.success_worldwide*100).toFixed(1) + "%";
  const img = document.getElementById('img'); img.src = "data:image/png;base64," + data.heatmap_base64; img.style.display="block";
  const tb = document.querySelector('#table tbody'); tb.innerHTML="";
  Object.entries(data.success_by_country).sort((a,b)=>b[1]-a[1]).slice(0,5).forEach(([c,v])=>{
    const tr = document.createElement('tr'); tr.innerHTML = `<td>${c}</td><td>${(v*100).toFixed(1)}%</td>`; tb.appendChild(tr);
  });
}

async function wif(){
  const base_payload = {
    genres: checked("genre"),
    platforms: checked("plat"),
    price_eur: n(document.getElementById('price').value),
    marketing_budget_k: n(document.getElementById('mk').value),
    is_sequel: document.getElementById('seq').checked,
    has_crossplay: document.getElementById('cross').checked,
    coop: document.getElementById('coop').checked
  };
  const variants = [
    {"price_eur": base_payload.price_eur - 10},
    {"platforms": Array.from(new Set([...base_payload.platforms, "PS5","Xbox"]))},
    {"marketing_budget_k": base_payload.marketing_budget_k + 80}
  ];
  const r = await fetch(API + "/whatif", {method:"POST", headers: headers(), body: JSON.stringify({base_payload, variants})});
  if(!r.ok){ let p; try{p=await r.json()}catch{p=await r.text()} alert("Error "+r.status+"\\n"+(typeof p==="string"?p:JSON.stringify(p))); return; }
  const data = await r.json();
  const tb = document.querySelector('#what tbody'); tb.innerHTML="";
  data.variants.forEach(v=>{
    const sign = v.delta >= 0 ? "▲" : "▼";
    const color = v.delta >= 0 ? "style='color:#5ce1e6;font-weight:700'" : "style='color:#f87171;font-weight:700'";
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${v.change}</td><td>${(v.success_worldwide*100).toFixed(1)}%</td><td ${color}>${sign} ${(v.delta*100).toFixed(1)} pp</td>`;
    tb.appendChild(tr);
  });
}
</script>
</body></html>
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
