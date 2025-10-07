from __future__ import annotations
import os, json, io, base64
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras as K

ARTIFACTS = os.environ.get("ARTIFACTS_DIR","artifacts")
MODEL = K.models.load_model(os.path.join(ARTIFACTS, "model_keras"))
META = json.load(open(os.path.join(ARTIFACTS,"meta.json"), "r"))
GENRES = META["GENRES"]; PLATFORMS = META["PLATFORMS"]; COUNTRIES = META["COUNTRIES"]

def vectorize_one(payload: dict) -> np.ndarray:
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
    ax.barh(labels, vals); ax.invert_yaxis(); ax.set_xlabel("Success prob")
    fig.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# multipliers por país para derivar un mapa desde el score mundial (demo)
AFF = {
    "JP": {"RPG":1.2, "Shooter":0.9},
    "US": {"Shooter":1.2, "Sports":1.1},
    "ES": {"Sports":1.15, "RPG":1.05},
    "BR": {"Sports":1.2, "Action":1.05},
    "FR": {}
}

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

class PredictOut(BaseModel):
    success_worldwide: float
    success_by_country: Dict[str,float]
    avg_playtime_by_country: Dict[str,float]
    heatmap_base64: str

app = FastAPI(title="GameRadar API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def check_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
def root():
    return {"status":"ok","docs":"/docs"}

@app.post("/predict", response_model=PredictOut, dependencies=[Depends(check_api_key)])
def predict(payload: PredictIn):
    X = vectorize_one(payload.model_dump())
    world = float(MODEL.predict(X, verbose=0)[0,0])

    # Derivar países con multiplicadores por género (demo vistosa)
    by_country = {}
    for c in COUNTRIES:
        mult = 1.0
        for g in payload.genres:
            mult *= AFF.get(c, {}).get(g, 1.0)
        by_country[c] = float(np.clip(world * mult, 0.01, 0.99))

    # Playtime simple (también demo)
    play = {}
    for c in COUNTRIES:
        base = 12.0 + (8.0 if "RPG" in payload.genres else 0.0) \
                      + (4.0 if "Strategy" in payload.genres else 0.0) \
                      - 0.04*(payload.price_eur - 39.0)
        play[c] = float(np.clip(base, 3.0, 60.0))

    heat_b64 = bars_png(by_country)
    return PredictOut(
        success_worldwide=world,
        success_by_country=by_country,
        avg_playtime_by_country=play,
        heatmap_base64=heat_b64
    )
