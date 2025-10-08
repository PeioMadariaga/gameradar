from __future__ import annotations
import os, json
import numpy as np
from tensorflow import keras as K

# Vocabularios y países (mantener cortito para demo)
GENRES = ["RPG","Adventure","Shooter","Sports","Action","Simulation","Strategy","Horror","Racing","Family","Indie","Platformer"]
PLATFORMS = ["PC","PS5","Xbox","Switch"]
COUNTRIES = ["ES", "US", "JP", "BR", "FR", "DE", "GB", "IT"]

def vectorize(batch):
    """ batch: dict con listas numpy. Devuelve X [n, d]. """
    def multi_hot(vals, vocab):
        out = np.zeros((len(vals), len(vocab)), dtype=np.float32)
        idx = {v:i for i,v in enumerate(vocab)}
        for r, items in enumerate(vals):
            for it in items:
                if it in idx: out[r, idx[it]] = 1.0
        return out
    Xg = multi_hot(batch["genres"], GENRES)
    Xp = multi_hot(batch["platforms"], PLATFORMS)
    nums = np.stack([batch["price_eur"], batch["marketing_budget_k"]], axis=1).astype(np.float32)
    bools = np.stack([batch["is_sequel"], batch["has_crossplay"], batch["coop"]], axis=1).astype(np.float32)
    return np.concatenate([Xg, Xp, nums, bools], axis=1)

def gen_synth(n=2000, rng=np.random.default_rng(7)):
    g = np.array([rng.choice(GENRES, size=rng.integers(1,3), replace=False).tolist() for _ in range(n)], dtype=object)
    p = np.array([rng.choice(PLATFORMS, size=rng.integers(1,3), replace=False).tolist() for _ in range(n)], dtype=object)
    price = rng.uniform(19,69,size=n).astype(np.float32)
    budget = rng.integers(20,400,size=n).astype(np.float32)
    sequel = rng.integers(0,2,size=n).astype(np.float32)
    cross  = rng.integers(0,2,size=n).astype(np.float32)
    coop   = rng.integers(0,2,size=n).astype(np.float32)
    # score sintético
    base = 0.45 \
        + 0.06*(np.array(["RPG" in gi for gi in g])) \
        + 0.05*(np.array(["Shooter" in gi for gi in g])) \
        + 0.04*(np.array(["Sports" in gi for gi in g])) \
        + 0.03*(np.array(["Action" in gi for gi in g])) \
        + 0.02*(np.array(["Adventure" in gi for gi in g])) \
        + 0.03*(np.array(["Strategy" in gi for gi in g])) \
        + 0.02*(np.array(["Simulation" in gi for gi in g])) \
        + 0.02*(np.array(["Family" in gi for gi in g])) \
        + 0.02*(np.array(["Racing" in gi for gi in g])) \
        + 0.01*(np.array(["Horror" in gi for gi in g])) \
        + 0.04*(np.array(["Switch" in pi for pi in p])) \
        + 0.03*cross + 0.02*coop \
        # --- Nuevo tratamiento de precio (óptimo ~60–80€) y marketing con tolerancia ---
        ideal_low  = 50.0   # por debajo, parece "low-cost"
        ideal_high = 80.0   # por encima, se percibe caro

        # Penalizaciones a dos lados (vectorizadas)
        pen_low  = np.maximum(0.0, ideal_low  - price) * 0.003   # < 40€
        pen_high = np.maximum(0.0, price - ideal_high) * 0.004   # > 85€

        # El marketing ayuda a "tolerar" precios altos (y da un pequeño empuje general)
        # Usamos el exceso sobre 120K para no favorecer en exceso valores muy pequeños.
        mk_boost = 0.0005 * (budget - 120)

        # Aplica: resta penalizaciones y suma boost de marketing
        base = base - pen_low - pen_high + mk_boost
        base = np.clip(base, 0.05, 0.95).astype(np.float32)

    base = np.clip(base, 0.05, 0.95).astype(np.float32)

    X = vectorize({
        "genres": g, "platforms": p,
        "price_eur": price, "marketing_budget_k": budget,
        "is_sequel": sequel, "has_crossplay": cross, "coop": coop
    })
    return X, base

def build_model(input_dim):
    inp = K.Input(shape=(input_dim,), name="x")
    x = K.layers.Dense(64, activation="relu")(inp)
    x = K.layers.Dropout(0.2)(x)
    x = K.layers.Dense(32, activation="relu")(x)
    out = K.layers.Dense(1, activation="sigmoid", name="success_world")(x)
    m = K.Model(inp, out)
    m.compile(optimizer=K.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

def main():
    X, y = gen_synth(3000)
    m = build_model(X.shape[1])
    m.fit(X, y, epochs=4, batch_size=64, verbose=2, validation_split=0.1)

    artifacts = os.environ.get("ARTIFACTS_DIR","artifacts")
    os.makedirs(artifacts, exist_ok=True)
    m.save(os.path.join(artifacts, "model.keras"))
    with open(os.path.join(artifacts, "meta.json"), "w") as f:
        json.dump({"GENRES":GENRES, "PLATFORMS":PLATFORMS, "COUNTRIES":COUNTRIES}, f)
    print("Artifacts saved in", artifacts)

if __name__ == "__main__":
    main()
