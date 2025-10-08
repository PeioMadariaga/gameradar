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
    # --- Rango de precios y tratamiento realista ---
    price = rng.uniform(39, 99, size=n).astype(np.float32)   # extiende algo el rango
    
    # Óptimo real ~70–80 €
    ideal_low, ideal_high = 70.0, 80.0
    
    # Penalización en forma de "U" (suave) fuera del intervalo óptimo
    # Cuanto más te alejas, más penaliza (cuadrático)
    pen_low  = np.maximum(0.0, ideal_low  - price)
    pen_high = np.maximum(0.0, price - ideal_high)
    price_penalty = 0.0008*(pen_low**2) + 0.0008*(pen_high**2)
    
    # Señales "premium": secuela / crossplay / coop reducen penalización por caro
    tolerance = 1.0 - 0.25*(0.4*cross + 0.3*coop + 0.3*sequel)
    price_penalty *= np.clip(tolerance, 0.7, 1.0)
    
    # Marketing ayuda un poco a compensar precios altos
    mk_boost = 0.0004 * (budget - 120)
    
    # Aplica efectos
    base = base - price_penalty + mk_boost
    
    # Recorte final
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
