"""
train.py — Train ML models on Aluminium Laser Experiment Results
=================================================================
Run AFTER completing experiments and filling in your CSV.

Usage:
    python train.py

Important note on sample size:
    27 samples (9 runs × 3 replicates) is small for ML.
    This script uses Leave-One-Out CV and careful settings to
    give honest performance estimates on small data.
    More replicates or additional confirmation runs = better model.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from config import (
        DATA_PATH, MODEL_DIR, SEED, FEATURES, FEATURE_LABELS,
        TARGETS, TARGET_KEYS, TARGET_LABELS, TARGET_UNITS, TARGET_COLORS,
    )
except ImportError:
    BASE_DIR    = Path(__file__).parent
    DATA_PATH   = BASE_DIR / "data" / "Al_Laser_Results.csv"
    MODEL_DIR   = BASE_DIR / "models"
    SEED        = 42
    FEATURES        = ["Laser_Power_W","Scanning_Speed_mm_s","Pulse_Frequency_kHz","Hatch_Spacing_um"]
    FEATURE_LABELS  = ["Laser Power (W)","Scanning Speed (mm/s)","Pulse Frequency (kHz)","Hatch Spacing (µm)"]
    TARGETS         = ["Microhardness_HV","Surface_Roughness_Ra_um"]
    TARGET_KEYS     = ["hardness","roughness"]
    TARGET_LABELS   = ["Microhardness","Surface Roughness (Ra)"]
    TARGET_UNITS    = ["HV","µm"]
    TARGET_COLORS   = ["#2ca02c","#1f77b4"]


def generate_demo_data(n_extra: int = 0) -> pd.DataFrame:
    """
    Physics-based synthetic data for a 50W fiber laser on Al.
    Used for testing the pipeline before real data is available.
    """
    rng = np.random.default_rng(SEED)

    # Base: 27 Taguchi measurements
    from config import TAGUCHI_LEVELS, TAGUCHI_L9
    rows = []
    for run_idx, (a, b, c, d) in enumerate(TAGUCHI_L9, start=1):
        p = TAGUCHI_LEVELS["Laser_Power_W"][a - 1]
        s = TAGUCHI_LEVELS["Scanning_Speed_mm_s"][b - 1]
        f = TAGUCHI_LEVELS["Pulse_Frequency_kHz"][c - 1]
        h = TAGUCHI_LEVELS["Hatch_Spacing_um"][d - 1]
        ed = (p * 1000) / (s * h)

        base_hv = 60 + 22 * np.exp(-0.5 * ((ed - 5) / 2.5) ** 2)
        base_ra = 0.7 + 0.35 * ed - 0.004 * f + 0.001 * h

        for rep in range(3):
            rows.append({
                "Laser_Power_W":           p,
                "Scanning_Speed_mm_s":     s,
                "Pulse_Frequency_kHz":     f,
                "Hatch_Spacing_um":        h,
                "Microhardness_HV":        float(np.clip(base_hv + rng.normal(0, 2), 50, 130)),
                "Surface_Roughness_Ra_um": float(np.clip(base_ra + rng.normal(0, 0.06), 0.1, 4.0)),
            })

    # Optional: add extra random confirmation points
    for _ in range(n_extra):
        p  = rng.uniform(20, 50)
        s  = rng.uniform(100, 500)
        f  = rng.uniform(20, 60)
        h  = rng.uniform(40, 120)
        ed = (p * 1000) / (s * h)
        rows.append({
            "Laser_Power_W":           p,
            "Scanning_Speed_mm_s":     s,
            "Pulse_Frequency_kHz":     f,
            "Hatch_Spacing_um":        h,
            "Microhardness_HV":        float(np.clip(60 + 22 * np.exp(-0.5 * ((ed - 5) / 2.5) ** 2) + rng.normal(0, 2), 50, 130)),
            "Surface_Roughness_Ra_um": float(np.clip(0.7 + 0.35 * ed + rng.normal(0, 0.08), 0.1, 4.0)),
        })

    return pd.DataFrame(rows)


def compare_models(X: np.ndarray, y: np.ndarray, label: str, unit: str) -> dict:
    """
    Compare RF, GBM, and Ridge on Leave-One-Out CV.
    For small datasets (n<50), LOO-CV gives the most honest estimate.
    Returns the best-performing model.
    """
    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=4, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=SEED
        ),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
    }

    loo     = LeaveOneOut()
    results = {}
    print(f"\n  {label} ({unit}) — LOO-CV comparison:")

    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=loo, scoring="r2")
        mean_r2 = scores.mean()
        mae_scores = cross_val_score(model, X, y, cv=loo,
                                     scoring="neg_mean_absolute_error")
        mean_mae = -mae_scores.mean()
        results[name] = {"r2": mean_r2, "mae": mean_mae, "model": model}
        print(f"    {name:<22}: LOO R² = {mean_r2:+.4f}   MAE = {mean_mae:.3f} {unit}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"    → Best: {best_name}")
    return results[best_name]["model"], results


def train_and_save():
    # ── Load data ──────────────────────────────────────────────────────────
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH).dropna(subset=TARGETS)
        using_real = True
        print(f"✅ Loaded real data: {len(df)} samples")
    else:
        print(f"⚠️  No real data found at {DATA_PATH.name}")
        print("    Running with SYNTHETIC demo data.\n")
        df = generate_demo_data()
        using_real = False

    n = len(df)
    print(f"   Dataset: {n} samples  ({'real' if using_real else 'synthetic'})")

    if n < 15:
        print("\n⚠️  WARNING: Very few samples. ML results will be unreliable.")
        print("   Consider adding more confirmation experiments.")

    X = df[FEATURES].values
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Model Selection & Training")
    print("=" * 60)

    best_models = {}
    all_results = {}

    for key, col, label, unit in zip(TARGET_KEYS, TARGETS, TARGET_LABELS, TARGET_UNITS):
        y = df[col].values
        best_model, results = compare_models(X, y, label, unit)

        # Final fit on ALL data
        best_model.fit(X, y)

        model_path = MODEL_DIR / f"model_{key}.pkl"
        joblib.dump(best_model, model_path)

        best_models[key] = best_model
        all_results[key] = results
        print(f"    Saved → {model_path}")

    # ── Feature importance plot (RF models only) ──────────────────────────
    print("\n  Generating feature importance plots …")
    fig, axes = plt.subplots(1, len(TARGET_KEYS), figsize=(5 * len(TARGET_KEYS), 4))
    if len(TARGET_KEYS) == 1:
        axes = [axes]

    for ax, key, label, color in zip(axes, TARGET_KEYS, TARGET_LABELS, TARGET_COLORS):
        model = best_models[key]
        if hasattr(model, "feature_importances_"):
            imp  = model.feature_importances_
        elif hasattr(model, "named_steps"):
            # Pipeline with Ridge — use |coef|
            coef = np.abs(model.named_steps["ridge"].coef_)
            imp  = coef / coef.sum()
        else:
            imp = np.ones(len(FEATURES)) / len(FEATURES)

        idx  = np.argsort(imp)
        ax.barh([FEATURE_LABELS[i] for i in idx], imp[idx], color=color, alpha=0.75)
        ax.set_title(f"{label}\n(feature importance)", fontsize=10)
        ax.set_xlabel("Relative importance", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        for i, (v, j) in enumerate(zip(imp[idx], idx)):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    plt.suptitle("Feature Importance — Al Laser Models", fontsize=12)
    plt.tight_layout()
    imp_path = MODEL_DIR / "feature_importance.png"
    plt.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {imp_path}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("  Run the app with:   python -m streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_save()
