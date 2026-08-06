"""
taguchi_analysis.py — Full Taguchi Analysis for Aluminium Laser Experiments
=============================================================================
Run AFTER you fill in Al_Experiment_Template.csv with your real measurements.

Rename the filled template to:   data/Al_Laser_Results.csv
Then run:                         python taguchi_analysis.py

What this script does:
  1. Loads your 27 measurements (9 runs × 3 replicates)
  2. Calculates S/N ratios for each run
  3. Calculates main effects (average S/N per factor level)
  4. Performs ANOVA to find statistically significant parameters
  5. Identifies optimal parameter combination
  6. Generates a PDF-ready analysis report with plots
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
RESULTS_CSV    = BASE_DIR / "data" / "Al_Laser_Results.csv"
OUTPUT_DIR     = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

FACTORS        = ["Laser_Power_W", "Scanning_Speed_mm_s", "Pulse_Frequency_kHz", "Hatch_Spacing_um"]
FACTOR_LABELS  = ["Power (W)", "Speed (mm/s)", "Frequency (kHz)", "Hatch (µm)"]
FACTOR_LETTERS = ["A", "B", "C", "D"]

TAGUCHI_LEVELS = {
    "Laser_Power_W":       [20.0, 35.0, 50.0],
    "Scanning_Speed_mm_s": [100.0, 300.0, 500.0],
    "Pulse_Frequency_kHz": [20.0, 40.0, 60.0],
    "Hatch_Spacing_um":    [40.0, 80.0, 120.0],
}

# L9 array mapping: each row = [A, B, C, D] levels (1-indexed)
L9 = np.array([
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
])


# ─────────────────────────────────────────────────────────────────────────────
# S/N RATIO FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def sn_larger_is_better(values: np.ndarray) -> float:
    """S/N = -10 * log10(mean(1/y²))  — for hardness (want HIGH)"""
    return -10 * np.log10(np.mean(1.0 / (values ** 2)))


def sn_smaller_is_better(values: np.ndarray) -> float:
    """S/N = -10 * log10(mean(y²))  — for roughness (want LOW)"""
    return -10 * np.log10(np.mean(values ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    if not RESULTS_CSV.exists():
        # Demo: generate synthetic results so you can see the analysis working
        print(f"⚠️  Results file not found: {RESULTS_CSV}")
        print("    Running with SYNTHETIC demo data — replace with your real measurements.\n")
        rng = np.random.default_rng(42)
        rows = []
        for run_idx, (a, b, c, d) in enumerate(L9, start=1):
            p   = TAGUCHI_LEVELS["Laser_Power_W"][a - 1]
            s   = TAGUCHI_LEVELS["Scanning_Speed_mm_s"][b - 1]
            f   = TAGUCHI_LEVELS["Pulse_Frequency_kHz"][c - 1]
            h   = TAGUCHI_LEVELS["Hatch_Spacing_um"][d - 1]
            ed  = (p * 1000) / (s * h)
            # Al physics: lower base hardness, modest ED effect
            base_hv = 60 + 20 * np.exp(-0.5 * ((ed - 5) / 2.5) ** 2)
            base_ra = 0.8 + 0.4 * ed - 0.005 * f

            for rep in range(1, 4):
                hv = float(np.clip(base_hv + rng.normal(0, 2), 50, 130))
                ra = float(np.clip(base_ra + rng.normal(0, 0.05), 0.1, 4.0))
                rows.append({
                    "Run": run_idx, "Replicate": rep,
                    "Laser_Power_W": p, "Scanning_Speed_mm_s": s,
                    "Pulse_Frequency_kHz": f, "Hatch_Spacing_um": h,
                    "Energy_Density_J_mm2": round(ed, 3),
                    "Microhardness_HV": round(hv, 1),
                    "Surface_Roughness_Ra_um": round(ra, 3),
                })
        return pd.DataFrame(rows)

    df = pd.read_csv(RESULTS_CSV)
    # Drop rows with missing measurements
    df = df.dropna(subset=["Microhardness_HV", "Surface_Roughness_Ra_um"])
    print(f"✅ Loaded {len(df)} measurements from {RESULTS_CSV.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TAGUCHI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sn_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-run mean, std, and S/N ratios."""
    records = []
    for run in sorted(df["Run"].unique()):
        sub = df[df["Run"] == run]
        hv  = sub["Microhardness_HV"].values
        ra  = sub["Surface_Roughness_Ra_um"].values

        row = L9[run - 1]  # [A, B, C, D] levels for this run

        records.append({
            "Run":              run,
            "A_Power_W":        TAGUCHI_LEVELS["Laser_Power_W"][row[0] - 1],
            "B_Speed_mm_s":     TAGUCHI_LEVELS["Scanning_Speed_mm_s"][row[1] - 1],
            "C_Freq_kHz":       TAGUCHI_LEVELS["Pulse_Frequency_kHz"][row[2] - 1],
            "D_Hatch_um":       TAGUCHI_LEVELS["Hatch_Spacing_um"][row[3] - 1],
            "HV_mean":          np.mean(hv),
            "HV_std":           np.std(hv, ddof=1),
            "Ra_mean":          np.mean(ra),
            "Ra_std":           np.std(ra, ddof=1),
            "SN_HV":            sn_larger_is_better(hv),    # larger is better
            "SN_Ra":            sn_smaller_is_better(ra),   # smaller is better
            "A_idx": row[0], "B_idx": row[1], "C_idx": row[2], "D_idx": row[3],
        })
    return pd.DataFrame(records)


def compute_main_effects(sn_table: pd.DataFrame, sn_col: str) -> dict:
    """For each factor, compute average S/N at each level."""
    effects = {}
    for factor_letter, col_idx in zip(FACTOR_LETTERS, ["A_idx", "B_idx", "C_idx", "D_idx"]):
        level_means = {}
        for lv in [1, 2, 3]:
            mask = sn_table[col_idx] == lv
            level_means[lv] = sn_table.loc[mask, sn_col].mean()
        effects[factor_letter] = level_means
        effects[f"{factor_letter}_delta"] = max(level_means.values()) - min(level_means.values())
    return effects


def anova(sn_table: pd.DataFrame, sn_col: str) -> pd.DataFrame:
    """Simple ANOVA on S/N ratios — calculates SS, DOF, MS, F, contribution %."""
    grand_mean = sn_table[sn_col].mean()
    ss_total   = ((sn_table[sn_col] - grand_mean) ** 2).sum()

    rows = []
    for fl, col_idx in zip(FACTOR_LETTERS, ["A_idx", "B_idx", "C_idx", "D_idx"]):
        ss_factor = 0
        for lv in [1, 2, 3]:
            mask    = sn_table[col_idx] == lv
            lv_mean = sn_table.loc[mask, sn_col].mean()
            n_lv    = mask.sum()
            ss_factor += n_lv * (lv_mean - grand_mean) ** 2
        dof = 2  # 3 levels → 2 DOF
        rows.append({
            "Factor": fl,
            "SS":     ss_factor,
            "DOF":    dof,
            "MS":     ss_factor / dof,
            "Contribution_%": 100 * ss_factor / ss_total if ss_total > 0 else 0,
        })

    anova_df          = pd.DataFrame(rows)
    ss_error          = ss_total - anova_df["SS"].sum()
    anova_df["F_ratio"] = anova_df["MS"] / (ss_error / max(ss_total - anova_df["SS"].sum(), 1e-9))
    return anova_df


def find_optimal_levels(effects_hv: dict, effects_ra: dict) -> dict:
    """Find level that maximises S/N for each factor for each objective."""
    optimal = {}
    for fl in FACTOR_LETTERS:
        lv_hv = max(effects_hv[fl], key=effects_hv[fl].get)
        lv_ra = max(effects_ra[fl], key=effects_ra[fl].get)
        factor_key = list(TAGUCHI_LEVELS.keys())[FACTOR_LETTERS.index(fl)]
        optimal[fl] = {
            "best_for_hardness":  TAGUCHI_LEVELS[factor_key][lv_hv - 1],
            "best_for_roughness": TAGUCHI_LEVELS[factor_key][lv_ra - 1],
            "conflict":           lv_hv != lv_ra,
        }
    return optimal


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_effects(effects: dict, sn_col_label: str, objective_label: str,
                      filename: str):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    level_vals = {
        "A": TAGUCHI_LEVELS["Laser_Power_W"],
        "B": TAGUCHI_LEVELS["Scanning_Speed_mm_s"],
        "C": TAGUCHI_LEVELS["Pulse_Frequency_kHz"],
        "D": TAGUCHI_LEVELS["Hatch_Spacing_um"],
    }
    for ax, fl, label in zip(axes, FACTOR_LETTERS, FACTOR_LABELS):
        x   = level_vals[fl]
        y   = [effects[fl][1], effects[fl][2], effects[fl][3]]
        ax.plot(x, y, "o-", color="#2ca02c", linewidth=2, markersize=7)
        ax.set_title(f"{fl}: {label}", fontsize=10)
        ax.set_xlabel("Level value", fontsize=9)
        ax.set_ylabel("S/N (dB)" if ax == axes[0] else "", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

    plt.suptitle(f"Main Effects Plot — S/N Ratio  ({objective_label})", fontsize=12)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sn_bar(sn_table: pd.DataFrame, sn_col: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    runs    = sn_table["Run"].astype(str)
    values  = sn_table[sn_col]
    colors  = ["#2ca02c" if v >= values.median() else "#1f77b4" for v in values]
    bars    = ax.bar(runs, values, color=colors, alpha=0.8)
    ax.axhline(values.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {values.mean():.2f} dB")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Experiment Run", fontsize=11)
    ax.set_ylabel("S/N Ratio (dB)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_response_surface(sn_table: pd.DataFrame):
    """Power vs Speed S/N response surface for hardness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sn_col, title in zip(
        axes,
        ["SN_HV", "SN_Ra"],
        ["Hardness S/N (larger-is-better)", "Roughness S/N (smaller-is-better)"],
    ):
        sc = ax.scatter(
            sn_table["A_Power_W"],
            sn_table["B_Speed_mm_s"],
            c=sn_table[sn_col],
            cmap="viridis",
            s=300,
            edgecolors="black",
            linewidth=0.7,
            zorder=5,
        )
        fig.colorbar(sc, ax=ax, label="S/N (dB)")
        for _, row in sn_table.iterrows():
            ax.annotate(f"Run {int(row['Run'])}", (row["A_Power_W"], row["B_Speed_mm_s"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_xlabel("Laser Power (W)", fontsize=10)
        ax.set_ylabel("Scanning Speed (mm/s)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)

    plt.suptitle("Power × Speed — S/N Response Map", fontsize=12)
    plt.tight_layout()
    path = OUTPUT_DIR / "response_surface.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(sn_table, anova_hv, anova_ra, effects_hv, effects_ra, optimal):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  TAGUCHI ANALYSIS REPORT — Aluminium Laser Surface Processing")
    print(sep)

    print("\n── S/N Ratios per Run ──────────────────────────────────────")
    cols = ["Run", "A_Power_W", "B_Speed_mm_s", "C_Freq_kHz", "D_Hatch_um",
            "HV_mean", "HV_std", "Ra_mean", "Ra_std", "SN_HV", "SN_Ra"]
    print(sn_table[cols].round(3).to_string(index=False))

    print("\n── ANOVA — Microhardness S/N ───────────────────────────────")
    print(anova_hv.round(4).to_string(index=False))
    top_hv = anova_hv.sort_values("Contribution_%", ascending=False).iloc[0]
    print(f"\n  Most influential factor for Hardness: "
          f"{top_hv['Factor']} ({top_hv['Contribution_%']:.1f}% contribution)")

    print("\n── ANOVA — Surface Roughness S/N ───────────────────────────")
    print(anova_ra.round(4).to_string(index=False))
    top_ra = anova_ra.sort_values("Contribution_%", ascending=False).iloc[0]
    print(f"\n  Most influential factor for Roughness: "
          f"{top_ra['Factor']} ({top_ra['Contribution_%']:.1f}% contribution)")

    print("\n── Optimal Parameter Levels ────────────────────────────────")
    factor_keys = list(TAGUCHI_LEVELS.keys())
    for i, fl in enumerate(FACTOR_LETTERS):
        opt = optimal[fl]
        conflict = " ← CONFLICT — pick based on priority" if opt["conflict"] else ""
        print(f"  {fl} ({factor_keys[i]})")
        print(f"     Best for Hardness  : {opt['best_for_hardness']}")
        print(f"     Best for Roughness : {opt['best_for_roughness']}{conflict}")

    print(f"\n{sep}")
    print("  Plots saved to: results/")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE SUMMARY EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def save_excel_summary(sn_table, anova_hv, anova_ra, df_raw):
    path = OUTPUT_DIR / "Taguchi_Summary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        sn_table.round(4).to_excel(writer, sheet_name="SN_Table", index=False)
        anova_hv.round(4).to_excel(writer, sheet_name="ANOVA_Hardness", index=False)
        anova_ra.round(4).to_excel(writer, sheet_name="ANOVA_Roughness", index=False)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Taguchi Analysis — 50W Laser on Aluminium")
    print("=" * 65)

    # 1. Load
    df = load_data()

    # 2. Compute S/N table
    sn_table = compute_sn_table(df)

    # 3. Main effects
    effects_hv = compute_main_effects(sn_table, "SN_HV")
    effects_ra = compute_main_effects(sn_table, "SN_Ra")

    # 4. ANOVA
    anova_hv = anova(sn_table, "SN_HV")
    anova_ra = anova(sn_table, "SN_Ra")

    # 5. Optimal levels
    optimal = find_optimal_levels(effects_hv, effects_ra)

    # 6. Print report
    print_report(sn_table, anova_hv, anova_ra, effects_hv, effects_ra, optimal)

    # 7. Plots
    print("Generating plots …")
    plot_sn_bar(sn_table, "SN_HV",
                "S/N Ratio per Run — Microhardness (Larger-is-Better)",
                "sn_hardness.png")
    plot_sn_bar(sn_table, "SN_Ra",
                "S/N Ratio per Run — Surface Roughness (Smaller-is-Better)",
                "sn_roughness.png")
    plot_main_effects(effects_hv, "SN_HV", "Microhardness — Larger is Better",
                      "main_effects_hardness.png")
    plot_main_effects(effects_ra, "SN_Ra", "Surface Roughness — Smaller is Better",
                      "main_effects_roughness.png")
    plot_response_surface(sn_table)

    # 8. Excel
    try:
        import openpyxl
        save_excel_summary(sn_table, anova_hv, anova_ra, df)
    except ImportError:
        print("  (openpyxl not installed — skipping Excel export. Run: pip install openpyxl)")

    print("\nDone. Open the results/ folder to see all plots and the summary spreadsheet.")
