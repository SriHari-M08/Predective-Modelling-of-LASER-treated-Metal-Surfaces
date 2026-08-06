from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "Al_Laser_Results.csv"   # fill in template → rename to this
MODEL_DIR  = BASE_DIR / "models"

# ── Model hyperparameters ────────────────────────────────────────────────────
SEED         = 42
N_ESTIMATORS = 100    # fewer trees — small dataset
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ── Feature / target definitions ─────────────────────────────────────────────
FEATURES = [
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Pulse_Frequency_kHz",
    "Hatch_Spacing_um",
]

FEATURE_LABELS = [
    "Laser Power (W)",
    "Scanning Speed (mm/s)",
    "Pulse Frequency (kHz)",
    "Hatch Spacing (µm)",
]

TARGETS       = ["Microhardness_HV", "Surface_Roughness_Ra_um"]
TARGET_KEYS   = ["hardness", "roughness"]
TARGET_LABELS = ["Microhardness", "Surface Roughness (Ra)"]
TARGET_UNITS  = ["HV", "µm"]
TARGET_COLORS = ["#2ca02c", "#1f77b4"]

# ── Parameter input ranges for 50W laser on Aluminium ────────────────────────
PARAM_RANGES = {
    "Laser_Power_W":        (20.0,  50.0,  35.0),   # min, max, default
    "Scanning_Speed_mm_s":  (100.0, 500.0, 300.0),
    "Pulse_Frequency_kHz":  (20.0,  60.0,  40.0),
    "Hatch_Spacing_um":     (40.0,  120.0, 80.0),
}

# ── Taguchi L9 design (for reference) ────────────────────────────────────────
# 4 parameters × 3 levels → 9 runs
TAGUCHI_LEVELS = {
    "Laser_Power_W":       [20.0, 35.0, 50.0],
    "Scanning_Speed_mm_s": [100.0, 300.0, 500.0],
    "Pulse_Frequency_kHz": [20.0, 40.0, 60.0],
    "Hatch_Spacing_um":    [40.0, 80.0, 120.0],
}

# L9 orthogonal array (index into TAGUCHI_LEVELS, 1-indexed)
TAGUCHI_L9 = [
    [1, 1, 1, 1],  # Run 1
    [1, 2, 2, 2],  # Run 2
    [1, 3, 3, 3],  # Run 3
    [2, 1, 2, 3],  # Run 4
    [2, 2, 3, 1],  # Run 5
    [2, 3, 1, 2],  # Run 6
    [3, 1, 3, 2],  # Run 7
    [3, 2, 1, 3],  # Run 8
    [3, 3, 2, 1],  # Run 9
]
