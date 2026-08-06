@'
# Predictive Modelling of LASER-treated Metal Surfaces

## Aluminium Laser Surface Processing — Taguchi DOE + ML

Complete workflow: design experiments → do physical runs → Taguchi analysis → ML model.

---

### Your machine: 50W laser on Aluminium — key facts

| Property | Aluminium | Ti-6Al-4V |
| :--- | :--- | :--- |
| **Reflectivity at 1064nm** | ~90% | ~60% |
| **Effective absorbed power** | ~5–10W of your 50W | Much more |
| **Thermal conductivity** | 205 W/m·K (very high) | 7 W/m·K |
| **Melting point** | 660°C | 1660°C |
| **Expected hardness range** | 55–110 HV (modest) | 300–550 HV |
| **Goal** | Surface texturing + oxide modification | Deep hardening |

> **Bottom line:** Your 50W laser will create measurable surface texture and modest hardness increases. Don't expect dramatic hardening — that's physically limited by Al's properties. Roughness and tribological changes will be more significant.

---

### Full workflow

#### Step 1 — Prepare samples
* Cut 27 identical Al coupons (e.g. 6061 alloy, 30×30×5mm).
* Sand all surfaces to 600-grit using the same technique.
* Clean with acetone, dry with nitrogen/air.
* Label coupons: Run1_Rep1 through Run9_Rep3.

#### Step 2 — Set up your laser
* Spot diameter: measure and note it (typically 50–200µm for a 50W fiber laser).
* Pattern: parallel scan lines covering a 10×10mm area.
* Focus: set to the surface (defocusing increases spot size and reduces intensity).
* Confirm your machine's power is calibrated (use a power meter if available).

#### Step 3 — Run the 9 experiments in the L9 order
For each run, use exactly these settings:

| Run | Power (W) | Speed (mm/s) | Frequency (kHz) | Hatch (µm) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 20 | 100 | 20 | 40 |
| **2** | 20 | 300 | 40 | 80 |
| **3** | 20 | 500 | 60 | 120 |
| **4** | 35 | 100 | 40 | 120 |
| **5** | 35 | 300 | 60 | 40 |
| **6** | 35 | 500 | 20 | 80 |
| **7** | 50 | 100 | 60 | 80 |
| **8** | 50 | 300 | 20 | 120 |
| **9** | 50 | 500 | 40 | 40 |

* Repeat each run 3 times on 3 separate coupons.

#### Step 4 — Measure each sample
Take measurements at 3 locations per sample and average them:
* **Microhardness:** Vickers, 100gf load (HV0.1), 10 second dwell time.
* **Surface roughness:** Ra in µm, profilometer cutoff = 0.8mm, evaluation length = 4mm.
* Record everything in `data/Al_Experiment_Template.csv`.

#### Step 5 — Enter results
1. Open `data/Al_Experiment_Template.csv`.
2. Fill in `Microhardness_HV` and `Surface_Roughness_Ra_um` for all 27 rows.
3. Save as `data/Al_Laser_Results.csv`.

#### Step 6 — Run Taguchi analysis

```bash
python taguchi_analysis.py
```

This produces:
* S/N ratios for each run.
* Main effects plots (which parameter matters most).
* ANOVA table (statistical significance).
* Optimal parameter recommendation.
* Excel summary: `results/Taguchi_Summary.xlsx`.
* Plots in `results/`.

#### Step 7 — Train the ML model

```bash
python train.py
```

* The script automatically picks the best model (RF vs GBM vs Ridge) using Leave-One-Out CV — appropriate for small datasets. 
* Saves the best model to `models/`.

#### Step 8 — Launch the prediction app

```bash
python -m streamlit run app.py
```

---

### Files

| File | Run when |
| :--- | :--- |
| `data/Al_Experiment_Template.csv` | Fill in during experiments |
| `taguchi_analysis.py` | After filling in results |
| `train.py` | After taguchi analysis |
| `app.py` | After training |
| `config.py` | Edit if you change parameter ranges |

---

### Why Taguchi and not full factorial?

* Full factorial with 4 parameters × 3 levels = 3⁴ = **81 experiments**.
* Taguchi L9 gives you = **9 experiments** (×3 replicates = 27 total).

The L9 orthogonal array is mathematically designed so that each level of each factor appears exactly the same number of times — this lets you isolate each parameter's effect independently, even though you're running far fewer experiments. You lose the ability to detect interaction effects (e.g. does Power × Speed interact?) but for a first study this tradeoff is always worth it.

---

### Warning about ML with 27 samples

27 data points is small for ML. The model will work but:
* Do NOT trust predictions far outside the training range.
* The uncertainty (±σ shown in the app) will be wide — that's honest.
* After the Taguchi analysis identifies the best parameters, run 3–5 "confirmation experiments" at the optimal settings and add those to your dataset before trusting the ML model.

This is standard practice in Taguchi methodology — always validate with confirmation runs.