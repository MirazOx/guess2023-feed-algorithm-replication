# Replicating the Chronological Feed experiment (Guess et al., 2023)

A **methodological replication** of:

> Guess, A. M., Malhotra, N., Pan, J., Barberá, P., Allcott, H., et al. (2023). *How do social media feed algorithms affect attitudes and behavior in an election campaign?* **Science** 381, 398. DOI: [10.1126/science.abp9364](https://doi.org/10.1126/science.abp9364)

This repo reconstructs the study's **analysis pipeline** on **synthetic data** and shows that it reproduces the paper's headline pattern. It does **not** use the real (restricted-access) data and makes **no claim about Facebook's actual effects** — see [`replication_log.md`](replication_log.md) for every divergence from the original.

---

## The finding being reproduced

During the 2020 US election, ~23,000 consenting Facebook users were randomly switched from the default **algorithmic feed** to a **reverse-chronological feed** for three months. The result:

| What changed | What didn't |
|---|---|
| **What people saw** — more political content, more from untrustworthy and cross-cutting sources (large, significant first-stage effects) | **Affective polarization** — no change |
| **On-platform behaviour** — political engagement fell | **Issue polarization, knowledge, participation, turnout** — no change |

In short: a feed change that dramatically altered *what users saw* and *what they did* left their *political attitudes* essentially untouched over the study window.

> ⚠️ **Important caveat (from the authors and Science).** Facebook's ranking algorithm was *in flux* during the study because of 2020 election "break-glass" measures — so the control "algorithmic feed" was not a stable baseline ([Thorp & Vinson, 2024 editorial](https://doi.org/10.1126/science.adt2983)). This is inherited by any replication. See log §2.

---

## What "replication" means here

The original data lives inside Meta's privacy infrastructure and is released only to vetted researchers via ICPSR/SOMAR. We cannot access it. So this is a **methodological reconstruction**:

1. **Simulate** a participant-level dataset with the same structure (arms, block randomisation, ~12% compliance bug, survey weights) and **plant the paper's reported effect sizes** as ground truth (from SM Tables S2/S4/S5).
2. **Rebuild the estimator** the paper uses — Lin (2013) regression, lasso-selected controls, HC2 robust SEs, weighted PATE vs unweighted SATE, sharpened FDR correction — with the robust-SE math written by hand so every step is visible.
3. **Show** the pipeline recovers the planted effects and reproduces the qualitative pattern (and the subtler PATE/SATE and FDR behaviour).

The honest test is *recovery of known effects + the right qualitative pattern*, **not** matching Meta's magnitudes (impossible with synthetic data).

---

## Results (synthetic, Facebook)

**First stage — what the feed shows (percentage points):** all six effects recovered close to planted truth and highly significant. ✔

**Primary hypotheses (weighted PATE, the main-text estimand):**

| Outcome | Planted | Recovered PATE | Significant after FDR? |
|---|---:|---:|---|
| H1a Affective polarization | 0.000 | −0.026 | No |
| H1b Issue polarization | 0.000 | +0.009 | No |
| H2a Election knowledge | 0.000 | −0.038 | No |
| H2b News knowledge | −0.025 | −0.040 | No (PATE) / **Yes (SATE)** |
| H3a Self-reported participation | −0.025 | −0.030 | No |
| H3b Self-reported turnout | 0.000 | −0.001 | No |
| **H3c On-platform engagement** | **−0.118** | **−0.154** | **Yes** |

The only significant primary outcome is **H3c — a behaviour, not an attitude** — exactly as in the paper. H2b reproduces the paper's specific quirk where the *unweighted* SATE reaches significance but the *weighted* PATE does not.

Figures are in [`figures/`](figures/): forest plots for the first stage, primary, and secondary families (a blue tick marks the planted truth; red = significant after correction).

---

## Design & power: is the null actually informative?

A null result only means something if the study could have *detected* an effect. So this repo includes a **design/power analysis** ([`src/power_analysis.py`](src/power_analysis.py)) that asks: given the sample size, the covariate adjustment, and the survey weighting, how small an attitude effect could this design reliably find?

| Estimand | Std. error (SD units) | Minimum detectable effect (80% power) |
|---|---:|---:|
| PATE (weighted, main-text) | ~0.020 | **~0.057 SD** |
| SATE (unweighted) | ~0.012 | ~0.033 SD |

The design has ~69% power to detect a 0.05 SD attitude effect under the PATE and ~99% under the SATE. Every observed attitudinal effect is *below* the detectable bound. So the nulls are **informative**: they are consistent with true effects smaller than roughly 0.05 SD — "no effect larger than a small bound," not "we couldn't tell." The closed-form power curve is confirmed by a simulation that plants known effects and re-runs the estimator (the false-positive rate lands at ~5%, as it should). See `figures/fig_power.png`.

This is the part that distinguishes reading a result from understanding it.

---

## Repo structure

```
guess2023-feed-algorithm-replication/
├── README.md                  # this file
├── replication_log.md         # every divergence from the original (read this!)
├── requirements.txt
├── notebooks/
│   └── guess2023_replication.ipynb   # self-contained, Colab-ready walkthrough
├── src/
│   ├── config.py              # outcome definitions + planted effect sizes (from the SM)
│   ├── simulate_data.py       # synthetic dataset generator
│   ├── estimators.py          # Lin estimator, HC2/HC1 SEs, sharpened FDR, IV check
│   ├── power_analysis.py      # minimum-detectable-effect + power curve (analytic + simulation)
│   └── run_replication.py     # end-to-end: simulate -> estimate -> tables -> figures
├── data/                      # generated synthetic CSV (created on first run)
├── results/                   # results tables (CSV)
└── figures/                   # forest plots (PNG)
```

## How to run

**Google Colab (easiest):** open `notebooks/guess2023_replication.ipynb` and Run All. Everything is self-contained; `numpy / pandas / matplotlib / scikit-learn` are preinstalled in Colab.

**Locally:**

```bash
pip install -r requirements.txt
python src/run_replication.py          # writes data/, results/, figures/
python src/power_analysis.py           # writes the design/power analysis + fig_power.png
```

The `src/` modules reproduce the notebook's logic in importable form and add a no-scikit-learn fallback so the pipeline runs even on a minimal install.

---

## Methods glossary (the four techniques used)

- **Lasso covariate selection** — lets the data choose which controls to adjust for by shrinking weak coefficients to zero; used only to pick controls ("post-lasso"), so the treatment estimate stays unbiased.
- **Lin (2013) estimator** — regression adjustment with `treatment × centred-covariate` interactions; never less efficient than a simple difference in means.
- **HC2 / HC1 robust standard errors** — heteroskedasticity-robust "sandwich" SEs; built by hand in `estimators.py` so the math is visible.
- **Sharpened FDR (Benjamini–Krieger–Yekutieli 2006)** — a two-stage false-discovery-rate correction that controls false positives across many hypotheses with more power than plain Benjamini–Hochberg.

---

## Limitations

This is a teaching/method replication on fabricated data. The numbers demonstrate that the *pipeline* works and that the *qualitative pattern* emerges; they are **not evidence about Facebook**. The full list of departures from the original — synthetic data, the algorithm-in-flux caveat, simplified stratification/weighting/FDR-binning, skipped scale construction, omitted heterogeneity and Instagram analyses — is in [`replication_log.md`](replication_log.md).

## Citing the original

If you build on this, cite the original paper (DOI above) and note the Dec 2024 / Mar 2026 errata and the [Science editorial on the algorithm context](https://doi.org/10.1126/science.adt2983). This replication is independent and not affiliated with the authors or Meta.
