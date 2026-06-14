# Replication log — Guess et al. (2023), Chronological Feed experiment

This file tracks **every place our replication departs from the original study**, and why. It is deliberately the most detailed document in the repo: knowing where a replication falls short is the part that shows the paper was actually understood, not just re-run.

Original paper: Guess, A. M., et al. (2023). *How do social media feed algorithms affect attitudes and behavior in an election campaign?* **Science** 381, 398. DOI: [10.1126/science.abp9364](https://doi.org/10.1126/science.abp9364). Methods drawn from the 327-page supplementary materials (v3, with errata of Dec 2024 and Mar 2026).

Each entry is tagged by how much it could change conclusions: **[major]**, **[moderate]**, **[minor]**.

> **How to describe this work (please read before citing or presenting).**
> This is a **methodological reconstruction on synthetic data**, not an empirical replication. It demonstrates that the paper's methods section can be read closely and re-implemented, and that the analysis pipeline recovers known effects and reproduces the study's qualitative pattern. It makes **no empirical claim about Facebook, Instagram, or real users** — the data is simulated and the effect sizes are planted from the published tables. Always present it as "a methodological reconstruction on synthetic data," never as "I replicated the Facebook study."

---

## 1. Data: synthetic, not the real thing **[major]**

The original data is restricted-access — Meta's *US 2020 Facebook & Instagram Election Study*, released only to vetted researchers through ICPSR/SOMAR under a signed agreement, with participant platform data now permanently de-linked from identifiers (per the Dec 2024 erratum). We do not have it.

**What we did instead:** generated a synthetic participant-level dataset (`src/simulate_data.py`) that mirrors the study's *structure* — sample sizes, treatment arms, block randomisation, compliance, weights, and outcome families — and **planted the paper's reported Facebook effect sizes as ground truth** (`src/config.py`, sourced from SM Tables S2/S4/S5).

**Consequence:** we cannot claim to reproduce Meta's *magnitudes* — that is impossible with fabricated data. What we *can* honestly claim is that (a) the analysis pipeline correctly recovers known effects (mean absolute recovery error ≈ 0.017 SD across outcomes), and (b) the study's **qualitative pattern** falls out of that pipeline: huge first-stage exposure effects, null attitudinal effects, a real behavioural (engagement) effect, plus the PATE/SATE and FDR nuances. Robustness was checked across multiple simulation seeds (see §13).

## 2. The algorithm was in flux — the control condition was not fixed **[major, inherited]**

This is the original authors' own caveat, surfaced in the Science editorial we were given ([Thorp & Vinson, 2024](https://doi.org/10.1126/science.adt2983)). During the study window (24 Sep – 23 Dec 2020), Facebook had "break-glass" election emergency measures in place that made the default ranking algorithm *less polarising and more reliable* than usual. So the control group's "algorithmic feed" was a moving target, not a stable baseline. The authors acknowledged in an eLetter that a differently-tweaked algorithm could have produced different results, while maintaining their conclusions stand.

**Consequence for us:** this limitation is baked into the original estimand and therefore into anything we reconstruct. Our synthetic "control" is, if anything, *cleaner* than reality (a fixed data-generating process), so it cannot capture this instability. Any claim of the form "chronological vs. *the* algorithm" should read "chronological vs. *the algorithm as it happened to be configured in autumn 2020*."

## 3. Weighted standard errors: HC1 instead of HC2 **[minor]**

The paper uses **HC2** robust standard errors. HC2 is well-defined for OLS but not standard for weighted least squares. For the **weighted PATE** we therefore use **HC1**, the conventional robust analogue for WLS; for the unweighted SATE we use HC2 exactly as the paper does (`src/estimators.py::_ols_robust`). At n ≈ 23k the HC1/HC2 difference is negligible.

## 4. Covariate selection: lasso, with a non-lasso fallback **[minor]**

The paper selects controls with cross-validated lasso (`cv.glmnet`, 10 folds, seed 2020). The **notebook** uses scikit-learn's `LassoCV` with the same settings — faithful. The **`src/estimators.py` module** adds a fallback (a simple correlation screen, |r| > 0.02) for environments without scikit-learn installed. When the fallback is used, the selected control set differs slightly from a true lasso, which can move estimates at the third decimal. The documented method is lasso; the fallback exists only so the pipeline never hard-fails.

## 5. Stratification is simplified **[moderate]**

The real study block-randomises within an elaborate stratification scheme (SM §S9.1). We approximate strata as **age-group × party-ID (12 blocks)** for readability (`assign_treatment_blockwise`). This is enough to demonstrate within-stratum randomisation and balance, but it is not the original blocking, so covariate balance and the stratum dummies in the regression are coarser than the paper's.

## 6. Multiple-comparison correction: per-family, not cumulative bins **[moderate]**

The paper bins hypotheses and applies **sharpened FDR cumulatively**: K1 (primary) alone, then K2 (secondary) adjusted *together with* K1, then K3 (tertiary) with K1+K2 (SM §S1.11). We implement the sharpened BKY-2006 procedure faithfully (`sharpened_fdr`) but apply it **within each family separately** (primary among primary, secondary among secondary). This is a defensible simplification but it is *not* their exact binning, so our adjusted q-values are not directly comparable to theirs outcome-by-outcome.

## 7. Survey weights are modelled, not raked **[moderate]**

Real weights rake the AmeriSpeak/recruited sample to population margins (SM §S9.5). We instead *construct* weights from a demographic model plus log-normal noise (`draw_weights`), tuned so the **design effect ≈ 3** — i.e. PATE variance roughly 3× SATE variance, reproducing the paper's pattern where weighted SEs exceed unweighted ones. The weights are realistic in *spread* but are not derived from real population targets, and PATE point estimates therefore inherit our weighting model rather than theirs.

## 8. Outcome scales are generated directly, skipping factor analysis **[moderate]**

The paper builds composite outcome scales (affective polarization, issue polarization, trust indices, etc.) via **principal-components analysis with varimax rotation** over many survey items, dropping items that fail to load (SM §S1.5, §S5). We skip scale *construction* entirely and generate each composite as a single latent factor + noise. So we replicate the **analysis of** the scales, not the **building of** them. Anyone extending this should reconstruct at least one scale (e.g. affective polarization from feeling thermometers) to replicate that step.

## 9. Compliance / IV is a just-identified Wald ratio **[minor]**

For the ~11.9% Facebook non-compliance (the web-version bug, SM §S1.8) the paper re-estimates with a full IV framework using weights and covariates to recover population-level LATEs. We implement the **simple Wald / 2SLS ratio** (reduced-form ÷ first-stage, `iv_complier_effect`), which conveys the idea and direction (LATE slightly larger than ITT) without the full covariate-adjusted IV. We also do not reproduce the post-14-Nov-2020 "100% compliance" time split.

## 10. Inference uses the normal approximation **[minor]**

p-values and confidence intervals use the standard normal (z) rather than a t-distribution or finite-sample correction. At n ≈ 23k this is indistinguishable from the paper's approach.

## 11. Scope: a representative subset of outcomes and analyses **[moderate]**

We implement the research question (6 feed-composition outcomes), all **7 primary** hypotheses (H1a–H3c), and **8 of the ~13+ secondary** hypotheses. We **omit**: tertiary (K3) outcomes, heterogeneous-treatment-effect analyses (SM §S2.3, §S4), off-platform passive-tracking outcomes (web visits/domain ideology, SM §S8), the political-violence battery's full treatment, and the parallel **Instagram** analysis (we generate IG data but the notebook analyses Facebook, which carries the full classifier set). The "No Reshares" and other intervention arms are out of scope by design — they are separate papers.

## 12. Differential attrition and missing-data rules not modelled **[minor]**

The paper documents ~19.5% attrition (balanced across arms, SM §S1.9) and a pre-registered missing-data recode (mean/mode for <10% missingness, SM §S1.10). Our synthetic data has no attrition and no missingness, so these procedures are described in the writeup but not exercised in code.

## 13. Robustness of the qualitative replication

Because the result depends on a random draw, we re-ran the primary family across simulation seeds {2020, 7, 99}. In all three: **H3c on-platform engagement was significant** (adj. p between 1e-9 and 1e-14) and the cleanly-null attitudes (H1a, H1b, H2a, turnout) were **non-significant under the PATE** (min adj. p ≥ 0.12). The headline pattern is not a lucky seed. The reproduction of H2b (PATE-null but SATE-significant) is more seed-sensitive because it sits near the boundary by construction — which is itself faithful to the original, where that outcome is exactly the borderline case.

---

### Bottom line

What transfers from this replication: the **analysis pipeline** (Lin estimator, robust SEs, lasso selection, sharpened FDR, PATE/SATE, IV-for-compliance) and the **qualitative finding** (strong first stage, null attitudes, real behavioural effect). What does **not** transfer: the actual magnitudes (synthetic data), the exact weighting/stratification/scale-construction, and any robustness to the real-world instability of the 2020 control algorithm. Treat the numbers here as a demonstration of method, not as evidence about Facebook.
