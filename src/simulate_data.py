"""
simulate_data.py
================
Generate a synthetic participant-level dataset that *mimics the structure* of
the Guess et al. (2023) Chronological Feed experiment.

We cannot use the real data (it is restricted-access, held inside Meta's privacy
infrastructure and only released to vetted researchers through ICPSR/SOMAR).
So we build a stand-in that has the same moving parts:

  - pre-treatment covariates (demographics, partisanship, baseline outcomes)
  - block randomisation into Control vs Chronological Feed
  - imperfect compliance in the treatment group
  - survey weights (so we can compute a weighted PATE vs unweighted SATE)
  - outcomes with KNOWN, planted treatment effects taken from the paper

Because we plant the effects ourselves (in config.py), the point of the
exercise is NOT "did we get the same number as Meta" (impossible with fake
data) but "does our analysis pipeline correctly recover the effects we put in,
and reproduce the paper's qualitative pattern?" That is what a methodological
replication can honestly claim.

Every random draw is seeded so the dataset is identical run to run.
"""

import numpy as np
import pandas as pd

import config


def _draw_covariates(rng, n):
    """Draw pre-treatment covariates for n participants.

    These are the variables that (a) we block-randomise on and (b) feed to the
    lasso covariate selector later. Distributions are rough but plausible for a
    US adult social-media sample; exact shapes do not affect a synthetic study.
    """
    df = pd.DataFrame(index=range(n))

    df["age_group"] = rng.choice(
        ["18-29", "30-44", "45-65", "65+"], size=n, p=[0.22, 0.30, 0.33, 0.15]
    )
    df["gender"] = rng.choice(
        ["Female", "Male", "Other"], size=n, p=[0.53, 0.45, 0.02]
    )
    df["race"] = rng.choice(
        ["White_NH", "Black_NH", "Hispanic", "AAPI", "Other"],
        size=n, p=[0.62, 0.12, 0.16, 0.06, 0.04],
    )
    df["party_id"] = rng.choice(
        ["Democrat", "Republican", "Independent"], size=n, p=[0.40, 0.35, 0.25]
    )
    # 7-point ideology, correlated with party (1 = very liberal, 7 = very conservative)
    party_centre = df["party_id"].map(
        {"Democrat": 3.0, "Republican": 5.2, "Independent": 4.1}
    )
    df["ideology"] = np.clip(rng.normal(party_centre, 1.0), 1, 7)

    df["college_degree"] = rng.binomial(1, 0.36, size=n)
    df["political_interest"] = rng.integers(1, 5, size=n)           # 1-4
    df["turnout_2016"] = rng.binomial(1, 0.70, size=n)

    # Self-reported news consumption channels (0-1 intensity scores).
    for ch in ["news_tv", "news_cable", "news_online", "news_social", "news_paper"]:
        df[ch] = np.clip(rng.beta(2, 5, size=n), 0, 1)

    df["pol_participation_pre"] = rng.integers(0, 7, size=n)        # 0-6 acts
    df["digital_literacy"] = np.clip(rng.normal(0, 1, size=n), -3, 3)

    return df


def _assign_treatment_blockwise(rng, df, n_treat, n_control):
    """Block-randomise treatment within strata.

    The paper randomises within strata so that treatment and control are
    balanced on key covariates. We define a stratum as (age_group x party_id),
    which gives 12 blocks, and within each block assign the global treatment
    share. This is a readable stand-in for their more elaborate blocking.
    """
    df = df.copy()
    df["stratum"] = df["age_group"].astype(str) + " | " + df["party_id"].astype(str)

    treat_share = n_treat / (n_treat + n_control)
    treatment = np.zeros(len(df), dtype=int)

    for _, idx in df.groupby("stratum").groups.items():
        idx = np.array(list(idx))
        k = int(round(len(idx) * treat_share))
        chosen = rng.choice(idx, size=k, replace=False)
        treatment[chosen] = 1

    df["treatment"] = treatment
    return df


def _draw_compliance(rng, df, noncompliance_rate):
    """Compliance = share of feed views actually shown in chronological order.

    Control units never get the chrono feed (~0). Treatment compliers get
    almost all views in chrono order (~0.97). A fraction of treatment units are
    non-compliers (the web-bug group) and sit much lower. This column is the
    instrument-able 'dose' used in the IV / CACE extension.
    """
    df = df.copy()
    n = len(df)
    pct = np.zeros(n)

    is_treat = df["treatment"].values == 1
    # Compliers: high chrono share.
    pct[is_treat] = np.clip(rng.normal(0.97, 0.03, size=is_treat.sum()), 0, 1)
    # Flip a random subset of treated units to non-compliers (web bug).
    treat_idx = np.where(is_treat)[0]
    n_nc = int(round(len(treat_idx) * noncompliance_rate))
    nc_idx = rng.choice(treat_idx, size=n_nc, replace=False)
    pct[nc_idx] = np.clip(rng.normal(0.35, 0.20, size=n_nc), 0, 1)
    # Control: essentially no chrono exposure.
    pct[~is_treat] = np.clip(rng.normal(0.01, 0.01, size=(~is_treat).sum()), 0, 1)

    df["pct_views_chrono"] = pct
    df["complier"] = (df["pct_views_chrono"] >= 0.80).astype(int)
    return df


def _draw_weights(rng, df):
    """Survey (post-stratification) weights.

    Real weights rake the sample to population margins. We approximate by
    giving each demographic cell a target population share and weighting by
    (population share / sample share). The spread in weights is what makes the
    weighted PATE noisier (larger SE) than the unweighted SATE, exactly the
    pattern the paper shows.
    """
    df = df.copy()
    # Under-represented groups (younger, non-white, less educated) get up-weighted.
    # The spread here is deliberately wide so the weighted PATE has a realistic
    # "design effect" (variance inflation) of roughly 3x relative to the
    # unweighted SATE -- matching the paper, where PATE standard errors are
    # noticeably larger than SATE standard errors. The design effect is
    # 1 + CV(weight)^2, so we target a weight coefficient of variation ~1.4.
    base = np.ones(len(df))
    base *= df["age_group"].map({"18-29": 2.2, "30-44": 1.1, "45-65": 0.7, "65+": 0.8}).values
    base *= df["race"].map(
        {"White_NH": 0.75, "Black_NH": 1.5, "Hispanic": 1.7, "AAPI": 1.3, "Other": 1.1}
    ).values
    base *= np.where(df["college_degree"].values == 1, 0.7, 1.35)
    # Multiplicative log-normal noise so weights are continuous, then normalise.
    base *= rng.lognormal(0.0, 0.85, size=len(df))
    df["weight"] = base / base.mean()
    return df


def _covariate_signal(df, rng, strength=0.45):
    """A reproducible linear index of covariates, standardised to ~N(0,1).

    Used to give outcomes realistic predictability from pre-treatment
    variables, so the lasso step and covariate adjustment actually do something.
    """
    x = (
        0.5 * (df["ideology"].values - 4)
        + 0.4 * (df["political_interest"].values - 2.5)
        + 0.3 * (df["pol_participation_pre"].values - 3)
        + 0.6 * df["digital_literacy"].values
        + 0.4 * df["turnout_2016"].values
    )
    x = (x - x.mean()) / x.std()
    return strength * x


def _add_rq_outcomes(df, rng):
    """First-stage feed-composition outcomes (percentage points)."""
    df = df.copy()
    signal = _covariate_signal(df, rng)
    for name, spec in config.RQ_OUTCOMES.items():
        noise = rng.normal(0, spec["control_sd"], size=len(df))
        y = (
            spec["control_mean"]
            + spec["true_effect"] * df["treatment"].values   # planted PATE
            + spec["control_sd"] * 0.3 * signal               # covariate signal
            + noise
        )
        df[name] = np.clip(y, 0, 100)
    return df


def _add_standardized_outcomes(df, rng):
    """Attitude / behaviour outcomes in SD units, with pre-treatment versions.

    Each post-treatment outcome = (persistent person component carried from the
    pre-treatment wave) + (planted treatment effect) + noise, then z-scored.
    The pre-treatment column ('<name>_pre') is what the lasso will lean on.
    """
    df = df.copy()
    for name, spec in config.all_standardized_outcomes().items():
        # Stable person trait, partly explained by covariates.
        trait = _covariate_signal(df, rng) + rng.normal(0, 0.9, size=len(df))
        pre = trait + rng.normal(0, 0.6, size=len(df))                  # Wave 1/2 measure
        post = (
            0.65 * trait                                               # carry-over
            + spec["true_effect"] * df["treatment"].values            # planted effect
            + rng.normal(0, 0.75, size=len(df))                       # fresh noise
        )
        # Standardise so effects are interpretable in SD units.
        df[name + "_pre"] = (pre - pre.mean()) / pre.std()
        df[name] = (post - post.mean()) / post.std()
    return df


def simulate_platform(platform, n_treat, n_control, noncompliance_rate, seed):
    """Build one platform's worth of synthetic participants."""
    rng = np.random.default_rng(seed)
    n = n_treat + n_control

    df = _draw_covariates(rng, n)
    df = _assign_treatment_blockwise(rng, df, n_treat, n_control)
    df = _draw_compliance(rng, df, noncompliance_rate)
    df = _draw_weights(rng, df)
    df = _add_rq_outcomes(df, rng)
    df = _add_standardized_outcomes(df, rng)

    df.insert(0, "platform", platform)
    df.insert(1, "participant_id", [f"{platform[:2].upper()}{i:06d}" for i in range(n)])
    return df


def simulate_all(seed=config.RANDOM_SEED):
    """Generate Facebook + Instagram and return a single combined DataFrame.

    Facebook carries the full outcome set. Instagram is generated with the same
    machinery; in the real study a few feed-composition classifiers (cross-
    cutting / political-news) are Facebook-only, so we blank those for IG to
    stay faithful to what is actually measurable on each platform.
    """
    fb = simulate_platform("facebook", config.N_TREAT_FB, config.N_CONTROL_FB,
                           config.NONCOMPLIANCE_RATE_FB, seed)
    ig = simulate_platform("instagram", config.N_TREAT_IG, config.N_CONTROL_IG,
                           0.0, seed + 1)
    # Facebook-only feed classifiers -> not available on Instagram.
    for col in ["pct_cross_cutting", "pct_political_news"]:
        ig[col] = np.nan

    return pd.concat([fb, ig], ignore_index=True)


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    data = simulate_all()
    path = os.path.join(out_dir, "synthetic_participants.csv")
    data.to_csv(path, index=False)
    print(f"Wrote {len(data):,} rows to {path}")
    print(data.groupby(['platform', 'treatment']).size())
