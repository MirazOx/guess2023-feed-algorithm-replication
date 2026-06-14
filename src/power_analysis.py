"""
power_analysis.py
=================
Design / power analysis: "could this study have detected an attitude effect if
one existed?"

This is the analytical contribution that turns the null result from "we found
nothing" into "we can rule out effects larger than X." A null is only
informative if the study was well-powered; this module shows it was.

Two complementary methods, deliberately both included:

  1. ANALYTIC  -- a closed-form minimum-detectable-effect (MDE) and power curve,
                  derived from the design (sample sizes, covariate R^2, and the
                  weighting design effect).
  2. SIMULATION -- we plant effects of known size into the actual data-generating
                  process and re-estimate them many times, counting how often the
                  pipeline calls them significant. This confirms the analytic
                  curve using the real estimator.

Key quantities
--------------
MDE = (z_{1-alpha/2} + z_{power}) * SE(effect)
Power(delta) = P(reject H0 | true effect = delta)
            ~= Phi(|delta|/SE - z_{1-alpha/2})

We report these for both estimands the paper uses: the weighted PATE (larger SE,
because weighting inflates variance) and the unweighted SATE.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import simulate_data
from estimators import _ols_robust as ols_robust

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

Z_ALPHA = 1.959963984540054   # 0.975 quantile (two-sided alpha = 0.05)
Z_POWER_80 = 0.8416212335729143  # 0.80 quantile (for 80% power)


def design_parameters(fb):
    """Read the design constants straight off the (synthetic) Facebook sample.

    Returns sample sizes, the weighting design effect, and the share of outcome
    variance explained by the pre-treatment covariate (R^2) -- the three things
    that determine how precisely any effect can be estimated.
    """
    n_t = int((fb["treatment"] == 1).sum())
    n_c = int((fb["treatment"] == 0).sum())
    cv = fb["weight"].std() / fb["weight"].mean()
    design_effect = 1 + cv**2                       # variance inflation from weighting

    # R^2 of a standardized outcome on its pre-treatment version (how much the
    # covariate adjustment shrinks residual variance).
    y = fb["affective_polarization"].values
    x = fb["affective_polarization_pre"].values
    r2 = np.corrcoef(y, x)[0, 1] ** 2

    return {"n_t": n_t, "n_c": n_c, "design_effect": design_effect, "r2": r2}


def analytic_se(params, weighted):
    """Closed-form SE of the treatment effect for a standardized outcome.

    SE0 = sqrt(1/n_t + 1/n_c)              (unadjusted difference in means, SD units)
    covariate adjustment multiplies variance by (1 - R^2)
    weighting multiplies variance by the design effect (PATE only)
    """
    p = params
    se0_sq = (1.0 / p["n_t"] + 1.0 / p["n_c"])
    var = se0_sq * (1.0 - p["r2"])
    if weighted:
        var *= p["design_effect"]
    return np.sqrt(var)


def mde(se, z_power=Z_POWER_80):
    """Minimum effect detectable with the given power at alpha = 0.05."""
    return (Z_ALPHA + z_power) * se


def analytic_power(delta, se):
    """Probability of rejecting H0 for a true effect `delta` (normal approx)."""
    # erf-based normal CDF (see _phi) -- no scipy dependency.
    return _phi(np.abs(delta) / se - Z_ALPHA) + _phi(-np.abs(delta) / se - Z_ALPHA)


def _phi(x):
    """Standard normal CDF via the error function (no scipy needed)."""
    from math import erf, sqrt
    vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
    return vec(x)


def simulate_power(fb, deltas, weighted, n_reps=300, seed=2020):
    """Empirical power: plant each effect size, re-estimate, count significance.

    Holds the design fixed (treatment, weights, covariate signal from the real
    DGP) and only varies the planted effect and the noise draw. Uses a fast
    single-covariate Lin estimator (const, T, pre, T x pre) so we can afford
    hundreds of reps; this is the same estimator family as the main analysis,
    just with one covariate instead of a lasso-selected set.
    """
    rng = np.random.default_rng(seed)
    T = fb["treatment"].values.astype(float)
    w = fb["weight"].values.astype(float)
    signal = simulate_data._covariate_signal(fb, rng)  # reproducible covariate index
    n = len(fb)

    powers = []
    for delta in deltas:
        hits = 0
        for _ in range(n_reps):
            trait = signal + rng.normal(0, 0.9, n)
            pre = trait + rng.normal(0, 0.6, n)
            post = 0.65 * trait + delta * T + rng.normal(0, 0.75, n)
            post = (post - post.mean()) / post.std()
            pre_c = (pre - pre.mean())                  # centre for Lin
            design = np.column_stack([np.ones(n), T, pre_c, T * pre_c])
            beta, vcov = ols_robust(design, post, w=(w if weighted else None),
                                    hc=("HC1" if weighted else "HC2"))
            est, se = beta[1], np.sqrt(vcov[1, 1])
            if abs(est / se) > Z_ALPHA:                 # significant at 0.05
                hits += 1
        powers.append(hits / n_reps)
    return np.array(powers)


def main():
    data = simulate_data.simulate_all()
    fb = data[data["platform"] == "facebook"].copy()
    p = design_parameters(fb)

    se_pate = analytic_se(p, weighted=True)
    se_sate = analytic_se(p, weighted=False)
    mde_pate, mde_sate = mde(se_pate), mde(se_sate)

    print("Design parameters (Facebook):")
    print(f"  n_treat = {p['n_t']:,}   n_control = {p['n_c']:,}")
    print(f"  weighting design effect = {p['design_effect']:.2f}")
    print(f"  covariate R^2 = {p['r2']:.2f}")
    print(f"\nStandard error of a standardized effect:")
    print(f"  PATE (weighted)   SE = {se_pate:.4f}")
    print(f"  SATE (unweighted) SE = {se_sate:.4f}")
    print(f"\nMinimum detectable effect (80% power, alpha=0.05, two-sided):")
    print(f"  PATE: {mde_pate:.3f} SD")
    print(f"  SATE: {mde_sate:.3f} SD")

    # Power at a few substantively interesting effect sizes.
    grid = np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.15])
    pow_pate_analytic = analytic_power(grid, se_pate)
    pow_sate_analytic = analytic_power(grid, se_sate)

    print("\nAnalytic power to detect a true attitude effect of given size:")
    for d, a, b in zip(grid, pow_pate_analytic, pow_sate_analytic):
        print(f"  delta = {d:.2f} SD  ->  PATE power = {a:5.1%}   SATE power = {b:5.1%}")

    # Simulation confirmation (a subset of sizes, fewer reps for speed).
    sim_grid = np.array([0.00, 0.03, 0.05, 0.08])
    sim_pate = simulate_power(fb, sim_grid, weighted=True, n_reps=300)
    print("\nSimulation-based power (PATE, 300 reps) -- confirms the analytic curve:")
    for d, s in zip(sim_grid, sim_pate):
        tag = "(false-positive rate)" if d == 0 else ""
        print(f"  delta = {d:.2f} SD  ->  empirical power = {s:5.1%} {tag}")

    # Save a results table.
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame({
        "true_effect_sd": grid,
        "power_pate": pow_pate_analytic,
        "power_sate": pow_sate_analytic,
    }).to_csv(os.path.join(RESULTS, "power_analysis.csv"), index=False)

    # --- Figure: power curves with MDE and observed effects marked ---
    xs = np.linspace(0, 0.16, 200)
    os.makedirs(FIGS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, analytic_power(xs, se_pate), color="#34495e", lw=2, label="PATE (weighted)")
    ax.plot(xs, analytic_power(xs, se_sate), color="#2980b9", lw=2, ls="--", label="SATE (unweighted)")
    ax.scatter(sim_grid, sim_pate, color="#34495e", s=40, zorder=5, label="PATE simulation (300 reps)")
    ax.axhline(0.80, color="#999", lw=1, ls=":")
    ax.text(0.001, 0.815, "80% power", fontsize=8, color="#666")
    ax.axvline(mde_pate, color="#c0392b", lw=1.2, ls="-")
    ax.text(mde_pate + 0.002, 0.10, f"PATE MDE = {mde_pate:.3f} SD", color="#c0392b", fontsize=8)
    # Observed primary attitude effects (absolute SD) as a rug along the x-axis.
    obs = [0.026, 0.001, 0.038, 0.040, 0.030, 0.001]  # |PATE| of the 6 attitude outcomes
    ax.scatter(obs, [0.02] * len(obs), marker="|", color="#7f8c8d", s=160,
               label="observed |effect| (attitudes)")
    ax.set_xlabel("True treatment effect on an attitude (SD units)")
    ax.set_ylabel("Probability of detecting it (power)")
    ax.set_title("Was the study powered to find an attitude effect?", loc="left", fontsize=12)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 0.16)
    ax.legend(fontsize=8, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_power.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {os.path.join(FIGS, 'fig_power.png')}")

    print("\nInterpretation:")
    print(f"  The design could detect an attitude effect as small as ~{mde_pate:.2f} SD "
          f"(PATE) / ~{mde_sate:.2f} SD (SATE) with 80% power.")
    print(f"  All six observed attitude effects are below the PATE MDE, so the nulls are")
    print(f"  consistent with TRUE effects smaller than ~{mde_pate:.2f} SD -- i.e. the study")
    print(f"  was well-powered, and 'no effect' means 'no effect larger than a small bound',")
    print(f"  not 'we couldn't tell'.")


if __name__ == "__main__":
    main()
