"""
run_replication.py
==================
End-to-end driver: simulate -> estimate -> save tables -> draw figures.

Run from the repo root:  python src/run_replication.py
Outputs land in  data/ , results/ , and figures/ .

This mirrors, on synthetic data, the analysis behind the Facebook results in
Guess et al. (2023): the first-stage effects on feed composition (large and
significant) and the downstream effects on attitudes/behaviour (almost all
null after correction). See README.md and replication_log.md.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

import config
import simulate_data
import estimators

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
for d in (DATA, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)


def get_data():
    """Generate the synthetic data once and cache it to CSV."""
    path = os.path.join(DATA, "synthetic_participants.csv")
    if not os.path.exists(path):
        data = simulate_data.simulate_all()
        data.to_csv(path, index=False)
        print(f"Generated synthetic data -> {path}")
    else:
        data = pd.read_csv(path)
        print(f"Loaded cached synthetic data <- {path}")
    return data


def run_rq_first_stage(fb):
    """First-stage / RQ outcomes (feed composition, percentage points).

    The paper treats these as auxiliary and does NOT apply the FDR correction,
    so we just report the weighted PATE and its raw p-value.
    """
    rows = []
    for name, spec in config.RQ_OUTCOMES.items():
        r = estimators.lin_estimator(fb, name, weight_col="weight", pre_treatment_col=None)
        rows.append({
            "outcome": name, "label": spec["label"],
            "true_effect": spec["true_effect"],
            "pate": r["estimate"], "pate_se": r["se"],
            "pate_ci_low": r["ci_low"], "pate_ci_high": r["ci_high"],
            "pate_p_raw": r["p_raw"], "n": r["n"],
        })
    return pd.DataFrame(rows)


def forest_plot(res, title, unit, path, signif_col="pate_p_adj"):
    """Horizontal point-and-CI plot, the style of SM Figs S7-S9."""
    res = res.iloc[::-1].reset_index(drop=True)  # first outcome on top
    y = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(res) + 1.5))
    for i, row in res.iterrows():
        sig = (signif_col in row) and (row[signif_col] <= 0.05)
        color = "#c0392b" if sig else "#34495e"
        ax.plot([row["pate_ci_low"], row["pate_ci_high"]], [y[i], y[i]],
                color=color, lw=2, zorder=2)
        ax.scatter([row["pate"]], [y[i]], color=color, s=45, zorder=3)
        # Faint marker for the planted truth, for the synthetic check.
        if not np.isnan(row.get("true_effect", np.nan)):
            ax.scatter([row["true_effect"]], [y[i]], marker="|", color="#2980b9",
                       s=120, zorder=4)

    ax.axvline(0, color="#999999", lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(res["label"])
    ax.set_xlabel(f"Treatment effect (chronological feed), {unit}")
    ax.set_title(title, fontsize=11, loc="left")
    # Legend.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#c0392b", marker="o", lw=2, label="Significant (FDR adj. p<.05)"),
        Line2D([0], [0], color="#34495e", marker="o", lw=2, label="Not significant"),
        Line2D([0], [0], color="#2980b9", marker="|", lw=0, markersize=12, label="Planted true effect"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {path}")


def main():
    data = get_data()
    fb = data[data["platform"] == "facebook"].copy()

    print("\n=== First stage (RQ): effect on feed composition (Facebook) ===")
    rq = run_rq_first_stage(fb)
    print(rq[["label", "true_effect", "pate", "pate_se", "pate_p_raw"]].to_string(index=False))
    rq.to_csv(os.path.join(RESULTS, "results_rq_first_stage.csv"), index=False)

    print("\n=== Primary hypotheses H1-H3 (Facebook) ===")
    primary = estimators.run_outcome_family(fb, config.PRIMARY_OUTCOMES, q=0.05)
    print(primary[["label", "true_effect", "pate", "pate_p_raw", "pate_p_adj",
                   "sate", "sate_p_raw", "sate_p_adj"]].to_string(index=False))
    primary.to_csv(os.path.join(RESULTS, "results_primary.csv"), index=False)

    print("\n=== Secondary hypotheses (Facebook) ===")
    secondary = estimators.run_outcome_family(fb, config.SECONDARY_OUTCOMES, q=0.05)
    print(secondary[["label", "true_effect", "pate", "pate_p_raw", "pate_p_adj"]].to_string(index=False))
    secondary.to_csv(os.path.join(RESULTS, "results_secondary.csv"), index=False)

    # Figures.
    forest_plot(rq, "First stage: chronological feed changes what you see (Facebook)",
                "percentage points", os.path.join(FIGS, "fig_rq_first_stage.png"),
                signif_col="pate_p_raw")
    forest_plot(primary, "Primary outcomes: attitudes/behaviour (Facebook)",
                "SD units", os.path.join(FIGS, "fig_primary.png"))
    forest_plot(secondary, "Secondary outcomes (Facebook)",
                "SD units", os.path.join(FIGS, "fig_secondary.png"))

    print("\nDone. Tables in results/, figures in figures/.")


if __name__ == "__main__":
    main()
