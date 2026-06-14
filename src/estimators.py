"""
estimators.py
=============
The statistical machinery, written to be read line by line.

The paper estimates each treatment effect with a "saturated" specification:
Lin (2013) regression with covariates chosen by lasso, plus stratum dummies,
with HC2 robust standard errors, reporting a weighted PATE and an unweighted
SATE. P-values are then corrected for multiple comparisons with the "sharpened"
two-stage FDR procedure of Benjamini, Krieger & Yekutieli (2006).

Design choice for this replication: the regression and robust standard errors
are implemented from scratch in NumPy rather than called from statsmodels.
There are two reasons. (1) It keeps every step visible -- you can see exactly
how an HC2 standard error is built, which is the point of a methods replication.
(2) It has no heavy dependencies, so the same code runs in Colab and on a plain
Python install. The one place we use an external library is the lasso covariate
selection (scikit-learn's LassoCV), and even that has a transparent fallback so
the pipeline runs without scikit-learn installed. See replication_log.md.

Four ideas you may not have used before, each explained where it appears:
  1. LASSO covariate selection  -> select_covariates
  2. LIN (2013) estimator        -> lin_estimator
  3. HC2 / HC1 robust SEs        -> _ols_robust  (built by hand)
  4. SHARPENED FDR (BKY 2006)    -> sharpened_fdr
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Build a numeric design matrix of covariates from the raw participant data.
# ---------------------------------------------------------------------------
def build_covariate_matrix(df, pre_treatment_col=None):
    """Turn raw covariate columns into a numeric matrix (one-hot for factors).

    We include the families of pre-treatment variables the paper feeds to its
    lasso: demographics, partisanship/ideology, news consumption, baseline
    participation and digital literacy, plus the pre-treatment version of the
    outcome being modelled (if one exists). 'drop_first=True' avoids the dummy-
    variable trap (perfect collinearity among a factor's indicators).
    """
    numeric = [
        "ideology", "political_interest", "turnout_2016",
        "news_tv", "news_cable", "news_online", "news_social", "news_paper",
        "pol_participation_pre", "digital_literacy", "college_degree",
    ]
    factors = ["age_group", "gender", "race", "party_id"]

    X = df[numeric].copy()
    X = pd.concat([X, pd.get_dummies(df[factors], drop_first=True)], axis=1)
    if pre_treatment_col is not None and pre_treatment_col in df.columns:
        X[pre_treatment_col] = df[pre_treatment_col].values
    return X.astype(float)


# ---------------------------------------------------------------------------
# 1. LASSO covariate selection (post-lasso: select here, refit OLS later).
# ---------------------------------------------------------------------------
def select_covariates(X, y, seed=2020):
    """Pick which covariates to keep.

    WHAT LASSO DOES (one paragraph):
    Ordinary regression keeps every covariate. Lasso adds a penalty on the size
    of the coefficients, shrinking weak ones to exactly zero. The covariates
    whose coefficient survives at the cross-validated penalty are the ones we
    keep. Treatment is NOT in this model -- we use lasso only to choose
    *controls*, then estimate the treatment effect by OLS afterwards
    ("post-lasso"), which keeps the treatment estimate unbiased while soaking up
    residual variance.

    We use scikit-learn's LassoCV when it is available (this is what the paper
    does and what you get in Colab). If scikit-learn is not installed, we fall
    back to a transparent screen: keep covariates whose absolute correlation
    with the outcome exceeds 0.02. The fallback is only a convenience for
    environments without scikit-learn; the documented method is lasso.
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X.values)
        lasso = LassoCV(cv=10, random_state=seed, max_iter=5000).fit(Xs, y)
        keep = X.columns[np.abs(lasso.coef_) > 1e-8].tolist()
        return keep, "lasso"
    except Exception:
        # Fallback: simple correlation screen (no scikit-learn needed).
        corrs = X.apply(lambda col: np.corrcoef(col.values, y)[0, 1])
        keep = corrs.index[np.abs(corrs.values) > 0.02].tolist()
        return keep, "correlation_screen"


# ---------------------------------------------------------------------------
# 3. OLS / WLS with robust (HC2 or HC1) standard errors, built by hand.
# ---------------------------------------------------------------------------
def _ols_robust(X, y, w=None, hc="HC2"):
    """Least squares with heteroskedasticity-robust standard errors.

    Returns (beta, vcov) where beta are the coefficients and vcov is the
    robust variance-covariance matrix. Works for OLS (w=None) and weighted
    least squares (w given).

    The "robust" SE is a sandwich:  bread * meat * bread.
        bread = (X'WX)^-1
        meat  = X'W * diag(adjusted squared residuals) * W X
    For OLS, W = I. The adjustment to the squared residuals is what
    distinguishes the HC variants:
        HC1: multiply the whole thing by n/(n-k)         (a dof correction)
        HC2: divide residual_i^2 by (1 - h_i)            (h_i = leverage)
    HC2 is what the paper uses for OLS; for the weighted PATE we use HC1, the
    standard robust analogue for weighted regression (see replication_log.md).

    We use the pseudo-inverse (pinv) for (X'WX)^-1 so that harmless collinearity
    among controls (e.g. stratum dummies overlapping with demographics) does not
    crash the solve. This matches how statsmodels handles rank-deficiency.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    W = np.ones(n) if w is None else np.asarray(w, dtype=float)

    XtWX = X.T @ (X * W[:, None])
    XtWX_inv = np.linalg.pinv(XtWX)          # the "bread"
    beta = XtWX_inv @ (X.T @ (W * y))
    resid = y - X @ beta

    if hc == "HC2":
        # Leverages h_i = diag of the (weighted) hat matrix.
        # h_i = w_i * x_i' (X'WX)^-1 x_i
        h = W * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)
        h = np.clip(h, 0, 0.9999)            # guard against rounding to >=1
        adj = resid**2 / (1.0 - h)
    elif hc == "HC1":
        adj = resid**2 * (n / (n - k))
    else:
        raise ValueError("hc must be 'HC2' or 'HC1'")

    meat = X.T @ (X * (W**2 * adj)[:, None])
    vcov = XtWX_inv @ meat @ XtWX_inv
    return beta, vcov


def _normal_ci_p(est, se, alpha=0.05):
    """Two-sided CI and p-value using the normal approximation (large n)."""
    from math import erf, sqrt
    z = 1.959963984540054  # 0.975 quantile of N(0,1)
    ci = (est - z * se, est + z * se)
    # two-sided p = 2 * (1 - Phi(|est/se|)); Phi via erf.
    t = abs(est / se) if se > 0 else np.inf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(t / sqrt(2.0))))
    return ci, p


# ---------------------------------------------------------------------------
# 2. The LIN (2013) estimator, weighted or unweighted.
# ---------------------------------------------------------------------------
def lin_estimator(df, outcome, weight_col=None, pre_treatment_col=None, seed=2020):
    """Estimate one treatment effect the way the paper does.

    LIN (2013) in one paragraph:
    A plain "regress outcome on treatment + controls" assumes the controls
    relate to the outcome the same way in both arms. Lin's fix is to also
    include treatment x (centred control) interactions. Centring the controls
    first means the coefficient on `treatment` still reads as the average
    treatment effect, but the interactions let each arm have its own slopes.
    This is the standard regression-adjusted ATE and never does worse than the
    unadjusted difference in means.

    weight_col = None     -> unweighted -> SATE (sample average treatment effect)
    weight_col = "weight" -> weighted   -> PATE (population average treatment effect)
    """
    d = df.dropna(subset=[outcome]).copy()
    T = d["treatment"].values.astype(float)

    # Choose controls by lasso on the pooled sample.
    Xfull = build_covariate_matrix(d, pre_treatment_col)
    keep, method = select_covariates(Xfull, d[outcome].values, seed=seed)

    # Always include stratum dummies (block-randomisation indicators).
    stratum = pd.get_dummies(d["stratum"], prefix="stratum", drop_first=True).astype(float)

    parts = []
    if keep:
        Xc = Xfull[keep]
        Xc_centred = Xc - Xc.mean(axis=0)                 # centre -> Lin's trick
        inter = Xc_centred.mul(T, axis=0)
        inter.columns = [f"T_x_{c}" for c in inter.columns]
        parts += [Xc_centred, inter]
    parts.append(stratum)

    design = pd.concat(parts, axis=1)
    design.insert(0, "treatment", T)
    design.insert(0, "const", 1.0)
    design = design.astype(float)

    y = d[outcome].values.astype(float)
    t_index = list(design.columns).index("treatment")

    if weight_col is None:
        beta, vcov = _ols_robust(design.values, y, w=None, hc="HC2")
    else:
        w = d[weight_col].values.astype(float)
        beta, vcov = _ols_robust(design.values, y, w=w, hc="HC1")

    est = beta[t_index]
    se = np.sqrt(vcov[t_index, t_index])
    (ci_low, ci_high), p = _normal_ci_p(est, se)

    return {
        "outcome": outcome, "estimate": est, "se": se,
        "ci_low": ci_low, "ci_high": ci_high, "p_raw": p,
        "n": int(len(d)), "n_controls_selected": len(keep),
        "selection_method": method,
    }


def iv_complier_effect(df, outcome, dose_col="pct_views_chrono", weight_col=None):
    """Instrumental-variables / CACE estimate (the paper's compliance check).

    ~12% of treated Facebook users never actually got the chrono feed (a web
    bug), so the paper re-estimates instrumenting the *dose* (share of views in
    chrono order) with random assignment. This is the Wald / 2SLS ratio:
        LATE = (effect of assignment on outcome) / (effect of assignment on dose)
    The result is the effect among compliers, mechanically a bit larger than the
    intention-to-treat effect because it divides by the compliance rate.
    """
    d = df.dropna(subset=[outcome]).copy()
    T = d["treatment"].values.astype(float)
    Y = d[outcome].values.astype(float)
    D = d[dose_col].values.astype(float)
    w = np.ones(len(d)) if weight_col is None else d[weight_col].values.astype(float)

    def wmean(a, mask):
        return np.average(a[mask], weights=w[mask])

    itt_y = wmean(Y, T == 1) - wmean(Y, T == 0)   # reduced form
    itt_d = wmean(D, T == 1) - wmean(D, T == 0)   # first stage
    return {"outcome": outcome, "itt_outcome": itt_y, "itt_dose": itt_d,
            "late": itt_y / itt_d}


# ---------------------------------------------------------------------------
# 4. SHARPENED FDR (Benjamini, Krieger & Yekutieli 2006).
# ---------------------------------------------------------------------------
def sharpened_fdr(pvalues, q=0.05):
    """Two-stage 'sharpened' false-discovery-rate adjusted q-values.

    WHY: testing many hypotheses produces false positives by chance. FDR
    control limits the expected share of false discoveries among those called
    significant. Plain Benjamini-Hochberg is one round; the BKY (2006)
    'sharpened' version runs BH once to estimate how many nulls are actually
    true, then reruns BH using that smaller count for more power. The paper uses
    this (their refs 49-50).

    Returns adjusted q-values aligned with the input order. A test is
    significant at level q if its returned value is <= q.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    def bh_qvalues(p_sorted, m_eff):
        ranks = np.arange(1, len(p_sorted) + 1)
        q_raw = p_sorted * m_eff / ranks
        q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]   # enforce monotonicity
        return np.clip(q_mono, 0, 1)

    # Stage 1: ordinary BH to estimate the number of true nulls m0.
    stage1_q = bh_qvalues(p_sorted, m)
    r1 = int(np.sum(stage1_q <= q))
    m0_hat = m - r1

    # Stage 2: rerun BH with the estimated m0 (the "sharpening").
    adj_sorted = stage1_q if m0_hat == 0 else bh_qvalues(p_sorted, m0_hat)

    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def run_outcome_family(df, outcomes_spec, pre_treatment=True, q=0.05, seed=2020):
    """Estimate PATE and SATE for a family of outcomes and FDR-correct.

    Returns a tidy DataFrame: one row per outcome, with weighted PATE and
    unweighted SATE, raw and FDR-adjusted p-values, plus the planted truth.
    """
    rows = []
    for name, spec in outcomes_spec.items():
        pre_col = (name + "_pre") if pre_treatment and (name + "_pre") in df.columns else None
        pate = lin_estimator(df, name, weight_col="weight", pre_treatment_col=pre_col, seed=seed)
        sate = lin_estimator(df, name, weight_col=None, pre_treatment_col=pre_col, seed=seed)
        rows.append({
            "outcome": name, "label": spec["label"],
            "true_effect": spec.get("true_effect", np.nan),
            "pate": pate["estimate"], "pate_se": pate["se"],
            "pate_ci_low": pate["ci_low"], "pate_ci_high": pate["ci_high"],
            "pate_p_raw": pate["p_raw"],
            "sate": sate["estimate"], "sate_se": sate["se"], "sate_p_raw": sate["p_raw"],
            "n": pate["n"],
        })
    res = pd.DataFrame(rows)
    res["pate_p_adj"] = sharpened_fdr(res["pate_p_raw"].values, q=q)
    res["sate_p_adj"] = sharpened_fdr(res["sate_p_raw"].values, q=q)
    return res
