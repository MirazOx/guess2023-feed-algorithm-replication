"""
config.py
=========
Single source of truth for the replication.

Everything that the *simulation* needs to plant an effect, and everything the
*analysis* needs to look for, lives here. Keeping it in one place means the
synthetic data and the estimator can never silently disagree about what an
outcome is called or what its "true" effect is supposed to be.

All numbers are taken from the Guess et al. (2023) supplementary materials
(science.abp9364, Tables S2, S4, S5 for Facebook). See replication_log.md for
exactly where each value comes from and where we deliberately diverge.

Two kinds of outcome appear in the paper:

  * "first-stage" / RQ outcomes  -> measured in PERCENTAGE POINTS of the feed.
    These are the things the intervention is *supposed* to move (what content
    you see). The paper reports big, significant effects here.

  * attitude / behaviour outcomes -> measured in STANDARD-DEVIATION units
    (z-scores). These are the downstream things the paper mostly finds NO
    effect on. They are the headline null result.
"""

# Sample sizes for "study-completes" on Facebook (SM section S1.1).
# These are the participants we actually analyse.
N_CONTROL_FB = 16_159
N_TREAT_FB = 7_232

# Instagram, for the optional second-platform run.
N_CONTROL_IG = 12_514
N_TREAT_IG = 8_800

# Non-compliance on Facebook: ~11.9% of the treatment group kept seeing the
# algorithmic feed because of a web-version bug (SM section S1.8). We reproduce
# this so the instrumental-variables / compliance extension has something to do.
NONCOMPLIANCE_RATE_FB = 0.119

# Reproducibility.
RANDOM_SEED = 2020


# ---------------------------------------------------------------------------
# RQ ("research question") first-stage outcomes: effect on what the feed shows.
# Units: percentage points. "true_effect" is the PATE we plant; "control_mean"
# is a plausible baseline for the algorithmic-feed group (exact baseline is not
# important for a synthetic study, the treatment *shift* is what we recover).
# Source for true_effect: SM Table S4 (Facebook, PATE column).
# ---------------------------------------------------------------------------
RQ_OUTCOMES = {
    "pct_political_content":     {"label": "% of feed that is political content",   "control_mean": 12.0, "control_sd": 9.0,  "true_effect": +1.68},
    "pct_cross_cutting":         {"label": "% of feed from cross-cutting sources",  "control_mean": 20.0, "control_sd": 12.0, "true_effect": -2.48},
    "pct_political_news":        {"label": "% of feed that is political news",      "control_mean": 6.0,  "control_sd": 6.0,  "true_effect": +1.82},
    "pct_untrustworthy":         {"label": "% of feed from untrustworthy sources",  "control_mean": 1.3,  "control_sd": 2.0,  "true_effect": +1.59},
    "pct_uncivil":               {"label": "% of feed that is uncivil",             "control_mean": 6.0,  "control_sd": 4.0,  "true_effect": -1.33},
    "pct_slur_words":            {"label": "% of feed with slur words",             "control_mean": 0.10, "control_sd": 0.20, "true_effect": -0.02},
}

# ---------------------------------------------------------------------------
# Primary hypotheses H1-H3. Units: standard deviations.
# true_effect from SM Table S2 (Facebook, PATE). Note almost all are ~0; the
# only real effect is H3c (on-platform political engagement), which is a
# BEHAVIOURAL outcome, not an attitude.
# "family" groups outcomes for the multiple-comparison (FDR) correction.
# ---------------------------------------------------------------------------
PRIMARY_OUTCOMES = {
    "affective_polarization":   {"label": "H1a: Affective polarization",          "family": "primary", "true_effect":  0.000},
    "issue_polarization":       {"label": "H1b: Issue polarization",              "family": "primary", "true_effect":  0.000},
    "election_knowledge":       {"label": "H2a: Election knowledge",              "family": "primary", "true_effect":  0.000},
    "news_knowledge":           {"label": "H2b: News knowledge",                 "family": "primary", "true_effect": -0.025},
    "self_participation":       {"label": "H3a: Self-reported participation",     "family": "primary", "true_effect": -0.025},
    "self_turnout":             {"label": "H3b: Self-reported turnout",           "family": "primary", "true_effect":  0.000},
    "onplatform_engagement":    {"label": "H3c: On-platform pol. engagement",    "family": "primary", "true_effect": -0.118},
}

# ---------------------------------------------------------------------------
# Secondary hypotheses (subset). Units: standard deviations.
# true_effect from SM Table S5 (Facebook, PATE). We keep the two that matter
# for showing the FDR machinery working:
#   * partisan_news_clicks (+0.107): survives the correction.
#   * trust_social_media   (+0.041): raw p ~ 0.02 but FDR pushes it to n.s.
# ---------------------------------------------------------------------------
SECONDARY_OUTCOMES = {
    "factual_discernment":      {"label": "SH1a: Knowledge / information",        "family": "secondary", "true_effect":  0.000},
    "trust_media":              {"label": "SH2a: Trust in media (excl. social)",  "family": "secondary", "true_effect":  0.000},
    "trust_social_media":       {"label": "SH2b: Trust in social media info",     "family": "secondary", "true_effect": +0.041},
    "confidence_institutions":  {"label": "SH2c: Confidence in institutions",     "family": "secondary", "true_effect":  0.000},
    "perceived_polarization":   {"label": "SH3a: Perceived polarization",         "family": "secondary", "true_effect":  0.000},
    "partisan_news_clicks":     {"label": "SH3b: Partisan news clicks",           "family": "secondary", "true_effect": +0.107},
    "political_efficacy":       {"label": "SH4: Political efficacy",              "family": "secondary", "true_effect":  0.000},
    "belief_election_legit":    {"label": "SH6: Belief in election legitimacy",   "family": "secondary", "true_effect":  0.000},
}


def all_standardized_outcomes():
    """Primary + secondary outcomes, all in SD units, as one dict."""
    merged = {}
    merged.update(PRIMARY_OUTCOMES)
    merged.update(SECONDARY_OUTCOMES)
    return merged
