"""
Motorcycle insurance pricing: claim frequency and severity models.

Data: Wasa insurance (Sweden) partial casco motorcycle policies, 1994-1998,
from Ohlsson & Johansson (2010), "Non-life insurance pricing with GLMs".
Run:  python analysis.py   (needs data/ohlsson.csv, see README)
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
RNG = 42
OUT = "results"
import os
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREY = "#4c7bd9", "#e0883a", "#8a8f98"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "text.color": GREY, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "axes.edgecolor": GREY, "grid.color": GREY, "grid.alpha": 0.25, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ----------------------------------------------------------------- 1. data
raw = pd.read_csv("data/ohlsson.csv").drop(columns="rownames")
raw = raw.rename(columns={
    "agarald": "owner_age", "kon": "sex", "zon": "zone", "mcklass": "bike_class",
    "fordald": "bike_age", "bonuskl": "bonus", "duration": "exposure",
    "antskad": "claims", "skadkost": "cost"})
n_raw, n_zero = len(raw), int((raw.exposure == 0).sum())
zero_claims = int(raw.loc[raw.exposure == 0, "claims"].sum())
df = raw[raw.exposure > 0].reset_index(drop=True)

def band(s, edges, labels):
    return pd.cut(s, edges, labels=labels, right=False).astype(str)

df["age_band"] = band(df.owner_age, [0, 25, 30, 40, 50, 60, 200],
                      ["<25", "25-29", "30-39", "40-49", "50-59", "60+"])
df["bike_age_band"] = band(df.bike_age, [0, 2, 5, 10, 15, 25, 200],
                           ["0-1", "2-4", "5-9", "10-14", "15-24", "25+"])
df["sex"] = df.sex.map({"K": "F", "M": "M"})  # K = kvinna (woman), M = man
df["zone"] = df.zone.astype(str)
df["bike_class"] = df.bike_class.astype(str)
df["bonus"] = df.bonus.astype(str)
df["freq"] = df.claims / df.exposure
CAT = ["age_band", "sex", "zone", "bike_class", "bike_age_band", "bonus"]

summary = {
    "rows_raw": n_raw, "rows_zero_exposure_dropped": n_zero,
    "claims_in_dropped_rows": zero_claims, "rows_used": len(df),
    "policy_years": round(float(df.exposure.sum()), 0),
    "claims": int(df.claims.sum()),
    "overall_frequency": float(df.claims.sum() / df.exposure.sum()),
    "mean_severity": float(df.cost.sum() / df.claims.sum()),
}

# ----------------------------------------------------- 2. nested CV on frequency
def pdev(y, mu, w):
    return mean_poisson_deviance(y, np.clip(mu, 1e-9, None), sample_weight=w)

def make_glm(alpha):
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return make_pipeline(pre, PoissonRegressor(alpha=alpha, max_iter=1000))

def make_gbm(lr, depth, leaf):
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return make_pipeline(pre, HistGradientBoostingRegressor(
        loss="poisson", learning_rate=lr, max_depth=depth, min_samples_leaf=leaf,
        max_iter=150, random_state=RNG))

def fit(model, tr, kind="glm"):
    key = "poissonregressor" if kind == "glm" else "histgradientboostingregressor"
    model.fit(tr[CAT], tr.freq, **{f"{key}__sample_weight": tr.exposure})
    return model

GLM_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
GBM_GRID = [(lr, d, leaf) for lr in (0.03, 0.08) for d in (2, 3) for leaf in (200, 800)]

def inner_select(train, grid, builder, kind):
    inner = KFold(3, shuffle=True, random_state=RNG)
    best, best_s = None, np.inf
    for p in grid:
        p = p if isinstance(p, tuple) else (p,)
        s = []
        for a, b in inner.split(train):
            m = fit(builder(*p), train.iloc[a], kind)
            s.append(pdev(train.freq.iloc[b], m.predict(train.iloc[b][CAT]), train.exposure.iloc[b]))
        if np.mean(s) < best_s:
            best, best_s = p, np.mean(s)
    return best

rows, oof_store = [], {}
for rep in range(2):
    outer = KFold(5, shuffle=True, random_state=RNG + rep)
    oof = {"glm": np.zeros(len(df)), "gbm": np.zeros(len(df)), "null": np.zeros(len(df))}
    for k, (tr_i, te_i) in enumerate(outer.split(df)):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        null = tr.claims.sum() / tr.exposure.sum()
        pg = inner_select(tr, GLM_GRID, make_glm, "glm")
        pb = inner_select(tr, GBM_GRID, make_gbm, "gbm")
        glm, gbm = fit(make_glm(*pg), tr, "glm"), fit(make_gbm(*pb), tr, "gbm")
        mu_g, mu_b = glm.predict(te[CAT]), gbm.predict(te[CAT])
        oof["glm"][te_i], oof["gbm"][te_i], oof["null"][te_i] = mu_g, mu_b, null
        d0 = pdev(te.freq, np.full(len(te), null), te.exposure)
        dg, db = pdev(te.freq, mu_g, te.exposure), pdev(te.freq, mu_b, te.exposure)
        rows.append({"rep": rep, "fold": k, "null": d0, "glm": dg, "gbm": db})
        print(f"rep {rep} fold {k}: null {d0:.4f} glm {dg:.4f} gbm {db:.4f}", flush=True)
    oof_store[rep] = oof

cv = pd.DataFrame(rows)
cv["glm_gain"] = 1 - cv.glm / cv.null
cv["gbm_gain"] = 1 - cv.gbm / cv.null
cv["gbm_minus_glm"] = cv.gbm - cv.glm
def mse(x):
    return float(x.std(ddof=1) / np.sqrt(len(x)))
cv_summary = {
    "folds": len(cv),
    "deviance_null": float(cv.null.mean()), "deviance_glm": float(cv.glm.mean()),
    "deviance_gbm": float(cv.gbm.mean()),
    "glm_gain_pct": float(100 * cv.glm_gain.mean()), "glm_gain_se": float(100 * mse(cv.glm_gain)),
    "gbm_gain_pct": float(100 * cv.gbm_gain.mean()), "gbm_gain_se": float(100 * mse(cv.gbm_gain)),
    "gbm_minus_glm_mean": float(cv.gbm_minus_glm.mean()),
    "gbm_minus_glm_se": mse(cv.gbm_minus_glm),
    "gbm_better_in_folds": int((cv.gbm_minus_glm < 0).sum()),
}

# --------------------------------------------- 3. out-of-fold calibration / lift
oof = oof_store[0]
def decile_table(pred):
    t = df[["exposure", "claims"]].copy()
    t["pred"] = pred * df.exposure
    t["risk"] = pred
    t["bin"] = pd.qcut(t.risk.rank(method="first"), 10, labels=False)
    g = t.groupby("bin").agg(exposure=("exposure", "sum"), claims=("claims", "sum"), pred=("pred", "sum"))
    g["actual_freq"], g["pred_freq"] = g.claims / g.exposure, g.pred / g.exposure
    return g

def gini(pred):
    t = df[["exposure", "claims"]].copy()
    t["risk"] = pred
    t = t.sort_values("risk")
    cum_e = np.cumsum(t.exposure) / t.exposure.sum()
    cum_c = np.cumsum(t.claims) / t.claims.sum()
    return float(1 - 2 * np.trapezoid(cum_c, cum_e))

dec = {"glm": decile_table(oof["glm"]), "gbm": decile_table(oof["gbm"])}
gini_res = {"glm": gini(oof["glm"]), "gbm": gini(oof["gbm"])}
top_vs_bottom = {m: float(dec[m].actual_freq.iloc[-1] / dec[m].actual_freq.iloc[0]) for m in dec}

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, m, c in zip(ax, ["glm", "gbm"], [BLUE, ORANGE]):
    g = dec[m]
    x = np.arange(10)
    a.bar(x - 0.2, g.actual_freq * 100, 0.4, color=GREY, alpha=0.8, label="Actual")
    a.bar(x + 0.2, g.pred_freq * 100, 0.4, color=c, label="Predicted")
    a.set_title("Poisson GLM" if m == "glm" else "Gradient boosting", color=GREY)
    a.set_xlabel("Risk decile (1 = lowest predicted risk)")
    a.set_xticks(x); a.set_xticklabels(x + 1)
    a.grid(axis="y")
ax[0].set_ylabel("Claims per 100 policy years")
ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/calibration.png", dpi=170); plt.close()

# ---------------------------------------------------- 4. GLM relativities
base = {"age_band": "40-49", "sex": "M", "zone": "4", "bike_class": "3",
        "bike_age_band": "10-14", "bonus": "7"}
formula = "claims ~ " + " + ".join(
    f"C({c}, Treatment(reference='{base[c]}'))" for c in CAT)
glm_sm = smf.glm(formula, data=df, family=sm.families.Poisson(),
                 offset=np.log(df.exposure)).fit()
ci = glm_sm.conf_int()
rel = []
for name in glm_sm.params.index:
    if name == "Intercept":
        continue
    fac = name.split("(")[1].split(",")[0]
    lvl = name.split("T.")[1].rstrip("]")
    rel.append({"factor": fac, "level": lvl, "rel": float(np.exp(glm_sm.params[name])),
                "lo": float(np.exp(ci.loc[name, 0])), "hi": float(np.exp(ci.loc[name, 1])),
                "p": float(glm_sm.pvalues[name]),
                "claims": int(df.loc[df[fac] == lvl, "claims"].sum()),
                "exposure": float(df.loc[df[fac] == lvl, "exposure"].sum())})
rel = pd.DataFrame(rel)
rel.to_csv(f"{OUT}/glm_relativities.csv", index=False)

# factor-level significance (likelihood-ratio test dropping each factor)
lr = {}
for c in CAT:
    f2 = "claims ~ " + " + ".join(
        f"C({k}, Treatment(reference='{base[k]}'))" for k in CAT if k != c)
    m2 = smf.glm(f2, data=df, family=sm.families.Poisson(), offset=np.log(df.exposure)).fit()
    stat = 2 * (glm_sm.llf - m2.llf)
    dof = df[c].nunique() - 1
    lr[c] = {"chi2": float(stat), "dof": int(dof), "p": float(stats.chi2.sf(stat, dof))}

plot_f = [("zone", "Zone"), ("bike_class", "Bike class (power-to-weight)"),
          ("age_band", "Owner age"), ("bike_age_band", "Bike age"), ("bonus", "Bonus class"), ("sex", "Sex")]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
for a, (f, title) in zip(axes.ravel(), plot_f):
    lv = [base[f]] + list(rel[rel.factor == f].level)
    order = sorted(lv, key=lambda s: (len(s), s)) if f not in ("age_band", "bike_age_band") else \
        (["<25", "25-29", "30-39", "40-49", "50-59", "60+"] if f == "age_band"
         else ["0-1", "2-4", "5-9", "10-14", "15-24", "25+"])
    y = np.arange(len(order))
    vals, los, his = [], [], []
    for l in order:
        if l == base[f]:
            vals.append(1); los.append(1); his.append(1)
        else:
            r = rel[(rel.factor == f) & (rel.level == l)].iloc[0]
            vals.append(r.rel); los.append(r.lo); his.append(r.hi)
    a.axvline(1, color=GREY, lw=1, ls="--")
    a.errorbar(vals, y, xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
               fmt="o", color=BLUE, ecolor=GREY, capsize=2, ms=5)
    a.set_yticks(y); a.set_yticklabels(order); a.set_xscale("log")
    a.set_xticks([0.25, 0.5, 1, 2, 4]); a.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(0.1, 12)
    a.set_title(title, color=GREY, fontsize=11); a.grid(axis="x")
plt.tight_layout(); plt.savefig(f"{OUT}/relativities.png", dpi=170); plt.close()

# --------------------------------------------------------- 5. CV comparison fig
fig, a = plt.subplots(figsize=(6.4, 3.6))
a.bar(["Poisson GLM", "Gradient\nboosting"],
      [cv_summary["glm_gain_pct"], cv_summary["gbm_gain_pct"]],
      yerr=[cv_summary["glm_gain_se"], cv_summary["gbm_gain_se"]],
      color=[BLUE, ORANGE], capsize=5, width=0.5, ecolor=GREY)
a.set_ylabel("Deviance reduction vs\nno-model baseline (%)")
a.grid(axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/cv_comparison.png", dpi=170); plt.close()

# --------------------------------------------------------------- 6. severity
c = df[df.claims > 0].copy()
c["sev"] = c.cost / c.claims
sev_null = smf.glm("sev ~ 1", data=c, family=sm.families.Gamma(sm.families.links.Log()),
                   var_weights=c.claims).fit()
sev_tests = {}
for f in CAT:
    m = smf.glm(f"sev ~ C({f})", data=c, family=sm.families.Gamma(sm.families.links.Log()),
                var_weights=c.claims).fit()
    # F-style test using scaled deviance (gamma dispersion estimated)
    dof = c[f].nunique() - 1
    F = ((sev_null.deviance - m.deviance) / dof) / (m.deviance / m.df_resid)
    sev_tests[f] = {"F": float(F), "dof": int(dof), "p": float(stats.f.sf(F, dof, m.df_resid))}
sev_summary = {
    "claim_rows": int(len(c)), "claims": int(c.claims.sum()),
    "mean": float(c.cost.sum() / c.claims.sum()),
    "median_row_severity": float(c.sev.median()),
    "p90_row_severity": float(c.sev.quantile(0.9)),
    "tests": sev_tests,
}

# ----------------------------------------------------- 7. pure premium (OOF)
#   Severity depends mainly on bike age (newer bikes cost more per claim), so a
#   Gamma GLM on bike age, zone and owner age is compared with a constant mean,
#   using the same outer folds as repetition 0 of the frequency CV.
df["zone_s"] = df.zone.replace({"5": "5-7", "6": "5-7", "7": "5-7"})
SEV_F = "sev ~ C(bike_age_band) + C(zone_s) + C(age_band)"
gam = lambda: sm.families.Gamma(sm.families.links.Log())

def gamma_dev(y, mu, w):
    return float(2 * np.sum(w * ((y - mu) / mu - np.log(y / mu))) / np.sum(w))

sev_glm_oof, sev_const_oof, g_const, g_glm = np.zeros(len(df)), np.zeros(len(df)), [], []
for tr_i, te_i in KFold(5, shuffle=True, random_state=RNG).split(df):
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    ctr = tr[tr.claims > 0].copy()
    ctr["sev"] = ctr.cost / ctr.claims
    const = ctr.cost.sum() / ctr.claims.sum()
    msev = smf.glm(SEV_F, data=ctr, family=gam(), var_weights=ctr.claims).fit()
    sev_glm_oof[te_i], sev_const_oof[te_i] = msev.predict(te), const
    tc = te[te.claims > 0]
    y = (tc.cost / tc.claims).values
    g_const.append(gamma_dev(y, np.full(len(tc), const), tc.claims.values))
    g_glm.append(gamma_dev(y, msev.predict(tc).values, tc.claims.values))
sev_cv = {"gamma_dev_constant": float(np.mean(g_const)), "gamma_dev_glm": float(np.mean(g_glm)),
          "glm_better_in_folds": int(sum(a < b for a, b in zip(g_glm, g_const))),
          "model": "Gamma GLM, log link: bike age band + zone group + owner age band"}

def pp_quintiles(sev_pred):
    pp = df[["exposure", "claims", "cost"]].copy()
    pp["pred"] = oof["glm"] * sev_pred * pp.exposure
    pp["bin"] = pd.qcut((oof["glm"] * sev_pred).argsort().argsort(), 5, labels=False)
    g = pp.groupby("bin").agg(exposure=("exposure", "sum"), cost=("cost", "sum"), pred=("pred", "sum"))
    g["actual_pp"], g["pred_pp"] = g.cost / g.exposure, g.pred / g.exposure
    return g[["actual_pp", "pred_pp"]].round(0).to_dict("list")

pure_premium = {"constant_severity": pp_quintiles(sev_const_oof),
                "severity_glm": pp_quintiles(sev_glm_oof)}
sev_summary["cv"] = sev_cv
q = pure_premium["severity_glm"]
fig, a = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(5)
a.bar(x - 0.2, q["actual_pp"], 0.4, color=GREY, alpha=0.8, label="Actual cost")
a.bar(x + 0.2, q["pred_pp"], 0.4, color=BLUE, label="Predicted (frequency x severity)")
a.set_xticks(x); a.set_xticklabels(["1\nlowest", "2", "3", "4", "5\nhighest"])
a.set_xlabel("Predicted pure-premium quintile"); a.set_ylabel("Cost per policy year (SEK)")
a.legend(frameon=False); a.grid(axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/pure_premium.png", dpi=170); plt.close()

# ------------------------------------------------ 8. data summaries for charts
def by(f):
    g = df.groupby(f).agg(exposure=("exposure", "sum"), claims=("claims", "sum"))
    g["freq"] = g.claims / g.exposure
    return g
raw_zone, raw_class = by("zone"), by("bike_class")

results = {
    "summary": summary, "cv": cv_summary, "gini": gini_res,
    "top_vs_bottom_decile": top_vs_bottom, "lr_tests": lr, "severity": sev_summary,
    "pure_premium_quintiles": pure_premium,
    "decile_glm": dec["glm"].round(5).reset_index().to_dict("list"),
    "decile_gbm": dec["gbm"].round(5).reset_index().to_dict("list"),
    "raw_zone": raw_zone.round(5).reset_index().to_dict("list"),
    "raw_class": raw_class.round(5).reset_index().to_dict("list"),
}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
cv.to_csv(f"{OUT}/cv_folds.csv", index=False)
print(json.dumps({k: results[k] for k in ["summary", "cv", "gini", "top_vs_bottom_decile",
                                          "lr_tests", "severity", "pure_premium_quintiles"]}, indent=1))
