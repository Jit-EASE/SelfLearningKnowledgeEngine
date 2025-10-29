#!/usr/bin/env python3
"""
AgriSense-KE: Self-Learning Agri-Food Knowledge Engine (Dash)

Features
--------
- Self-learning loop: logs feedback and periodically retrains (1) a text intent classifier, (2) a simple outcome model.
- Agentic-RAG: local TF-IDF retrieval over uploaded KB docs + optional OpenAI (gpt-4o-mini) grounded synthesis.
- QLDPC & QSTP: illustrative stochastic simulators for scenario experiments (placeholders for quantum narratives).
- Geo: Folium (Leaflet) map for all 26 counties (ROI); popups include concise NLP explanations (local or OpenAI).
- Econometrics: DiD and event-study helpers with robust fallbacks when variation is insufficient.
"""

import os
import io
import json
import math
import base64
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Dash / Viz
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Stats / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Folium (Leaflet)
import folium
import branca

# Optional OpenAI (new SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------
APP_TITLE = "AgriSense-KE: Self-Learning Agri-Food Knowledge Engine"
DEFAULT_SEED = 42

YEARS = list(range(2018, 2025))
QUARTERS = [1, 2, 3, 4]

COUNTIES = [
    "Carlow","Cavan","Clare","Cork","Donegal","Dublin","Galway","Kerry","Kildare",
    "Kilkenny","Laois","Leitrim","Limerick","Longford","Louth","Mayo","Meath","Monaghan",
    "Offaly","Roscommon","Sligo","Tipperary","Waterford","Westmeath","Wexford","Wicklow"
]

# Approximate county centroids (fallback when GeoJSON unavailable)
COUNTY_COORDS = {
    "Carlow": (52.8365, -6.9341),
    "Cavan": (53.9890, -7.3600),
    "Clare": (52.8600, -8.9800),
    "Cork": (51.8985, -8.4756),
    "Donegal": (54.6500, -8.1000),
    "Dublin": (53.3498, -6.2603),
    "Galway": (53.2707, -9.0568),
    "Kerry": (52.1545, -9.5667),
    "Kildare": (53.1589, -6.9097),
    "Kilkenny": (52.6541, -7.2448),
    "Laois": (53.0340, -7.3000),
    "Leitrim": (54.3070, -8.0000),
    "Limerick": (52.6638, -8.6267),
    "Longford": (53.7270, -7.8000),
    "Louth": (53.9500, -6.5400),
    "Mayo": (53.9000, -9.2900),
    "Meath": (53.6050, -6.6500),
    "Monaghan": (54.2490, -6.9700),
    "Offaly": (53.2730, -7.4900),
    "Roscommon": (53.6270, -8.1900),
    "Sligo": (54.2697, -8.4691),
    "Tipperary": (52.4730, -7.9400),
    "Waterford": (52.2590, -7.1100),
    "Westmeath": (53.5340, -7.3500),
    "Wexford": (52.3360, -6.4620),
    "Wicklow": (52.9810, -6.3670),
}

DATA_DIR = os.path.dirname(__file__)
GEOJSON_PATH = os.path.join(DATA_DIR, "ireland_counties.geojson")  # optional if you have polygons
MEMORY_LOG_PATH = os.path.join(DATA_DIR, "agent_feedback.jsonl")
KB_STORE_PATH = os.path.join(DATA_DIR, "rag_docs.json")

# OpenAI
OPENAI_MODEL = os.getenv("AFKE_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY and OpenAI is not None:
    try:
        _openai_client = OpenAI()
        _openai_enabled = True
    except Exception:
        _openai_client = None
        _openai_enabled = False
else:
    _openai_client = None
    _openai_enabled = False

# In-memory stores
RAG_DOCS: List[dict] = []
RAG_VEC = None
RAG_VECT = None
AGENT_LOG: List[dict] = []
INTENT_CLF = None             # self-learning text intent classifier
OUTCOME_MODEL = None          # (pipeline, control_vars) trained on panel

np.random.seed(DEFAULT_SEED)

# Dash version guard for allow_duplicate support
try:
    _ = Output("tmp", "children", allow_duplicate=True)
    _ALLOW_DUP_SUPPORTED = True
except TypeError:
    _ALLOW_DUP_SUPPORTED = False

def OUT(id_, prop, allow_duplicate=False):
    """Compatibility wrapper to use allow_duplicate only if Dash supports it."""
    if allow_duplicate and _ALLOW_DUP_SUPPORTED:
        return Output(id_, prop, allow_duplicate=True)
    return Output(id_, prop)

# -----------------------------
# SYNTHETIC PANEL GENERATOR
# -----------------------------
def make_synthetic_panel(seed=DEFAULT_SEED, n_farms=320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    farms = pd.DataFrame({
        "farm_id": np.arange(1, n_farms + 1),
        "county": rng.choice(COUNTIES, n_farms),
        "farm_type": rng.choice(["Dairy","Beef","Mixed","Tillage","Sheep"], n_farms, p=[0.35,0.25,0.2,0.1,0.1]),
        "herd_size": (rng.normal(90, 30, n_farms).clip(5, 500)).astype(int),
    })
    idx = pd.MultiIndex.from_product([farms.farm_id, YEARS, QUARTERS], names=["farm_id","year","q"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(farms, on="farm_id", how="left")
    panel["date"] = pd.PeriodIndex.from_fields(year=panel["year"].to_numpy(),
                                               quarter=panel["q"].to_numpy()).to_timestamp(how="end")

    county_effect = panel["county"].map({c:i for i,c in enumerate(COUNTIES)}).astype(float)
    time_trend = (panel["year"] - min(YEARS)) + (panel["q"]-1)/4
    noise = rng.normal(0, 1, len(panel))

    panel["rain_mm"] = (rng.gamma(6, 12, len(panel)) + county_effect*1.5 + (panel["q"]==1)*15).round(1)
    panel["gdd"] = (rng.normal(1400, 150, len(panel)) - (panel["q"]==1)*80 + (panel["q"]==3)*60).round(0)
    panel["input_price_idx"] = (100 + 2*time_trend + rng.normal(0, 3, len(panel))).round(1)

    adopt_year = rng.choice([2020,2021,2022,2023,None], size=len(farms), p=[0.15,0.25,0.25,0.2,0.15])
    adopt_q = rng.choice([1,2,3,4], size=len(farms))
    panel = panel.merge(pd.DataFrame({"farm_id": farms.farm_id,
                                      "adopt_year": adopt_year, "adopt_q": adopt_q}),
                        on="farm_id", how="left")
    panel["adopt_year"] = panel["adopt_year"].astype("Int64")
    panel["adopt_q"] = panel["adopt_q"].astype("Int64")

    has_adopt = panel["adopt_year"].notna()
    ay = panel["adopt_year"].fillna(-10).astype(int)
    aq = panel["adopt_q"].fillna(1).astype(int)
    after = (panel["year"] > ay)
    same = ((panel["year"] == ay) & (panel["q"] >= aq))
    panel["treated"] = (has_adopt & (after | same)).astype(int)

    lagged_treat = panel.groupby("farm_id")["treated"].shift(1).fillna(0)
    policy_effect = -6.5  # emissions drop after adoption
    panel["emissions_kg_ha"] = (
        320 + 0.2*panel["herd_size"] - 0.03*panel["rain_mm"] + 0.02*panel["gdd"]
        + 0.6*county_effect + 1.0*time_trend + noise*5 + policy_effect*lagged_treat
    ).round(2)

    yield_base = 20 + 0.02*panel["gdd"] - 0.008*panel["rain_mm"] + (panel["farm_type"]=="Dairy")*3.5
    panel["milk_yield_lcd"] = (yield_base + 0.3*panel["treated"] + rng.normal(0,0.9,len(panel))).round(2)

    panel["quarter"] = panel["year"].astype(str) + "Q" + panel["q"].astype(str)

    return panel[[
        "farm_id","county","farm_type","herd_size","year","q","quarter","date",
        "treated","adopt_year","adopt_q","rain_mm","gdd","input_price_idx",
        "emissions_kg_ha","milk_yield_lcd"
    ]]

# -----------------------------
# RAG + SELF-LEARNING
# -----------------------------
def rag_build_index(docs: List[dict]) -> bool:
    """Build TF-IDF index from docs; persist for warm start."""
    global RAG_DOCS, RAG_VEC, RAG_VECT
    RAG_DOCS = docs[:]
    corpus = [d.get("text","") for d in RAG_DOCS]
    if not corpus:
        RAG_VEC = RAG_VECT = None
        return False
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    RAG_VEC, RAG_VECT = vec, X
    try:
        with open(KB_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(RAG_DOCS, f)
    except Exception:
        pass
    return True

def rag_retrieve(query: str, k: int = 5) -> List[dict]:
    if RAG_VECT is None or RAG_VEC is None or not RAG_DOCS:
        return []
    qv = RAG_VEC.transform([query])
    sims = (qv @ RAG_VECT.T).toarray().ravel()
    idx = sims.argsort()[::-1][:k]
    return [{"doc": RAG_DOCS[i], "score": float(sims[i])} for i in idx if sims[i] > 0]

def retrain_intent_clf() -> bool:
    """Retrain a small intent classifier from feedback rows labeled with 'label_intent'."""
    global INTENT_CLF
    if not os.path.exists(MEMORY_LOG_PATH):
        return False
    texts, intents = [], []
    with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                if rec.get("query") and rec.get("label_intent"):
                    texts.append(rec["query"])
                    intents.append(rec["label_intent"])
            except Exception:
                continue
    if not texts:
        return False
    pipeline = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
                             SGDClassifier(max_iter=1000, tol=1e-3))
    pipeline.fit(texts, intents)
    INTENT_CLF = pipeline
    return True

def retrain_outcome_model(panel: pd.DataFrame,
                          outcome: str,
                          control_vars: List[str]) -> bool:
    """Train a simple ElasticNet predictor for the outcome."""
    global OUTCOME_MODEL
    if panel is None or panel.empty:
        return False
    X = panel[control_vars].select_dtypes(include=[np.number]).fillna(0.0)
    y = panel[outcome].astype(float).fillna(0.0)
    if X.shape[0] < 50:
        return False
    pipe = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, random_state=DEFAULT_SEED, max_iter=2000))
    pipe.fit(X, y)
    OUTCOME_MODEL = (pipe, control_vars)
    return True

# -----------------------------
# QUANTUM-LIKE SIMS (placeholders)
# -----------------------------
def simulate_qldpc(n_qubits: int = 127, rounds: int = 10, phys_error: float = 0.001) -> dict:
    """Rough logical error estimator (illustrative)."""
    distance = int(math.sqrt(max(1, n_qubits)))
    logical_error = phys_error * math.exp(-0.5 * distance) * (1 + 0.1 * rounds/10)
    rng = np.random.default_rng(DEFAULT_SEED)
    est = float((rng.random(1000) < logical_error).mean())
    return {"n_qubits": n_qubits, "rounds": rounds, "phys_error": phys_error, "logical_error_est": est}

def simulate_qstp(num_nodes: int = 6, base_fid: float = 0.99, hops: int = 3) -> dict:
    """QSTP chain fidelity (illustrative)."""
    rng = np.random.default_rng(DEFAULT_SEED)
    eff = (base_fid ** hops) * (1 - 0.01 * max(0, num_nodes - 3))
    return {"nodes": num_nodes, "hops": hops, "base_fidelity": base_fid,
            "est_fidelity": max(0.0, float(eff + rng.normal(0, 0.004)))}

# -----------------------------
# DID / EVENT-STUDY HELPERS
# -----------------------------
def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    req = ["farm_id","county","farm_type","year","q","treated","emissions_kg_ha","milk_yield_lcd"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["q"] = df["q"].astype(int)
    df["treated"] = df["treated"].astype(int).clip(0,1)
    if "adopt_year" in df.columns: df["adopt_year"] = df["adopt_year"].astype("Int64")
    if "adopt_q" in df.columns: df["adopt_q"] = df["adopt_q"].astype("Int64")
    if "quarter" not in df.columns:
        df["quarter"] = df["year"].astype(str) + "Q" + df["q"].astype(str)
    if "date" not in df.columns:
        df["date"] = pd.PeriodIndex.from_fields(year=df["year"].to_numpy(),
                                                quarter=df["q"].to_numpy()).to_timestamp(how="end")
    return df

def make_event_study(
    df: pd.DataFrame,
    outcome: str,
    controls: List[str],
    min_per_bucket: int = 25,
    rel_cap: int = 6
) -> pd.DataFrame:
    """
    Robust event-study builder.
    - Buckets rel into [-rel_cap, +rel_cap]
    - Drops sparse buckets (< min_per_bucket)
    - Auto-selects baseline (prefer -1; else nearest pre-period; else global min)
    - If OLS design still singular/too wide, falls back to a simple binned estimator
      that reports mean(y_tilde | rel=b) - mean(y_tilde | rel=baseline) with HC1-like CI
      via a quick sandwich approximation.

    Returns tidy DataFrame: rel, coef, ci_low, ci_high
    """
    if "rel" not in df.columns:
        return pd.DataFrame(columns=["rel","coef","ci_low","ci_high"])

    tmp = df.dropna(subset=["rel", outcome]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["rel","coef","ci_low","ci_high"])

    # Cap to manageable window and drop non-integers
    tmp["rel_bucket"] = tmp["rel"].astype(int).clip(-abs(rel_cap), abs(rel_cap))

    # Two-way demeaning (farm x quarter) to absorb FE
    def _tw_demean(s, g1, g2):
        g1m = s.groupby(g1).transform("mean")
        g2m = s.groupby(g2).transform("mean")
        return s - g1m - g2m + s.mean()

    y = tmp[outcome].astype(float)
    tmp["y_tilde"] = _tw_demean(y, tmp["farm_id"], tmp["quarter"])

    # Keep only buckets with enough support
    counts = tmp["rel_bucket"].value_counts()
    keep = counts[counts >= int(min_per_bucket)].index.tolist()
    tmp = tmp[tmp["rel_bucket"].isin(keep)].copy()
    if tmp["rel_bucket"].nunique() < 2:
        return pd.DataFrame(columns=["rel","coef","ci_low","ci_high"])

    # Choose baseline: -1 if present; else max pre-period (<0); else smallest rel
    if -1 in keep:
        baseline_val = -1
    else:
        pre = [b for b in keep if b < 0]
        baseline_val = (max(pre) if pre else min(keep))

    # --- Attempt OLS with dummies (preferred) ---
    try:
        d_rel = pd.get_dummies(tmp["rel_bucket"], prefix="rel", drop_first=False)
        # Drop baseline
        base_col = f"rel_{baseline_val}"
        if base_col in d_rel.columns:
            d_rel = d_rel.drop(columns=[base_col])

        # Drop any all-zero columns (should not exist, but just in case)
        d_rel = d_rel.loc[:, d_rel.sum(axis=0) > 0]

        # Guard: rows must exceed params by a margin
        X = sm.add_constant(d_rel, has_constant="add")
        y_tilde = tmp["y_tilde"].to_numpy()
        if X.shape[0] > X.shape[1] + 5:
            model = sm.OLS(y_tilde, X).fit(cov_type="HC1")
            rows = []
            for c in d_rel.columns:
                if c in model.params.index:
                    rel_k = int(c.split("_", 1)[1])
                    b = float(model.params[c]); se = float(model.bse[c])
                    rows.append({
                        "rel": rel_k,
                        "coef": b,
                        "ci_low": b - 1.96*se,
                        "ci_high": b + 1.96*se
                    })
            if rows:
                out = pd.DataFrame(rows).sort_values("rel")
                # Ensure the baseline itself appears as 0 effect with tiny CI (visual anchor)
                out = pd.concat([
                    pd.DataFrame([{"rel": baseline_val, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0}]),
                    out
                ], ignore_index=True).sort_values("rel")
                return out
    except Exception:
        pass

    # --- Fallback: simple binned estimator around baseline ---
    # Compute mean differences vs baseline on y_tilde, with rough HC1-like SE
    rows = []
    grp = tmp.groupby("rel_bucket", as_index=False)["y_tilde"].agg(["mean","var","count"]).reset_index()
    grp.columns = ["rel","mean","var","n"]
    if (grp["rel"] == baseline_val).any():
        mu0 = float(grp.loc[grp["rel"]==baseline_val, "mean"].iloc[0])
        v0  = float(grp.loc[grp["rel"]==baseline_val, "var"].iloc[0])
        n0  = max(1.0, float(grp.loc[grp["rel"]==baseline_val, "n"].iloc[0]))
        for _, r in grp.iterrows():
            rel_k = int(r["rel"]); mu = float(r["mean"]); v = float(r["var"]); n = max(1.0, float(r["n"]))
            if rel_k == baseline_val:
                rows.append({"rel": rel_k, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0})
            else:
                diff = mu - mu0
                # SE of difference of means (heteroskedastic-friendly-ish)
                se = math.sqrt((v/max(1.0, n-1))/n + (v0/max(1.0, n0-1))/n0)
                rows.append({"rel": rel_k, "coef": diff, "ci_low": diff - 1.96*se, "ci_high": diff + 1.96*se})
        return pd.DataFrame(rows).sort_values("rel")

    # If even baseline summary missing, nothing to show
    return pd.DataFrame(columns=["rel","coef","ci_low","ci_high"])
# -----------------------------
# FOLIUM MAP + NLP POPUPS
# -----------------------------
def build_county_popup_text(county: str, avg_outcome: float, treat_rate: float, outcome_name: str) -> str:
    """Return HTML popup using local template or OpenAI one-liner if enabled."""
    base = (f"<b>{county}</b><br>"
            f"Avg {outcome_name}: {avg_outcome:.2f}<br>"
            f"Policy adoption (treated share): {treat_rate:.2f}<br>")
    template = (f"{base}<hr><i>Local explanation:</i> This county’s {outcome_name} sits near its recent average. "
                f"With adoption at {treat_rate*100:.1f}%, incremental effects typically emerge over ~2–4 quarters, "
                f"holding weather, prices, and herd constant.")

    if _openai_enabled and _openai_client is not None:
        try:
            prompt = (f"Produce one crisp sentence for a farmer in {county}: "
                      f"{outcome_name}={avg_outcome:.2f}, adoption_rate={treat_rate:.2f}. "
                      f"<= 18 words, plain English.")
            msgs = [{"role": "system", "content": "You write concise, practical advice."},
                    {"role": "user", "content": prompt}]
            resp = _openai_client.chat.completions.create(model=OPENAI_MODEL, messages=msgs,
                                                          temperature=0.2, max_tokens=40)
            sent = resp.choices[0].message.content.strip()
            return template + f"<br><b>AI note:</b> {sent}"
        except Exception:
            return template
    return template

def build_folium_map(panel_df: pd.DataFrame, outcome: str, county_filter: Optional[str]) -> str:
    """Return HTML string for embedding Folium map."""
    m = folium.Map(location=[53.35, -7.9], zoom_start=6, tiles="CartoDB positron")

    # aggregate by county
    if panel_df is None or panel_df.empty:
        panel_df = make_synthetic_panel()
    agg = panel_df.groupby("county", as_index=False).agg({outcome: "mean", "treated": "mean"})
    agg = agg.rename(columns={outcome: "avg_outcome", "treated": "treat_rate"})

    # try polygons if available
    if os.path.exists(GEOJSON_PATH):
        try:
            gj = json.load(open(GEOJSON_PATH, "r", encoding="utf-8"))
            folium.Choropleth(
                geo_data=gj, name="choropleth", data=agg,
                columns=["county", "avg_outcome"], key_on="feature.properties.NAME",
                fill_color="YlGnBu", fill_opacity=0.75, line_opacity=0.2
            ).add_to(m)
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                name = props.get("NAME") or props.get("name") or props.get("county")
                if not name: continue
                row = agg[agg["county"].str.lower() == str(name).lower()]
                if not row.empty:
                    a = float(row["avg_outcome"].iloc[0]); t = float(row["treat_rate"].iloc[0])
                    html_popup = build_county_popup_text(name, a, t, outcome)
                    folium.GeoJson(feat, name=name, tooltip=name,
                                   popup=folium.Popup(html_popup, max_width=420)).add_to(m)
            return m._repr_html_()
        except Exception:
            pass

    # fallback: centroid markers
    mn, mx = float(agg["avg_outcome"].min()), float(agg["avg_outcome"].max())
    rng = max(mx - mn, 1e-6)
    palette = px.colors.sequential.Viridis

    for _, r in agg.iterrows():
        c = r["county"]
        lat, lon = COUNTY_COORDS.get(c, (53.35, -7.9))
        a = float(r["avg_outcome"]); t = float(r["treat_rate"])
        color_idx = int(min(9, max(0, (a - mn)/rng * 9)))
        popup = build_county_popup_text(c, a, t, outcome)
        folium.CircleMarker(
            location=[lat, lon], radius=8 + 12*t, color=None, fill=True,
            fill_color=palette[color_idx], fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=420)
        ).add_to(m)

    if county_filter and county_filter != "All" and county_filter in COUNTY_COORDS:
        lat, lon = COUNTY_COORDS[county_filter]
        m.location = [lat, lon]
        m.zoom_start = 7

    return m._repr_html_()

# -----------------------------
# AGENT PLANNER + ANSWER
# -----------------------------
def agent_plan_local(query: str) -> str:
    q = (query or "").lower()
    if INTENT_CLF is not None:
        try:
            return str(INTENT_CLF.predict([query])[0])
        except Exception:
            pass
    if any(x in q for x in ["did", "difference-in-differences", "treated effect", "treatment effect"]):
        return "did"
    if any(x in q for x in ["event study", "dynamic effect", "lead", "lag"]):
        return "event_study"
    if any(x in q for x in ["map", "choropleth", "county map", "where"]):
        return "map"
    return "rag"

def agent_answer(query: str,
                 panel_records: List[dict],
                 outcome: str,
                 controls: List[str],
                 county_filter: Optional[str],
                 se_type: str,
                 use_openai_llm: bool,
                 openai_temp: float) -> Tuple[str, Optional[object], List[dict]]:
    """Run retrieval + suitable tool; optionally synthesize final answer via OpenAI.
       Returns (text, figure_or_html, hits)
    """
    hits = rag_retrieve(query, k=6)
    intent = agent_plan_local(query)

    df = pd.DataFrame(panel_records) if panel_records else pd.DataFrame()
    if not df.empty and county_filter and county_filter != "All":
        df = df.query("county == @county_filter").copy()

    tool_text = None
    fig_payload = None

    if intent == "did" and not df.empty:
        try:
            rhs = ["treated"] + [c for c in controls if c in df.columns] + ["C(farm_id)", "C(quarter)"]
            model = ols(f"{outcome} ~ " + " + ".join(rhs), data=df).fit(cov_type="HC1")
            b = float(model.params.get("treated", np.nan))
            se = float(model.bse.get("treated", np.nan))
            p = float(model.pvalues.get("treated", np.nan))
            tool_text = f"DiD: treated={b:.3f} (SE={se:.3f}, p={p:.3g})"
        except Exception as e:
            tool_text = f"DiD failed: {e}"

    elif intent == "event_study" and not df.empty:
        if {"adopt_year","q"}.issubset(df.columns):
            adopt_q = (df["adopt_year"].fillna(0).astype(int)*4 + df.get("adopt_q",4).fillna(0).astype(int))
            current_q = df["year"]*4 + df["q"]
            rel = (current_q - adopt_q).astype(int)
            rel = rel.where(df["treated"].eq(1) | adopt_q.gt(0), np.nan)
            df_es = df.assign(rel=rel)
        else:
            df_es = df.assign(rel=np.nan)
        es = make_event_study(df_es, outcome, controls)
        if not es.empty:
            tool_text = f"Event-study estimated at {len(es)} points."
            fig_payload = px.scatter(es, x="rel", y="coef",
                                     error_y=es["ci_high"]-es["coef"],
                                     error_y_minus=es["coef"]-es["ci_low"]).update_layout(
                                         title="Event-study (relative to adoption)")
        else:
            tool_text = "Event-study unavailable (insufficient variation)."

    elif intent == "map":
        html_map = build_folium_map(df if not df.empty else pd.DataFrame(panel_records), outcome, county_filter)
        tool_text = "Interactive county map prepared."
        fig_payload = html_map  # HTML string

    else:  # RAG-only
        tool_text = "Retrieved top-K documents."

    # Base answer with citations
    citations = "\n".join([f"- {h['doc']['name']} (score {h['score']:.2f})" for h in hits]) or "- (no KB docs indexed)"
    base_ans = f"Tool: {intent} -> {tool_text}\n\nKey docs:\n{citations}"
    final_ans = base_ans

    # Optional OpenAI synthesis
    if use_openai_llm and _openai_enabled and _openai_client is not None:
        try:
            kb_text = "\n\n".join([(h["doc"].get("text","")[:1200]) for h in hits[:4]]) or "(no docs)"
            prompt = (
                f"Question: {query}\nOutcome: {outcome}; Controls: {', '.join(controls) if controls else '(none)'}\n\n"
                f"Tool output:\n{tool_text}\n\nDocuments:\n{kb_text}\n\n"
                "Write 3–6 sentences grounded strictly in the provided context and a one-line recommendation. "
                "Cite doc names in brackets like [doc]."
            )
            msgs = [{"role":"system","content":"You are an econometrics-savvy assistant for Irish agri-food policy."},
                    {"role":"user","content":prompt}]
            resp = _openai_client.chat.completions.create(model=OPENAI_MODEL, messages=msgs,
                                                          temperature=float(openai_temp or 0.2), max_tokens=380)
            llm = resp.choices[0].message.content.strip()
            final_ans = llm + "\n\n" + base_ans
        except Exception:
            final_ans = base_ans + "\n\n(LLM synthesis unavailable.)"

    return final_ans, fig_payload, hits

# -----------------------------
# DASH APP
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = APP_TITLE

DEFAULT_PANEL = make_synthetic_panel()

controls_card = dbc.Card(
    dbc.CardBody([
        html.H5("Planner Controls", className="mb-2"),
        dcc.DatePickerSingle(id="week_date", display_format="YYYY-MM-DD", className="mb-2"),
        html.Label("Projection horizon (weeks)"),
        dcc.Slider(id="horizon_wk", min=0, max=12, step=1, value=2, tooltip={"placement":"bottom"}),
        html.Label("Rotation length (days)", className="mt-2"),
        dcc.Slider(id="rotation_days", min=10, max=40, step=1, value=18, tooltip={"placement":"bottom"}),
        html.Label("Nitrogen plan", className="mt-2"),
        dcc.Dropdown(id="nitrogen_plan",
                     options=[{"label":x,"value":x} for x in ["Low","Medium","High"]],
                     value="Medium"),
        html.Hr(),

        html.Label("Agentic RAG (OpenAI)"),
        dcc.Checklist(id="use_openai",
                      options=[{"label":"Use OpenAI for answers","value":"on"}],
                      value=[], className="mb-2"),
        html.Label("Creativity"),
        dcc.Slider(id="openai_temp", min=0.0, max=1.2, step=0.05, value=0.2,
                   tooltip={"placement":"bottom"}),
        html.Hr(),

        html.Label("Outcome"),
        dcc.RadioItems(
            id="outcome",
            options=[{"label":"Emissions (kg CO₂e/ha)","value":"emissions_kg_ha"},
                     {"label":"Milk yield (L/cow/day)","value":"milk_yield_lcd"}],
            value="emissions_kg_ha",
            className="mb-2"
        ),
        html.Label("County filter"),
        dcc.Dropdown(id="county_filter",
                     options=[{"label":"All","value":"All"}] + [{"label":c,"value":c} for c in COUNTIES],
                     value="All", className="mb-2"),
        html.Label("Controls"),
        dcc.Checklist(
            id="controls_vars",
            options=[{"label":"Rain (mm)","value":"rain_mm"},
                     {"label":"GDD","value":"gdd"},
                     {"label":"Input price index","value":"input_price_idx"},
                     {"label":"Herd size","value":"herd_size"}],
            value=["rain_mm","gdd","input_price_idx"],
            className="mb-2"
        ),
        html.Label("SE type"),
        dcc.Dropdown(id="se_type",
                     options=[{"label":"HC1 (robust)","value":"HC1"},
                              {"label":"Cluster by Farm","value":"cluster_farm"},
                              {"label":"Cluster by County","value":"cluster_county"}],
                     value="HC1", className="mb-3"),
        dbc.Button("Run Analysis", id="run_btn", color="primary", className="w-100 mb-3"),

        html.Small("Upload panel CSV (.csv/.xlsx)"),
        dcc.Upload(id="upload-data",
                   children=html.Div(["Drag & drop or ", html.A("select file")]),
                   multiple=False, className="mb-2"),
        html.Small("Optional: upload county GeoJSON (properties.NAME must match county)."),
        dcc.Upload(id="upload-geo",
                   children=html.Div(["Upload GeoJSON"]),
                   multiple=False, className="mb-2"),
        dbc.Button("Use Synthetic Data", id="use_synth", color="secondary", outline=True, className="w-100"),
    ])
)

agent_card = dbc.Card(
    dbc.CardBody([
        html.H5("Agentic Assistant"),
        html.Small("Upload KB docs and ask questions. Agent retrieves, runs tools, and may synthesize with OpenAI."),
        dcc.Upload(id="upload-kb",
                   children=html.Div(["Upload .txt/.md/.csv KB files"]),
                   multiple=True, className="mb-2"),
        dbc.Button("Build KB", id="build_kb", color="secondary", className="mb-2"),
        html.Div(id="kb_status", className="mb-2"),
        dcc.Textarea(id="agent_query", style={"width":"100%","height":"110px"},
                     placeholder="Ask about treatment effect, event-study, mapping, scenarios..."),
        dbc.Button("Ask Agent", id="ask_btn", color="primary", className="mt-2"),
        html.Div(id="agent_llm_hint", className="text-muted mt-1", style={"fontSize":"12px"}),
        html.Div(id="agent_answer", style={"whiteSpace":"pre-wrap","marginTop":"10px"}),
        html.Div(id="agent_fig_html"),
        html.Div([
            dbc.Button("👍", id="agent_up", color="success", size="sm", className="me-1"),
            dbc.Button("👎", id="agent_down", color="danger", size="sm"),
        ], className="mt-2"),
        html.Small(id="agent_feedback_status", className="text-muted")
    ])
)

center_card = dbc.Card(
    dbc.CardBody([
        html.H4("Geo Map (Leaflet / Folium)"),
        html.Div(id="folium_map_iframe"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Estimated Growth (kgDM/ha/d)"), html.H3(id="kpi_growth")])), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Estimated Cover (kgDM/ha)"), html.H3(id="kpi_cover")])), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("QLDPC logical error • QSTP fidelity"),
                                           html.Pre(id="quantum_out", style={"fontSize":"12px"})])), md=4),
        ], className="mt-2"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="ts_plot"), md=6),
            dbc.Col(dcc.Graph(id="es_plot"), md=6),
        ])
    ])
)

app.layout = dbc.Container([
    html.H2(APP_TITLE, className="mt-3"),
    dbc.Row([
        dbc.Col(controls_card, md=3),
        dbc.Col(center_card, md=6),
        dbc.Col(agent_card, md=3),
    ]),
    dcc.Store(id="panel_store"),
    dcc.Store(id="geo_store"),
    dcc.Store(id="kb_store"),
    dcc.Interval(id="heartbeat", interval=45*1000, n_intervals=0)  # periodic self-learning tick
], fluid=True)

# -----------------------------
# PARSERS
# -----------------------------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
        if filename.lower().endswith(('.xlsx','.xls')):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        return None
    return None

def parse_geojson(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        return json.loads(decoded.decode('utf-8', errors='ignore'))
    except Exception:
        return None

def parse_textlike(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.bdecode(content_string) if False else base64.b64decode(content_string)  # safe
    lower = filename.lower()
    try:
        if lower.endswith(('.txt','.md')):
            return decoded.decode('utf-8', errors='ignore')
        if lower.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
            return df.head(500).to_csv(index=False)
    except Exception:
        return None

# -----------------------------
# CALLBACKS
# -----------------------------
@callback(
    OUT("panel_store","data"),
    OUT("geo_store","data"),
    OUT("kb_store","data"),
    OUT("kb_status","children"),
    Input("use_synth","n_clicks"),
    Input("upload-data","contents"), State("upload-data","filename"),
    Input("upload-geo","contents"), State("upload-geo","filename"),
    Input("upload-kb","contents"), State("upload-kb","filename"),
    prevent_initial_call=False
)
def load_sources(use_synth_clicks, data_contents, data_filename, geo_contents, geo_filename, kb_contents, kb_filenames):
    panel = None; geo = None; kb_docs = []; msgs = []
    if data_contents:
        try:
            df = parse_contents(data_contents, data_filename)
            df = ensure_types(df)
            panel = df; msgs.append(f"Panel loaded: {data_filename} ({len(df)} rows)")
        except Exception as e:
            msgs.append(f"Panel upload failed: {e}")
    if geo_contents:
        gj = parse_geojson(geo_contents, geo_filename)
        if gj:
            geo = gj; msgs.append("GeoJSON loaded.")
    if kb_contents and kb_filenames:
        for c,f in zip(kb_contents, kb_filenames):
            txt = parse_textlike(c,f)
            if txt:
                kb_docs.append({"id": f, "name": f, "text": txt, "meta": {}})
        if kb_docs:
            rag_build_index(kb_docs); msgs.append(f"KB indexed: {len(kb_docs)} docs")
    if panel is None:
        panel = DEFAULT_PANEL; msgs.append("Using synthetic panel.")
    return panel.to_dict("records"), geo, (RAG_DOCS or []), dbc.Alert(" | ".join(msgs), color="info")

@callback(
    OUT("kb_status","children", allow_duplicate=True),
    Input("heartbeat","n_intervals"),
    State("panel_store","data"),
    prevent_initial_call=True
)
def periodic_self_learning(n, panel_records):
    notes = []
    try:
        if retrain_intent_clf(): notes.append("Intent model retrained.")
    except Exception: notes.append("Intent retrain error.")
    try:
        df = pd.DataFrame(panel_records) if panel_records else pd.DataFrame()
        if not df.empty and retrain_outcome_model(df, "emissions_kg_ha", ["rain_mm","gdd","input_price_idx","herd_size"]):
            notes.append("Outcome model retrained.")
    except Exception: notes.append("Outcome retrain error.")
    return dbc.Alert(" ".join(notes) or "Self-learning idle.", color="secondary")

@callback(
    OUT("ts_plot","figure"),
    OUT("es_plot","figure"),
    OUT("kpi_growth","children"),
    OUT("kpi_cover","children"),
    OUT("quantum_out","children"),
    Input("run_btn","n_clicks"),
    State("panel_store","data"),
    State("outcome","value"),
    State("controls_vars","value"),
    State("county_filter","value"),
    prevent_initial_call=True
)
def run_analysis(n, panel_records, outcome, controls, county_filter):
    if not panel_records:
        return px.scatter(), px.scatter(), "-", "-", "-"
    df = pd.DataFrame(panel_records)
    if county_filter and county_filter != "All":
        df = df.query("county == @county_filter").copy()

    # Time series
    ts = df.groupby(["date","treated"], as_index=False)[outcome].mean()
    ts_fig = px.line(ts, x="date", y=outcome, color="treated", labels={"treated":"Treated"}, markers=True)

    # Event-study (robust, auto-bucketing)
    if {"adopt_year","q"}.issubset(df.columns):
        adopt_q = (df["adopt_year"].fillna(0).astype(int)*4 + df.get("adopt_q",4).fillna(0).astype(int))
        current_q = df["year"]*4 + df["q"]
        rel = (current_q - adopt_q).astype(int)
        # keep both pre and post for adopters; drop never-treated
        rel = rel.where(df["treated"].eq(1) | adopt_q.gt(0), np.nan)
        df_es = df.assign(rel=rel)
    else:
        df_es = df.assign(rel=np.nan)

    # Try a tighter cap first; if still empty, loosen criteria
    es = make_event_study(df_es, outcome, controls, min_per_bucket=25, rel_cap=6)
    if es.empty:
        es = make_event_study(df_es, outcome, controls, min_per_bucket=15, rel_cap=8)

    if not es.empty:
        es_fig = px.scatter(
            es, x="rel", y="coef",
            error_y=(es["ci_high"] - es["coef"]),
            error_y_minus=(es["coef"] - es["ci_low"]),
            title="Event-study (relative to adoption; baseline shown at 0)"
        ).update_traces(mode="markers+lines")
    else:
        es_fig = px.scatter(title="Event-study unavailable (insufficient support even after pooling)")

    # KPIs (stylized)
    gdd = float(df.get("gdd", pd.Series([1200])).mean())
    rain = float(df.get("rain_mm", pd.Series([80])).mean())
    herd = float(df.get("herd_size", pd.Series([100])).mean())
    growth = max(0.0, (gdd/40.0 - rain/200.0))
    cover = int(round(herd*18 + rain*5))

    # Quantum-like outputs
    qldpc = simulate_qldpc(n_qubits=127, rounds=10, phys_error=0.001)
    qstp = simulate_qstp(num_nodes=6, base_fid=0.99, hops=3)
    qtxt = json.dumps({"QLDPC": qldpc, "QSTP": qstp})

    return ts_fig, es_fig, f"{growth:.1f}", f"{cover}", qtxt

@callback(
    OUT("folium_map_iframe","children"),
    Input("panel_store","data"),
    State("outcome","value"),
    State("county_filter","value"),
    prevent_initial_call=False
)
def update_map(panel_records, outcome, county_filter):
    df = pd.DataFrame(panel_records) if panel_records else pd.DataFrame()
    map_html = build_folium_map(df, outcome, county_filter)
    return html.Iframe(srcDoc=map_html, style={"width":"100%","height":"600px","border":"none"})

@callback(
    OUT("kb_status","children", allow_duplicate=True),
    Input("build_kb","n_clicks"),
    State("upload-kb","contents"),
    State("upload-kb","filename"),
    prevent_initial_call=True
)
def build_kb_cb(n, contents_list, filenames):
    if not contents_list:
        return dbc.Alert("No KB files uploaded.", color="warning")
    docs = []
    for c,f in zip(contents_list, filenames):
        # parse text-like; accept first 500 rows of CSV to avoid massive tokenization
        try:
            content_type, content_string = c.split(',')
            decoded = base64.b64decode(content_string)
            if f.lower().endswith(('.txt','.md')):
                txt = decoded.decode('utf-8', errors='ignore')
            elif f.lower().endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
                txt = df.head(500).to_csv(index=False)
            else:
                txt = None
            if txt:
                docs.append({"id": f, "name": f, "text": txt, "meta": {}})
        except Exception:
            continue
    ok = rag_build_index(docs)
    return dbc.Alert(f"KB built: {len(docs)} docs" if ok else "KB build failed.",
                     color="success" if ok else "danger")

@callback(
    OUT("agent_llm_hint","children"),
    Input("use_openai","value"),
    prevent_initial_call=False
)
def show_llm_hint(val):
    on = (val is not None) and ("on" in val)
    if on and _openai_enabled:
        return f"Using OpenAI {OPENAI_MODEL} for answer synthesis."
    if on and not _openai_enabled:
        return "OpenAI not configured (set OPENAI_API_KEY). Falling back to local agent."
    return "Local agent mode (no OpenAI)."

# *** FIXED: remove duplicate writer to agent_feedback_status.children ***
@callback(
    OUT("agent_answer","children"),
    OUT("agent_fig_html","children"),
    Input("ask_btn","n_clicks"),
    State("agent_query","value"),
    State("panel_store","data"),
    State("outcome","value"),
    State("controls_vars","value"),
    State("county_filter","value"),
    State("se_type","value"),
    State("use_openai","value"),
    State("openai_temp","value"),
    prevent_initial_call=True
)
def ask_agent_cb(n, query, panel_records, outcome, controls, county_filter, se_type, use_openai, openai_temp):
    if not query or not query.strip():
        return "Ask a question first.", None
    use_llm = (use_openai is not None) and ("on" in use_openai)
    text, fig_payload, _hits = agent_answer(query.strip(), panel_records or [], outcome,
                                            controls or [], county_filter, se_type, use_llm, openai_temp or 0.2)
    fig_html = None
    if isinstance(fig_payload, str) and fig_payload.strip().startswith("<"):
        fig_html = html.Iframe(srcDoc=fig_payload, style={"width":"100%","height":"350px","border":"none"})
    elif fig_payload is not None:
        fig_html = dcc.Graph(figure=fig_payload)
    return text, fig_html

@callback(
    OUT("agent_feedback_status","children"),
    Input("agent_up","n_clicks"),
    Input("agent_down","n_clicks"),
    State("agent_query","value"),
    State("agent_answer","children"),
    prevent_initial_call=True
)
def agent_feedback_cb(up, down, q, a):
    trig = ctx.triggered_id
    if not q or not a:
        return no_update
    label = 1 if trig == "agent_up" else 0
    rec = {"query": q, "answer": a, "label": label, "ts": datetime.utcnow().isoformat()}
    try:
        with open(MEMORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return "Feedback recorded."
    except Exception:
        return "Could not record feedback."

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Use run_server for Dash; host binding for external access
    app.run(host="0.0.0.0", port=8050, debug=True)
