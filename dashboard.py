MODE = "streamlit"  # options: "streamlit", "api", "batch"

#PART A: BACKEND (FASTAPI)
#1. API Schemas (schemas.py)
from pydantic import BaseModel
from typing import Dict, List
import shap
import joblib
import numpy as np

class SourceInput(BaseModel):
    source_id: str
    features: Dict[str, float]
    reliability_series: List[float]

class OptimizationRequest(BaseModel):
    sources: List[SourceInput]
    seed: int = 42

class Assignment(BaseModel):
    source_id: str
    task: str
    expected_risk: float

class OptimizationResponse(BaseModel):
    policies: Dict[str, List[Assignment]]
    emv: Dict[str, float]
    evpi: float
    audit_log: Dict

#2. ML Layer
#i) XGBoost behavior classifier
#ml/xgb_behavior.py
import joblib
import numpy as np

MODEL_VERSION = "xgb_v4"
xgb_model = joblib.load("models/xgb_behavior.pkl")

BEHAVIOR_CLASSES = [
    "cooperative",
    "uncertain",
    "coerced",
    "deceptive"
]

def predict_behavior_probs(features: dict):
    x = np.array(list(features.values())).reshape(1, -1)
    probs = xgb_model.predict_proba(x)[0]

    return dict(zip(BEHAVIOR_CLASSES, probs.tolist()))

#SHAP service
#ml shap explainer.py
xgb_model = joblib.load("models/xgb_behavior.pkl")

# TreeExplainer is correct for XGBoost
explainer = shap.TreeExplainer(xgb_model)

FEATURE_NAMES = [
    "task_success_rate",
    "corroboration_score",
    "report_timeliness",
    "handler_confidence",
    "ci_flag",
    "scenario_probability"
]

def explain_source(features: dict):
    x = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)

    shap_values = explainer.shap_values(x)

    # Multi-class output → return per class
    explanation = {}
    for i, cls in enumerate(xgb_model.classes_):
        explanation[str(cls)] = {
            FEATURE_NAMES[j]: float(shap_values[i][0][j])
            for j in range(len(FEATURE_NAMES))
        }

    return explanation

#3. GRU regressor (realibility + deception)
# ml/gru_scores.py
import tensorflow as tf
import numpy as np

MODEL_VERSION = "gru_v2"
gru_model = tf.keras.models.load_model("models/gru_reliability_deception")

def predict_gru_scores(series):
    ts = np.array(series).reshape(1, -1, 1)
    reliability, deception = gru_model.predict(ts, verbose=0)[0]
    return float(reliability), float(deception)

#4. Optimization layer (Pyomo TSSP)
# optimization/tssp_model.py
# optimization/tssp_model.py
from pyomo.environ import *

REC_COST = {
    "cooperative": 0,
    "uncertain": 20,
    "coerced": 50,
    "deceptive": 100
}

TASKS = ["Task_A", "Task_B", "Task_C"]

def solve_tssp(sources, behavior_probs, reliability, deception):

    m = ConcreteModel()
    m.S = Set(initialize=sources)
    m.T = Set(initialize=TASKS)
    m.B = Set(initialize=REC_COST.keys())

    m.x = Var(m.S, m.T, domain=Binary)
    m.y = Var(m.S, m.T, m.B, domain=NonNegativeReals)

    def stage1_cost(s):
        return 10 * (1 - reliability[s]) + 15 * deception[s]

    def objective(m):
        stage1 = sum(stage1_cost(s) * m.x[s, t]
                     for s in m.S for t in m.T)

        stage2 = sum(
            behavior_probs[s][b] * REC_COST[b] * m.y[s, t, b]
            for s in m.S for t in m.T for b in m.B
        )

        reward = sum(5 * reliability[s] * m.x[s, t]
                     for s in m.S for t in m.T)

        return stage1 + stage2 - reward

    m.Obj = Objective(rule=objective, sense=minimize)

    m.Assign = Constraint(
        m.S, rule=lambda m, s: sum(m.x[s, t] for t in m.T) == 1
    )

    m.Link = Constraint(
        m.S, m.T, m.B,
        rule=lambda m, s, t, b: m.y[s, t, b] <= m.x[s, t]
    )

    SolverFactory("cbc").solve(m)

    assignments = []
    for s in m.S:
        for t in m.T:
            if m.x[s, t].value > 0.5:
                expected_risk = sum(
                    behavior_probs[s][b] * REC_COST[b]
                    for b in m.B
                )
                assignments.append({
                    "source_id": s,
                    "task": t,
                    "expected_risk": round(expected_risk, 2)
                })

    return assignments

#Baseline solver wrapper
#optimization/baselines.py
#deterministic
def solve_deterministic(sources, reliability):
    return [
        {
            "source_id": s,
            "task": "Task_A",
            "expected_risk": 0.0
        }
        for s in sources
    ]

#uniform
from optimization.tssp_model import solve_tssp

def solve_uniform(sources, reliability, deception):
    uniform_probs = {
        s: {b: 0.25 for b in ["cooperative","uncertain","coerced","deceptive"]}
        for s in sources
    }
    return solve_tssp(sources, uniform_probs, reliability, deception)

#EMV + EVPI Utilities
# optimization/emv.py

def compute_emv(assignments):
    return round(sum(a["expected_risk"] for a in assignments), 2)

def compute_evpi(ml_emv, uniform_emv):
    return round(uniform_emv - ml_emv, 2)

#SOURCE-LEVEL EVPI
from copy import deepcopy

def compute_source_evpi(
    source_id,
    base_assignments,
    behavior_probs,
    recourse_cost
):
    base_emv = compute_emv(
        base_assignments,
        behavior_probs,
        recourse_cost
    )

    # Perfect info assumption: source behavior known → zero uncertainty cost
    perfect_probs = deepcopy(behavior_probs)
    perfect_probs[source_id] = {
        "cooperative": 1.0,
        "uncertain": 0.0,
        "coerced": 0.0,
        "deceptive": 0.0,
    }

    perfect_emv = compute_emv(
        base_assignments,
        perfect_probs,
        recourse_cost
    )

    return base_emv - perfect_emv
#Risk vs coverage scatter plot
#optmization/metrics.py
def compute_coverage(assignments):
    return len(assignments)

def compute_expected_risk(assignments, behavior_probs, recourse_cost):
    risk = 0
    for s, t in assignments:
        for b, p in behavior_probs[s].items():
            risk += p * recourse_cost[b]
    return risk

#return per policy
def compute_tradeoff(ml_assignments, det_assignments, uni_assignments, behavior_probs, recourse_cost):
    return {
        "ml": {
            "coverage": compute_coverage(ml_assignments),
            "risk": compute_expected_risk(ml_assignments, behavior_probs, recourse_cost)
        },
        "deterministic": {
            "coverage": compute_coverage(det_assignments),
            "risk": compute_expected_risk(det_assignments, behavior_probs, recourse_cost)
        },
        "uniform": {
            "coverage": compute_coverage(uni_assignments),
            "risk": compute_expected_risk(uni_assignments, behavior_probs, recourse_cost)
        }
    }

#5. FastAPI Entry point
## main.py
from fastapi import FastAPI
from schemas import OptimizationRequest
from ml.xgb_behavior import predict_behavior_probs
from ml.gru_scores import predict_gru_scores
from optimization.tssp_model import solve_tssp
from optimization.baselines import solve_deterministic, solve_uniform
from optimization.emv import compute_emv, compute_evpi
import uuid, time
from ml.shap_explainer import explain_source
from schemas import SourceInput
import hashlib
from datetime import datetime

app = FastAPI()

@app.post("/optimize")
def optimize(req: OptimizationRequest):

    behavior_probs, reliability, deception = {}, {}, {}

    for src in req.sources:
        behavior_probs[src.source_id] = predict_behavior_probs(src.features)
        r, d = predict_gru_scores(src.reliability_series)
        reliability[src.source_id] = r
        deception[src.source_id] = d

    sources = [s.source_id for s in req.sources]

    ml = solve_tssp(sources, behavior_probs, reliability, deception)
    det = solve_deterministic(sources, reliability)
    uni = solve_uniform(sources, reliability, deception)

    ml_emv = compute_emv(ml)
    det_emv = compute_emv(det)
    uni_emv = compute_emv(uni)

    audit_log = {
        "run_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "seed": req.seed,
        "models": {
            "xgboost": "xgb_v4",
            "gru": "gru_v2"
        }
    }

    return {
        "policies": {
            "ml_tssp": ml,
            "deterministic": det,
            "uniform": uni
        },
        "emv": {
            "ml_tssp": ml_emv,
            "deterministic": det_emv,
            "uniform": uni_emv
        },
        "evpi": compute_evpi(ml_emv, uni_emv),
        "audit_log": audit_log
    }

#SHAP Service
@app.post("/explain")
def explain(source: SourceInput):
    shap_values = explain_source(source.features)

    return {
        "source_id": source.source_id,
        "shap_values": shap_values,
        "model": "xgb_behavior_v4"
    }

#GRU drift monitoring timeline
#logging gru outputs
gru_log = {
    "source_id": src.source_id,
    "timestamp": datetime.utcnow().isoformat(),
    "reliability": r,
    "deception": d,
    "model_version": "gru_v2"
}
#add endpoints
@app.get("/drift/{source_id}")
def get_drift(source_id: str):
    records = []
    with open("logs/gru_drift.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["source_id"] == source_id:
                records.append(rec)
    return records

# Persist (file / db)
with open("logs/gru_drift.jsonl", "a") as f:
    f.write(json.dumps(gru_log) + "\n")

#Export SHAP + decision logs to signed JSON

def generate_audit_log(payload, results, shap_values):
    log = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": payload,
        "results": results,
        "shap": shap_values,
        "models": {
            "xgb": "xgb_behavior_v4",
            "gru": "gru_v2"
        }
    }

    hash_value = hashlib.sha256(
        json.dumps(log, sort_keys=True).encode()
    ).hexdigest()

    log["hash"] = hash_value

    return log

#expose end point
@app.post("/export_audit")
def export_audit(payload: dict):
    log = generate_audit_log(
        payload["inputs"],
        payload["results"],
        payload["shap"]
    )
    return log


#B. PART B: FRONTEND (STREAMLIT)
#frontend/app.py
import streamlit as st
from api import run_optimization
import shap
import matplotlib.pyplot as plt
from api import explain_source

st.set_page_config(
    page_title="ML–TSSP HUMINT Tasking Dashboard",
    layout="wide"
)

st.title("ML–TSSP HUMINT Source Tasking Optimisation Dashboard")

def explain_source(source):
    r = requests.post("http://backend:8000/explain", json=source)
    return r.json()


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None

# -------------------------------------------------
# Source input panel
# -------------------------------------------------
st.header("Source Profiles")

sources = []

num_sources = st.slider(
    "Number of sources to simulate",
    min_value=1,
    max_value=10,
    value=3
)

for i in range(num_sources):
    st.subheader(f"Source {i + 1}")

    col1, col2 = st.columns(2)

    with col1:
        features = {
            "task_success_rate": st.slider(
                "Task Success Rate",
                0.0, 1.0, 0.6,
                key=f"tsr_{i}"
            ),
            "corroboration_score": st.slider(
                "Corroboration Score",
                0.0, 1.0, 0.5,
                key=f"cor_{i}"
            ),
            "report_timeliness": st.slider(
                "Report Timeliness",
                0.0, 1.0, 0.5,
                key=f"time_{i}"
            )
        }

    with col2:
        st.caption("GRU-predicted reliability trajectory")
        reliability_ts = [0.6, 0.65, 0.7, 0.68]
        st.line_chart(reliability_ts)

    sources.append({
        "source_id": f"SRC_{i + 1:03d}",
        "features": features,
        "reliability_series": reliability_ts
    })

# -------------------------------------------------
# Run optimisation
# -------------------------------------------------
st.divider()

if st.button("Run Optimisation"):
    payload = {
        "sources": sources,
        "seed": 42
    }

    with st.spinner("Running ML–TSSP optimisation…"):
        st.session_state.results = run_optimization(payload)

    st.success("Optimisation completed")

results = st.session_state.results

# -------------------------------------------------
# Results section
# -------------------------------------------------
if results is not None:
    tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(
    ["ML–TSSP", "Deterministic", "Uniform", "SHAP Explanations", "EVPI Ranking", "Risk vs Coverage","Reliability & Deception Drift"]
)

    # ---------------- ML–TSSP ----------------
    with tab1:
        st.subheader("Optimised ML–TSSP Policy")

        st.table(results["policies"]["ml_tssp"])

        st.metric(
            "Expected Operational Risk (EMV)",
            f"{results['emv']['ml_tssp']:.2f}"
        )

    # ---------------- Deterministic ----------------
    with tab2:
        st.subheader("Deterministic Assignment")
        st.caption("Ignores uncertainty and recourse")

        st.table(results["policies"]["deterministic"])

        st.metric(
            "Expected Operational Risk (EMV)",
            f"{results['emv']['deterministic']:.2f}"
        )

    # ---------------- Uniform ----------------
    with tab3:
        st.subheader("Uniform-Probability TSSP")
        st.caption("Assumes equal likelihood of behavioural outcomes")

        st.table(results["policies"]["uniform"])

        st.metric(
            "Expected Operational Risk (EMV)",
            f"{results['emv']['uniform']:.2f}"
        )

    # -------------------------------------------------
    # EVPI panel
    # -------------------------------------------------
    st.divider()
    st.header("Value of Information")

    evpi = results["emv"]["uniform"] - results["emv"]["ml_tssp"]

    st.metric(
        "EVPI (Operational Value of ML)",
        f"{evpi:.2f}"
    )

    st.caption(
        "EVPI is computed relative to a uniform-uncertainty baseline. "
        "Higher values indicate greater benefit from ML-driven uncertainty modelling."
    )

    # -------------------------------------------------
    # Audit & transparency
    # -------------------------------------------------
    with st.expander("Audit Metadata"):
        st.json(results.get("audit_log", {}))


#SHAP tab UI
with tab4:
    st.subheader("Source-level ML Explanations (SHAP)")

    selected_source = st.selectbox(
        "Select Source",
        [s["source_id"] for s in sources]
    )

    source_data = next(
        s for s in sources if s["source_id"] == selected_source
    )

    if st.button("Explain Decision"):
        explanation = explain_source(source_data)

        st.caption(
            f"SHAP explanation for XGBoost behavior classifier "
            f"(Model: {explanation['model']})"
        )

        behavior = st.selectbox(
            "Select Behavior Class",
            explanation["shap_values"].keys()
        )

        shap_dict = explanation["shap_values"][behavior]

        fig, ax = plt.subplots()
        ax.barh(
            list(shap_dict.keys()),
            list(shap_dict.values())
        )
        ax.set_title(
            f"Feature impact for behavior: {behavior}"
        )
        ax.set_xlabel("SHAP value")

        st.pyplot(fig)

        st.info(
            "Positive values push the prediction toward this behavior. "
            "Negative values reduce likelihood."
        )

#source evpi ranking
tab5 = st.tabs(
    ["EVPI Ranking"]
)[0]

with tab5:
    st.subheader("Source-Level EVPI Ranking")

    evpi_df = (
        pd.DataFrame(
            results["source_evpi"].items(),
            columns=["Source", "EVPI"]
        )
        .sort_values("EVPI", ascending=False)
    )

    st.table(evpi_df)

    st.caption(
        "Higher EVPI indicates greater operational value "
        "from resolving uncertainty about that source."
    )
#Scatter plot
tab6 = st.tabs(["Risk vs Coverage"])[0]

with tab6:
    st.subheader("Task Coverage vs Expected Risk")

    trade = results["tradeoff"]

    df = pd.DataFrame([
        {"Policy": k, "Coverage": v["coverage"], "Risk": v["risk"]}
        for k, v in trade.items()
    ])

    st.scatter_chart(
        df,
        x="Risk",
        y="Coverage",
        color="Policy"
    )

    st.caption(
        "Preferred policies achieve higher coverage with lower expected risk."
    )
#GRU drift timeline
tab7 = st.tabs(["GRU Drift"])[0]

with tab7:
    st.subheader("Reliability & Deception Drift")

    src = st.selectbox(
        "Select Source",
        [s["source_id"] for s in sources]
    )

    drift = requests.get(
        f"http://backend:8000/drift/{src}"
    ).json()

    if drift:
        df = pd.DataFrame(drift)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.line_chart(
            df.set_index("timestamp")[["reliability", "deception"]]
        )
st.markdown(
    "<hr style='margin-top:2rem;'>"
    "<p style='text-align:center; font-size:0.85em; color:gray;'>"
    "© 2026 ML–TSSP Research Prototype. All rights reserved."
    "</p>",
    unsafe_allow_html=True
)
