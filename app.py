"""Online, synthetic-only demonstration of industrial anomaly detection.
No SWaT records, attack schedules, or pretrained SWaT artifacts are included.
Run: python -m streamlit run app.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Industrial Anomaly Explainer · Demo", page_icon="⚙️", layout="wide")
st.markdown("""<style>
.block-container {padding-top:1.6rem;max-width:1400px;padding-bottom:3rem}
[data-testid="stMetric"] {background:rgba(95,119,161,.10);border:1px solid rgba(115,142,181,.16);padding:16px 18px;border-radius:14px}
[data-testid="stMetricLabel"] {font-weight:600}
.hero {padding:26px 30px;border-radius:18px;background:linear-gradient(120deg,#132d4b,#164e63);color:#f4fbff;margin-bottom:20px}
.hero h1 {font-size:2.1rem;line-height:1.15;color:#fff;margin:0 0 10px 0}
.hero p {color:#d4eaf1;font-size:1rem;margin:0}
.hero .eyebrow {font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:#91dbe7;font-weight:700;margin-bottom:12px}
.section-note {padding:14px 18px;border-radius:12px;background:rgba(83,125,154,.10);border-left:4px solid #48a5c1;margin:14px 0}
.small-note {font-size:.85rem;opacity:.85}
</style>""", unsafe_allow_html=True)

FEATURES = ["FIT101","LIT101","MV101","P101","P102","AIT201","AIT202","AIT203","FIT201","MV201",
"P201","P202","P203","AIT301","AIT302","AIT303","FIT301","LIT301","MV301","P301",
"P302","AIT401","AIT402","FIT401","LIT401","P401","P402","P403","P404","UV401",
"AIT501","AIT502","AIT503","AIT504","FIT501","FIT502","FIT503","FIT504","P501","P502",
"PIT501","PIT502","PIT503","FIT601","P601"]
# These names are illustrative industrial-sensor labels, not an assertion of exact SWaT feature identity.

@st.cache_resource(show_spinner="Building synthetic demonstration model…")
def build_demo():
    rng = np.random.default_rng(42)
    n, k = 5500, len(FEATURES)
    t = np.arange(n)
    latent = np.column_stack([np.sin(t/90), np.cos(t/140), np.sin(t/240), np.cos(t/55)])
    weights = rng.normal(0, .5, size=(4,k))
    baseline = rng.uniform(1, 12, size=k)
    spread = rng.uniform(.25, 2, size=k)
    normal = baseline + (latent @ weights + rng.normal(0,.18,(n,k))) * spread
    normal = pd.DataFrame(normal, columns=FEATURES)
    # Independent held-out synthetic monitoring sequence; disturbance labels are generated, not real attacks.
    m = 1500
    tt = np.arange(n,n+m)
    monitoring = baseline + (np.column_stack([np.sin(tt/90),np.cos(tt/140),np.sin(tt/240),np.cos(tt/55)]) @ weights
                            + rng.normal(0,.18,(m,k))) * spread
    labels = np.zeros(m, dtype=int)
    for start, end, cols, strength in [(240,320,[1,3,7],8),(670,760,[0,17,24],10),(1090,1190,[6,30,43],7)]:
        labels[start:end] = 1
        monitoring[start:end,cols] += strength * spread[cols]
    monitoring = pd.DataFrame(monitoring, columns=FEATURES)
    scaler = StandardScaler().fit(normal)
    scaled_normal = scaler.transform(normal)
    model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto", random_state=42, n_jobs=-1)
    model.fit(scaled_normal)
    normal_scores = -model.decision_function(scaled_normal)
    threshold = float(np.percentile(normal_scores, 99))
    monitor_scores = -model.decision_function(scaler.transform(monitoring))
    return normal, monitoring, labels, model, scaler, normal_scores, monitor_scores, threshold

@st.cache_resource(show_spinner=False)
def explainer_for(_model):
    return shap.TreeExplainer(_model)

def score_rows(frame, model, scaler, threshold):
    scaled = scaler.transform(frame[FEATURES])
    scores = -model.decision_function(scaled)
    return scores, (scores > threshold).astype(int)

def explain_one(frame, model, scaler):
    transformed = scaler.transform(frame[FEATURES])
    magnitudes = np.abs(np.asarray(explainer_for(model).shap_values(transformed)).reshape(-1))
    return pd.DataFrame({"Sensor":FEATURES,"Absolute SHAP magnitude":magnitudes,
                         "Reading":frame.iloc[0][FEATURES].to_numpy(dtype=float)}).nlargest(10,"Absolute SHAP magnitude")

def influence_chart(frame, model, scaler):
    top = explain_one(frame,model,scaler)
    st.bar_chart(top.set_index("Sensor")["Absolute SHAP magnitude"],horizontal=True)
    st.dataframe(top,hide_index=True,width="stretch")
    st.caption("Absolute SHAP magnitudes rank influence on TreeExplainer's internal model output, not causal effects or signed contributions to the displayed anomaly score.")

normal, monitoring, labels, model, scaler, normal_scores, monitor_scores, threshold = build_demo()
flags = (monitor_scores > threshold).astype(int)
st.markdown("""<div class="hero"><div class="eyebrow">Research prototype · Industrial anomaly detection</div>
<h1>Industrial Anomaly Explainer</h1><p>Explainable machine learning for multivariate industrial sensor monitoring · Isolation Forest + SHAP</p></div>""", unsafe_allow_html=True)
st.info("**Two clearly separated components:** The **Research Results**, **Nine-Model Comparison** and **Research SHAP** tabs report results from locally run SWaT research experiments (using different evaluation protocols). The other interactive tabs run an **independent synthetic-data demonstration**. No SWaT records, attack schedules, or SWaT-trained model artifacts are hosted here.", icon="ℹ️")
with st.sidebar:
    st.header("Demo observation controls")
    sample = st.selectbox("Quick example",["Injected disturbance example","Undisturbed example","Custom row"])
    default = 280 if sample=="Injected disturbance example" else 100
    index = int(st.number_input("Observation index",0,len(monitoring)-1,default if sample!="Custom row" else 280,1))
    window = st.slider("Temporal window (each side)",30,300,100,10)
    st.divider()
    st.caption(f"{len(FEATURES)} synthetic features · {model.n_estimators} trees")
    st.caption("Threshold: normal training score 99th percentile")

research, comparison, research_shap, summary, observation, explain, temporal, context, method, explorer, simulator = st.tabs([
    "🎓 Research Results", "⚖️ Nine-Model Comparison", "🧬 Research SHAP", "📊 Demo Overview", "🔎 Observation", "🧠 SHAP Influence", "📈 Temporal Context",
    "🚨 Disturbance Context", "ℹ️ Methodology", "🔬 Anomaly Explorer", "🎛️ Sensor Simulator"])

with research:
    st.subheader("Research results · original SWaT-based prototype")
    st.info("These original Isolation Forest figures use a different training/calibration protocol from the nine-model research comparison. Compare the nine algorithms using the **Model Comparison** tab, not against the figures below.")
    st.caption("These are results reported from the separate, locally executed SWaT research implementation. They are NOT computed by the hosted synthetic demonstration.")
    r1,r2,r3,r4=st.columns(4)
    r1.metric("Precision", "86.02%")
    r2.metric("Recall", "61.81%")
    r3.metric("F1-score", "71.93%")
    r4.metric("Average precision (AP)", "0.7343")
    st.markdown("#### Evaluation at a glance")
    st.bar_chart(pd.DataFrame({"Metric": ["Precision", "Recall", "F1-score", "Average precision"],
                               "Value": [0.8602,0.6181,0.7193,0.7343]}).set_index("Metric"),
                 horizontal=True, x_label="Metric value (0–1)")
    st.caption("Average precision (AP) summarises the precision–recall curve; it is not the same as ROC-AUC. Values above are rounded reported results.")
    st.markdown("#### Research pipeline")
    st.markdown("""**SWaT multivariate sensor data (authorised local access)** → **preprocessing** → **Isolation Forest anomaly scoring** → **normal-data threshold calibration** → **evaluation against available labels** → **SHAP-based sensor influence analysis**.""")
    a,b=st.columns(2)
    with a:
        st.markdown("**Dataset and preprocessing**")
        st.markdown("""- SWaT normal and attack datasets were used in the local research environment.
- 45 process variables were retained after removing six constant features.
- `StandardScaler` was fitted on normal training observations only.
- No restricted SWaT data is included in this hosted application.""")
    with b:
        st.markdown("**Model and interpretation**")
        st.markdown("""- Isolation Forest with 200 trees and `random_state=42`.
- Higher anomaly scores indicate more unusual observations.
- The research decision threshold was set at the 99th percentile of normal training scores.
- Absolute SHAP magnitudes indicate relative influence on the explainer's internal model output, **not** causal effects.""")
    st.markdown("#### Guided demonstration workflow")
    st.markdown("**Start here:** Review the original SWaT metrics above → open **Nine-Model Comparison** for the nine-model study and **Research SHAP** for local explanation summaries → open **Demo Overview** to inspect synthetic monitoring → use **Anomaly Explorer** for model-flagged readings → visit **Sensor Simulator** to enter all 45 readings manually, generate synthetic streams, or upload a non-restricted CSV → inspect SHAP influence. Every interactive prediction uses the separate synthetic-trained model.")
    st.markdown("#### What can be tested on this website?")
    st.markdown("The remaining tabs let reviewers inspect synthetic observations, explore detected disturbances, change sensor values, generate synthetic monitoring sequences and upload **non-restricted** CSV samples. This validates the interaction design and analysis workflow, **not** the reported SWaT performance figures.")
    st.warning("**Research limitations:** The online model is separately trained on synthetic readings. Its predictions, threshold and disturbance scenarios cannot be interpreted as SWaT results or validated real-world attack detection.")
    st.markdown("#### Reproducibility and demonstration")
    st.markdown("The original SWaT-trained prototype and its research evaluation are maintained locally for authorised demonstration. This website intentionally does not provide the restricted dataset or pretrained SWaT artifacts.")
    st.link_button("View public synthetic demo source code", "https://github.com/Ainuddin28/industrial-anomaly-detection")


with comparison:
    st.subheader("Nine-model comparison · local SWaT research")
    st.caption("Offline, aggregate experimental results from the locally executed SWaT study. No restricted data or trained SWaT models are included.")
    st.info("**Different protocols:** The original prototype's 71.93% F1 is from an earlier experiment. The nine-model study reports 63.82% F1 for Isolation Forest under another training/calibration protocol. Point-wise and window-based models also differ in evaluation alignment and training subsets; rankings are descriptive, not a controlled universal benchmark.")
    nine = pd.read_csv(Path(__file__).resolve().parent / "nine_model_aggregate.csv")
    x,y,z = st.columns(3)
    x.metric("Models evaluated", "9")
    y.metric("Highest F1 · Isolation Forest", "63.82%")
    z.metric("Highest recall · One-Class SVM", "90.49%")
    st.markdown("#### Nine-model results")
    st.dataframe(nine.rename(columns={"F1_percent":"F1 (%)","AP_percent":"Average precision (%)","Precision_percent":"Precision (%)","Recall_percent":"Recall (%)","Evaluation_type":"Evaluation type"}).style.format({"F1 (%)":"{:.2f}","Average precision (%)":"{:.2f}","Precision (%)":"{:.2f}","Recall (%)":"{:.2f}"},na_rep="Not verified"),hide_index=True,use_container_width=True)
    st.caption("Some metrics are omitted where exact values were not available in the supplied screenshots. F1 and AP are rounded for display; underlying measurements and evaluation choices are documented locally.")
    st.markdown("#### F1 ranking")
    st.bar_chart(nine.sort_values("F1_percent").set_index("Model")["F1_percent"], horizontal=True, x_label="F1 (%)")
    st.markdown("#### Average precision ranking")
    st.bar_chart(nine.sort_values("AP_percent").set_index("Model")["AP_percent"], horizontal=True, x_label="Average precision (%)")
    st.markdown("#### Precision versus recall")
    st.bar_chart(nine.set_index("Model")[["Precision_percent","Recall_percent"]].rename(columns={"Precision_percent":"Precision (%)","Recall_percent":"Recall (%)"}),stack=False)
    st.markdown("#### Research interpretation")
    st.markdown("**Isolation Forest** produced the strongest F1 (63.82%) and average precision (72.98%) in this set of experiments, with 60.96% precision and 66.96% recall. **One-Class SVM** showed high recall (90.49%) but low precision (14.82%). The compact **USAD** implementation ranked second in F1 (30.36%). These results reflect the tested configurations only.")
    st.warning("**Methodological caveat:** Point-wise and window-based detectors are presented together for research overview, not as an exactly matched head-to-head comparison. Window-based models align their labels with the final timestep and exclude the first window−1 test rows. The USAD implementation is preliminary, not a reproduction of published benchmark settings.")
    st.download_button("⬇ Download nine-model aggregate metrics",nine.to_csv(index=False).encode("utf-8"),file_name="nine_model_aggregate_metrics.csv",mime="text/csv",key="nine_download")
    with st.expander("Earlier three-model experiment · full aggregate confusion counts"):
        original = pd.read_csv(Path(__file__).resolve().parent / "model_comparison.csv")
        st.dataframe(original,hide_index=True,use_container_width=True)
        st.caption("Earlier three-model aggregate results retained for traceability. No SWaT observation-level data is hosted.")

with research_shap:
    st.subheader("Research SHAP · local Isolation Forest")
    st.info("These are **aggregate, research-derived SHAP influence summaries** for selected local SWaT observations. The public interactive SHAP tab uses a separate synthetic-trained Isolation Forest. The explainer values are for the forest's internal output, **not** signed anomaly scores or physical causes.")
    st.markdown("#### True-positive attack observation")
    attack = pd.DataFrame({"Sensor":["AIT501","PIT501","PIT503","FIT502","FIT501","FIT503","FIT504","LIT401","MV304","UV401"],"Absolute SHAP magnitude":[0.609746,0.567487,0.550689,0.533861,0.526948,0.522674,0.495088,0.450556,0.442061,0.389468]})
    st.bar_chart(attack.set_index("Sensor"),horizontal=True)
    st.markdown("#### True-negative normal observation")
    normal = pd.DataFrame({"Sensor":["AIT201","AIT202","AIT504","LIT101","AIT501","AIT402","PIT503","FIT502","P205","FIT401"],"Absolute SHAP magnitude":[0.451255,0.397481,0.338790,0.309995,0.250296,0.244757,0.184622,0.152240,0.122929,0.119226]})
    st.bar_chart(normal.set_index("Sensor"),horizontal=True)
    st.caption("Values transcribed from locally executed run_shap_comparison.py terminal output. These are selected examples, not population-level feature importance or causal attribution.")
    st.markdown("**Interpretation:** AIT501, PIT501 and PIT503 have prominent absolute SHAP magnitudes in the selected detected attack; AIT201 and AIT202 are prominent in the selected correctly classified normal observation. Magnitude alone does not show which physical component caused an anomaly.")
    st.warning("No raw SWaT readings, attack schedules, row-level exports, or SWaT-trained model files are included in this public dashboard.")

with summary:
    st.subheader("Synthetic monitoring dashboard")
    st.caption("All numbers and plots in this tab are generated from the hosted synthetic model, not the original SWaT evaluation.")
    a,b,c,d=st.columns(4)
    a.metric("Monitoring observations",f"{len(monitoring):,}")
    b.metric("Injected disturbance rows",f"{int(labels.sum()):,}")
    c.metric("Model flagged rows",f"{int(flags.sum()):,}")
    d.metric("Decision threshold",f"{threshold:.6f}")
    st.caption("Injected disturbance labels are artificial scenarios, not real cyberattacks or SWaT ground truth.")
    counts=pd.crosstab(pd.Series(labels,name="Synthetic disturbance"),pd.Series(flags,name="Model flag"))
    st.subheader("Synthetic disturbance / detection comparison")
    st.dataframe(counts,width="stretch")
    st.line_chart(pd.DataFrame({"Anomaly score":monitor_scores,"Decision threshold":threshold}),
                  x_label="Synthetic observation index",y_label="Anomaly score")

with observation:
    score=float(monitor_scores[index]); pred=int(flags[index]); injected=int(labels[index])
    st.subheader(f"Synthetic observation {index:,}")
    a,b,c,d=st.columns(4)
    a.metric("Anomaly score",f"{score:.6f}")
    b.metric("Threshold",f"{threshold:.6f}")
    c.metric("Model prediction","ANOMALY" if pred else "NORMAL")
    d.metric("Injected disturbance?","YES" if injected else "NO")
    st.dataframe(pd.DataFrame({"Sensor":FEATURES,"Reading":monitoring.iloc[index].to_numpy()}),hide_index=True,width="stretch")

with explain:
    st.subheader("SHAP influence · selected synthetic observation")
    influence_chart(monitoring.iloc[[index]],model,scaler)

with temporal:
    st.subheader("Anomaly scores near selected synthetic observation")
    lo,hi=max(0,index-window),min(len(monitoring),index+window+1)
    view=pd.DataFrame({"Anomaly score":monitor_scores[lo:hi],"Decision threshold":threshold},index=np.arange(lo,hi)-index)
    st.line_chart(view,x_label="Observations relative to selected row",y_label="Anomaly score")
    st.metric("Flagged in window",int(flags[lo:hi].sum()))

with context:
    st.subheader("Synthetic disturbance context")
    st.info("This tab describes **injected hypothetical disturbances**, not documented SWaT attacks.")
    scenarios=[(240,320,"Synthetic level/valve/quality disturbance"),(670,760,"Synthetic flow/level disturbance"),(1090,1190,"Synthetic quality/flow disturbance")]
    match=[(start,end,description) for start,end,description in scenarios if start<=index<end]
    if match:
        start,end,description=match[0]
        st.write(f"**Scenario:** {description}  ")
        st.write(f"**Synthetic rows:** {start}–{end-1}  ")
        st.write(f"**Model prediction:** {'ANOMALY' if flags[index] else 'NORMAL'}")
    else:
        st.success("No hypothetical disturbance was injected at this row.")
    st.caption("Injected changes are not a physical plant simulation and do not establish a cyberattack.")

with method:
    st.subheader("Hosted demonstration methodology")
    st.info("This methodology describes the synthetic model powering the interactive tabs. See **Research Results** for the separate SWaT research pipeline.")
    st.markdown("""**Source:** reproducibly generated synthetic multivariate sensor readings; no SWaT records are bundled.  
**Model:** Isolation Forest (200 trees, `random_state=42`) trained only on synthetic normal readings.  
**Preprocessing:** StandardScaler fitted on synthetic normal readings only.  
**Score:** negative sklearn `decision_function`; higher means more anomalous.  
**Threshold:** 99th percentile of synthetic normal training scores.  
**Explanations:** absolute SHAP magnitudes for TreeExplainer internal output.  
**Evaluation:** synthetic injected disturbances only; no SWaT precision, recall, F1, or PR-AUC claims.  
**Limitations:** not live telemetry, a validated digital twin, or a production safety system.""")

with explorer:
    st.subheader("Explore synthetic monitoring observations")
    f=st.selectbox("Filter",["All observations","Predicted anomalies","Injected disturbances","Unflagged observations"])
    options=np.arange(len(monitoring))
    if f=="Predicted anomalies": options=options[flags==1]
    elif f=="Injected disturbances": options=options[labels==1]
    elif f=="Unflagged observations": options=options[flags==0]
    order=st.selectbox("Sort",["Highest anomaly score","Lowest anomaly score","Closest to threshold"])
    key=(-monitor_scores[options] if order=="Highest anomaly score" else
         monitor_scores[options] if order=="Lowest anomaly score" else np.abs(monitor_scores[options]-threshold))
    chosen=options[np.argsort(key)[:100]]
    st.caption(f"{len(options):,} matching observations; showing up to 100")
    st.dataframe(pd.DataFrame({"Row":chosen,"Anomaly score":monitor_scores[chosen],
                               "Model prediction":np.where(flags[chosen],"ANOMALY","NORMAL"),
                               "Injected disturbance":labels[chosen].astype(bool)}),hide_index=True,width="stretch")
    if len(chosen):
        picked=int(st.selectbox("Inspect a row",chosen.tolist()))
        st.write(f"**Row {picked}** · score {monitor_scores[picked]:.6f} · {'ANOMALY' if flags[picked] else 'NORMAL'}")
        influence_chart(monitoring.iloc[[picked]],model,scaler)

with simulator:
    st.subheader("Manual observation lab · edit all 45 sensor readings")
    st.caption("All 45 synthetic-model input features are editable in one table. Start from a synthetic baseline or create a fully custom observation. This is NOT the locally trained SWaT model.")
    if "lab_values" not in st.session_state:
        st.session_state["lab_values"] = normal.iloc[100].to_numpy(dtype=float).copy()
        st.session_state["lab_reference"] = normal.iloc[100].to_numpy(dtype=float).copy()
        st.session_state["lab_result"] = None
    st.markdown("#### 1 · Set up an observation")
    mode = st.selectbox("Starting point", ["Synthetic normal example", "Synthetic disturbed example", "Blank values (all zero)", "Keep my current edits"], key="lab_mode")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        baseline_index = int(st.number_input("Baseline observation index", min_value=0, max_value=1499, value=280, step=1, key="lab_baseline_index", disabled=mode in ("Blank values (all zero)","Keep my current edits")))
    with col_b:
        st.write(" ")
        if st.button("↺ Load selected starting point", use_container_width=True, key="lab_load"):
            if mode == "Synthetic normal example":
                vals = normal.iloc[baseline_index % len(normal)].to_numpy(dtype=float)
            elif mode == "Synthetic disturbed example":
                vals = monitoring.iloc[baseline_index].to_numpy(dtype=float)
            elif mode == "Blank values (all zero)":
                vals = np.zeros(len(FEATURES), dtype=float)
            else:
                vals = st.session_state["lab_values"].copy()
            st.session_state["lab_values"] = vals.copy()
            st.session_state["lab_reference"] = vals.copy()
            st.session_state["lab_result"] = None
            st.session_state.pop("lab_editor", None)
            st.rerun()
    with col_c:
        st.write(" ")
        if st.button("↩ Reset to original synthetic baseline", use_container_width=True, key="lab_reset"):
            vals = normal.iloc[100].to_numpy(dtype=float)
            st.session_state["lab_values"] = vals.copy()
            st.session_state["lab_reference"] = vals.copy()
            st.session_state["lab_result"] = None
            st.session_state.pop("lab_editor", None)
            st.rerun()
    st.markdown("#### 2 · Enter all sensor values")
    st.info("**Edit the Value column** for any or all of the 45 sensors. Use the starting-point button to load an example, then edit values directly. The reference and difference columns are informational.")
    reference = np.asarray(st.session_state["lab_reference"],dtype=float)
    values = np.asarray(st.session_state["lab_values"],dtype=float)
    editor_df = pd.DataFrame({"Sensor": FEATURES, "Value": values, "Reference": reference, "Change": values-reference})
    edited = st.data_editor(editor_df, hide_index=True, use_container_width=True, num_rows="fixed", key="lab_editor",
                            disabled=["Sensor","Reference","Change"],
                            column_config={"Sensor":st.column_config.TextColumn("Sensor",width="medium"),
                                           "Value":st.column_config.NumberColumn("Value · editable",format="%.6f",required=True),
                                           "Reference":st.column_config.NumberColumn("Starting value",format="%.6f"),
                                           "Change":st.column_config.NumberColumn("Change at last run",format="%.6f")})
    try:
        entered = pd.to_numeric(edited["Value"],errors="coerce").to_numpy(dtype=float)
        valid = len(entered)==len(FEATURES) and np.isfinite(entered).all()
    except (ValueError,TypeError):
        valid=False
    if not valid:
        st.error("All 45 sensor readings must be finite numeric values. Correct the highlighted entries before running.")
    else:
        changes=int(np.count_nonzero(~np.isclose(entered,reference,rtol=1e-9,atol=1e-9)))
        st.caption(f"{len(FEATURES)} sensor inputs · {changes} different from the loaded reference · values are illustrative and may not match physical equipment ranges.")
    st.markdown("#### 3 · Analyse your custom observation")
    if st.button("▶ Analyse all 45 readings",type="primary",disabled=not valid,key="lab_run"):
        test = pd.DataFrame([entered],columns=FEATURES)
        scores,preds=score_rows(test,model,scaler,threshold)
        baseline_frame=pd.DataFrame([reference],columns=FEATURES)
        baseline_score,baseline_pred=score_rows(baseline_frame,model,scaler,threshold)
        st.session_state["lab_values"]=entered.copy()
        st.session_state["lab_result"]={"values":entered.copy(),"reference":reference.copy(),
                                        "score":float(scores[0]),"pred":int(preds[0]),
                                        "base_score":float(baseline_score[0]),"base_pred":int(baseline_pred[0])}
    result=st.session_state.get("lab_result")
    if result is not None:
        st.success("Analysis complete · results from the hosted synthetic-trained Isolation Forest.")
        r1,r2,r3,r4=st.columns(4)
        r1.metric("Custom prediction","ANOMALY" if result["pred"] else "NORMAL")
        r2.metric("Custom anomaly score",f'{result["score"]:.6f}',delta=f'{result["score"]-result["base_score"]:+.6f} vs reference',delta_color="inverse")
        r3.metric("Decision threshold",f'{threshold:.6f}')
        r4.metric("Reference prediction","ANOMALY" if result["base_pred"] else "NORMAL")
        st.caption("Higher scores mean more unusual observations. The threshold is calibrated on synthetic normal training data. A prediction is not a physical safety assessment.")
        comparison=pd.DataFrame({"Sensor":FEATURES,"Reference":result["reference"],"Custom":result["values"],"Difference":result["values"]-result["reference"]})
        changed=comparison.loc[~np.isclose(comparison["Difference"],0,atol=1e-9,rtol=1e-9)]
        st.markdown("##### Sensor changes")
        if changed.empty:
            st.info("No sensor readings differ from the selected reference. Try changing several values and analyse again.")
        else:
            st.dataframe(changed,hide_index=True,use_container_width=True)
            top_diff=changed.assign(AbsChange=changed["Difference"].abs()).nlargest(12,"AbsChange")
            st.bar_chart(top_diff.set_index("Sensor")["Difference"],horizontal=True)
        st.markdown("##### SHAP sensor influence · custom observation")
        custom_frame=pd.DataFrame([result["values"]],columns=FEATURES)
        influence_chart(custom_frame,model,scaler)
        st.download_button("⬇ Download custom observation + prediction",data=custom_frame.assign(Anomaly_score=result["score"],Predicted_anomaly=result["pred"]).to_csv(index=False),file_name="custom_synthetic_observation_prediction.csv",mime="text/csv",key="lab_download")
        st.caption("If you change table values, click Analyse again to update the results. SHAP magnitudes are not causal attributions.")
    st.divider()
    st.subheader("Automatic synthetic sensor generator")
    a,b,c=st.columns(3)
    with a:
        count=st.slider("Generated readings",30,600,180,30)
        seed=int(st.number_input("Seed",0,999999,42))
    with b:
        variation=st.slider("Variation (% of sensor spread)",0,20,2)
        inject=st.checkbox("Inject hypothetical disturbance",value=True)
    with c:
        magnitude=st.slider("Disturbance strength (standard deviations)",1.,12.,8.,.5,disabled=not inject)
        fraction=st.slider("Disturbance duration (%)",5,60,25,5,disabled=not inject)
    affected=st.multiselect("Disturbed sensors",FEATURES,default=["LIT101","FIT101","LIT301"],disabled=not inject)
    if st.button("⚙️ Generate & analyse sensor stream",type="primary"):
        rng=np.random.default_rng(seed)
        start=int(rng.integers(0,len(normal)-count+1))
        values=normal.iloc[start:start+count].to_numpy().copy()
        spread=normal.std().to_numpy()
        values+=rng.normal(0,variation/100,size=values.shape)*spread
        injected=np.zeros(count,dtype=bool)
        if inject and affected:
            duration=max(1,round(count*fraction/100))
            begin=(count-duration)//2
            injected[begin:begin+duration]=True
            for sensor in affected:
                j=FEATURES.index(sensor)
                values[injected,j]+=float(rng.choice([-1,1]))*magnitude*spread[j]
        frame=pd.DataFrame(values,columns=FEATURES)
        scores,preds=score_rows(frame,model,scaler,threshold)
        st.session_state["generated"]=(frame,scores,preds,injected)
    generated=st.session_state.get("generated")
    if generated is not None:
        frame,scores,preds,injected=generated
        a,b,c=st.columns(3)
        a.metric("Generated rows",len(frame))
        b.metric("Flagged anomalies",int(preds.sum()))
        c.metric("Injected disturbance rows",int(injected.sum()))
        st.line_chart(pd.DataFrame({"Anomaly score":scores,"Decision threshold":threshold}),x_label="Generated step")
        chart_sensors=st.multiselect("Plot sensors",FEATURES,default=["LIT101","FIT101"])
        if chart_sensors: st.line_chart(frame[chart_sensors],x_label="Generated step")
        inspect=int(st.number_input("Inspect generated step",0,len(frame)-1,int(np.argmax(scores))))
        influence_chart(frame.iloc[[inspect]],model,scaler)
        output=frame.copy()
        output.insert(0,"Generated_step",np.arange(len(frame)))
        output["Anomaly_score"]=scores
        output["Predicted_anomaly"]=preds
        output["Injected_disturbance"]=injected.astype(int)
        st.download_button("⬇ Download synthetic predictions CSV",output.to_csv(index=False),"synthetic_predictions.csv","text/csv")
    st.divider()
    st.subheader("Upload sensor readings · CSV batch detection")
    st.caption("Upload a CSV with all 45 demonstration feature names. Uploaded data is processed by the hosted server; do not upload restricted SWaT data without authorisation.")
    st.download_button("⬇ Download synthetic CSV template",normal.head(10).to_csv(index=False),"synthetic_sensor_template.csv","text/csv")
    uploaded=st.file_uploader("Choose CSV",type="csv")
    if uploaded is not None:
        try:
            upload_df=pd.read_csv(uploaded)
            missing=[name for name in FEATURES if name not in upload_df.columns]
            if missing: st.error("Missing required columns: "+", ".join(missing))
            elif upload_df.empty: st.error("The CSV has no rows.")
            elif len(upload_df)>50000: st.error("Maximum 50,000 rows per upload.")
            else:
                numeric=upload_df[FEATURES].apply(pd.to_numeric,errors="coerce")
                if not np.isfinite(numeric.to_numpy(dtype=float)).all(): st.error("Sensor readings must all be finite numbers.")
                elif st.button("▶ Analyse uploaded CSV",type="primary"):
                    scores,preds=score_rows(numeric,model,scaler,threshold)
                    result=upload_df.copy()
                    result["Anomaly_score"]=scores
                    result["Predicted_anomaly"]=preds
                    st.success(f"Analysed {len(result):,} rows; flagged {int(preds.sum()):,} anomalies.")
                    st.dataframe(result.head(250),width="stretch")
                    st.download_button("⬇ Download predictions CSV",result.to_csv(index=False),"demo_batch_predictions.csv","text/csv")
        except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as exc:
            st.error(f"Could not read CSV: {exc}")
st.divider()
st.caption("Research metrics: original local SWaT evaluation · Interactive predictions: independent synthetic model · No restricted SWaT data hosted · Academic demonstration, not a production monitoring system.")
