# Industrial Anomaly Explainer — Research Showcase + Synthetic Interactive Demo

A Streamlit showcase for a locally executed SWaT-based anomaly detection research prototype, plus a **separately trained, synthetic-only interactive demonstration**.

## Hosted app
https://industrial-anomaly-explainer-demo.streamlit.app/

## Features
- **Model Comparison:** Aggregate offline SWaT evaluation of Isolation Forest, One-Class SVM and MLP reconstruction autoencoder; comparative metrics, false alarms, methodology, limitations and CSV export. The separate experiment uses a different protocol from the original prototype results.
- **Research Results:** Reported metrics from the separate, locally executed SWaT study (precision 86.02%, recall 61.81%, F1 71.93%, average precision 0.7343). These are not recalculated by the hosted app.
- **Demo Overview:** Synthetic monitoring scores and injected hypothetical disturbances.
- **Observation / SHAP Influence / Temporal Context / Disturbance Context:** Explain individual synthetic observations.
- **Anomaly Explorer:** Browse and inspect model-flagged synthetic observations.
- **Sensor Simulator:** Edit **all 45** sensor values in a single table, load synthetic example baselines or start with zeros, run anomaly prediction, compare against the reference, inspect SHAP magnitudes, and download custom predictions. Also generate streams and analyse uploaded **non-restricted** CSV files.

## Local setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important research and data limitations
- **No original SWaT records, attack schedules, or SWaT-trained model artifacts are hosted or included in this repository.**
- The public model is trained exclusively on synthetic data. Its anomaly scores, thresholds, and explanations **do not** reproduce SWaT evaluation or validate real-world attack detection.
- Manual values are scored by the synthetic-trained model and may fall outside its training distribution; they are not a plant simulator or physical safety assessment.
- SHAP absolute magnitudes rank influence on TreeExplainer's internal model output, not signed or causal contributions to the displayed anomaly score.
- CSV uploads are processed by the hosted service. **Do not upload restricted SWaT data.**

## Technical details
The synthetic demo generates 5,500 normal training readings and 1,500 held-out monitoring readings across 45 illustrative features. A StandardScaler is fitted on synthetic normal data, followed by a 200-tree Isolation Forest. Anomalies are flagged above the 99th percentile of synthetic normal training scores.

## Three-model comparison files
`model_comparison.csv` and `experiment_metadata.json` contain only aggregate metrics and experiment settings from a separate local SWaT experiment. No SWaT observation rows, model weights, attack schedules or sensor traces are distributed. The online synthetic demo continues to use its own Isolation Forest.

The original prototype's Isolation Forest metrics and the three-model experiment's Isolation Forest metrics have different training/calibration protocols and should not be directly compared.
