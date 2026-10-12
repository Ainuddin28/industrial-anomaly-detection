# Industrial Anomaly Explainer — Synthetic Online Demo

This Streamlit application demonstrates Isolation Forest anomaly detection, SHAP influence ranking, sensor editing, synthetic stream generation, and CSV batch prediction.

**IMPORTANT:** This version uses **entirely synthetic data and a separately trained synthetic model**. It is not the SWaT-trained research prototype, does not contain SWaT observations, and does not reproduce or validate the project's reported SWaT metrics. The feature names are illustrative. It is not live industrial monitoring or a physically validated simulator.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository. On Streamlit Community Cloud, create an app using the repository's `main` branch and `app.py` as the entrypoint. Verify the hosted app and its sharing settings before giving access to assessors.

## Data protection

No original SWaT spreadsheets, attack schedule, or SWaT-derived trained model are included. Do not upload restricted SWaT sensor CSVs into the public demo without permission: uploaded readings are processed by the remote host.

## Demonstration

Eight tabs: Overview, Observation, SHAP influence, Temporal context, Disturbance context, Methodology, Anomaly Explorer, Sensor Simulator. The simulator includes manual what-if readings, synthetic monitoring generation, and CSV batch analysis. A synthetic CSV template is downloadable from the app.
