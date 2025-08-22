from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from ml_service.utils import load_model, load_json

app = FastAPI(title="Smart Mesh ML API", version="0.1.0")

# =============================
# Load models at startup
# =============================
try:
    link_model = load_model("link_quality.pkl")
    link_features = load_json("link_quality_features.json")["order"]
    print("✅ Loaded link_quality model")
except Exception as e:
    print("❌ Failed to load link_quality model:", e)
    link_model = None
    link_features = []

try:
    demand_model = load_model("bandwidth_forecast.pkl")
    demand_features = load_json("bandwidth_features.json")["order"]
    print("✅ Loaded bandwidth_forecast model")
except Exception as e:
    print("❌ Failed to load bandwidth_forecast model:", e)
    demand_model = None
    demand_features = []

try:
    anomaly_model = load_model("anomaly_iforest.pkl")
    anomaly_features = load_json("anomaly_features.json")["order"]
    print("✅ Loaded anomaly model")
except Exception as e:
    print("❌ Failed to load anomaly model:", e)
    anomaly_model = None
    anomaly_features = []

try:
    sos_model = load_model("sos_text_clf.pkl")
    print("✅ Loaded SOS text classifier")
except Exception as e:
    print("❌ Failed to load SOS text classifier:", e)
    sos_model = None


# =============================
# Request Schemas
# =============================
class LinkSample(BaseModel):
    rssi: float
    ping_ms: float
    jitter_ms: float
    throughput_mbps: float
    battery_pct: float
    hops: int
    mobility: int  # 0 static, 1 moving
    congestion: float  # 0..1
    packet_loss_pct: float


class LinkRequest(BaseModel):
    samples: List[LinkSample]


class DemandRequest(BaseModel):
    hour_of_day: int
    day_of_week: int   # 0=Mon .. 6=Sun
    last_hour_mb: float
    avg_6h_mb: float


class AnomalyRequest(BaseModel):
    bytes_shared_mb: float
    session_minutes: float
    peers_connected: int
    avg_speed_mbps: float
    variance_speed: float
    claimed_tokens: float
    location_change_km: float


class SOSRequest(BaseModel):
    text: str


# =============================
# Endpoints
# =============================
@app.get("/health")
def health():
    global link_model, demand_model, anomaly_model, sos_model
    return {
        "status": "ok",
        "models": {
            "link_quality": link_model is not None,
            "bandwidth_forecast": demand_model is not None,
            "anomaly": anomaly_model is not None,
            "sos_nlp": sos_model is not None
        }
    }


@app.get("/models/info")
def models_info():
    global link_features, demand_features, anomaly_features
    return {
        "link_quality_features": link_features,
        "bandwidth_features": demand_features,
        "anomaly_features": anomaly_features
    }


@app.post("/predict/link_quality")
def predict_link_quality(req: LinkRequest):
    global link_model, link_features
    if link_model is None:
        return {"error": "link_quality model not loaded. Train and place models in ml_service/models."}

    X = np.array([[getattr(s, f) for f in link_features] for s in req.samples], dtype=float)
    preds = link_model.predict(X).tolist()

    probas = []
    if hasattr(link_model, "predict_proba"):
        probas = link_model.predict_proba(X)[:, 1].tolist()
    elif hasattr(link_model, "decision_function"):
        z = link_model.decision_function(X)
        probas = (1 / (1 + np.exp(-z))).tolist()
    else:
        probas = [None] * len(preds)

    return {"preds": preds, "probas": probas}


@app.post("/predict/usage")
def predict_usage(req: DemandRequest):
    global demand_model
    if demand_model is None:
        return {"error": "bandwidth_forecast model not loaded."}

    x = np.array([[req.hour_of_day, req.day_of_week, req.last_hour_mb, req.avg_6h_mb]], dtype=float)
    pred = float(demand_model.predict(x)[0])
    return {"pred_mb": pred}


@app.post("/predict/anomaly")
def predict_anomaly(req: AnomalyRequest):
    global anomaly_model
    if anomaly_model is None:
        return {"error": "anomaly model not loaded."}

    x = np.array([[req.bytes_shared_mb, req.session_minutes, req.peers_connected,
                   req.avg_speed_mbps, req.variance_speed, req.claimed_tokens, req.location_change_km]], dtype=float)

    label = int(anomaly_model.predict(x)[0])  # -1 anomaly, 1 normal
    score = float(anomaly_model.decision_function(x)[0])
    return {"is_anomaly": (label == -1), "score": score}


@app.post("/predict/sos")
def predict_sos(req: SOSRequest):
    global sos_model
    if sos_model is None:
        return {"error": "sos model not loaded."}

    if hasattr(sos_model, "predict_proba"):
        p = float(sos_model.predict_proba([req.text])[0][1])
        return {"emergency": p >= 0.5, "proba": p}
    else:
        pred = int(sos_model.predict([req.text])[0])
        return {"emergency": bool(pred), "proba": None}
