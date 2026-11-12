"""FastAPI inference + audit service.

Run from project root:   uvicorn backend.main:app --reload
Docs:                    http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.models import TransactionLog
from backend.schemas import LogOut, PredictRequest, PredictResponse
from ml.config import (DECISION_THRESHOLD, FEATURES, HIGH_THRESHOLD, LOW_THRESHOLD,
                       MODEL_NAMES, MODELS_DIR)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise RuntimeError("models/scaler.pkl not found - run `python -m ml.train` first.")
    STATE["scaler"] = joblib.load(scaler_path)
    STATE["models"] = {n: joblib.load(MODELS_DIR / f"{n}.pkl")
                       for n in MODEL_NAMES if (MODELS_DIR / f"{n}.pkl").exists()}
    yield
    STATE.clear()


app = FastAPI(title="Credit Card Fraud Risk Scoring API", version="1.0", lifespan=lifespan)


def risk_level(p: float) -> str:
    if p < LOW_THRESHOLD:
        return "LOW"
    if p < HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(STATE.get("models", {}))}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    model = STATE["models"].get(req.algorithm)
    if model is None:
        raise HTTPException(404, f"Model '{req.algorithm}' is not trained/loaded.")

    row = pd.DataFrame([[req.features[f] for f in FEATURES]], columns=FEATURES)
    scaled = pd.DataFrame(STATE["scaler"].transform(row), columns=FEATURES)
    prob = float(model.predict_proba(scaled)[0, 1])

    log = TransactionLog(
        algorithm=req.algorithm,
        amount=req.features["Amount"],
        fraud_probability=prob,
        risk_level=risk_level(prob),
        is_flagged=prob >= DECISION_THRESHOLD,
        features_json=json.dumps(req.features),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return PredictResponse(
        id=log.id, algorithm=log.algorithm, fraud_probability=round(prob, 6),
        risk_score=round(prob * 100, 2), risk_level=log.risk_level,
        is_flagged=log.is_flagged,
    )


@app.get("/logs", response_model=List[LogOut])
def get_logs(limit: int = Query(100, ge=1, le=1000),
             risk: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH)$"),
             db: Session = Depends(get_db)):
    q = db.query(TransactionLog)
    if risk:
        q = q.filter(TransactionLog.risk_level == risk)
    return q.order_by(TransactionLog.id.desc()).limit(limit).all()


@app.get("/logs/summary")
def logs_summary(db: Session = Depends(get_db)):
    total = db.query(func.count(TransactionLog.id)).scalar() or 0
    flagged = db.query(func.count(TransactionLog.id)).filter(TransactionLog.is_flagged).scalar() or 0
    by_risk = dict(db.query(TransactionLog.risk_level, func.count(TransactionLog.id))
                   .group_by(TransactionLog.risk_level).all())
    return {"total": total, "flagged": flagged, "by_risk": by_risk}


@app.get("/metrics")
def metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found - run `python -m ml.train` first.")
    return json.loads(path.read_text())
