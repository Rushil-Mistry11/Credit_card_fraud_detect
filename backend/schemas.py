from datetime import datetime
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ml.config import FEATURES


class PredictRequest(BaseModel):
    algorithm: Literal["svc", "decision_tree", "naive_bayes", "mlp"] = "mlp"
    features: Dict[str, float]

    @field_validator("features")
    @classmethod
    def check_features(cls, v):
        missing = [f for f in FEATURES if f not in v]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return v


class PredictResponse(BaseModel):
    id: int
    algorithm: str
    fraud_probability: float
    risk_score: float          # 0-100
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    is_flagged: bool


class LogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    algorithm: str
    amount: float
    fraud_probability: float
    risk_level: str
    is_flagged: bool
