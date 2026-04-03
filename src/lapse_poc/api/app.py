from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from lapse_poc.data.preprocessing import TabularPreprocessor
from lapse_poc.models.tabular import TabularNet
from lapse_poc.settings import NUM_COLS


# get version number from pyproject.toml
def _get_version() -> str:
    try:
        import tomli

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)
        return pyproject_data["tool"]["poetry"]["version"]
    except Exception:
        return "unknown"


app = FastAPI(title="Lapse Prediction PoC API", version=_get_version())


class PredictRequest(BaseModel):
    """Request model for lapse prediction."""
    tenure_qtr: int = Field(..., ge=1, description="Policy tenure in quarters")
    gender: str
    premium_frequency: str
    risk_state: str
    underwriting_age: str
    living_place: str
    acc_death_rider: str
    annual_premium: float


class PredictResponse(BaseModel):
    """Response model for lapse prediction."""
    lapse_prob: float


def _load_artifacts():
    env_path = os.getenv("LAPSE_POC_MODEL_DIR")
    if env_path is None:
        artifact_dir = Path(__file__).parent.parent / "artifacts"
    else:
        artifact_dir = Path(env_path)

    pre_dict = joblib.load(artifact_dir / "preprocessor.joblib")
    pre = TabularPreprocessor.from_dict(pre_dict)

    model = TabularNet(cat_cardinalities=pre.cat_cardinalities(), n_num=len(NUM_COLS))
    state = torch.load(artifact_dir / "model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()

    return pre, model


_PRE: Optional[TabularPreprocessor] = None  # noqa: UP045
_MODEL: Optional[TabularNet] = None  # noqa: UP045


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict lapse probability for a single policy snapshot.

    Args:
        req: PredictRequest object with policy features.
    Returns:
        PredictResponse object with lapse probability.
    """
    global _PRE, _MODEL
    if _PRE is None or _MODEL is None:
        _PRE, _MODEL = _load_artifacts()

    # align to feature contract
    row = {
        "tenure_qtr": req.tenure_qtr,
        "annual_premium": req.annual_premium,
        "gender": req.gender,
        "premium_frequency": req.premium_frequency,
        "risk_state": req.risk_state,
        "underwriting_age": req.underwriting_age,
        "living_place": req.living_place,
        "acc_death_rider": req.acc_death_rider,
    }

    df = pd.DataFrame([row])
    x_cat, x_num = _PRE.transform(df)

    with torch.no_grad():
        logits = _MODEL(torch.from_numpy(x_cat), torch.from_numpy(x_num))
        prob = float(torch.sigmoid(logits).item())

    return PredictResponse(lapse_prob=prob)
