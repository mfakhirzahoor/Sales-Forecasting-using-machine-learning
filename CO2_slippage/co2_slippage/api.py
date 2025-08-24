from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import FEATURE_COLUMNS
from .model import load_trained_model
from .optimize import suggest_optimal_setpoints

app = FastAPI(title="CO2 Slippage API", version="0.1.0")


class Features(BaseModel):
    pressure_bar: float = Field(..., ge=0)
    temperature_c: float = Field(...)
    gas_flow_rate_nm3h: float = Field(..., ge=0)
    solvent_concentration_pct: float = Field(..., ge=0)
    contact_time_s: float = Field(..., ge=0)
    packing_height_m: float = Field(..., ge=0)


@app.get("/")
async def root():
    return {"status": "ok", "message": "CO2 slippage prediction service"}


@app.post("/predict")
async def predict(features: Features):
    try:
        model = load_trained_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    x = [[getattr(features, col) for col in FEATURE_COLUMNS]]
    yhat = float(model.predict(x)[0])
    return {"prediction_co2_slippage_pct": yhat}


@app.post("/optimize")
async def optimize():
    try:
        model = load_trained_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    setpoints = suggest_optimal_setpoints(model)
    return {"suggested_setpoints": setpoints}