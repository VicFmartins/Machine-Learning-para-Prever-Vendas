#!/usr/bin/env python3
"""
Gelato Magico - Scoring local e compativel com Azure ML endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

model = None
features: List[str] = []
model_metadata: Dict[str, Any] = {}


def init() -> None:
    global model, features, model_metadata

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "model_artifacts"))
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"

    model = joblib.load(model_path)

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            model_metadata = json.load(file)
            features = model_metadata.get("features", ["temperature_celsius"])
    else:
        model_metadata = {"model_name": "gelato-magico-model"}
        features = ["temperature_celsius"]

    logger.info("Modelo carregado com features: %s", features)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "temperature_celsius" not in prepared.columns:
        raise ValueError("A coluna 'temperature_celsius' e obrigatoria.")

    prepared["temperature_celsius"] = prepared["temperature_celsius"].clip(-20, 50)

    if "date" in prepared.columns:
        prepared["date"] = pd.to_datetime(prepared["date"])
        prepared["day_of_week"] = prepared["date"].dt.dayofweek
        prepared["month"] = prepared["date"].dt.month
        prepared["is_weekend"] = (prepared["day_of_week"] >= 5).astype(int)
    else:
        now = datetime.now()
        prepared["day_of_week"] = now.weekday()
        prepared["month"] = now.month
        prepared["is_weekend"] = int(now.weekday() >= 5)

    prepared["temp_squared"] = prepared["temperature_celsius"] ** 2
    prepared["temp_high"] = (prepared["temperature_celsius"] >= 30).astype(int)
    prepared["temp_low"] = (prepared["temperature_celsius"] <= 15).astype(int)
    return prepared


def parse_input_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    raise ValueError("Formato de payload invalido. Use objeto JSON ou lista de objetos.")


def run(raw_data: Any) -> str:
    try:
        input_df = parse_input_payload(raw_data)
        prepared_df = prepare_features(input_df)
        x_data = prepared_df[features]
        predictions = np.maximum(model.predict(x_data), 0)

        response = {
            "predictions": [round(float(value), 2) for value in predictions],
            "model_info": {
                "model_name": model_metadata.get("model_name", "gelato-magico-model"),
                "features_used": features,
                "prediction_timestamp": datetime.now().isoformat(),
                "num_predictions": int(len(predictions)),
            },
        }
        return json.dumps(response, ensure_ascii=False)
    except Exception as exc:
        logger.exception("Falha durante o scoring")
        return json.dumps(
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "timestamp": datetime.now().isoformat(),
            },
            ensure_ascii=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executar previsao local do modelo.")
    parser.add_argument("--input", type=str, default="", help="JSON inline ou caminho para arquivo JSON.")
    args = parser.parse_args()

    init()

    payload = args.input
    if payload and Path(payload).exists():
        payload = Path(payload).read_text(encoding="utf-8")
    elif not payload:
        payload = json.dumps(
            {
                "data": [
                    {"temperature_celsius": 25.5, "date": "2024-01-01"},
                    {"temperature_celsius": 31.2, "date": "2024-01-02"},
                    {"temperature_celsius": 35.0, "date": "2024-01-03"},
                ]
            }
        )

    print(run(payload))


if __name__ == "__main__":
    main()
