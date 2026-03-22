#!/usr/bin/env python3
"""
Gelato Magico - Avaliacao local do modelo treinado.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from train import generate_synthetic_data, preprocess_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliar modelo de previsao de vendas.")
    parser.add_argument("--model-dir", type=str, default="model_artifacts", help="Diretorio com model.pkl e metadata.json.")
    parser.add_argument("--test-data", type=str, default="data/test_data.csv", help="Dataset de teste.")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Diretorio de saida.")
    return parser.parse_args()


def load_model(model_dir: Path) -> Tuple[object, Dict]:
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_path}")

    model = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, metadata


def load_test_data(test_data_path: Path) -> pd.DataFrame:
    if not test_data_path.exists():
        logger.warning("Base de teste ausente em %s. Gerando base sintetica.", test_data_path)
        return generate_synthetic_data(test_data_path, random_state=123)

    return pd.read_csv(test_data_path)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
    }


def save_plots(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_pred - y_true

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=2)
    plt.xlabel("Vendas reais")
    plt.ylabel("Vendas preditas")
    plt.title("Predicao vs real")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Residuo")
    plt.ylabel("Frequencia")
    plt.title("Distribuicao dos residuos")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_report(metrics: Dict[str, float], metadata: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "model_metadata": metadata,
        "interpretation": {
            "performance_level": (
                "Excelente" if metrics["r2"] > 0.9 else "Bom" if metrics["r2"] > 0.8 else "Regular"
            ),
            "bias_warning": abs(metrics["mean_residual"]) > max(metrics["std_residual"] * 0.5, 1.0),
        },
    }

    with (output_dir / "evaluation_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with (output_dir / "evaluation_report.txt").open("w", encoding="utf-8") as file:
        file.write("# Relatorio de avaliacao\n")
        file.write(f"Data: {report['evaluation_timestamp']}\n\n")
        for metric_name, metric_value in metrics.items():
            file.write(f"- {metric_name}: {metric_value:.4f}\n")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    model, metadata = load_model(model_dir)
    test_df = load_test_data(Path(args.test_data))
    x_test, y_test, features = preprocess_data(test_df)

    expected_features = metadata.get("features", features)
    x_test = x_test[expected_features]

    predictions = np.maximum(model.predict(x_test), 0)
    metrics = calculate_metrics(y_test, predictions)

    save_plots(y_test, predictions, output_dir)
    save_report(metrics, metadata, output_dir)

    logger.info("=" * 60)
    logger.info("AVALIACAO CONCLUIDA")
    logger.info("RMSE: %.2f", metrics["rmse"])
    logger.info("MAE: %.2f", metrics["mae"])
    logger.info("R2: %.3f", metrics["r2"])
    logger.info("Resultados salvos em: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
