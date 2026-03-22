#!/usr/bin/env python3
"""
Gelato Magico - Treinamento local e opcionalmente com MLflow.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path("model_artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar modelo de previsao de vendas de sorvete.")
    parser.add_argument("--data", type=str, default="data/icecream_sales.csv", help="Caminho para o dataset.")
    parser.add_argument("--experiment-name", type=str, default="gelato-magico-experiment", help="Nome do experimento MLflow.")
    parser.add_argument("--model-name", type=str, default="gelato-magico-model", help="Nome logico do modelo.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcao dos dados reservada para teste.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Diretorio de saida dos artefatos.")
    parser.add_argument("--enable-mlflow", action="store_true", help="Ativa rastreamento em MLflow.")
    parser.add_argument("--register-model", action="store_true", help="Registra o modelo no registry do MLflow.")
    return parser.parse_args()


def generate_synthetic_data(data_path: Path, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    temperatures = np.clip(rng.normal(26, 7, len(dates)), 8, 42)
    weekends = (dates.dayofweek >= 5).astype(int)
    months = dates.month.to_numpy()
    season_boost = np.where(np.isin(months, [11, 12, 1, 2]), 18, 0)
    weekend_boost = weekends * 22
    noise = rng.normal(0, 18, len(dates))
    sales = 85 + (7.5 * temperatures) + weekend_boost + season_boost + noise
    sales = np.maximum(sales, 0)

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature_celsius": np.round(temperatures, 2),
            "sales_units": np.round(sales, 2),
        }
    )

    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info("Dataset sintetico gerado em %s", data_path)
    return df


def load_data(data_path: str, random_state: int) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        logger.warning("Dataset nao encontrado em %s. Gerando base sintetica.", path)
        return generate_synthetic_data(path, random_state)

    df = pd.read_csv(path)
    logger.info("Dataset carregado com %s linhas e %s colunas", df.shape[0], df.shape[1])
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    processed = df.copy()

    if processed.isnull().sum().any():
        logger.warning("Valores ausentes detectados. Removendo linhas incompletas.")
        processed = processed.dropna()

    if "date" in processed.columns:
        processed["date"] = pd.to_datetime(processed["date"])
        processed["day_of_week"] = processed["date"].dt.dayofweek
        processed["month"] = processed["date"].dt.month
        processed["is_weekend"] = (processed["day_of_week"] >= 5).astype(int)

    processed["temp_squared"] = processed["temperature_celsius"] ** 2
    processed["temp_high"] = (processed["temperature_celsius"] >= 30).astype(int)
    processed["temp_low"] = (processed["temperature_celsius"] <= 15).astype(int)

    features = [
        "temperature_celsius",
        "day_of_week",
        "month",
        "is_weekend",
        "temp_squared",
        "temp_high",
        "temp_low",
    ]
    features = [feature for feature in features if feature in processed.columns]

    x_data = processed[features]
    y_data = processed["sales_units"]

    logger.info("Features utilizadas: %s", features)
    return x_data, y_data, features


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> Tuple[Dict, str, Dict[str, Dict]]:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }

    results: Dict[str, Dict] = {}

    for model_name, model in models.items():
        logger.info("Treinando %s", model_name)
        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        cv_scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error",
        )

        results[model_name] = {
            "model": model,
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "test_r2": float(r2_score(y_test, y_pred_test)),
            "cv_rmse": float(np.sqrt(-cv_scores.mean())),
        }

    best_model_name = min(results, key=lambda item: results[item]["test_rmse"])
    return results[best_model_name], best_model_name, results


def maybe_log_mlflow(
    enabled: bool,
    model_info: Dict,
    best_model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    args: argparse.Namespace,
) -> str | None:
    if not enabled:
        logger.info("MLflow desativado. Pulando tracking remoto/local.")
        return None

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise ImportError(
            "MLflow nao esta instalado. Instale manualmente se quiser usar --enable-mlflow."
        ) from exc

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"gelato-magico-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.log_param("model_type", best_model_name)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        for metric_name in ("train_rmse", "test_rmse", "test_mae", "test_r2", "cv_rmse"):
            mlflow.log_metric(metric_name, model_info[metric_name])

        mlflow.sklearn.log_model(
            sk_model=model_info["model"],
            artifact_path="model",
            registered_model_name=args.model_name if args.register_model else None,
        )

        predictions = model_info["model"].predict(x_test)
        comparison = pd.DataFrame({"actual": y_test, "predicted": predictions})
        comparison_path = Path(args.artifact_dir) / "mlflow_predictions_sample.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.head(50).to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path))

        return mlflow.active_run().info.run_id


def save_artifacts(
    model_info: Dict,
    model_name: str,
    features: List[str],
    artifact_dir: Path,
    all_results: Dict[str, Dict],
) -> Tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.pkl"
    metadata_path = artifact_dir / "metadata.json"

    joblib.dump(model_info["model"], model_path)

    metadata = {
        "model_name": model_name,
        "features": features,
        "metrics": {
            "train_rmse": model_info["train_rmse"],
            "test_rmse": model_info["test_rmse"],
            "test_mae": model_info["test_mae"],
            "test_r2": model_info["test_r2"],
            "cv_rmse": model_info["cv_rmse"],
        },
        "candidates": {
            name: {
                "test_rmse": values["test_rmse"],
                "test_mae": values["test_mae"],
                "test_r2": values["test_r2"],
            }
            for name, values in all_results.items()
        },
        "timestamp": datetime.now().isoformat(),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_path, metadata_path


def main() -> None:
    args = parse_args()

    df = load_data(args.data, args.random_state)
    x_data, y_data, features = preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    best_model_info, best_model_name, all_results = train_models(
        x_train,
        y_train,
        x_test,
        y_test,
        args.random_state,
    )

    run_id = maybe_log_mlflow(
        enabled=args.enable_mlflow,
        model_info=best_model_info,
        best_model_name=best_model_name,
        x_test=x_test,
        y_test=y_test,
        features=features,
        args=args,
    )

    model_path, metadata_path = save_artifacts(
        best_model_info,
        best_model_name,
        features,
        Path(args.artifact_dir),
        all_results,
    )

    logger.info("=" * 60)
    logger.info("TREINAMENTO CONCLUIDO")
    logger.info("Melhor modelo: %s", best_model_name)
    logger.info("RMSE teste: %.2f", best_model_info["test_rmse"])
    logger.info("MAE teste: %.2f", best_model_info["test_mae"])
    logger.info("R2 teste: %.3f", best_model_info["test_r2"])
    logger.info("Modelo salvo em: %s", model_path)
    logger.info("Metadata salva em: %s", metadata_path)
    if run_id:
        logger.info("MLflow Run ID: %s", run_id)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
