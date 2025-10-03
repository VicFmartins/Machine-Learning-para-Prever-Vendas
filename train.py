#!/usr/bin/env python3
"""
Gelato Mágico - Script de Treinamento
Treina modelo de regressão para predição de vendas de sorvete baseado em temperatura
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Azure ML
from azureml.core import Workspace, Run, Dataset, Model
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Treinar modelo de predição de vendas de sorvete')
    parser.add_argument('--data', type=str, default='data/icecream_sales.csv', 
                       help='Caminho para o dataset')
    parser.add_argument('--experiment-name', type=str, default='gelato-magico-experiment',
                       help='Nome do experimento MLflow')
    parser.add_argument('--model-name', type=str, default='gelato-magico-model',
                       help='Nome do modelo no registry')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporção dos dados para teste')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state para reprodutibilidade')
    parser.add_argument('--register-model', action='store_true',
                       help='Registrar modelo no MLflow Model Registry')
    
    return parser.parse_args()

def load_data(data_path):
    """Carrega e valida os dados"""
    logger.info(f"Carregando dados de: {data_path}")
    
    if not os.path.exists(data_path):
        # Gerar dados sintéticos para demonstração
        logger.info("Arquivo não encontrado. Gerando dados sintéticos...")
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        temperatures = np.random.normal(25, 8, len(dates))  # Temp média 25°C, std 8°C
        # Relação linear com ruído
        sales = 100 + 8 * temperatures + np.random.normal(0, 20, len(dates))
        sales = np.maximum(sales, 0)  # Vendas não podem ser negativas
        
        df = pd.DataFrame({
            'date': dates,
            'temperature_celsius': temperatures,
            'sales_units': sales
        })
        
        # Salvar dados sintéticos
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Dados sintéticos salvos em: {data_path}")
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
    return df

def preprocess_data(df):
    """Pré-processamento dos dados"""
    logger.info("Iniciando pré-processamento...")
    
    # Verificar valores ausentes
    if df.isnull().sum().any():
        logger.warning("Valores ausentes encontrados. Removendo...")
        df = df.dropna()
    
    # Features de data (se houver coluna date)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Features principais
    features = ['temperature_celsius']
    
    # Adicionar features derivadas se existirem colunas de data
    if 'day_of_week' in df.columns:
        features.extend(['day_of_week', 'month', 'is_weekend'])
    
    X = df[features]
    y = df['sales_units']
    
    logger.info(f"Features utilizadas: {features}")
    logger.info(f"Estatísticas da variável alvo - Min: {y.min():.2f}, Max: {y.max():.2f}, Média: {y.mean():.2f}")
    
    return X, y, features

def train_models(X_train, y_train, X_test, y_test):
    """Treina múltiplos modelos e seleciona o melhor"""
    
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Treinando modelo: {name}")
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse
        }
        
        logger.info(f"{name} - Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.3f}")
    
    # Selecionar melhor modelo (menor RMSE de teste)
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
    best_model_info = results[best_model_name]
    
    logger.info(f"Melhor modelo selecionado: {best_model_name}")
    
    return best_model_info, best_model_name, results

def log_mlflow_experiment(model_info, model_name, X_test, y_test, features, args):
    """Registra experimento no MLflow"""
    
    with mlflow.start_run(run_name=f"gelato-magico-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        
        # Log parâmetros
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("features", features)
        
        # Log métricas
        mlflow.log_metric("train_rmse", model_info['train_rmse'])
        mlflow.log_metric("test_rmse", model_info['test_rmse'])
        mlflow.log_metric("test_mae", model_info['test_mae'])
        mlflow.log_metric("test_r2", model_info['test_r2'])
        mlflow.log_metric("cv_rmse", model_info['cv_rmse'])
        
        # Log modelo
        mlflow.sklearn.log_model(
            sk_model=model_info['model'],
            artifact_path="model",
            registered_model_name=args.model_name if args.register_model else None
        )
        
        # Criar e salvar gráfico de predições vs real
        import matplotlib.pyplot as plt
        
        y_pred_test = model_info['model'].predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Vendas Reais')
        plt.ylabel('Vendas Preditas')
        plt.title(f'Predições vs Real - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Salvar gráfico
        plot_path = "predictions_vs_actual.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Log feature importance (se disponível)
        if hasattr(model_info['model'], 'feature_importances_'):
            importance_dict = dict(zip(features, model_info['model'].feature_importances_))
            for feature, importance in importance_dict.items():
                mlflow.log_metric(f"importance_{feature}", importance)
        
        logger.info("Experimento registrado no MLflow com sucesso!")
        
        return mlflow.active_run().info.run_id

def save_model_artifacts(model_info, model_name, features):
    """Salva artefatos do modelo para deployment"""
    
    # Criar diretório de saída
    output_dir = "model_artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar modelo
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model_info['model'], model_path)
    
    # Salvar metadata
    metadata = {
        'model_name': model_name,
        'features': features,
        'metrics': {
            'test_rmse': model_info['test_rmse'],
            'test_mae': model_info['test_mae'],
            'test_r2': model_info['test_r2']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Artefatos salvos em: {output_dir}")
    
    return model_path, metadata_path

def main():
    """Função principal"""
    
    args = parse_args()
    
    # Configurar MLflow
    mlflow.set_experiment(args.experiment_name)
    
    try:
        # Carregar dados
        df = load_data(args.data)
        
        # Pré-processamento
        X, y, features = preprocess_data(df)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        
        logger.info(f"Dados divididos - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
        
        # Treinar modelos
        best_model_info, best_model_name, all_results = train_models(X_train, y_train, X_test, y_test)
        
        # Log no MLflow
        run_id = log_mlflow_experiment(best_model_info, best_model_name, X_test, y_test, features, args)
        
        # Salvar artefatos
        model_path, metadata_path = save_model_artifacts(best_model_info, best_model_name, features)
        
        # Resultado final
        logger.info("=" * 60)
        logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 60)
        logger.info(f"Melhor modelo: {best_model_name}")
        logger.info(f"Test RMSE: {best_model_info['test_rmse']:.2f}")
        logger.info(f"Test MAE: {best_model_info['test_mae']:.2f}")
        logger.info(f"Test R²: {best_model_info['test_r2']:.3f}")
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"Modelo salvo em: {model_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise e

if __name__ == "__main__":
    main()