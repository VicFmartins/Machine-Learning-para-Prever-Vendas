#!/usr/bin/env python3
"""
Gelato Mágico - Script de Avaliação
Avalia o desempenho do modelo em dados de teste
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# ML Libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Avaliar modelo de predição de vendas de sorvete')
    parser.add_argument('--model-name', type=str, default='gelato-magico-model',
                       help='Nome do modelo no MLflow Registry')
    parser.add_argument('--test-data', type=str, default='data/test_data.csv',
                       help='Caminho para dados de teste')
    parser.add_argument('--stage', type=str, default='Production',
                       choices=['Staging', 'Production', 'None'],
                       help='Estágio do modelo no registry')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Diretório para salvar resultados')
    
    return parser.parse_args()

def load_model_from_registry(model_name, stage):
    """Carrega modelo do MLflow Model Registry"""
    
    client = MlflowClient()
    
    try:
        # Buscar versão do modelo no estágio especificado
        model_version = client.get_latest_versions(model_name, stages=[stage])[0]
        
        logger.info(f"Carregando modelo {model_name} versão {model_version.version} (estágio: {stage})")
        
        # Carregar modelo
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Buscar metadata do run
        run = client.get_run(model_version.run_id)
        model_metadata = {
            'version': model_version.version,
            'stage': stage,
            'run_id': model_version.run_id,
            'metrics': run.data.metrics,
            'params': run.data.params
        }
        
        return model, model_metadata
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo do registry: {str(e)}")
        raise e

def load_test_data(test_data_path):
    """Carrega dados de teste"""
    
    if not os.path.exists(test_data_path):
        logger.warning(f"Arquivo de teste não encontrado: {test_data_path}")
        logger.info("Gerando dados de teste sintéticos...")
        
        # Gerar dados de teste sintéticos
        np.random.seed(123)  # Seed diferente do treino
        dates = pd.date_range('2024-06-01', '2024-06-30', freq='D')
        temperatures = np.random.normal(28, 6, len(dates))
        sales = 100 + 8 * temperatures + np.random.normal(0, 25, len(dates))
        sales = np.maximum(sales, 0)
        
        test_df = pd.DataFrame({
            'date': dates,
            'temperature_celsius': temperatures,
            'sales_units': sales
        })
        
        # Salvar dados de teste
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        test_df.to_csv(test_data_path, index=False)
        logger.info(f"Dados de teste sintéticos salvos em: {test_data_path}")
    else:
        test_df = pd.read_csv(test_data_path)
    
    return test_df

def prepare_features(df):
    """Prepara features para avaliação"""
    
    features_df = df.copy()
    
    # Processar data
    if 'date' in features_df.columns:
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    
    return features_df

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de avaliação"""
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2': r2_score(y_true, y_pred),
        'mean_residual': np.mean(y_pred - y_true),
        'std_residual': np.std(y_pred - y_true)
    }
    
    return metrics

def create_evaluation_plots(y_true, y_pred, output_dir):
    """Cria gráficos de avaliação"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predições vs Real
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Vendas Reais')
    plt.ylabel('Vendas Preditas')
    plt.title('Predições vs Real')
    plt.grid(True, alpha=0.3)
    
    # 2. Resíduos
    residuals = y_pred - y_true
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predições')
    plt.ylabel('Resíduos')
    plt.title('Gráfico de Resíduos')
    plt.grid(True, alpha=0.3)
    
    # 3. Distribuição dos resíduos
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Resíduos')
    plt.grid(True, alpha=0.3)
    
    # 4. QQ Plot dos resíduos
    from scipy import stats
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot dos Resíduos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos salvos em: {plot_path}")
    return plot_path

def create_time_series_plot(df, y_true, y_pred, output_dir):
    """Cria gráfico de séries temporais se houver coluna de data"""
    
    if 'date' not in df.columns:
        return None
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], y_true, label='Real', linewidth=2)
    plt.plot(df['date'], y_pred, label='Predito', linewidth=2, alpha=0.8)
    plt.xlabel('Data')
    plt.ylabel('Vendas (unidades)')
    plt.title('Vendas Reais vs Preditas ao Longo do Tempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plot_path = os.path.join(output_dir, 'time_series_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de série temporal salvo em: {plot_path}")
    return plot_path

def generate_evaluation_report(metrics, model_metadata, output_dir):
    """Gera relatório de avaliação"""
    
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_info': model_metadata,
        'metrics': metrics,
        'interpretation': {
            'performance_level': 'Excelente' if metrics['r2'] > 0.9 else 'Bom' if metrics['r2'] > 0.8 else 'Regular' if metrics['r2'] > 0.6 else 'Ruim',
            'rmse_interpretation': f"Erro médio de {metrics['rmse']:.1f} unidades",
            'mape_interpretation': f"Erro percentual médio de {metrics['mape']:.1f}%",
            'bias_check': 'Modelo sem viés significativo' if abs(metrics['mean_residual']) < metrics['std_residual']/2 else 'Possível viés no modelo'
        }
    }
    
    # Salvar relatório JSON
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Criar relatório em texto
    text_report = f"""
# RELATÓRIO DE AVALIAÇÃO - GELATO MÁGICO
Data da Avaliação: {report['evaluation_timestamp']}

## INFORMAÇÕES DO MODELO
- Nome: {model_metadata.get('model_name', 'N/A')}  
- Versão: {model_metadata.get('version', 'N/A')}
- Estágio: {model_metadata.get('stage', 'N/A')}
- Run ID: {model_metadata.get('run_id', 'N/A')}

## MÉTRICAS DE AVALIAÇÃO
- RMSE: {metrics['rmse']:.2f} unidades
- MAE: {metrics['mae']:.2f} unidades  
- MAPE: {metrics['mape']:.2f}%
- R²: {metrics['r2']:.4f}
- Resíduo Médio: {metrics['mean_residual']:.2f}
- Desvio dos Resíduos: {metrics['std_residual']:.2f}

## INTERPRETAÇÃO
- Nível de Performance: {report['interpretation']['performance_level']}
- {report['interpretation']['rmse_interpretation']}
- {report['interpretation']['mape_interpretation']}
- {report['interpretation']['bias_check']}

## RECOMENDAÇÕES
"""
    
    if metrics['r2'] < 0.8:
        text_report += "- Considerar re-treinamento com mais features\n"
    if metrics['mape'] > 15:
        text_report += "- Erro percentual alto, revisar dados de entrada\n"
    if abs(metrics['mean_residual']) > metrics['std_residual']/2:
        text_report += "- Investigar possível viés sistemático\n"
    
    text_report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(text_report_path, 'w') as f:
        f.write(text_report)
    
    logger.info(f"Relatório salvo em: {report_path}")
    return report_path, text_report_path

def main():
    """Função principal"""
    
    args = parse_args()
    
    try:
        # Carregar modelo
        model, model_metadata = load_model_from_registry(args.model_name, args.stage)
        
        # Carregar dados de teste
        test_df = load_test_data(args.test_data)
        
        # Preparar features
        features_df = prepare_features(test_df)
        
        # Determinar features do modelo (usar as mesmas do treinamento)
        available_features = ['temperature_celsius', 'day_of_week', 'month', 'is_weekend']
        model_features = [f for f in available_features if f in features_df.columns]
        
        X_test = features_df[model_features]
        y_test = test_df['sales_units']
        
        logger.info(f"Avaliando com {len(model_features)} features: {model_features}")
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Garantir valores não-negativos
        
        # Calcular métricas
        metrics = calculate_metrics(y_test, y_pred)
        
        # Criar diretório de saída
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Criar gráficos
        plots_path = create_evaluation_plots(y_test, y_pred, args.output_dir)
        time_series_path = create_time_series_plot(test_df, y_test, y_pred, args.output_dir)
        
        # Gerar relatório
        report_path, text_report_path = generate_evaluation_report(metrics, model_metadata, args.output_dir)
        
        # Log no MLflow (opcional)
        with mlflow.start_run(run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            mlflow.log_artifact(plots_path)
            if time_series_path:
                mlflow.log_artifact(time_series_path)
            mlflow.log_artifact(report_path)
        
        # Resultado final
        logger.info("=" * 60)
        logger.info("AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
        logger.info("=" * 60)
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"MAE: {metrics['mae']:.2f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Resultados salvos em: {args.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Erro durante avaliação: {str(e)}")
        raise e

if __name__ == "__main__":
    main()