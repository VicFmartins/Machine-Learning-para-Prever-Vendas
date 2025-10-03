#!/usr/bin/env python3
"""
Gelato Mágico - Script de Scoring
Script para inferência do modelo em produção (Azure ML Endpoint)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """
    Função chamada quando o endpoint é inicializado
    Carrega o modelo e metadata necessários
    """
    global model, features, model_metadata
    
    try:
        # Caminho do modelo no Azure ML
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', './model_artifacts'), 'model.pkl')
        metadata_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', './model_artifacts'), 'metadata.json')
        
        # Carregar modelo
        logger.info(f"Carregando modelo de: {model_path}")
        model = joblib.load(model_path)
        
        # Carregar metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
                features = model_metadata.get('features', ['temperature_celsius'])
        else:
            # Default features se metadata não existir
            features = ['temperature_celsius', 'day_of_week', 'month', 'is_weekend']
            model_metadata = {
                'model_name': 'gelato-magico-model',
                'features': features,
                'status': 'loaded_without_metadata'
            }
        
        logger.info(f"Modelo carregado com sucesso. Features: {features}")
        logger.info(f"Metadata: {model_metadata}")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar modelo: {str(e)}")
        raise e

def run(raw_data):
    """
    Função principal de inferência
    
    Args:
        raw_data: JSON string com os dados de entrada
        
    Returns:
        JSON com as predições e metadata
    """
    try:
        # Parse dos dados de entrada
        logger.info(f"Dados recebidos: {raw_data}")
        data = json.loads(raw_data)
        
        # Converter para DataFrame
        if isinstance(data, dict):
            if 'data' in data:
                # Formato: {"data": [{"temperature_celsius": 30, "date": "2024-01-01"}, ...]}
                input_data = pd.DataFrame(data['data'])
            else:
                # Formato: {"temperature_celsius": 30, "date": "2024-01-01"}
                input_data = pd.DataFrame([data])
        elif isinstance(data, list):
            # Formato: [{"temperature_celsius": 30}, ...]
            input_data = pd.DataFrame(data)
        else:
            raise ValueError("Formato de dados inválido. Use dict ou list.")
        
        # Preparar features
        input_features = prepare_features(input_data)
        
        # Validar features
        if not all(col in input_features.columns for col in features):
            missing_features = [f for f in features if f not in input_features.columns]
            raise ValueError(f"Features obrigatórias ausentes: {missing_features}")
        
        # Selecionar apenas as features do modelo
        X = input_features[features]
        
        logger.info(f"Features preparadas: {X.columns.tolist()}")
        logger.info(f"Shape dos dados: {X.shape}")
        
        # Fazer predições
        predictions = model.predict(X)
        
        # Garantir que predições sejam não-negativas (vendas não podem ser negativas)
        predictions = np.maximum(predictions, 0)
        
        # Preparar resposta
        response = {
            'predictions': predictions.tolist(),
            'model_info': {
                'model_name': model_metadata.get('model_name', 'gelato-magico-model'),
                'features_used': features,
                'prediction_timestamp': datetime.now().isoformat(),
                'num_predictions': len(predictions)
            }
        }
        
        # Adicionar informações detalhadas se solicitado
        if 'include_details' in data and data['include_details']:
            response['input_data'] = input_features.to_dict(orient='records')
            response['model_metadata'] = model_metadata
        
        logger.info(f"Predições geradas: {len(predictions)} valores")
        logger.info(f"Predições: {predictions[:5]}...")  # Log apenas os primeiros 5 valores
        
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"Erro durante inferência: {str(e)}")
        return json.dumps(error_response)

def prepare_features(df):
    """
    Prepara as features a partir dos dados de entrada
    
    Args:
        df: DataFrame com dados de entrada
        
    Returns:
        DataFrame com features preparadas
    """
    
    # Criar cópia dos dados
    features_df = df.copy()
    
    # Processar coluna de data se existir
    if 'date' in features_df.columns:
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    else:
        # Se não há data, usar data atual
        current_date = datetime.now()
        features_df['day_of_week'] = current_date.weekday()
        features_df['month'] = current_date.month
        features_df['is_weekend'] = int(current_date.weekday() >= 5)
    
    # Validar temperatura
    if 'temperature_celsius' in features_df.columns:
        # Aplicar limites razoáveis de temperatura
        features_df['temperature_celsius'] = features_df['temperature_celsius'].clip(-20, 50)
    else:
        raise ValueError("Coluna 'temperature_celsius' é obrigatória")
    
    # Features derivadas opcionais
    if 'temperature_celsius' in features_df.columns:
        features_df['temp_squared'] = features_df['temperature_celsius'] ** 2
        features_df['temp_high'] = (features_df['temperature_celsius'] > 30).astype(int)
        features_df['temp_low'] = (features_df['temperature_celsius'] < 15).astype(int)
    
    logger.info(f"Features preparadas: {features_df.columns.tolist()}")
    
    return features_df

def validate_input(data):
    """
    Valida os dados de entrada
    
    Args:
        data: dados de entrada
        
    Returns:
        bool: True se válido, False caso contrário
    """
    
    required_fields = ['temperature_celsius']
    
    if isinstance(data, dict):
        if 'data' in data:
            # Validar cada item na lista de dados
            for item in data['data']:
                if not all(field in item for field in required_fields):
                    return False
        else:
            # Validar item único
            if not all(field in data for field in required_fields):
                return False
    elif isinstance(data, list):
        # Validar cada item na lista
        for item in data:
            if not all(field in item for field in required_fields):
                return False
    else:
        return False
    
    return True

# Exemplo de teste local
if __name__ == "__main__":
    # Simular inicialização
    init()
    
    # Dados de teste
    test_data = {
        "data": [
            {"temperature_celsius": 25.5, "date": "2024-01-01"},
            {"temperature_celsius": 30.2, "date": "2024-01-02"},
            {"temperature_celsius": 35.1, "date": "2024-01-03"}
        ],
        "include_details": True
    }
    
    # Testar predição
    result = run(json.dumps(test_data))
    print("Resultado da predição:")
    print(json.dumps(json.loads(result), indent=2))