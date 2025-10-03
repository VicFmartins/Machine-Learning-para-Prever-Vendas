# Machine-Learning-para-Prever-Vendas

Visão Geral
Este projeto desenvolve um modelo de regressão para prever vendas diárias de sorvete com base na temperatura ambiente, utilizando Azure Machine Learning e MLflow para um pipeline completo de MLOps.
Cenário
A sorveteria Gelato Mágico precisa otimizar sua produção diária baseada na temperatura prevista, reduzindo desperdícios e maximizando lucros através de predições precisas.
 Arquitetura
 azure-ml-icecream-sales-prediction
├── 📁 data/
│   └── icecream_sales.csv
├── 📁 src/
│   ├── train.py
│   ├── score.py
│   └── evaluate.py
├── 📁 notebooks/
│   └── eda_analysis.ipynb
├── 📁 configs/
│   ├── conda.yaml
│   └── endpoint_config.yaml
├── 📁 inputs/
│   └── business_insights.txt
├── requirements.txt
├── MLproject
└── README.md

 Objetivos
 Treinar modelo de regressão para prever vendas baseado em temperatura
 Implementar rastreamento completo com MLflow
 Deploy de endpoint para inferência em tempo real
 Pipeline reproduzível e escalável
 Métricas Esperadas
•	RMSE: < 50 unidades
•	MAE: < 35 unidades
•	R²: > 0.85
🛠️ Stack Tecnológica
•	Azure Machine Learning: Workspace, Compute, Endpoints
•	MLflow: Tracking, Model Registry, Artifacts
•	Scikit-learn: Algoritmos de regressão
•	Python: 3.8+
•	Pandas, NumPy: Manipulação de dados
 Quick Start
1. Configuração do Ambiente
# Clone o repositório
git clone https://github.com/seu-usuario/azure-ml-icecream-sales-prediction.git
cd azure-ml-icecream-sales-prediction

# Instale dependências
pip install -r requirements.txt

2. Treinamento do Modelo
# Execute o treinamento
python src/train.py --data data/icecream_sales.csv --experiment-name gelato-magico

3. Deploy do Endpoint
# Configure o endpoint
az ml online-endpoint create -f configs/endpoint_config.yaml

 Resultados
Experimento no Azure ML Studio

MLflow Model Registry

Endpoint em Produção
 
 Insights Obtidos
•	Correlação forte entre temperatura e vendas (R² = 0.92)
•	AutoML identificou modelo Gradient Boosting como melhor performer
•	Temperatura acima de 30°C gera picos de demanda de +40%
•	Sazonalidade semanal impacta significativamente as vendas
 Possibilidades de Evolução
•	Incluir features de umidade e precipitação
•	Implementar validação temporal (TimeSeriesSplit)
•	A/B testing entre versões do modelo
•	Dashboard BI para planejamento de produção
•	Monitoramento de drift automático
 Estrutura dos Dados
date,temperature_celsius,sales_units
2024-01-01,25.5,180
2024-01-02,28.2,220
2024-01-03,32.1,310

