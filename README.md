# Machine Learning para Prever Vendas

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![Status](https://img.shields.io/badge/status-MVP%20executavel-2e8b57)

Projeto de regressão para prever vendas diárias de sorvete com base na temperatura e em variáveis derivadas de calendário, com fluxo local de treinamento, avaliação, scoring e integração opcional com MLflow.

## Visão Geral

O cenário usado é o da sorveteria fictícia **Gelato Mágico**, que precisa prever demanda para:

- reduzir desperdício de produção;
- planejar compras de insumos;
- ajustar equipe em dias de pico;
- transformar previsão climática em decisão operacional.

O projeto foi melhorado para ficar realmente executável localmente, sem depender obrigatoriamente de Azure Machine Learning para funcionar.

## O Que o Projeto Faz

- carrega uma base de vendas e temperatura;
- gera features simples de calendário;
- treina múltiplos modelos de regressão;
- escolhe automaticamente o melhor pelo menor RMSE;
- salva artefatos do modelo;
- avalia o desempenho em base de teste;
- executa inferência local via JSON;
- suporta tracking com MLflow quando habilitado.

## Estrutura Atual

```text
.
├── README.md
├── train.py
├── evaluate.py
├── score.py
├── MLproject
├── conda.yaml
├── endpoint-config.yaml
├── requirements.txt
└── data/
    ├── icecream_sales.csv
    └── sample_request.json
```

## Principais Arquivos

### `train.py`

Responsável por:

- carregar dados reais ou gerar base sintética;
- criar features;
- treinar `LinearRegression`, `RandomForestRegressor` e `GradientBoostingRegressor`;
- comparar métricas;
- salvar `model.pkl` e `metadata.json`;
- logar no MLflow opcionalmente.

### `evaluate.py`

Responsável por:

- carregar o modelo salvo localmente;
- avaliar em base de teste;
- gerar métricas de regressão;
- salvar relatório e gráficos.

### `score.py`

Responsável por:

- carregar o modelo treinado;
- receber payload JSON;
- preparar as mesmas features do treinamento;
- devolver previsões em formato compatível com uso local ou endpoint.

## Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- MLflow (opcional)

## Como Executar

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd Machine-Learning-para-Prever-Vendas
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```bash
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Treine o modelo

```bash
python train.py --data data/icecream_sales.csv
```

Se você quiser ativar tracking com MLflow:

```bash
pip install mlflow
python train.py --data data/icecream_sales.csv --enable-mlflow
```

### 5. Avalie o modelo

```bash
python evaluate.py --model-dir model_artifacts --test-data data/icecream_sales.csv
```

### 6. Execute previsões locais

```bash
python score.py --input data/sample_request.json
```

## Exemplo de Payload

```json
{
  "data": [
    {
      "temperature_celsius": 24.5,
      "date": "2024-07-01"
    },
    {
      "temperature_celsius": 30.0,
      "date": "2024-07-02"
    }
  ]
}
```

## Exemplo de Saída

```json
{
  "predictions": [189.42, 244.17],
  "model_info": {
    "model_name": "gradient_boosting",
    "features_used": ["temperature_celsius", "day_of_week", "month", "is_weekend", "temp_squared", "temp_high", "temp_low"],
    "prediction_timestamp": "2026-03-22T00:00:00",
    "num_predictions": 2
  }
}
```

## Melhorias Aplicadas Nesta Versão

- remoção da dependência obrigatória de Azure para o fluxo principal;
- correção de inconsistências entre treino, avaliação e scoring;
- inclusão de base inicial e payload de exemplo;
- atualização de dependências para versões mais realistas;
- correção do `MLproject` e do `endpoint-config.yaml`;
- documentação alinhada com o que o repositório realmente entrega;
- suporte local mais forte para portfólio e demonstração.

## Próximos Passos

- adicionar validação temporal;
- incluir mais features climáticas;
- salvar gráficos comparativos do treino;
- adicionar testes automatizados;
- criar dashboard simples para previsões;
- conectar o pipeline a previsão do tempo real.

## Observação

O projeto está pronto como MVP local de machine learning aplicado a negócio. Azure ML continua como possibilidade de evolução, mas o repositório agora já entrega valor e demonstração prática sem depender de infraestrutura externa para começar.
