# Backend

[![Python application test with Pytest](https://github.com/4Banks/backend/actions/workflows/pytest.yml/badge.svg?branch=dev)](https://github.com/4Banks/backend/actions/workflows/pytest.yml)

Este repositório contém o código backend para uma aplicação de machine learning. A aplicação foi projetada para lidar com várias tarefas relacionadas à pré-processamento de dados, treinamento e avaliação de modelos.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```bash
backend/
├── app
│   ├── dataset_balancer.py
│   ├── dataset_manager.py
│   ├── image_manager.py
│   ├── json_manager.py
│   ├── main.py
│   ├── missing_data_treater.py
│   ├── outliers_detector.py
│   ├── outliers_treater.py
│   ├── superficial_analysis.py
├── tests
│   ├── test_dataset_balancer.py
│   ├── test_dataset_manager.py
│   ├── ...
├── requirements.txt
├── .gitignore
├── README.md
```

## Visão Geral da Aplicação

A aplicação backend foi desenvolvida utilizando a linguagem Python e o framework FastAPI. A aplicação é responsável por receber os dados de entrada, realizar o pré-processamento dos dados, treinar e avaliar os modelos de machine learning e retornar os resultados para o frontend.

A aplicação é dividida em vários módulos, cada um responsável por uma tarefa específica:

- `dataset_balancer.py`: Contém funções para balancear o conjunto de dados usando várias técnicas como subamostragem aleatória, superamostragem aleatória, SMOTE, Borderline SMOTE e ADASYN.
- `dataset_manager.py`: Lida com operações relacionadas ao carregamento e salvamento de conjuntos de dados do/para o Google Cloud Storage.
- `image_manager.py`: Gerencia operações relacionadas à criação e salvamento de - imagens de árvores de decisão.
- `json_manager.py`: Lida com operações relacionadas ao salvamento de dados JSON n o -Google Cloud Storage.
- `main.py`: Contém a função principal para treinamento e avaliação de modelos de - machine learning.
- `missing_data_treater`.py: Fornece uma função para tratar dados faltantes em um - DataFrame.
- `outliers_detector.py`: Contém uma função para detectar outliers no conjunto de dados usando vários métodos como Z-score, Robust Z-score, IQR e Winsorization.
- `outliers_treater.py`: Fornece uma função para tratar outliers em um DataFrame.
- `superficial_analysis.py`: Contém uma função para gerar estatísticas básicas sobre um DataFrame.

## Bibliotecas Chave

A aplicação usa várias bibliotecas Python para tarefas de processamento de dados e machine learning:

- `pandas`: Usado para manipulação e análise de dados.
- `numpy`: Usado para cálculos numéricos.
- `scikit-learn`: Usado para tarefas de machine learning como pré-processamento de dados, treinamento de modelos e avaliação.
- `imbalanced-learn`: Usado para lidar com conjuntos de dados desbalanceados.
- `google-cloud-storage`: Usado para interagir com o Google Cloud Storage.
- `xgboost`: Usado para treinar modelos XGBoost.
- `lightgbm`: Usado para treinar modelos LightGBM.
- `graphviz`: Usado para criar imagens de árvores de decisão.

## Como Executar a Aplicação

Para executar a aplicação, é necessário ter o Python 3.8 ou superior instalado. Além disso, é necessário instalar as dependências do projeto. Para isso, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

Após instalar as dependências, é possível executar a aplicação usando o seguinte comando:

```bash
uvicorn app.main:app --reload
```

## Como Executar os Testes

Para executar os testes, é necessário ter o Python 3.8 ou superior instalado. Além disso, é necessário instalar as dependências do projeto. Para isso, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

Após instalar as dependências, é possível executar os testes usando o seguinte comando:

```bash
pytest
```
