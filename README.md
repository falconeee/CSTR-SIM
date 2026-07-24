# Detecção de Falhas (CSTR & TE) com Modelos Não-Supervisionados

Este repositório aborda o problema de detecção de falhas e anomalias industriais com foco em duas bases de dados amplamente conhecidas e utilizadas: **CSTR** (Continuous Stirred Tank Reactor) e **TE** (Tennessee Eastman).

O objetivo principal deste projeto é treinar modelos de inteligência artificial de forma **não-supervisionada** nos dados de operação normal do processo e, em seguida, prever as anomalias nas bases de teste (dados com falha). O modelo analisa o erro de reconstrução e gera um índice para identificar desvios da normalidade, com base em um limiar (Threshold) calibrado. Vários modelos estão disponíveis para uso (MSCVAE, PCA, LSTM_AE, entre outros). No caso específico do modelo PCA, foi implementada uma análise aprimorada baseada no estado da arte com Q-statistic e contribuição por RBC (Reconstruction-Based Contribution).

## 🗂 Estrutura de Diretórios
- `data/` -> Contém as bases de dados otimizadas em formato de alta performance `.parquet` (`CSTR_data/` e `TE_data/`). Arquivos legados estão organizados nas subpastas `csv_files/` e `mat_files/`.
- `src/` -> Códigos fonte e implementações dos modelos preditivos (MSCVAE, PCA, etc).
- `output/` -> Diretório gerado automaticamente que armazena as predições (gráficos de índice Phi, reconstrução das variáveis) e as métricas calculadas em formato `.json`.
- `notebooks/` -> Pasta contendo os notebooks interativos de experimentação (`CSTR-Detection.ipynb`, `TE-Detection.ipynb` e `CSTR-SIM.ipynb`).
- `dashboard/` -> Contém o painel interativo (Digital Twin) construído com Streamlit para simulação visual do CSTR em tempo real.
- `run_detection.py` -> Script principal em Python que realiza o treinamento, detecção massiva das falhas, plotagem e extração automática de métricas (F1-score, Recall, etc).

## 🚀 Como Executar

### 1. Configuração do Ambiente
Recomenda-se o uso de um ambiente virtual (que já está configurado na pasta `venv`). Ative o ambiente ou certifique-se de usar o interpretador diretamente e instalar os pacotes essenciais:

```bash
# Se o venv não estiver ativo
source venv/bin/activate 

# Instale os requerimentos do projeto caso ainda não possua (Ex: pandas, plotly, scikit-learn, kaleido)
pip install pandas numpy scipy plotly scikit-learn kaleido
```

### 2. Executando as Detecções
O arquivo `run_detection.py` automatiza todo o processo (treinamento e testes em lote). O único parâmetro obrigatório é `--dataset` (**CSTR** ou **TE**). Você também pode controlar a calibração do modelo e o treinamento diretamente via linha de comando.

**Parâmetros disponíveis:**
* `--dataset`: Qual base utilizar (`CSTR` ou `TE`). **Obrigatório**.
* `--model`: Qual modelo utilizar (MSCVAE, MSCVAE_v2, PCA, ANN_AE, CNN_AE, LSTM_AE, MSCRED, OmniAnomaly, TranAD, USAD). Padrão: `MSCVAE`.
* `--gain`: Multiplicador de sensibilidade do limiar de alerta. Valores maiores reduzem alarmes falsos (menos sensível), valores menores aumentam a sensibilidade para anomalias sutis. Padrão: `1.0`.
* `--epochs`: Número de épocas no treinamento. Padrão: `50` para CSTR, `100` para TE.

```bash
# Executar com os valores padrão para a base CSTR (utilizará o MSCVAE)
python run_detection.py --dataset CSTR

# Executar a detecção utilizando PCA customizando o ganho do limiar
python run_detection.py --dataset CSTR --model PCA --gain 1.2

# Executar a detecção para a base TE customizando épocas, ganho do limiar e modelo
python run_detection.py --dataset TE --model LSTM_AE --gain 1.2 --epochs 120
```

### 3. Entendendo a Saída
Ao executar o script, os resultados serão salvos automaticamente na pasta `output/` obedecendo à seguinte estrutura dinâmica:
`output/<NOME_DATASET>_<DATA_E_HORA>/<NOME_DATASET>/`

Para cada arquivo de falha (ex: `falha10.csv` ou `fault_01.mat`), o script salvará:
* `<nome_da_falha>_predict.png`: O gráfico contendo a linha temporal do Índice gerado pelo modelo sobreposto à linha do Limiar (Threshold). Todo momento que o índice fica acima do limiar denota um alerta de anomalia.
* `<nome_da_falha>_reconstruction.png`: Gráfico interativo comparando as variáveis originais que mais contribuíram para a falha com a sua respectiva tentativa de reconstrução pelo modelo. O título do gráfico inclui o percentual exato de contribuição de cada variável.

Além dos gráficos, o script gerará também um consolidado global:
* `metrics_<DATASET>.json`: Um arquivo contendo os parâmetros originais da execução (metadados como dataset, model, gain, epochs e timestamp) e todas as métricas detalhadas (**Precisão**, **Recall**, **F1-Score** e **FPR** - False Positive Rate) e a matriz de confusão crua (TP, FP, TN, FN) para todas as simulações, considerando os verdadeiros rótulos de momento de inserção das anomalias.

### 4. Dashboard Interativo (Streamlit)
Você pode executar o Digital Twin interativo do CSTR para visualizar dinâmicas de falha em tempo real através do navegador:
```bash
streamlit run dashboard/app.py
```
