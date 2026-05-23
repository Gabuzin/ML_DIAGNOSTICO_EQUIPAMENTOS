# ML Diagnostico de Equipamentos IoT

Projeto de Machine Learning para diagnosticar problemas em equipamentos IoT, principalmente bombas e carreteis, a partir de leituras normalizadas de sensores.

O projeto possui dois fluxos principais:

- Treinamento dos modelos com base no dataset `dataset_iot_normalizado.xlsx`.
- Aplicacao Streamlit para analisar um JSON recebido do equipamento e comparar predicoes de dois modelos.

## Tecnologias

- Python
- pandas
- scikit-learn
- Streamlit
- Plotly
- openpyxl

## Estrutura do Projeto

```text
.
|-- app_streamlit_iot.py
|-- dataset_iot_normalizado.xlsx
|-- requirements.txt
|-- treinar_modelos_iot.py
`-- modelos_treinados/
    |-- dataset_com_labels_corrigidos.csv
    |-- encoder_label.pkl
    |-- features_modelo.pkl
    |-- matriz_confusao_arvore_decisao.csv
    |-- matriz_confusao_random_forest.csv
    |-- metricas_modelos.csv
    |-- modelo_arvore_decisao.pkl
    `-- modelo_random_forest.pkl
```

## Objetivo

O sistema transforma dados brutos dos equipamentos em features numericas e usa modelos de classificacao para identificar possiveis falhas.

As principais classes de diagnostico sao:

- `BOMBA_OFFLINE`
- `CARRETEL_OFFLINE`
- `ASPERSOR_OFFLINE`
- `GPS_BOMBA_PROBLEMA`
- `GPS_ASPERSOR_PROBLEMA`
- `GPS_CARRETEL_PROBLEMA`
- `PRESSAO_BOMBA_NULA`
- `RECOLHIMENTO_ZERADO`
- `OK`

## Instalacao

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual.

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Treinamento dos Modelos

O treinamento e feito pelo arquivo:

```bash
python treinar_modelos_iot.py
```

Esse script executa as seguintes etapas:

1. Carrega o dataset `dataset_iot_normalizado.xlsx`.
2. Aplica regras de diagnostico corrigidas.
3. Cria a coluna `diagnosticos_corrigidos`, com todos os problemas encontrados.
4. Cria a coluna `label_corrigido`, com um unico diagnostico principal por linha.
5. Prepara as features usadas pelos modelos.
6. Treina os modelos:
   - Arvore de Decisao
   - Random Forest
7. Salva os modelos, metricas e arquivos auxiliares na pasta `modelos_treinados/`.

## Regras de Labels Corrigidos

A funcao `aplicar_labels_corrigidos` cria os labels finais usados no treinamento.

Ela usa a funcao `gerar_diagnosticos_corrigidos` para analisar cada linha do dataset. Uma linha pode ter mais de um diagnostico, por exemplo:

```text
PRESSAO_BOMBA_NULA;GPS_BOMBA_PROBLEMA
```

Como o modelo single-label precisa de apenas uma classe por linha, o projeto usa uma lista de prioridade chamada `PRIORIDADE_SINGLE_LABEL`.

Quando ha mais de um problema, o primeiro diagnostico encontrado nessa lista de prioridade vira o `label_corrigido`.

## Features Usadas no Treinamento

As principais colunas usadas como entrada dos modelos sao:

- `project_numeric_id`
- `equipamento_id`
- `bomba_operando`
- `gps_valido`
- `gps_aspersor_valido`
- `gps_carretel_valido`
- `pressao_num`
- `rpm_num`
- `pressao_aspersor_num`
- `pressao_carretel_num`
- `velocidade_recolhimento_num`
- `metragem_mangueira_num`
- `null_count`

Essas features sao salvas em `modelos_treinados/features_modelo.pkl` para garantir que a aplicacao use a mesma ordem de colunas no momento da predicao.

## Arquivos Gerados

Apos o treinamento, os principais arquivos gerados sao:

- `modelo_arvore_decisao.pkl`: modelo treinado de Arvore de Decisao.
- `modelo_random_forest.pkl`: modelo treinado de Random Forest.
- `encoder_label.pkl`: codificador que converte labels numericos para texto.
- `features_modelo.pkl`: lista das features usadas no treinamento.
- `metricas_modelos.csv`: metricas comparativas dos modelos.
- `matriz_confusao_arvore_decisao.csv`: matriz de confusao da Arvore de Decisao.
- `matriz_confusao_random_forest.csv`: matriz de confusao do Random Forest.
- `dataset_com_labels_corrigidos.csv`: dataset final com os diagnosticos gerados.

## Aplicacao Streamlit

A aplicacao permite colar um JSON recebido do equipamento e visualizar o diagnostico previsto pelos modelos.

Execute:

```bash
streamlit run app_streamlit_iot.py
```

A interface faz:

- leitura do JSON informado;
- extracao das bombas e carreteis;
- montagem das features esperadas pelos modelos;
- predicao com Arvore de Decisao;
- predicao com Random Forest;
- comparacao entre os dois modelos;
- exibicao da confianca de cada predicao;
- exportacao do resultado em CSV.

## Formato de Entrada da Aplicacao

A aplicacao espera um JSON com a chave `dado`, contendo uma lista de pacotes recebidos do equipamento.

Exemplo simplificado:

```json
{
  "ver": "1",
  "projectId": "uuid-do-projeto",
  "device": "Device/RADCOM_EDC9",
  "dado": [
    ["2026-05-22 23:24:56,000", "101", "00.00"]
  ]
}
```

O arquivo `app_streamlit_iot.py` ja possui um exemplo completo preenchido na caixa de texto da interface.

## Observacoes

- O arquivo `dataset_iot_normalizado.xlsx` precisa estar na raiz do projeto para rodar o treinamento.
- A pasta `modelos_treinados/` precisa conter os arquivos `.pkl` para que o Streamlit consiga fazer predicoes.
- Sempre que o dataset ou as regras de diagnostico forem alterados, e recomendado executar novamente `python treinar_modelos_iot.py`.
