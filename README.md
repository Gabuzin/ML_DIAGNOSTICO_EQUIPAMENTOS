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

## Como os Modelos Sao Gerados no Codigo

Toda a geracao dos modelos acontece no arquivo `treinar_modelos_iot.py`. O ponto de entrada e a funcao `main()`, que organiza o processo completo.

### 1. Carregamento do dataset

A funcao `carregar_dataset()` le o arquivo Excel:

```python
df = carregar_dataset(ARQUIVO_DATASET)
```

O arquivo usado como entrada e:

```text
dataset_iot_normalizado.xlsx
```

Esse arquivo contem as leituras ja normalizadas dos equipamentos, como pressao, RPM, GPS, velocidade de recolhimento, metragem da mangueira e identificacao do equipamento.

### 2. Geracao dos diagnosticos corrigidos

Depois de carregar os dados, o codigo chama:

```python
df = aplicar_labels_corrigidos(df)
```

Essa etapa cria os diagnosticos que serao usados como resposta esperada do modelo.

Primeiro, cada linha do dataset passa pela funcao:

```python
gerar_diagnosticos_corrigidos(linha)
```

Essa funcao aplica regras de negocio para identificar problemas em bombas e carreteis.

Exemplos de regras:

- bomba com pressao zerada, RPM zerado e GPS invalido vira `BOMBA_OFFLINE`;
- bomba com GPS invalido vira `GPS_BOMBA_PROBLEMA`;
- carretel com pressao, aspersor, recolhimento e GPS invalidos vira `CARRETEL_OFFLINE`;
- carretel com pressao do carretel positiva e recolhimento zerado vira `RECOLHIMENTO_ZERADO`;
- se nenhuma falha for encontrada, o diagnostico vira `OK`.

Como uma mesma linha pode ter mais de um problema, o codigo cria duas colunas:

- `diagnosticos_corrigidos`: guarda todos os diagnosticos encontrados.
- `label_corrigido`: guarda apenas um diagnostico principal.

O `label_corrigido` e escolhido pela lista `PRIORIDADE_SINGLE_LABEL`. Essa lista define qual falha tem prioridade quando existem varios problemas na mesma leitura.

### 3. Preparacao das features

Depois dos labels, o codigo prepara as colunas de entrada do modelo:

```python
X, features = preparar_features(df)
```

`X` representa os dados que o modelo usa para aprender. Ele contem somente as colunas numericas escolhidas em `FEATURES_BASE`.

Exemplos:

- `equipamento_id`
- `bomba_operando`
- `gps_valido`
- `pressao_num`
- `rpm_num`
- `pressao_aspersor_num`
- `pressao_carretel_num`
- `velocidade_recolhimento_num`
- `null_count`

As features tambem sao salvas em:

```text
modelos_treinados/features_modelo.pkl
```

Isso garante que, na hora de usar o modelo no Streamlit, as colunas sejam montadas na mesma ordem usada durante o treinamento.

### 4. Conversao dos labels para numeros

Modelos do scikit-learn trabalham melhor com classes numericas. Por isso, dentro da funcao `treinar_single_label()`, o codigo usa:

```python
encoder = LabelEncoder()
y = encoder.fit_transform(y_texto.astype(str))
```

O `LabelEncoder` converte os nomes dos diagnosticos para numeros.

Exemplo conceitual:

```text
BOMBA_OFFLINE -> 0
CARRETEL_OFFLINE -> 1
OK -> 2
```

Esse encoder e salvo em:

```text
modelos_treinados/encoder_label.pkl
```

Ele e necessario para transformar a resposta numerica do modelo de volta para texto.

### 5. Divisao entre treino e teste

O dataset e dividido em duas partes:

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

A divisao usada e:

- 75% dos dados para treino;
- 25% dos dados para teste.

O parametro `stratify=y` mantem a proporcao das classes nas duas partes. Isso ajuda a evitar que uma classe fique concentrada apenas no treino ou apenas no teste.

### 6. Criacao dos modelos

O codigo cria dois modelos:

```python
modelos = {
    "arvore_decisao": DecisionTreeClassifier(...),
    "random_forest": RandomForestClassifier(...),
}
```

Os modelos treinados sao:

- `DecisionTreeClassifier`: Arvore de Decisao.
- `RandomForestClassifier`: Random Forest.

A Arvore de Decisao cria uma estrutura de decisoes baseada nas features.

O Random Forest cria varias arvores de decisao e combina os resultados. Por isso, normalmente e mais robusto que uma unica arvore.

### 7. Treinamento

Cada modelo e treinado com:

```python
modelo.fit(X_train, y_train)
```

Nessa etapa, o modelo aprende a relacao entre:

- entradas: features dos equipamentos;
- saida esperada: `label_corrigido`.

Em outras palavras, ele aprende padroes como:

```text
pressao = 0, rpm = 0, gps invalido -> BOMBA_OFFLINE
```

### 8. Avaliacao

Depois do treinamento, o modelo faz predicoes nos dados de teste:

```python
y_pred = modelo.predict(X_test)
```

Em seguida, o codigo calcula metricas:

- acuracia;
- precisao;
- recall;
- F1-score;
- matriz de confusao.

As metricas ajudam a comparar o desempenho da Arvore de Decisao e do Random Forest.

### 9. Salvamento dos modelos

Por fim, cada modelo treinado e salvo em arquivo `.pkl`:

```python
with open(os.path.join(PASTA_SAIDA, f"modelo_{nome}.pkl"), "wb") as f:
    pickle.dump(modelo, f)
```

Os principais arquivos finais sao:

```text
modelos_treinados/modelo_arvore_decisao.pkl
modelos_treinados/modelo_random_forest.pkl
modelos_treinados/encoder_label.pkl
modelos_treinados/features_modelo.pkl
modelos_treinados/metricas_modelos.csv
```

Esses arquivos sao usados depois pela aplicacao `app_streamlit_iot.py` para carregar os modelos e fazer novas predicoes a partir de um JSON.

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
  "device": "Device/XXXXX",
  "dado": [
    ["2026-05-22 23:24:56,000", "101", "00.00"...]
  ]
}
```

O arquivo `app_streamlit_iot.py` ja possui um exemplo completo preenchido na caixa de texto da interface.

## Observacoes

- O arquivo `dataset_iot_normalizado.xlsx` precisa estar na raiz do projeto para rodar o treinamento.
- A pasta `modelos_treinados/` precisa conter os arquivos `.pkl` para que o Streamlit consiga fazer predicoes.
- Sempre que o dataset ou as regras de diagnostico forem alterados, e recomendado executar novamente `python treinar_modelos_iot.py`.
