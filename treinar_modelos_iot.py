"""
================================================================================
ETAPA 1: CARREGAMENTO E PREPARAÇÃO DE DADOS
================================================================================

Este script treina modelos de Machine Learning para diagnosticar problemas
em equipamentos IoT (bombas e carretéis).

ENTRADA:
  - dataset_iot_normalizado.xlsx

SAÍDAS (em modelos_treinados/):
  - modelo_arvore_decisao.pkl          -> Modelo treinado (Árvore de Decisão)
  - modelo_random_forest.pkl           -> Modelo treinado (Random Forest)
  - encoder_label.pkl                  -> Codificador de labels
  - metricas_modelos.csv               -> Métricas de desempenho
  - dataset_com_labels_corrigidos.csv  -> Dataset com diagnósticos

================================================================================
ETAPA 2: DEFINIÇÕES DE DIAGNÓSTICOS
================================================================================

Os diagnósticos possíveis são:
- BOMBA_OFFLINE: Pressão e RPM zerados
- CARRETEL_OFFLINE: Pressão do carretel zerada
- ASPERSOR_OFFLINE: Pressão do aspersor zerada
- GPS_BOMBA_PROBLEMA: GPS da bomba inválido
- GPS_ASPERSOR_PROBLEMA: GPS do aspersor inválido
- GPS_CARRETEL_PROBLEMA: GPS do carretel inválido
- PRESSAO_BOMBA_NULA: Pressão da bomba nula
- RECOLHIMENTO_ZERADO: Recolhimento não está marcando

================================================================================
"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ARQUIVO_DATASET = "dataset_iot_normalizado.xlsx"
PASTA_SAIDA = "modelos_treinados"
RANDOM_STATE = 42

# Ordem de prioridade para criar uma classe única.
PRIORIDADE_SINGLE_LABEL = [
    "BOMBA_OFFLINE",
    "CARRETEL_OFFLINE",
    "ASPERSOR_OFFLINE",
    "RECOLHIMENTO_ZERADO",
    "GPS_BOMBA_PROBLEMA",
    "GPS_ASPERSOR_PROBLEMA",
    "GPS_CARRETEL_PROBLEMA",
    "PRESSAO_BOMBA_NULA",
]


FEATURES_BASE = [
    "project_numeric_id",
    "equipamento_id",
    "bomba_operando",
    "gps_valido",
    "gps_aspersor_valido",
    "gps_carretel_valido",
    "pressao_num",
    "rpm_num",
    "pressao_aspersor_num",
    "pressao_carretel_num",
    "velocidade_recolhimento_num",
    "metragem_mangueira_num",
    "null_count",
]


def caminho_absoluto_dataset(nome_arquivo: str) -> str:
    """Permite rodar o script tanto da pasta atual quanto da pasta do próprio arquivo."""
    if os.path.exists(nome_arquivo):
        return nome_arquivo

    pasta_script = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(pasta_script, nome_arquivo)
    if os.path.exists(caminho):
        return caminho

    raise FileNotFoundError(f"Dataset não encontrado: {nome_arquivo}")


def carregar_dataset(caminho: str) -> pd.DataFrame:
    caminho = caminho_absoluto_dataset(caminho)
    df = pd.read_excel(caminho)
    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df


def valor_numero(linha: pd.Series, coluna: str, padrao: float = 0.0) -> float:
    valor = linha.get(coluna, padrao)
    valor = pd.to_numeric(valor, errors="coerce")
    if pd.isna(valor):
        return padrao
    return float(valor)


def valor_int(linha: pd.Series, coluna: str, padrao: int = 0) -> int:
    return int(valor_numero(linha, coluna, padrao))


def eh_bomba(linha: pd.Series) -> bool:
    tipo = str(linha.get("tipo", "")).strip().lower()
    equipamento_id = valor_int(linha, "equipamento_id", -1)
    return tipo == "bomba" or equipamento_id in [10, 11, 20, 21, 30, 31]


def eh_carretel(linha: pd.Series) -> bool:
    tipo = str(linha.get("tipo", "")).strip().lower()
    equipamento_id = valor_int(linha, "equipamento_id", -1)
    return tipo == "carretel" or equipamento_id in [22, 24, 26, 32, 34, 36]


def gerar_diagnosticos_corrigidos(linha: pd.Series) -> list[str]:
    """
    Regras corrigidas conforme a lógica real do equipamento.

    Bomba:
    - Se pressão = 0, rpm = 0: e gps = null bomba offline.
    - Se GPS inválido/Null: problema de GPS da bomba.
    - Se pressão = 0, mas não está totalmente offline: pressão nula.

    Carretel:
    - Se tudo estiver 0 ou null carretel offline.
    - Se gps aspersor = 0 e pressão do aspersor = 0: aspersor offline.
    - GPS do aspersor Null: problema de GPS do aspersor.
    - GPS do carretel Null: problema de GPS do carretel.
    - Se pressão do carretel > 0 e recolhimento = 0: não está marcando recolhimento.
    """
    diagnosticos = []

    if eh_bomba(linha):
        pressao = valor_numero(linha, "pressao_num")
        rpm = valor_numero(linha, "rpm_num")
        gps_valido = valor_int(linha, "gps_valido", 1)

        bomba_offline = pressao <= 0 and rpm <= 0 and gps_valido == 0

        if bomba_offline:
            diagnosticos.append("BOMBA_OFFLINE")
        else:
            if pressao <= 0:
                diagnosticos.append("PRESSAO_BOMBA_NULA")

        if gps_valido == 0:
            diagnosticos.append("GPS_BOMBA_PROBLEMA")

    elif eh_carretel(linha):
        pressao_aspersor = valor_numero(linha, "pressao_aspersor_num")
        pressao_carretel = valor_numero(linha, "pressao_carretel_num")
        velocidade_recolhimento = valor_numero(linha, "velocidade_recolhimento_num")
        gps_aspersor_valido = valor_int(linha, "gps_aspersor_valido", 1)
        gps_carretel_valido = valor_int(linha, "gps_carretel_valido", 1)

        

        if (pressao_carretel <= 0 and pressao_aspersor <= 0 and velocidade_recolhimento <= 0 
            and gps_aspersor_valido == 0 and gps_carretel_valido == 0):
            diagnosticos.append("CARRETEL_OFFLINE")
        else:
            if pressao_aspersor <= 0 and gps_aspersor_valido == 0:
                diagnosticos.append("ASPERSOR_OFFLINE")

            if velocidade_recolhimento <= 0 and pressao_carretel > 1.5:
                diagnosticos.append("RECOLHIMENTO_ZERADO")

            if gps_aspersor_valido == 0:
                diagnosticos.append("GPS_ASPERSOR_PROBLEMA")

            if gps_carretel_valido == 0:
                diagnosticos.append("GPS_CARRETEL_PROBLEMA")

    if not diagnosticos:
        diagnosticos.append("OK")

    return diagnosticos


def aplicar_labels_corrigidos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lista_diagnosticos = df.apply(gerar_diagnosticos_corrigidos, axis=1)
    df["diagnosticos_corrigidos"] = lista_diagnosticos.apply(lambda x: ";".join(x))

    # Escolhe um label único por prioridade
    def escolher_label_unico(diagnosticos: list[str]) -> str:
        for label in PRIORIDADE_SINGLE_LABEL:
            if label in diagnosticos:
                return label
        return "OK"

    df["label_corrigido"] = lista_diagnosticos.apply(escolher_label_unico)

    return df


def preparar_features(df: pd.DataFrame):
    features = [col for col in FEATURES_BASE if col in df.columns]

    if not features:
        raise ValueError("Nenhuma feature válida foi encontrada no dataset.")

    X = df[features].copy()
    X = X.fillna(-1)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1)

    return X, features


def treinar_single_label(X: pd.DataFrame, y_texto: pd.Series):
    """
    ============================================================================
    TREINAMENTO DE MODELOS SINGLE-LABEL
    ============================================================================
    
    Esta função realiza o treinamento de dois modelos de classificação:
    1. Árvore de Decisão (Decision Tree)
    2. Floresta Aleatória (Random Forest)
    
    Ambos são treinados para diagnosticar problemas em equipamentos.
    
    Parâmetros:
      X: DataFrame com as features (características dos equipamentos)
      y_texto: Series com o diagnóstico de cada equipamento
    
    Retorna:
      metricas_df: DataFrame com as métricas de desempenho
      encoder: Objeto que mapeia diagnósticos para números
    """
    
    print("\n" + "=" * 80)
    print("TREINAMENTO DOS MODELOS")
    print("=" * 80)
    
    # ========================================================================
    # PASSO 1: CODIFICAÇÃO DE LABELS
    # ========================================================================
    print("\n>>> PASSO 1: Codificando diagnósticos em números")
    
    # O LabelEncoder converte diagnósticos em texto para números
    # Exemplo: "BOMBA_OFFLINE" -> 0, "CARRETEL_OFFLINE" -> 1, etc.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_texto.astype(str))
    
    print(f"    Classes encontradas: {list(encoder.classes_)}")
    print(f"    Total de amostras: {len(y)}")

    # ========================================================================
    # PASSO 2: DIVISÃO EM DADOS DE TREINO E TESTE
    # ========================================================================
    print("\n>>> PASSO 2: Dividindo dados em treino (75%) e teste (25%)")
    
    # Split: 75% para treinar, 25% para testar
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,              # 25% dos dados para teste
        random_state=RANDOM_STATE,   # Reprodutibilidade
        stratify=y,                  # Mantém proporção de classes
    )
    
    print(f"    Treino: {len(X_train)} amostras")
    print(f"    Teste:  {len(X_test)} amostras")

    # ========================================================================
    # PASSO 3: DEFINIÇÃO DOS MODELOS
    # ========================================================================
    print("\n>>> PASSO 3: Criando modelos com hiperparâmetros otimizados")
    
    modelos = {
        # Árvore de Decisão
        "arvore_decisao": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=10,              # Profundidade máxima da árvore
            min_samples_split=10,      # Mínimo de amostras para dividir
            min_samples_leaf=5,        # Mínimo de amostras em folhas
            class_weight="balanced",   # Balanceia classes desiguais
        ),
        
        # Floresta Aleatória (500 árvores)
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=250,          # Número de árvores na floresta
            max_depth=14,              # Profundidade máxima
            min_samples_split=8,       # Mínimo de amostras para dividir
            min_samples_leaf=3,        # Mínimo de amostras em folhas
            class_weight="balanced",   # Balanceia classes desiguais
            n_jobs=-1,                 # Usa todos os processadores
        ),
    }

    resultados = []

    # ========================================================================
    # PASSO 4: TREINAMENTO E AVALIAÇÃO DOS MODELOS
    # ========================================================================
    print("\n>>> PASSO 4: Treinando e avaliando modelos")
    
    for nome, modelo in modelos.items():
        print("\n" + "-" * 80)
        print(f"Treinando: {nome}")
        print("-" * 80)

        # ====================================================================
        # TREINAMENTO: Modelo aprende com dados de treino
        # ====================================================================
        print(f"  ✓ Ajustando modelo com {len(X_train)} amostras...")
        modelo.fit(X_train, y_train)
        
        # ====================================================================
        # PREDIÇÃO: Faz diagnósticos nos dados de teste
        # ====================================================================
        print(f"  ✓ Fazendo predições em {len(X_test)} amostras de teste...")
        y_pred = modelo.predict(X_test)

        # ====================================================================
        # CÁLCULO DE MÉTRICAS: Avalia desempenho do modelo
        # ====================================================================
        print("  ✓ Calculando métricas...")
        
        # Acurácia: percentual de acertos
        acuracia = accuracy_score(y_test, y_pred)
        
        # Precisão: quando o modelo prevê um diagnóstico, acerta com que frequência
        precisao = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Recall: de todos os casos de um diagnóstico, o modelo acerta quantos
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # F1-Score: média harmônica entre precisão e recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        resultados.append({
            "modelo": nome,
            "acuracia": acuracia,
            "precisao_weighted": precisao,
            "recall_weighted": recall,
            "f1_score_weighted": f1,
        })

        # Exibe as métricas
        print(f"\n  MÉTRICAS:")
        print(f"    • Acurácia:  {acuracia:.4f} ({acuracia*100:.2f}%)")
        print(f"    • Precisão:  {precisao:.4f}")
        print(f"    • Recall:    {recall:.4f}")
        print(f"    • F1-Score:  {f1:.4f}")
        
        # Relatório detalhado por classe
        print(f"\n  Relatório detalhado por diagnóstico:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=encoder.classes_, 
            zero_division=0
        ))

        # ====================================================================
        # SALVAMENTO DO MODELO TREINADO
        # ====================================================================
        print(f"  ✓ Salvando modelo em disco...")
        
        # Salva a matriz de confusão (mostra erros do modelo)
        matriz = confusion_matrix(y_test, y_pred)
        matriz_df = pd.DataFrame(
            matriz, 
            index=encoder.classes_, 
            columns=encoder.classes_
        )
        matriz_df.to_csv(
            os.path.join(PASTA_SAIDA, f"matriz_confusao_{nome}.csv"), 
            encoding="utf-8-sig"
        )

        # Salva o modelo em arquivo .pkl (pickle)
        with open(os.path.join(PASTA_SAIDA, f"modelo_{nome}.pkl"), "wb") as f:
            pickle.dump(modelo, f)
        
        print(f"    ✓ Arquivo: modelo_{nome}.pkl")
        print(f"    ✓ Matriz: matriz_confusao_{nome}.csv")

    # ========================================================================
    # PASSO 5: SALVAMENTO DE RESULTADOS FINAIS
    # ========================================================================
    print("\n>>> PASSO 5: Salvando resultados finais")
    
    # Cria tabela com métricas de todos os modelos
    metricas_df = pd.DataFrame(resultados).sort_values(
        by="f1_score_weighted", 
        ascending=False
    )
    metricas_df.to_csv(
        os.path.join(PASTA_SAIDA, "metricas_modelos.csv"), 
        index=False, 
        encoding="utf-8-sig"
    )
    print("  ✓ Arquivo: metricas_modelos.csv")

    # Salva o encoder para usar na predição posterior
    with open(os.path.join(PASTA_SAIDA, "encoder_label.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    print("  ✓ Arquivo: encoder_label.pkl")

    return metricas_df, encoder





def main():
    """
    ============================================================================
    FUNÇÃO PRINCIPAL - Orquestra todo o fluxo de treinamento
    ============================================================================
    
    Realiza 4 etapas principais:
    1. CARREGAMENTO: Lê dados do arquivo Excel
    2. PREPARAÇÃO: Gera diagnósticos e prepara features
    3. TREINAMENTO: Treina os modelos
    4. SALVAMENTO: Persiste modelos e métricas
    """
    
    # Criação da pasta de saída
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # ========================================================================
    # ETAPA 1: CARREGAMENTO DE DADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 1: CARREGAMENTO DE DADOS")
    print("=" * 80)
    
    # Carrega o arquivo Excel do dataset
    df = carregar_dataset(ARQUIVO_DATASET)
    # Resultado: DataFrame com todas as colunas originais do dataset
    
    # ========================================================================
    # ETAPA 2: PREPARAÇÃO E PROCESSAMENTO DE DADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 2: PREPARAÇÃO E PROCESSAMENTO DE DADOS")
    print("=" * 80)
    
    # Aplica as regras de diagnóstico e gera labels corrigidos
    # Cria a coluna "label_corrigido" com os diagnósticos
    df = aplicar_labels_corrigidos(df)
    
    # Exibe estatísticas dos labels
    print("\nDistribuição do label corrigido:")
    print(df["label_corrigido"].value_counts())

    # Extrai as features (características) que serão usadas para treinar
    # Features: pressão, RPM, GPS, etc.
    X, features = preparar_features(df)
    
    print("\nFeatures usadas no treinamento:")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")

    # Salva o dataset processado com os labels corrigidos
    df.to_csv(
        os.path.join(PASTA_SAIDA, "dataset_com_labels_corrigidos.csv"), 
        index=False, 
        encoding="utf-8-sig"
    )
    print(f"\n✓ Dataset processado salvo em: {PASTA_SAIDA}/dataset_com_labels_corrigidos.csv")
    
    # Salva as features usadas no treinamento
    with open(os.path.join(PASTA_SAIDA, "features_modelo.pkl"), "wb") as f:
        pickle.dump(features, f)
    print(f"✓ Features salvas em: {PASTA_SAIDA}/features_modelo.pkl")

    # ========================================================================
    # ETAPA 3: TREINAMENTO DOS MODELOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 3: TREINAMENTO DOS MODELOS DE MACHINE LEARNING")
    print("=" * 80)
    
    # Treina modelos de classificação single-label
    # (uma única classe de diagnóstico por equipamento)
    metricas_single, encoder = treinar_single_label(X, df["label_corrigido"])

    # ========================================================================
    # ETAPA 4: RESULTADOS E RESUMO
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 4: RESULTADOS E RESUMO DO TREINAMENTO")
    print("=" * 80)
    
    print("\nArquivos gerados:")
    print(f"  ✓ Pasta de saída: {PASTA_SAIDA}/")
    print(f"    - modelo_arvore_decisao.pkl")
    print(f"    - modelo_random_forest.pkl")
    print(f"    - encoder_label.pkl")
    print(f"    - metricas_modelos.csv")
    print(f"    - dataset_com_labels_corrigidos.csv")

    print("\nMétricas de desempenho dos modelos:")
    print(metricas_single.to_string())

    print("\nClasses de diagnóstico treinadas:")
    for classe in encoder.classes_:
        print(f"  - {classe}")


if __name__ == "__main__":
    main()
