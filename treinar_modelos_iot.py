"""
Treinamento corrigido para diagnóstico IoT.

Correção principal:
- O diagnóstico NÃO considera carretel offline apenas porque GPS ou pressão do aspersor veio zerado.
- Carretel offline = pressão do carretel zerada.
- Aspersor offline = pressão do aspersor zerada, mas pressão do carretel existe.
- GPS Null é tratado como problema de GPS, não como offline.
- Também gera diagnóstico multi-label, pois um equipamento pode ter mais de uma falha ao mesmo tempo.

Entrada esperada:
- dataset_iot_normalizado.xlsx

Saídas geradas em modelos_treinados/:
- modelo_arvore_decisao.pkl                  -> modelo single-label compatível com o Streamlit antigo
- modelo_random_forest.pkl                   -> modelo single-label compatível com o Streamlit antigo
- encoder_label.pkl                          -> encoder do single-label
- modelo_multilabel_arvore_decisao.pkl       -> modelo multi-label recomendado
- modelo_multilabel_random_forest.pkl        -> modelo multi-label recomendado
- labels_multilabel.pkl                      -> lista de diagnósticos multi-label
- features_modelo.pkl                        -> features usadas
- metricas_modelos.csv                       -> métricas single-label
- metricas_modelos_multilabel.csv            -> métricas multi-label por diagnóstico
- dataset_com_labels_corrigidos.csv          -> dataset com labels corrigidos
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
)


ARQUIVO_DATASET = "dataset_iot_normalizado.xlsx"
PASTA_SAIDA = "modelos_treinados"
RANDOM_STATE = 42

# Diagnósticos possíveis no modelo multi-label.
LABELS_MULTILABEL = [
    "BOMBA_OFFLINE",
    "GPS_BOMBA_PROBLEMA",
    "PRESSAO_BOMBA_NULA",
    "CARRETEL_OFFLINE",
    "ASPERSOR_OFFLINE",
    "GPS_ASPERSOR_PROBLEMA",
    "GPS_CARRETEL_PROBLEMA",
    "RECOLHIMENTO_ZERADO",
]

# Ordem de prioridade para criar uma classe única.
# Isso é usado apenas para manter compatibilidade com o Streamlit antigo.
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
    - Se pressão = 0 e rpm = 0: bomba offline.
    - Se GPS inválido/Null: problema de GPS da bomba.
    - Se pressão = 0, mas não está totalmente offline: pressão nula.

    Carretel:
    - Se pressão do carretel = 0: carretel offline.
    - Se pressão do carretel > 0 e pressão do aspersor = 0: aspersor offline.
    - GPS do aspersor Null: problema de GPS do aspersor.
    - GPS do carretel Null: problema de GPS do carretel.
    - Se pressão do carretel > 0 e recolhimento = 0: não está marcando recolhimento.
    """
    diagnosticos = []

    if eh_bomba(linha):
        pressao = valor_numero(linha, "pressao_num")
        rpm = valor_numero(linha, "rpm_num")
        gps_valido = valor_int(linha, "gps_valido", 1)

        bomba_offline = pressao <= 0 and rpm <= 0

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

        carretel_offline = pressao_carretel <= 0

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

    # Colunas binárias para multi-label.
    for label in LABELS_MULTILABEL:
        df[label] = lista_diagnosticos.apply(lambda diagnosticos: int(label in diagnosticos))

    # Label único por prioridade, para compatibilidade com o app atual.
    def escolher_label_unico(diagnosticos: list[str]) -> str:
        for label in PRIORIDADE_SINGLE_LABEL:
            if label in diagnosticos:
                return label
        return "OK"

    df["label_corrigido"] = lista_diagnosticos.apply(escolher_label_unico)
    df["is_falha_corrigido"] = (df["label_corrigido"] != "OK").astype(int)
    df["is_ok_corrigido"] = (df["label_corrigido"] == "OK").astype(int)

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
    """Treina modelos single-label para manter compatibilidade com o Streamlit antigo."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_texto.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelos = {
        "arvore_decisao": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=250,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    resultados = []

    print("\n" + "=" * 80)
    print("TREINAMENTO SINGLE-LABEL - compatível com o Streamlit atual")

    for nome, modelo in modelos.items():
        print("\n" + "-" * 80)
        print(f"Treinando: {nome}")

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        resultados.append({
            "modelo": nome,
            "acuracia": acuracia,
            "precisao_weighted": precisao,
            "recall_weighted": recall,
            "f1_score_weighted": f1,
        })

        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nRelatório por classe:")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

        matriz = confusion_matrix(y_test, y_pred)
        matriz_df = pd.DataFrame(matriz, index=encoder.classes_, columns=encoder.classes_)
        matriz_df.to_csv(os.path.join(PASTA_SAIDA, f"matriz_confusao_{nome}.csv"), encoding="utf-8-sig")

        with open(os.path.join(PASTA_SAIDA, f"modelo_{nome}.pkl"), "wb") as f:
            pickle.dump(modelo, f)

    metricas_df = pd.DataFrame(resultados).sort_values(by="f1_score_weighted", ascending=False)
    metricas_df.to_csv(os.path.join(PASTA_SAIDA, "metricas_modelos.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(PASTA_SAIDA, "encoder_label.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    return metricas_df, encoder


def treinar_multilabel(X: pd.DataFrame, y_multi: pd.DataFrame):
    """Treina modelos multi-label, recomendado para o seu caso real."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_multi,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    modelos = {
        "arvore_decisao": MultiOutputClassifier(
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
            )
        ),
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=250,
                max_depth=14,
                min_samples_split=8,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
    }

    resultados = []

    print("\n" + "=" * 80)
    print("TREINAMENTO MULTI-LABEL - recomendado para diagnóstico real")

    for nome, modelo in modelos.items():
        print("\n" + "-" * 80)
        print(f"Treinando: {nome}")

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred, columns=LABELS_MULTILABEL, index=y_test.index)

        exact_match = accuracy_score(y_test, y_pred_df)
        hamming = hamming_loss(y_test, y_pred_df)
        f1_micro = f1_score(y_test, y_pred_df, average="micro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred_df, average="macro", zero_division=0)

        print(f"Exact Match Accuracy: {exact_match:.4f}")
        print(f"Hamming Loss:         {hamming:.4f}")
        print(f"F1 Micro:             {f1_micro:.4f}")
        print(f"F1 Macro:             {f1_macro:.4f}")

        resultados.append({
            "modelo": nome,
            "diagnostico": "GERAL",
            "exact_match_accuracy": exact_match,
            "hamming_loss": hamming,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precisao": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        })

        for label in LABELS_MULTILABEL:
            precisao = precision_score(y_test[label], y_pred_df[label], zero_division=0)
            recall = recall_score(y_test[label], y_pred_df[label], zero_division=0)
            f1 = f1_score(y_test[label], y_pred_df[label], zero_division=0)
            acuracia = accuracy_score(y_test[label], y_pred_df[label])

            resultados.append({
                "modelo": nome,
                "diagnostico": label,
                "exact_match_accuracy": np.nan,
                "hamming_loss": np.nan,
                "f1_micro": np.nan,
                "f1_macro": np.nan,
                "acuracia_binaria": acuracia,
                "precisao": precisao,
                "recall": recall,
                "f1_score": f1,
            })

        with open(os.path.join(PASTA_SAIDA, f"modelo_multilabel_{nome}.pkl"), "wb") as f:
            pickle.dump(modelo, f)

    metricas_df = pd.DataFrame(resultados)
    metricas_df.to_csv(os.path.join(PASTA_SAIDA, "metricas_modelos_multilabel.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(PASTA_SAIDA, "labels_multilabel.pkl"), "wb") as f:
        pickle.dump(LABELS_MULTILABEL, f)

    return metricas_df


def salvar_metadados(features: list[str]):
    with open(os.path.join(PASTA_SAIDA, "features_modelo.pkl"), "wb") as f:
        pickle.dump(features, f)


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    df = carregar_dataset(ARQUIVO_DATASET)
    df = aplicar_labels_corrigidos(df)

    print("\nDistribuição do label original:")
    if "label" in df.columns:
        print(df["label"].value_counts())
    else:
        print("Coluna label original não encontrada.")

    print("\nDistribuição do label corrigido single-label:")
    print(df["label_corrigido"].value_counts())

    print("\nQuantidade por diagnóstico multi-label:")
    print(df[LABELS_MULTILABEL].sum().sort_values(ascending=False))

    X, features = preparar_features(df)
    salvar_metadados(features)

    print("\nFeatures usadas no treinamento:")
    for feature in features:
        print(f"- {feature}")

    df.to_csv(os.path.join(PASTA_SAIDA, "dataset_com_labels_corrigidos.csv"), index=False, encoding="utf-8-sig")

    metricas_single, encoder = treinar_single_label(X, df["label_corrigido"])
    metricas_multi = treinar_multilabel(X, df[LABELS_MULTILABEL])

    print("\n" + "=" * 80)
    print("TREINAMENTO FINALIZADO")
    print(f"Arquivos salvos em: {PASTA_SAIDA}")

    print("\nComparativo single-label:")
    print(metricas_single)

    print("\nClasses single-label:")
    for classe in encoder.classes_:
        print(f"- {classe}")

    print("\nObservação:")
    print("Use o modelo single-label se quiser manter o Streamlit atual.")
    print("Use o modelo multi-label se quiser mostrar todas as falhas ao mesmo tempo, que é o ideal para seu caso real.")


if __name__ == "__main__":
    main()
