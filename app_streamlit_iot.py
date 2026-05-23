"""
Aplicação Streamlit - Comparação de modelos ML para IoT
Modelos comparados: Árvore de Decisão x Random Forest

Como usar:
1. Coloque este arquivo na mesma pasta do treinamento.
2. Garanta a estrutura:
   app_streamlit_iot.py
   modelos_treinados/
      modelo_arvore_decisao.pkl
      modelo_random_forest.pkl
      encoder_label.pkl
      features_modelo.pkl
      metricas_modelos.csv
3. Execute:
   streamlit run app_streamlit_iot.py
"""

import json
import os
import pickle
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PASTA_MODELOS = "modelos_treinados"

EXEMPLO_JSON = """{
  "ver": "1",
  "projectId": "ahhvgsajwrjfhgsdfg-1234-5678-9101-abcdef123456",
  "device": "Device/XXXXX",
  "dado": [["2026-05-22 23:24:56,000", "101", "00.00", "00000000", "Null", "0", "20", "3.30", "1273", "240", "00000100", "Null", "21", "0.00", "0", "0", "00000000", "Null", "22", "0.00", "Null", "8.35", "00000001", "-19.387531/-48.317593", "50", "100", "0M00", "24", "0.00", "-19.387531/-48.317593", "9.33", "00000010", "-19.387531/-48.317593", "0", "0", "0M00", "26", "0.00", "Null", "0.00", "00000010", "Null", "0", "0", "0M00"]]
}"""


def carregar_pickle(nome_arquivo: str):
    caminho = os.path.join(PASTA_MODELOS, nome_arquivo)
    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        st.stop()
    with open(caminho, "rb") as arquivo:
        return pickle.load(arquivo)


@st.cache_resource
def carregar_modelos():
    arvore = carregar_pickle("modelo_arvore_decisao.pkl")
    floresta = carregar_pickle("modelo_random_forest.pkl")
    encoder = carregar_pickle("encoder_label.pkl")
    features = carregar_pickle("features_modelo.pkl")
    return arvore, floresta, encoder, features


def para_float(valor: Any, padrao: float = -1.0) -> float:
    if valor is None:
        return padrao
    texto = str(valor).strip().replace(",", ".")
    if texto.lower() in ["null", "nan", "none", "", "-"]:
        return padrao
    try:
        return float(texto)
    except ValueError:
        return padrao


def para_int(valor: Any, padrao: int = -1) -> int:
    numero = para_float(valor, padrao=float(padrao))
    try:
        return int(numero)
    except Exception:
        return padrao


def gps_valido(valor: Any) -> int:
    texto = str(valor).strip()
    if texto.lower() in ["null", "nan", "none", "", "-"]:
        return 0
    if "/" in texto:
        return 1
    return 0


def contar_nulls(lista: List[Any]) -> int:
    total = 0
    for item in lista:
        if str(item).strip().lower() in ["null", "nan", "none", "", "-"]:
            total += 1
    return total


def extrair_linhas_do_json(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforma o JSON bruto do equipamento em linhas compatíveis com as features do modelo.

    Interpretação usada:
    - Cada lista dentro de 'dado' representa um pacote recebido via MQTT/LoRa.
    - A partir da posição 6 aparecem duas bombas em blocos de 6 campos.
    - A partir da posição 18 aparecem grupos de carreteis com 9 campos.
    - Exemplo: 20/21 são bombas; 22/24/26 são carreteis.
    """
    pacotes = payload.get("dado", [])
    if not isinstance(pacotes, list) or len(pacotes) == 0:
        raise ValueError("O JSON precisa ter a chave 'dado' contendo uma lista de leituras.")

    linhas = []
    project_uuid = payload.get("projectId", "")
    device = payload.get("device", "")

    for pacote in pacotes:
        if not isinstance(pacote, list) or len(pacote) < 12:
            continue

        timestamp = pacote[0]
        project_numeric_id = para_int(pacote[1], padrao=0)
        bomba_operando = para_int(pacote[5], padrao=0)

        # ============================================================
        # Estrutura real do pacote RADCOM_EDC9
        # ============================================================
        # Cabeçalho: posições 0 a 5
        # Bomba 1: posições 6 a 11
        # Bomba 2: posições 12 a 17
        # Carreteis: a partir da posição 18, em blocos de 9 campos
        #
        # Isso evita interpretar o ID 21 como carretel.
        # IDs típicos:
        # - Bombas: 20 e 21, ou 10/11, ou 30/31
        # - Carreteis: 22/24/26, ou 32/34/36 etc.

        # Bombas: dois blocos fixos de 6 campos.
        for idx_bomba in [6, 12]:
            if idx_bomba + 5 >= len(pacote):
                continue

            trecho_bomba = pacote[idx_bomba:idx_bomba + 6]
            bomba_id = para_int(trecho_bomba[0])

            if bomba_id <= 0:
                continue

            bomba_pressao = para_float(trecho_bomba[1])
            bomba_rpm = para_float(trecho_bomba[2])
            bomba_gps = trecho_bomba[5]

            linhas.append({
                "timestamp": timestamp,
                "project_uuid": project_uuid,
                "project_numeric_id": project_numeric_id,
                "device": device,
                "tipo": "bomba",
                "equipamento_id": bomba_id,
                "bomba_operando": bomba_operando,
                "gps_valido": gps_valido(bomba_gps),
                "gps_aspersor_valido": -1,
                "gps_carretel_valido": -1,
                "pressao_num": bomba_pressao,
                "rpm_num": bomba_rpm,
                "pressao_aspersor_num": -1,
                "pressao_carretel_num": -1,
                "velocidade_recolhimento_num": -1,
                "metragem_mangueira_num": -1,
                "null_count": contar_nulls(trecho_bomba),
                "diagnostico_regras": diagnostico_regras(
                    tipo="bomba",
                    gps_bomba=bomba_gps,
                    pressao_bomba=bomba_pressao,
                    rpm=bomba_rpm,
                    bomba_operando=bomba_operando,
                ),
            })

        # Carreteis: após as duas bombas, grupos fixos de 9 campos.
        idx = 18
        while idx + 8 < len(pacote):
            trecho = pacote[idx:idx + 9]
            carretel_id = para_int(trecho[0])

            if carretel_id <= 0:
                idx += 9
                continue

            pressao_aspersor = para_float(trecho[1])
            gps_asp = trecho[2]
            pressao_carretel = para_float(trecho[3])
            gps_carr = trecho[5]
            velocidade = para_float(trecho[6])
            metragem = para_float(trecho[7])

            linhas.append({
                "timestamp": timestamp,
                "project_uuid": project_uuid,
                "project_numeric_id": project_numeric_id,
                "device": device,
                "tipo": "carretel",
                "equipamento_id": carretel_id,
                "bomba_operando": bomba_operando,
                "gps_valido": 1,
                "gps_aspersor_valido": gps_valido(gps_asp),
                "gps_carretel_valido": gps_valido(gps_carr),
                "pressao_num": -1,
                "rpm_num": -1,
                "pressao_aspersor_num": pressao_aspersor,
                "pressao_carretel_num": pressao_carretel,
                "velocidade_recolhimento_num": velocidade,
                "metragem_mangueira_num": metragem,
                "null_count": contar_nulls(trecho),
                "diagnostico_regras": diagnostico_regras(
                    tipo="carretel",
                    gps_aspersor=gps_asp,
                    gps_carretel=gps_carr,
                    pressao_aspersor=pressao_aspersor,
                    pressao_carretel=pressao_carretel,
                    velocidade=velocidade,
                    metragem=metragem,
                    bomba_operando=bomba_operando,
                ),
            })
            idx += 9

    if not linhas:
        raise ValueError("Não foi possível extrair equipamentos do JSON informado.")

    return pd.DataFrame(linhas)


def diagnostico_regras(**kwargs) -> str:
    """Diagnóstico auxiliar baseado em regras simples para explicar o resultado ao usuário."""
    problemas = []
    tipo = kwargs.get("tipo")
    bomba_operando = kwargs.get("bomba_operando", 0)

    if tipo == "bomba":
        if gps_valido(kwargs.get("gps_bomba")) == 0:
            problemas.append("GPS da bomba veio Null/inválido")
        if kwargs.get("pressao_bomba", -1) == 0:
            problemas.append("pressão da bomba zerada")
        if bomba_operando == 1 and kwargs.get("rpm", -1) <= 0:
            problemas.append("bomba marcada como operando, mas RPM zerado")

    if tipo == "carretel":
        if gps_valido(kwargs.get("gps_aspersor")) == 0:
            problemas.append("GPS do aspersor veio Null/inválido")
        if gps_valido(kwargs.get("gps_carretel")) == 0:
            problemas.append("GPS do carretel veio Null/inválido")
        if kwargs.get("pressao_aspersor", -1) == 0:
            problemas.append("pressão do aspersor zerada")
        if kwargs.get("pressao_carretel", -1) == 0:
            problemas.append("pressão do carretel zerada")
        if bomba_operando == 1 and kwargs.get("velocidade", -1) == 0 and kwargs.get("metragem", -1) == 0:
            problemas.append("recolhimento zerado com bomba operando")

    return "OK pelas regras básicas" if not problemas else "; ".join(problemas)


def preparar_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df.copy()
    for col in features:
        if col not in X.columns:
            X[col] = -1
    X = X[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1)
    return X


def prever(modelo, X: pd.DataFrame, encoder) -> List[str]:
    pred = modelo.predict(X)
    return list(encoder.inverse_transform(pred))


def prever_probabilidade(modelo, X: pd.DataFrame, encoder) -> List[float]:
    if not hasattr(modelo, "predict_proba"):
        return [0.0] * len(X)
    probas = modelo.predict_proba(X)
    return [float(max(p) * 100) for p in probas]


def status_visual(label: str) -> str:
    return "✅ Sem problema" if label == "OK" else "⚠️ Possível problema"


def main():
    st.set_page_config(
        page_title="ML IoT - Comparação de Modelos",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 Machine Learning aplicado à IoT")
    st.subheader("Comparação entre Árvore de Decisão e Random Forest")

    st.write(
        "Cole um JSON recebido do equipamento. O sistema extrai as leituras, "
        "classifica cada bomba/carretel e compara o resultado dos dois modelos."
    )

    arvore, floresta, encoder, features = carregar_modelos()

    with st.sidebar:
        st.header("Modelos carregados")
        st.success("Árvore de Decisão")
        st.success("Random Forest")
        st.write("Classes possíveis:")
        st.dataframe(pd.DataFrame({"classe": list(encoder.classes_)}), hide_index=True)

        metricas_path = os.path.join(PASTA_MODELOS, "metricas_modelos.csv")
        if os.path.exists(metricas_path):
            st.write("Métricas do treinamento:")
            metricas = pd.read_csv(metricas_path)
            st.dataframe(metricas, hide_index=True)

    texto_json = st.text_area(
        "JSON de entrada",
        value=EXEMPLO_JSON,
        height=260,
    )

    analisar = st.button("Analisar JSON", type="primary")

    if analisar:
        try:
            payload = json.loads(texto_json)
            df_linhas = extrair_linhas_do_json(payload)
            X = preparar_features(df_linhas, features)

            df_linhas["resultado_arvore_decisao"] = prever(arvore, X, encoder)
            df_linhas["confianca_arvore_%"] = prever_probabilidade(arvore, X, encoder)
            df_linhas["resultado_random_forest"] = prever(floresta, X, encoder)
            df_linhas["confianca_random_forest_%"] = prever_probabilidade(floresta, X, encoder)
            df_linhas["status_arvore"] = df_linhas["resultado_arvore_decisao"].apply(status_visual)
            df_linhas["status_random_forest"] = df_linhas["resultado_random_forest"].apply(status_visual)
            df_linhas["modelos_concordam"] = (
                df_linhas["resultado_arvore_decisao"] == df_linhas["resultado_random_forest"]
            )

            total = len(df_linhas)
            falhas_arvore = int((df_linhas["resultado_arvore_decisao"] != "OK").sum())
            falhas_floresta = int((df_linhas["resultado_random_forest"] != "OK").sum())
            concordancia = float(df_linhas["modelos_concordam"].mean() * 100)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equipamentos analisados", total)
            col2.metric("Falhas - Árvore", falhas_arvore)
            col3.metric("Falhas - Random Forest", falhas_floresta)
            col4.metric("Concordância", f"{concordancia:.1f}%")

            st.markdown("### Resultado comparativo por equipamento")
            colunas_exibir = [
                "tipo",
                "equipamento_id",
                "status_arvore",
                "resultado_arvore_decisao",
                "confianca_arvore_%",
                "status_random_forest",
                "resultado_random_forest",
                "confianca_random_forest_%",
                "modelos_concordam",
                "diagnostico_regras",
            ]
            st.dataframe(
                df_linhas[colunas_exibir].style.format({
                    "confianca_arvore_%": "{:.2f}",
                    "confianca_random_forest_%": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Interpretação")
            for _, linha in df_linhas.iterrows():
                equipamento = f"{linha['tipo']} ID {linha['equipamento_id']}"
                if linha["resultado_random_forest"] == "OK" and linha["resultado_arvore_decisao"] == "OK":
                    st.success(f"{equipamento}: os dois modelos classificaram como OK.")
                elif linha["modelos_concordam"]:
                    st.warning(
                        f"{equipamento}: os dois modelos indicaram {linha['resultado_random_forest']}. "
                        f"Observação: {linha['diagnostico_regras']}"
                    )
                else:
                    st.info(
                        f"{equipamento}: os modelos divergiram. "
                        f"Árvore: {linha['resultado_arvore_decisao']} | "
                        f"Random Forest: {linha['resultado_random_forest']}. "
                        f"Regras auxiliares: {linha['diagnostico_regras']}"
                    )

            st.markdown("### Features enviadas para os modelos")
            st.dataframe(X, use_container_width=True, hide_index=True)

            csv = df_linhas.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Baixar resultado em CSV",
                data=csv,
                file_name="resultado_classificacao_iot.csv",
                mime="text/csv",
            )

        except json.JSONDecodeError as erro:
            st.error(f"JSON inválido: {erro}")
        except Exception as erro:
            st.error(f"Erro ao analisar o JSON: {erro}")


if __name__ == "__main__":
    main()
