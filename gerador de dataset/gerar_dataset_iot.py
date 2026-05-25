"""
Gerador de dataset sintético para telemetria RADCOM / motobombas / carreteis.

Entrada:
    - Planilha real de IDs: TABELA IDS 2025.xlsx

Saídas:
    - dataset_iot_normalizado.csv
    - dataset_iot_normalizado.xlsx
    - relatorios_json_sinteticos.jsonl

Instalação:
    pip install pandas openpyxl

Uso:
    python gerar_dataset_iot.py --ids "TABELA IDS 2025.xlsx" --amostras 1000

Observação:
    O script gera:
    1) uma tabela normalizada por equipamento;
    2) relatórios JSON sintéticos seguindo a gramática informada;
    3) labels de falhas para treinamento supervisionado.
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES PRINCIPAIS
# ============================================================

FALHAS_CARRETEL = [
    None,
    None,
    None,
    "CARRETEL_OFFLINE",
    "ASPERSOR_OFFLINE",
    "GPS_ASPERSOR_PROBLEMA",
    "PRESSAO_ASPERSOR_SUSPEITA",
    "GPS_CARRETEL_PROBLEMA",
    "RECOLHIMENTO_ZERADO_COM_BOMBA_OPERANDO",
]

FALHAS_BOMBA = [
    None,
    None,
    None,
    None,
    "BOMBA_OFFLINE",
    "GPS_BOMBA_PROBLEMA",
    "PRESSAO_BOMBA_NULA",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def is_empty(value):
    return pd.isna(value) or value is None or str(value).strip() == ""


def clean_text(value):
    if is_empty(value):
        return ""
    return str(value).strip()


def clean_id(value):
    if is_empty(value):
        return None
    try:
        return str(int(float(value)))
    except Exception:
        return clean_text(value)


def gps_random():
    lat = random.uniform(-19.410000, -19.370000)
    lon = random.uniform(-48.340000, -48.280000)
    return f"{lat:.6f}/{lon:.6f}"


def projeto_id_random():
    return str(random.randint(1, 99))


def safe_float(value):
    if value in [None, "Null"]:
        return None
    try:
        return float(value)
    except Exception:
        return None


def gps_valido(gps):
    return gps not in [None, "Null", ""] and "/" in str(gps)


# ============================================================
# LEITURA DA PLANILHA REAL DE IDs
# ============================================================

def classificar_tipo(nome):
    nome = clean_text(nome).lower()

    if "bom" in nome or "motobomba" in nome or "m.bom" in nome:
        return "bomba"

    if "carretel" in nome:
        return "carretel"

    if "canhao" in nome or "canhão" in nome:
        return "aspersor"

    if "gerador" in nome:
        return "gerador"

    if "hidrometro" in nome or "hidr" in nome:
        return "hidrometro"

    if "repetidora" in nome:
        return "repetidora"

    return "outro"


def extrair_modulos_planilha(caminho_xlsx):
    """
    Lê a planilha TABELA IDS 2025.xlsx e tenta extrair módulos automaticamente.

    Formato esperado, baseado no arquivo enviado:
        linha de cabeçalho com nomes dos equipamentos
        linha ID
        linha Canal
        linha N° Frota

    Retorna uma lista de módulos:
        [
            {
                "usina": "...",
                "modulo": "...",
                "equipamentos": [
                    {"nome": "M.Bom. 1", "tipo": "bomba", "id": "3", "canal": "5", "frota": "280039"},
                    ...
                ]
            }
        ]
    """
    xls = pd.ExcelFile(caminho_xlsx)
    modulos = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(caminho_xlsx, sheet_name=sheet_name, header=None)

        for i in range(len(df)):
            row = df.iloc[i].tolist()

            # Detecta linha de ID
            has_id_marker = any(clean_text(cell).lower() == "id" for cell in row)

            if not has_id_marker:
                continue

            # Cabeçalho geralmente está na linha anterior
            if i == 0:
                continue

            header = df.iloc[i - 1].tolist()
            canal_row = df.iloc[i + 1].tolist() if i + 1 < len(df) else []
            frota_row = df.iloc[i + 2].tolist() if i + 2 < len(df) else []

            usina = clean_text(row[0]) or clean_text(df.iloc[i - 1, 0]) or sheet_name
            modulo = clean_text(row[1]) or clean_text(row[0]) or f"MODULO_{i}"

            equipamentos = []

            for col in range(len(row)):
                nome_eq = clean_text(header[col]) if col < len(header) else ""
                id_eq = clean_id(row[col])

                if not id_eq:
                    continue

                # ignora a célula textual "ID"
                if str(id_eq).lower() == "id":
                    continue

                tipo = classificar_tipo(nome_eq)

                # ignora colunas sem nome de equipamento
                if tipo == "outro" and nome_eq == "":
                    continue

                canal = clean_id(canal_row[col]) if col < len(canal_row) else None
                frota = clean_id(frota_row[col]) if col < len(frota_row) else None

                equipamentos.append({
                    "nome": nome_eq,
                    "tipo": tipo,
                    "id": id_eq,
                    "canal": canal,
                    "frota": frota,
                })

            if equipamentos:
                modulos.append({
                    "sheet": sheet_name,
                    "usina": usina,
                    "modulo": modulo,
                    "equipamentos": equipamentos,
                })

    return modulos


def selecionar_bombas_e_carreteis(modulo):
    bombas = [e for e in modulo["equipamentos"] if e["tipo"] == "bomba"]
    carreteis = [e for e in modulo["equipamentos"] if e["tipo"] == "carretel"]
    aspersores = [e for e in modulo["equipamentos"] if e["tipo"] == "aspersor"]

    # Limita para a gramática atual: 2 bombas e até 3 carreteis.
    bombas = bombas[:2]
    carreteis = carreteis[:3]
    aspersores = aspersores[:3]

    # Caso algum módulo venha incompleto, gera fallback por padrão.
    if len(bombas) < 2:
        base = random.choice([10, 20, 30, 40, 50])
        bombas = [
            {"nome": "MotoBomba 1", "tipo": "bomba", "id": str(base), "canal": None, "frota": None},
            {"nome": "MotoBomba 2", "tipo": "bomba", "id": str(base + 1), "canal": None, "frota": None},
        ]

    if len(carreteis) < 3:
        base = int(bombas[0]["id"]) if str(bombas[0]["id"]).isdigit() else random.choice([10, 20, 30])
        carreteis = [
            {"nome": "Carretel 1", "tipo": "carretel", "id": str(base + 2), "canal": None, "frota": None},
            {"nome": "Carretel 2", "tipo": "carretel", "id": str(base + 4), "canal": None, "frota": None},
            {"nome": "Carretel 3", "tipo": "carretel", "id": str(base + 6), "canal": None, "frota": None},
        ]

    return bombas, carreteis, aspersores


# ============================================================
# GERAÇÃO DE EQUIPAMENTOS
# ============================================================

def gerar_bomba(eq, falha=None):
    if falha == "BOMBA_OFFLINE":
        return {
            "tipo": "bomba",
            "equipamento_id": eq["id"],
            "pressao": "Null",
            "rpm": "Null",
            "status_1": "Null",
            "status_2": "Null",
            "gps": "Null",
            "label": "BOMBA_OFFLINE",
        }

    pressao = round(random.uniform(1.20, 1.60), 2)
    rpm = random.choice([0, random.randint(700, 1100)])
    status_1 = random.randint(50, 90)
    status_2 = random.choice(["00000000", "00000100"])
    gps = gps_random()
    label = "OK"

    if falha == "GPS_BOMBA_PROBLEMA":
        gps = "Null"
        label = "GPS_BOMBA_PROBLEMA"

    elif falha == "PRESSAO_BOMBA_NULA":
        pressao = "Null"
        label = "PRESSAO_BOMBA_NULA"

    return {
        "tipo": "bomba",
        "equipamento_id": eq["id"],
        "pressao": pressao,
        "rpm": rpm,
        "status_1": status_1,
        "status_2": status_2,
        "gps": gps,
        "label": label,
    }


def gerar_carretel(eq, bomba_operando=False, falha=None):
    velocidade_ok = random.randint(1, 10) if bomba_operando else 0
    metragem = random.randint(1, 300)
    codigo = f"{random.randint(3000, 4999)}M00"

    if falha == "CARRETEL_OFFLINE":
        return {
            "tipo": "carretel",
            "equipamento_id": eq["id"],
            "pressao_aspersor": "0.00",
            "gps_aspersor": "Null",
            "pressao_carretel": "0.00",
            "status": "00000010",
            "gps_carretel": "Null",
            "velocidade_recolhimento": 0,
            "metragem_mangueira": 0,
            "codigo": "0M00",
            "label": "CARRETEL_OFFLINE",
        }

    pressao_aspersor = round(random.uniform(0.01, 0.20), 2)
    gps_aspersor = gps_random()
    pressao_carretel = round(random.uniform(0.08, 0.50), 2)
    status = "00000011"
    gps_carretel = gps_random()
    velocidade = velocidade_ok
    label = "OK"

    if falha == "ASPERSOR_OFFLINE":
        pressao_aspersor = "Null"
        gps_aspersor = "Null"
        label = "ASPERSOR_OFFLINE"

    elif falha == "GPS_ASPERSOR_PROBLEMA":
        pressao_aspersor = round(random.uniform(0.01, 0.20), 2)
        gps_aspersor = "Null"
        label = "GPS_ASPERSOR_PROBLEMA"

    elif falha == "PRESSAO_ASPERSOR_SUSPEITA":
        # Caso informado: 0.00 com coordenada válida tende a indicar sensor/pressão suspeita.
        pressao_aspersor = "0.00"
        gps_aspersor = gps_random()
        label = "PRESSAO_ASPERSOR_SUSPEITA"

    elif falha == "GPS_CARRETEL_PROBLEMA":
        gps_carretel = "Null"
        label = "GPS_CARRETEL_PROBLEMA"

    elif falha == "RECOLHIMENTO_ZERADO_COM_BOMBA_OPERANDO":
        # Se bomba está operando, velocidade não deveria ser zero.
        if bomba_operando:
            velocidade = 0
            label = "RECOLHIMENTO_ZERADO_COM_BOMBA_OPERANDO"
        else:
            label = "OK"

    return {
        "tipo": "carretel",
        "equipamento_id": eq["id"],
        "pressao_aspersor": pressao_aspersor,
        "gps_aspersor": gps_aspersor,
        "pressao_carretel": pressao_carretel,
        "status": status,
        "gps_carretel": gps_carretel,
        "velocidade_recolhimento": velocidade,
        "metragem_mangueira": metragem,
        "codigo": codigo,
        "label": label,
    }


# ============================================================
# CONVERSÃO PARA JSON DA GRAMÁTICA ORIGINAL
# ============================================================

def montar_vetor_json(timestamp, project_numeric_id, bombas, carreteis):
    """
    Monta o vetor interno do campo "dado".

    Gramática:
        [
            timestamp,
            project_numeric_id,
            "00.00", "00000000", "Null", "0",

            bomba1: id, pressao, rpm, status_1, status_2, gps
            bomba2: id, pressao, rpm, status_1, status_2, gps

            carretel1: id, pressao_aspersor, gps_aspersor, pressao_carretel, status,
                       gps_carretel, velocidade_recolhimento, metragem_mangueira, codigo

            carretel2: ...
            carretel3: ...
        ]
    """
    vetor = [
        timestamp,
        str(project_numeric_id),
        "00.00",
        "00000000",
        "Null",
        "0",
    ]

    for b in bombas:
        vetor.extend([
            str(b["equipamento_id"]),
            str(b["pressao"]),
            str(b["rpm"]),
            str(b["status_1"]),
            str(b["status_2"]),
            str(b["gps"]),
        ])

    for c in carreteis:
        vetor.extend([
            str(c["equipamento_id"]),
            str(c["pressao_aspersor"]),
            str(c["gps_aspersor"]),
            str(c["pressao_carretel"]),
            str(c["status"]),
            str(c["gps_carretel"]),
            str(c["velocidade_recolhimento"]),
            str(c["metragem_mangueira"]),
            str(c["codigo"]),
        ])

    return vetor


def gerar_amostra(modulo, timestamp):
    project_uuid = str(uuid.uuid4())
    project_numeric_id = projeto_id_random()
    device = f"Device/RADCOM_{random.randint(1000, 9999)}"

    bombas_cfg, carreteis_cfg, _ = selecionar_bombas_e_carreteis(modulo)

    bombas = []
    for bcfg in bombas_cfg:
        falha = random.choice(FALHAS_BOMBA)
        bombas.append(gerar_bomba(bcfg, falha))

    bomba_operando = any(
        safe_float(b["rpm"]) is not None and safe_float(b["rpm"]) > 0
        for b in bombas
    )

    carreteis = []
    for ccfg in carreteis_cfg:
        falha = random.choice(FALHAS_CARRETEL)
        carreteis.append(gerar_carretel(ccfg, bomba_operando, falha))

    vetor = montar_vetor_json(
        timestamp=timestamp,
        project_numeric_id=project_numeric_id,
        bombas=bombas,
        carreteis=carreteis,
    )

    payload = {
        "ver": "1",
        "projectId": project_uuid,
        "device": device,
        "dado": [vetor],
    }

    linhas_normalizadas = []

    for b in bombas:
        linhas_normalizadas.append({
            "timestamp": timestamp,
            "project_uuid": project_uuid,
            "project_numeric_id": project_numeric_id,
            "device": device,
            "sheet": modulo["sheet"],
            "usina": modulo["usina"],
            "modulo": modulo["modulo"],
            "tipo": "bomba",
            "equipamento_id": b["equipamento_id"],
            "pressao": b["pressao"],
            "rpm": b["rpm"],
            "status_1": b["status_1"],
            "status_2": b["status_2"],
            "gps": b["gps"],
            "pressao_aspersor": None,
            "gps_aspersor": None,
            "pressao_carretel": None,
            "gps_carretel": None,
            "velocidade_recolhimento": None,
            "metragem_mangueira": None,
            "codigo": None,
            "bomba_operando": int(bomba_operando),
            "gps_valido": int(gps_valido(b["gps"])),
            "label": b["label"],
        })

    for c in carreteis:
        linhas_normalizadas.append({
            "timestamp": timestamp,
            "project_uuid": project_uuid,
            "project_numeric_id": project_numeric_id,
            "device": device,
            "sheet": modulo["sheet"],
            "usina": modulo["usina"],
            "modulo": modulo["modulo"],
            "tipo": "carretel",
            "equipamento_id": c["equipamento_id"],
            "pressao": None,
            "rpm": None,
            "status_1": None,
            "status_2": c["status"],
            "gps": c["gps_carretel"],
            "pressao_aspersor": c["pressao_aspersor"],
            "gps_aspersor": c["gps_aspersor"],
            "pressao_carretel": c["pressao_carretel"],
            "gps_carretel": c["gps_carretel"],
            "velocidade_recolhimento": c["velocidade_recolhimento"],
            "metragem_mangueira": c["metragem_mangueira"],
            "codigo": c["codigo"],
            "bomba_operando": int(bomba_operando),
            "gps_valido": int(gps_valido(c["gps_carretel"])),
            "gps_aspersor_valido": int(gps_valido(c["gps_aspersor"])),
            "gps_carretel_valido": int(gps_valido(c["gps_carretel"])),
            "label": c["label"],
        })

    return payload, linhas_normalizadas


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def gerar_dataset(caminho_ids, qtd_amostras, pasta_saida):
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    modulos = extrair_modulos_planilha(caminho_ids)

    if not modulos:
        raise RuntimeError("Nenhum módulo foi encontrado na planilha de IDs.")

    inicio = datetime(2026, 5, 21, 11, 28, 13)

    todos_jsons = []
    todas_linhas = []

    for i in range(qtd_amostras):
        modulo = random.choice(modulos)
        timestamp = (inicio + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S,000")

        payload, linhas = gerar_amostra(modulo, timestamp)

        todos_jsons.append(payload)
        todas_linhas.extend(linhas)

    df = pd.DataFrame(todas_linhas)

    # Features auxiliares úteis para ML
    df["is_ok"] = (df["label"] == "OK").astype(int)
    df["is_falha"] = (df["label"] != "OK").astype(int)

    df["pressao_num"] = pd.to_numeric(df["pressao"], errors="coerce")
    df["rpm_num"] = pd.to_numeric(df["rpm"], errors="coerce")
    df["pressao_aspersor_num"] = pd.to_numeric(df["pressao_aspersor"], errors="coerce")
    df["pressao_carretel_num"] = pd.to_numeric(df["pressao_carretel"], errors="coerce")
    df["velocidade_recolhimento_num"] = pd.to_numeric(df["velocidade_recolhimento"], errors="coerce")
    df["metragem_mangueira_num"] = pd.to_numeric(df["metragem_mangueira"], errors="coerce")

    # null_count por linha/equipamento
    cols_check_null = [
        "pressao", "rpm", "gps",
        "pressao_aspersor", "gps_aspersor",
        "pressao_carretel", "gps_carretel",
        "velocidade_recolhimento", "metragem_mangueira"
    ]

    def contar_nulls(row):
        count = 0
        for col in cols_check_null:
            if col in row and str(row[col]) == "Null":
                count += 1
        return count

    df["null_count"] = df.apply(contar_nulls, axis=1)

    csv_path = pasta_saida / "dataset_iot_normalizado.csv"
    xlsx_path = pasta_saida / "dataset_iot_normalizado.xlsx"
    jsonl_path = pasta_saida / "relatorios_json_sinteticos.jsonl"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for payload in todos_jsons:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("Arquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")
    print(f"- {jsonl_path}")
    print()
    print("Resumo de labels:")
    print(df["label"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="Caminho da planilha TABELA IDS 2025.xlsx")
    parser.add_argument("--amostras", type=int, default=1000, help="Quantidade de pacotes JSON sintéticos")
    parser.add_argument("--saida", default="saida_dataset", help="Pasta de saída")

    args = parser.parse_args()

    gerar_dataset(
        caminho_ids=args.ids,
        qtd_amostras=args.amostras,
        pasta_saida=args.saida,
    )


if __name__ == "__main__":
    main()
