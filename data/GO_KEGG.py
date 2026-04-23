# -*- coding: utf-8 -*-

import io
import time
import requests
import pandas as pd

# ========= 这里直接改输入输出文件名 =========
input_file = "ab_output.emapper.annotations"
output_file = "ab_output_go_kegg_mapped.tsv"
# ==========================================


def read_eggnog_table(path):
    """读取 eggNOG 注释表，跳过 ## 开头的元信息行"""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##"):
                continue
            lines.append(line)

    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str).fillna("")

    if "#query" in df.columns:
        df = df.rename(columns={"#query": "gene_id"})

    return df


def split_ids(x):
    """拆分逗号分隔的编号"""
    x = str(x).strip()
    if x in {"", "-", "nan", "None"}:
        return []
    return [i.strip() for i in x.split(",") if i.strip() and i.strip() != "-"]


def fetch_go_names(go_ids):
    """
    批量查询 GO 名称
    返回:
        {"GO:0000166": "nucleotide binding (GO:0000166)"}
    """
    if not go_ids:
        return {}

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    result = {}

    batch_size = 100
    for i in range(0, len(go_ids), batch_size):
        batch = go_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids_str}"
        r = session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()

        for item in data.get("results", []):
            go_id = item.get("id", "")
            name = item.get("name", "")
            if go_id:
                result[go_id] = f"{name} ({go_id})"

        time.sleep(0.1)

    return result


def fetch_kegg_pathway_names(pathway_ids):
    """
    逐个查询 KEGG pathway 名称
    返回:
        {"ko02020": "Two-component system (ko02020)"}
    """
    if not pathway_ids:
        return {}
    session = requests.Session()
    result = {}
    for pid in pathway_ids:
        try:
            url = f"https://rest.kegg.jp/list/{pid}"
            r = session.get(url, timeout=60)

            if r.status_code == 200 and r.text.strip():
                line = r.text.strip().split("\n")[0]
                parts = line.split("\t")
                if len(parts) >= 2:
                    kegg_id = parts[0].strip()
                    desc = parts[1].strip()
                    result[pid] = f"{desc} ({kegg_id})"
                else:
                    result[pid] = pid
            else:
                result[pid] = pid

            time.sleep(0.1)

        except Exception:
            result[pid] = pid
    return result
def map_ids_to_names(id_string, mapping):
    """
    把一列里的多个ID映射成 'name (ID)' 格式
    """
    ids = split_ids(id_string)
    if not ids:
        return ""
    return "; ".join([mapping.get(x, x) for x in ids])

# ========= 主程序 =========
df = read_eggnog_table(input_file)

# 只保留三列
keep_cols = ["gene_id", "KEGG_Pathway", "GOs"]
df = df[[c for c in keep_cols if c in df.columns]].copy()

# 收集所有唯一 GO 和 KEGG pathway 编号
all_go_ids = sorted(set(i for x in df["GOs"] for i in split_ids(x))) if "GOs" in df.columns else []
all_kegg_ids = sorted(set(i for x in df["KEGG_Pathway"] for i in split_ids(x))) if "KEGG_Pathway" in df.columns else []

print(f"GO IDs found: {len(all_go_ids)}")
print(f"KEGG pathways found: {len(all_kegg_ids)}")

# 调用 API 查询
go_map = fetch_go_names(all_go_ids)
kegg_map = fetch_kegg_pathway_names(all_kegg_ids)

# 生成英文说明列
if "GOs" in df.columns:
    df["GO_terms_en"] = df["GOs"].apply(lambda x: map_ids_to_names(x, go_map))

if "KEGG_Pathway" in df.columns:
    df["KEGG_pathways_en"] = df["KEGG_Pathway"].apply(lambda x: map_ids_to_names(x, kegg_map))

# 保存
df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")
print(f"Done. Saved to: {output_file}")