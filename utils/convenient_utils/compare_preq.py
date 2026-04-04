#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量比较 p1_{taskid}_{strict/loose}.log 与 p1_woP_{taskid}_{strict/loose}.log 的分子相似度，
并输出：
1）所有 sample pair 的相似度与性质变化到一个 Excel（多 sheet）
2）每一对文件一张图片（所有重复 sample 的左右两列分子）
3）每一对文件的相似度均值 ± 标准差
4）按 taskid 的目标性质变化量均值 ± 标准差（base / wop / 差值）

依赖：
- rdkit
- pandas
- pillow (PIL)

运行方式示例（在项目根目录）：
    python batch_mol_similarity.py
"""

import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem import Crippen, QED, Descriptors, Lipinski


# ===================== 配置区域 =====================

# 非 woP 文件所在目录：p1_{taskid}_{strict/loose}.log
BASE_DIR = os.path.join("results", "opt_results", "gpt_35", "new", "1")

# woP 文件所在目录：p1_woP_{taskid}_{strict/loose}.log
WOP_DIR = os.path.join("results", "opt_results", "gpt_35", "new", "ablation", "P")

# 输出目录
OUT_DIR = os.path.join("results", "opt_results", "compare", "preq_mol_sim_output")
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_EXCEL_PATH = os.path.join(OUT_DIR, "molecule_similarity.xlsx")


# ===================== 任务 -> 性质与方向 =====================

def prop_direc_from_taskid(taskid: int):
    if taskid == 101:
        return ["logp"], ["decrease"]
    elif taskid == 102:
        return ["logp"], ["increase"]
    elif taskid == 103:
        return ["qed"], ["increase"]
    elif taskid == 104:
        return ["qed"], ["decrease"]
    elif taskid == 105:
        return ["tpsa"], ["decrease"]
    elif taskid == 106:
        return ["tpsa"], ["increase"]
    elif taskid == 107:
        return ["NumHAcceptors"], ["increase"]
    elif taskid == 108:
        return ["NumHDonors"], ["increase"]
    elif taskid == 201:
        return ["logp", "NumHAcceptors"], ["decrease", "increase"]
    elif taskid == 202:
        return ["logp", "NumHAcceptors"], ["increase", "increase"]
    elif taskid == 203:
        return ["logp", "NumHDonors"], ["decrease", "increase"]
    elif taskid == 204:
        return ["logp", "NumHDonors"], ["increase", "increase"]
    elif taskid == 205:
        return ["logp", "tpsa"], ["decrease", "decrease"]
    elif taskid == 206:
        return ["logp", "tpsa"], ["increase", "increase"]
    else:
        return [], []


# ===================== 工具函数 =====================

SAMPLE_RE = re.compile(r">>>>> Sample\s+(\d+):")


def ensure_dirs() -> None:
    os.makedirs(OUT_IMG_DIR, exist_ok=True)


def parse_sample_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SAMPLE_RE.match(line.strip())
            if m:
                ids.append(m.group(1))
    return ids


def parse_input_smiles(path: str, target_ids: Set[str]) -> Dict[str, str]:
    """
    从 Sample 行中提取输入分子 SMILES：">>>>> Sample {id}: <SMILES>"
    仅对 target_ids 中的 sample_id 提取。
    """
    res: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            m = SAMPLE_RE.match(t)
            if m:
                sid = m.group(1)
                if sid in target_ids and ":" in t:
                    smi_part = t.split(":", 1)[1].strip()
                    if smi_part:
                        smi = smi_part.split()[0]
                        res[sid] = smi
    return res


def parse_last_generated(path: str, target_ids: Set[str]) -> Dict[str, str]:
    """
    对指定文件，返回 {sample_id: 最后一个 Generated Result 的 SMILES}
    仅对 target_ids 中的 sample_id 提取。
    """
    res: Dict[str, str] = {}
    cur = None
    last = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            m = SAMPLE_RE.match(t)
            if m:
                if cur in target_ids and last is not None:
                    res[cur] = last
                cur = m.group(1)
                last = None
                continue
            if t.startswith("Generated Result:"):
                last = t[len("Generated Result:"):].strip()
        if cur in target_ids and last is not None:
            res[cur] = last
    return res


def parse_prequery_smiles(path: str, target_ids: Set[str], mode: str) -> Dict[str, str]:
    per_sample: Dict[str, List[str]] = {sid: [] for sid in target_ids}
    cur = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            m = SAMPLE_RE.match(t)
            if m:
                cur = m.group(1)
                continue
            if cur in target_ids and t.startswith("Pre-query DB:"):
                seg = t[len("Pre-query DB:"):].strip()
                if seg:
                    per_sample[cur].append(seg)

    res: Dict[str, str] = {}
    idx_mode = 0 if mode == "loose" else 1

    for sid, entries in per_sample.items():
        if not entries:
            continue

        # -------- 关键策略 --------
        # 两条：按 loose/strict 索引取
        # 一条：不区分 loose/strict，都用这一条
        if len(entries) == 1:
            seg = entries[0]
        else:
            if len(entries) <= idx_mode:
                continue
            seg = entries[idx_mode]
        # --------------------------

        first = seg.split()[0].strip()
        if not first or first.lower().startswith("fail"):
            continue

        res[sid] = first

    return res



def calc_props(mol: Chem.Mol) -> Dict[str, float]:
    """
    计算需要的分子性质：
    - logp
    - qed
    - tpsa
    - NumHAcceptors
    - NumHDonors
    """
    if mol is None:
        return {
            "logp": float("nan"),
            "qed": float("nan"),
            "tpsa": float("nan"),
            "NumHAcceptors": float("nan"),
            "NumHDonors": float("nan"),
        }
    return {
        "logp": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "tpsa": Descriptors.TPSA(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
    }


def process_file_pair(
    base_file: str, wop_file: str
) -> Tuple[List[dict], str, dict, List[dict]]:
    """
    对一对文件（base 与 woP），返回：
    - rows: 用于写 Excel 的合法样本记录（包含相似度和性质变化）
    - img_path: 该文件对的合成图片路径（如没有可画的 sample，则为 ""）
    - debug_info: 该文件对的统计信息（用于 debug_pairs）
    - invalid_rows: RDKit 解析失败的样本记录（用于 invalid_smiles）
    """
    base_path = os.path.join(BASE_DIR, base_file)
    wop_path = os.path.join(WOP_DIR, wop_file)

    debug_info = {
        "base_file": base_file,
        "wop_file": wop_file,
        "task_id": None,
        "mode": None,
        "n_ids_base": 0,
        "n_ids_wop": 0,
        "n_ids_common": 0,
        "n_rows_final": 0,
        "n_missing_input": 0,
        "n_missing_base_gen": 0,
        "n_missing_wop_gen": 0,
        "n_invalid_input_mol": 0,
        "n_invalid_base_mol": 0,
        "n_invalid_wop_mol": 0,
    }

    invalid_rows: List[dict] = []

    if not (os.path.isfile(base_path) and os.path.isfile(wop_path)):
        return [], "", debug_info, invalid_rows

    ids_base = set(parse_sample_ids(base_path))
    ids_wop = set(parse_sample_ids(wop_path))
    common_ids = sorted(list(ids_base & ids_wop), key=int)

    debug_info["n_ids_base"] = len(ids_base)
    debug_info["n_ids_wop"] = len(ids_wop)
    debug_info["n_ids_common"] = len(common_ids)

    if not common_ids:
        return [], "", debug_info, invalid_rows

    common_set = set(common_ids)

    mfile = re.match(r"p1_(\d+)_(strict|loose)\.log$", base_file)
    task_id = mfile.group(1) if mfile else ""
    mode = mfile.group(2) if mfile else ""
    debug_info["task_id"] = task_id
    debug_info["mode"] = mode

    input_smiles = parse_input_smiles(base_path, common_set)
    gen_wop = parse_last_generated(wop_path, common_set)
    pq_base = parse_prequery_smiles(base_path, common_set, mode)

    rows: List[dict] = []
    mols_for_grid: List[Chem.Mol] = []
    legends: List[str] = []

    for sid in common_ids:
        missing = False
        if sid not in input_smiles:
            debug_info["n_missing_input"] += 1
            missing = True
        if sid not in pq_base:
            debug_info["n_missing_base_gen"] += 1
            missing = True
        if sid not in gen_wop:
            debug_info["n_missing_wop_gen"] += 1
            missing = True
        if missing:
            continue

        smi_input = input_smiles[sid]
        smi_base = pq_base[sid]
        smi_wop = gen_wop[sid]

        m_input = Chem.MolFromSmiles(smi_input)
        m_base = Chem.MolFromSmiles(smi_base)
        m_wop = Chem.MolFromSmiles(smi_wop)

        invalid_flags = {
            "input_invalid": m_input is None,
            "base_invalid": m_base is None,
            "wop_invalid": m_wop is None,
        }

        if any(invalid_flags.values()):
            if invalid_flags["input_invalid"]:
                debug_info["n_invalid_input_mol"] += 1
            if invalid_flags["base_invalid"]:
                debug_info["n_invalid_base_mol"] += 1
            if invalid_flags["wop_invalid"]:
                debug_info["n_invalid_wop_mol"] += 1

            invalid_rows.append(
                {
                    "task_id": task_id,
                    "mode": mode,
                    "base_file": base_file,
                    "wop_file": wop_file,
                    "sample_id": sid,
                    "smiles_input": smi_input,
                    "smiles_base_generated": smi_base,
                    "smiles_wop_generated": smi_wop,
                    "input_invalid": invalid_flags["input_invalid"],
                    "base_invalid": invalid_flags["base_invalid"],
                    "wop_invalid": invalid_flags["wop_invalid"],
                }
            )
            continue

        fp_base = AllChem.GetMorganFingerprintAsBitVect(m_base, 2, 2048)
        fp_wop = AllChem.GetMorganFingerprintAsBitVect(m_wop, 2, 2048)
        tanimoto = DataStructs.TanimotoSimilarity(fp_base, fp_wop)
        dice = DataStructs.DiceSimilarity(fp_base, fp_wop)

        props_input = calc_props(m_input)
        props_wop = calc_props(m_wop)
        props_base = calc_props(m_base)

        row = {
            "task_id": task_id,
            "mode": mode,
            "base_file": base_file,
            "wop_file": wop_file,
            "sample_id": sid,
            "smiles_input": smi_input,
            "smiles_wop_generated": smi_wop,
            "smiles_base_generated": smi_base,
            "tanimoto_wop_vs_base": tanimoto,
            "dice_wop_vs_base": dice,
        }

        for prop_key in ["logp", "qed", "tpsa", "NumHAcceptors", "NumHDonors"]:
            v_in = props_input[prop_key]
            v_b = props_base[prop_key]
            v_w = props_wop[prop_key]

            row[f"{prop_key}_input"] = v_in
            row[f"{prop_key}_base"] = v_b
            row[f"{prop_key}_wop"] = v_w

            row[f"delta_base_{prop_key}"] = v_b - v_in
            row[f"delta_wop_{prop_key}"] = v_w - v_in
            row[f"delta_diff_{prop_key}"] = row[f"delta_wop_{prop_key}"] - row[f"delta_base_{prop_key}"]

        rows.append(row)

        mols_for_grid.append(m_wop)
        mols_for_grid.append(m_base)
        legends.append(f"{sid}-woP")
        legends.append(f"{sid}-base")

    debug_info["n_rows_final"] = len(rows)

    if not rows or not mols_for_grid:
        return rows, "", debug_info, invalid_rows

    img = Draw.MolsToGridImage(
        mols_for_grid,
        molsPerRow=2,
        subImgSize=(250, 250),
        legends=legends,
    )

    out_name = f"{os.path.splitext(base_file)[0]}__vs__{os.path.splitext(wop_file)[0]}.png"
    img_path = os.path.join(OUT_IMG_DIR, out_name)
    img.save(img_path)

    return rows, img_path, debug_info, invalid_rows


def main():
    ensure_dirs()

    all_rows: List[dict] = []
    debug_rows: List[dict] = []
    invalid_all: List[dict] = []

    for fname in os.listdir(BASE_DIR):
        m = re.match(r"p1_(\d+)_(strict|loose)\.log$", fname)
        if not m:
            continue
        task_id, mode = m.group(1), m.group(2)
        wop_name = f"p1_woP_{task_id}_{mode}.log"
        wop_path = os.path.join(WOP_DIR, wop_name)
        if not os.path.isfile(wop_path):
            continue

        rows, img_path, dbg, invalid_rows = process_file_pair(fname, wop_name)
        all_rows.extend(rows)
        debug_rows.append(dbg)
        invalid_all.extend(invalid_rows)

        print(
            f"Processed pair: {fname}  vs  {wop_name}  | "
            f"common_ids: {dbg['n_ids_common']}  | valid_rows: {dbg['n_rows_final']}  | "
            f"invalid_input/base/wop: {dbg['n_invalid_input_mol']}/"
            f"{dbg['n_invalid_base_mol']}/{dbg['n_invalid_wop_mol']}  | img: {img_path}"
        )

    if not all_rows and not invalid_all:
        print("没有任何可用的样本对，未生成 Excel。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    debug_df = pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame()
    invalid_df = pd.DataFrame(invalid_all) if invalid_all else pd.DataFrame()

    # 每对文件的相似度均值 ± 标准差（仅合法样本）
    if not df.empty:
        pair_stats = (
            df.groupby(["task_id", "mode", "base_file", "wop_file"])
            .agg(
                n_samples=("sample_id", "count"),
                tanimoto_mean=("tanimoto_wop_vs_base", "mean"),
                tanimoto_std=("tanimoto_wop_vs_base", "std"),
                dice_mean=("dice_wop_vs_base", "mean"),
                dice_std=("dice_wop_vs_base", "std"),
            )
            .reset_index()
        )
    else:
        pair_stats = pd.DataFrame()

    # 按 taskid 的目标性质变化量：base / wop / 差值（只用合法样本）
    task_prop_rows: List[dict] = []
    if not df.empty:
        for tid in sorted(df["task_id"].dropna().unique(), key=lambda x: int(x)):
            try:
                tid_int = int(tid)
            except Exception:
                continue
            props, direcs = prop_direc_from_taskid(tid_int)
            if not props:
                continue
            sub = df[df["task_id"] == tid]
            for prop, direc in zip(props, direcs):
                col_b = f"delta_base_{prop}"
                col_w = f"delta_wop_{prop}"
                col_d = f"delta_diff_{prop}"
                if col_b not in sub.columns or col_w not in sub.columns or col_d not in sub.columns:
                    continue
                vals_b = sub[col_b].dropna()
                vals_w = sub[col_w].dropna()
                vals_d = sub[col_d].dropna()
                if vals_b.empty or vals_w.empty or vals_d.empty:
                    continue
                task_prop_rows.append(
                    {
                        "task_id": tid,
                        "property": prop,
                        "direction_expected": direc,
                        "mean_delta_base": vals_b.mean(),
                        "std_delta_base": vals_b.std(ddof=1),
                        "mean_delta_wop": vals_w.mean(),
                        "std_delta_wop": vals_w.std(ddof=1),
                        "mean_delta_diff_wop_minus_base": vals_d.mean(),
                        "std_delta_diff_wop_minus_base": vals_d.std(ddof=1),
                        "n_samples": len(sub),
                    }
                )
    task_prop_df = pd.DataFrame(task_prop_rows)

    with pd.ExcelWriter(OUT_EXCEL_PATH) as writer:
        if not df.empty:
            df.to_excel(writer, index=False, sheet_name="samples")
            pair_stats.to_excel(writer, index=False, sheet_name="pair_stats")
            if not task_prop_df.empty:
                task_prop_df.to_excel(writer, index=False, sheet_name="task_prop_deltas")
        if not debug_df.empty:
            debug_df.to_excel(writer, index=False, sheet_name="debug_pairs")
        if not invalid_df.empty:
            invalid_df.to_excel(writer, index=False, sheet_name="invalid_smiles")

    print(f"结果已写入: {OUT_EXCEL_PATH}")
    print(f"配对图片保存在: {OUT_IMG_DIR}")


if __name__ == "__main__":
    main()

