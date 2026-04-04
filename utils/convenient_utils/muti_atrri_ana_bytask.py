#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_attr_from_logs_to_excel_cn.py

功能：
    从多个 log 文件解析双属性任务（taskid 201~206）的输入/输出分子，
    对 chatdrug 与 p1(=ours) 做统计，并输出一个“大 Excel”（列名全中文）。

关键变化（相对你之前版本）：
1) 不再把解析/消毒/属性计算失败的样本“跳过不计”，而是统一记为“不合法分子/不可评估样本”；
   同时在 Excel 中输出：
      - 总样本数
      - 可评估样本数
      - 不合法样本数
      - 不合法样本占比（不合法/总样本）
2) loose / strict 两套阈值下的成功/失败结构、失败内部占比、case③(只对一个属性)失败属性Δ统计仍然保留，
   这些统计默认在“可评估样本”上计算（因为不合法样本无法计算属性与阈值判定）。
3) 代码内部支持 --logs 通配符模式（glob 展开）：
      --logs "xxx/chatdrug_20*_strict.log" "xxx/p1_20*_strict.log"

log 格式假设：
    >Sample 0: ...   或  >>>>> Sample 0: ...  （任意数量的 '>'）
    Generated Result: <smiles>
同一个 Sample 可能多个 Generated Result：取最后一个作为最终输出。

输出 Excel：
    Sheet 1: 总览_宽表
    Sheet 2: 总览_长表
    Sheet 3: 文件级汇总
"""

import argparse
import re
from glob import glob
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# 关闭 RDKit 控制台错误/警告输出（避免 parse/kekulize/valence 在终端刷屏）
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    try:
        from rdkit import rdBase
        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")
    except Exception:
        pass


# ========================
#  一、task → 属性 / 方向 / 阈值
# ========================

def prop_direc_from_taskid(taskid: int) -> Tuple[List[str], List[str]]:
    """taskid -> (props[2], directions[2]) for multi-attr tasks 201~206."""
    if taskid == 201:
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
        raise ValueError(f"taskid {taskid} 不是 201~206")


# 任务 -> [ [loose阈值2个], [strict阈值2个] ]
task2threshold_list = {
    201: [[0, 0], [0.5, 1]],
    202: [[0, 0], [0.5, 1]],
    203: [[0, 0], [0.5, 1]],
    204: [[0, 0], [0.5, 1]],
    205: [[0, 0], [0.5, 10]],
    206: [[0, 0], [0.5, 10]],
}


def direction_str_to_sign(d: str) -> int:
    """'increase'/'decrease' -> +1/-1."""
    if d == "increase":
        return 1
    if d == "decrease":
        return -1
    return 0


def direction_str_cn(d: str) -> str:
    """方向字符串转中文."""
    if d == "increase":
        return "增大"
    if d == "decrease":
        return "减小"
    return d


# ========================
#  二、log 解析
# ========================

# 兼容 >>Sample / >>>>> Sample 等任意数量 '>'
SAMPLE_RE = re.compile(r"^>+\s*Sample\s+(\d+)\s*:\s*(.*)\s*$")


def extract_smiles_from_sample_rest(rest: str) -> str:
    """
    解析 Sample 行里的 input 部分，支持：
      - "input_drug, <SMILES>"
      - "<SMILES>"
      - "xxx, yyy, <SMILES>" -> 取最后一个逗号后的字段
    """
    if rest is None:
        return ""
    s = rest.strip()
    if not s:
        return ""
    if "," in s:
        s = s.rsplit(",", 1)[-1].strip()
    return s


def parse_log(text: str) -> List[Dict[str, Any]]:
    """
    解析 log 文本为样本块：
      [
        {"sample": int, "input": str, "generated": [str,...]},
        ...
      ]
    """
    samples = []
    cur = None

    for line in text.splitlines():
        m = SAMPLE_RE.match(line)
        if m:
            if cur is not None:
                samples.append(cur)
            sid = int(m.group(1))
            inp = extract_smiles_from_sample_rest(m.group(2))
            cur = {"sample": sid, "input": inp, "generated": []}
            continue

        if cur is not None and line.startswith("Generated Result:"):
            smi = line.split("Generated Result:", 1)[1].strip()
            if smi:
                cur["generated"].append(smi)

    if cur is not None:
        samples.append(cur)

    return samples


def largest_fragment_mol(smiles: str):
    """
    SMILES -> RDKit Mol
    多片段取重原子数最大者；sanitize 失败返回 None。
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None
    if not frags:
        return None
    best = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(best)
    except Exception:
        return None
    return best


# ========================
#  三、属性计算
# ========================

def calc_logp(mol: Chem.Mol) -> float:
    return float(Descriptors.MolLogP(mol))


def calc_qed(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


def calc_tpsa(mol: Chem.Mol) -> float:
    return float(Descriptors.TPSA(mol))


def calc_hba(mol: Chem.Mol) -> float:
    return float(Descriptors.NumHAcceptors(mol))


def calc_hbd(mol: Chem.Mol) -> float:
    return float(Descriptors.NumHDonors(mol))


PROP2FUNC = {
    "logp": calc_logp,
    "qed": calc_qed,
    "tpsa": calc_tpsa,
    "NumHAcceptors": calc_hba,
    "NumHDonors": calc_hbd,
}


def calc_props_for_pair(in_mol: Chem.Mol, out_mol: Chem.Mol, props: List[str]) -> Tuple[List[float], List[float]]:
    """计算 (input_mol, output_mol) 的指定属性值列表。"""
    in_vals = []
    out_vals = []
    for p in props:
        func = PROP2FUNC.get(p)
        if func is None:
            raise ValueError(f"未知属性: {p}")
        in_vals.append(func(in_mol))
        out_vals.append(func(out_mol))
    return in_vals, out_vals


# ========================
#  四、记录结构与文件名推断
# ========================

@dataclass
class Record:
    """一个可评估的样本（双属性任务的输入/输出分子对）。"""
    model: str
    taskid: int
    sample: int
    props: List[str]       # length 2
    dir_strs: List[str]    # length 2
    in_vals: List[float]   # length 2
    out_vals: List[float]  # length 2


def guess_model_from_path(path: Path) -> str:
    """从文件名推断模型：chatdrug -> chatdrug, p1 -> ours。"""
    name = path.name.lower()
    if "chatdrug" in name:
        return "chatdrug"
    if "p1" in name:
        return "ours"
    prefix = path.stem.split("_")[0]
    return "ours" if prefix == "p1" else prefix


def guess_taskid_from_path(path: Path) -> int:
    """从文件名解析 taskid，如 chatdrug_205_strict.log."""
    m = re.search(r"_(\d{3})_", path.name)
    if not m:
        raise ValueError(f"无法从文件名解析 taskid: {path.name}")
    return int(m.group(1))


# ========================
#  五、从单个 log 提取 records + 不合法统计
# ========================

def extract_records_from_log(path: Path) -> Tuple[List[Record], Dict[str, Any]]:
    """
    解析单个 log 文件：
      - 返回可评估的 Record 列表
      - 返回文件级汇总（总样本数/可评估/不合法，以及不合法原因拆分）

    不合法样本定义：
      - 无 Generated Result
      - 输入或输出 SMILES 解析/消毒失败
      - 属性计算失败
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    samples = parse_log(text)

    model = guess_model_from_path(path)
    taskid = guess_taskid_from_path(path)

    info = {
        "文件": str(path),
        "模型": model,
        "任务ID": taskid,
        "样本块数量": len(samples),

        "可评估样本数": 0,
        "不合法样本数": 0,

        "不合法_无生成结果": 0,
        "不合法_输入分子无效": 0,
        "不合法_输出分子无效": 0,
        "不合法_属性计算失败": 0,
    }

    if taskid not in task2threshold_list:
        # 非 201~206 文件：全部不参与统计（但这里也不算“不合法”，避免污染）
        return [], info

    props, dir_strs = prop_direc_from_taskid(taskid)

    records: List[Record] = []
    for s in samples:
        inp_smiles = s["input"]
        gen_list = s["generated"]

        if not gen_list:
            info["不合法样本数"] += 1
            info["不合法_无生成结果"] += 1
            continue

        out_smiles = gen_list[-1]

        in_mol = largest_fragment_mol(inp_smiles)
        if in_mol is None:
            info["不合法样本数"] += 1
            info["不合法_输入分子无效"] += 1
            continue

        out_mol = largest_fragment_mol(out_smiles)
        if out_mol is None:
            info["不合法样本数"] += 1
            info["不合法_输出分子无效"] += 1
            continue

        try:
            in_vals, out_vals = calc_props_for_pair(in_mol, out_mol, props)
        except Exception:
            info["不合法样本数"] += 1
            info["不合法_属性计算失败"] += 1
            continue

        records.append(
            Record(
                model=model,
                taskid=taskid,
                sample=int(s["sample"]),
                props=props,
                dir_strs=dir_strs,
                in_vals=in_vals,
                out_vals=out_vals,
            )
        )
        info["可评估样本数"] += 1

    return records, info


# ========================
#  六、统计（在“可评估样本”上计算）
# ========================

def stats_array(arr: List[float]) -> Dict[str, float]:
    """float 列表基础统计：样本数/均值/标准差/最小/最大。"""
    import math
    if not arr:
        return {"样本数": 0, "均值": 0.0, "标准差": 0.0, "最小值": 0.0, "最大值": 0.0}
    n = len(arr)
    mean = sum(arr) / n
    var = sum((x - mean) ** 2 for x in arr) / n
    return {
        "样本数": n,
        "均值": mean,
        "标准差": math.sqrt(var),
        "最小值": min(arr),
        "最大值": max(arr),
    }


def analyze_for_mode(records: List[Record], mode: str) -> Dict[str, Any]:
    """
    阈值达标口径（loose/strict）：
      - 四类计数：
          两个属性均达标 / 两个属性均失败 / 仅属性1达标 / 仅属性2达标
      - 失败样本内部占比（分母=失败样本=至少一个属性失败）
      - 仅对一个属性（单属性成功）时，失败属性的 Δ(out-in) 统计
    """
    assert mode in ("loose", "strict")
    idx = 0 if mode == "loose" else 1
    mode_cn = "宽松" if mode == "loose" else "严格"

    cat = Counter()
    failed_prop1_deltas = []  # 仅属性2达标 时，属性1失败的 Δ1
    failed_prop2_deltas = []  # 仅属性1达标 时，属性2失败的 Δ2

    for r in records:
        thresholds = task2threshold_list[r.taskid][idx]  # length 2

        ok = []
        for i in range(2):
            d = r.dir_strs[i]
            delta = r.out_vals[i] - r.in_vals[i]
            th = thresholds[i]
            if d == "increase":
                ok.append(delta >= th)
            else:  # decrease
                ok.append((r.in_vals[i] - r.out_vals[i]) >= th)

        p1_ok, p2_ok = ok[0], ok[1]

        if p1_ok and p2_ok:
            cat["两个属性均达标"] += 1
        elif (not p1_ok) and (not p2_ok):
            cat["两个属性均失败"] += 1
        elif p1_ok and (not p2_ok):
            cat["仅属性1达标"] += 1
            failed_prop2_deltas.append(r.out_vals[1] - r.in_vals[1])
        else:
            cat["仅属性2达标"] += 1
            failed_prop1_deltas.append(r.out_vals[0] - r.in_vals[0])

    # 失败样本内部占比：分母=两个属性均失败 + 仅属性1达标 + 仅属性2达标
    c_both_fail = cat["两个属性均失败"]
    c_only_p1 = cat["仅属性1达标"]
    c_only_p2 = cat["仅属性2达标"]
    fail_total = c_both_fail + c_only_p1 + c_only_p2

    if fail_total == 0:
        fail_ratios = {"两属性均失败占比": 0.0, "仅属性1达标占比": 0.0, "仅属性2达标占比": 0.0}
    else:
        fail_ratios = {
            "两属性均失败占比": c_both_fail / fail_total,
            "仅属性1达标占比": c_only_p1 / fail_total,
            "仅属性2达标占比": c_only_p2 / fail_total,
        }

    return {
        f"{mode_cn}_四类计数": dict(cat),
        f"{mode_cn}_失败内部占比": fail_ratios,
        f"{mode_cn}_单属性成功时失败属性Δ统计": {
            "属性1失败_Δ(out-in)": stats_array(failed_prop1_deltas),
            "属性2失败_Δ(out-in)": stats_array(failed_prop2_deltas),
        },
    }


def analyze_direction(records: List[Record]) -> Dict[str, Any]:
    """
    方向正确性口径（不看阈值）：
      - 四类计数：
          两属性方向均正确 / 两属性方向均错误 / 仅属性1方向正确 / 仅属性2方向正确
      - 方向失败内部占比（分母=方向失败样本=至少一个属性方向错）
    """
    cat = Counter()

    for r in records:
        dirs = [direction_str_to_sign(d) for d in r.dir_strs]
        deltas = [r.out_vals[i] - r.in_vals[i] for i in range(2)]
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in deltas]

        p1_ok = (signs[0] == dirs[0])
        p2_ok = (signs[1] == dirs[1])

        if p1_ok and p2_ok:
            cat["两属性方向均正确"] += 1
        elif (not p1_ok) and (not p2_ok):
            cat["两属性方向均错误"] += 1
        elif p1_ok and (not p2_ok):
            cat["仅属性1方向正确"] += 1
        else:
            cat["仅属性2方向正确"] += 1

    b = cat["两属性方向均错误"]
    c = cat["仅属性1方向正确"]
    d = cat["仅属性2方向正确"]
    fail_total = b + c + d

    if fail_total == 0:
        ratios = {"两属性方向均错误占比": 0.0, "仅属性1方向正确占比": 0.0, "仅属性2方向正确占比": 0.0}
    else:
        ratios = {
            "两属性方向均错误占比": b / fail_total,
            "仅属性1方向正确占比": c / fail_total,
            "仅属性2方向正确占比": d / fail_total,
        }

    return {
        "方向四类计数": dict(cat),
        "方向失败内部占比": ratios,
    }


def run_for_model(records: List[Record]) -> Dict[str, Any]:
    """对一个模型的一组可评估 records 做完整统计。"""
    out = {}
    out.update(analyze_for_mode(records, "loose"))
    out.update(analyze_for_mode(records, "strict"))
    out.update(analyze_direction(records))
    return out


# ========================
#  七、Excel 组织（列名全中文）
# ========================

def build_meta_for_scope(scope: str) -> Dict[str, Any]:
    """
    scope 为 ALL 或 task201..task206：
      - ALL：不填属性/方向/阈值
      - taskXXX：填属性1/属性2/方向/阈值
    """
    if scope == "ALL":
        return {
            "任务ID": "",
            "属性1": "",
            "属性2": "",
            "属性1方向": "",
            "属性2方向": "",
            "宽松阈值1": "",
            "宽松阈值2": "",
            "严格阈值1": "",
            "严格阈值2": "",
        }
    m = re.match(r"^task(\d{3})$", scope)
    if not m:
        return {
            "任务ID": "",
            "属性1": "",
            "属性2": "",
            "属性1方向": "",
            "属性2方向": "",
            "宽松阈值1": "",
            "宽松阈值2": "",
            "严格阈值1": "",
            "严格阈值2": "",
        }
    tid = int(m.group(1))
    props, dirs = prop_direc_from_taskid(tid)
    th_loose = task2threshold_list[tid][0]
    th_strict = task2threshold_list[tid][1]
    return {
        "任务ID": tid,
        "属性1": props[0],
        "属性2": props[1],
        "属性1方向": direction_str_cn(dirs[0]),
        "属性2方向": direction_str_cn(dirs[1]),
        "宽松阈值1": th_loose[0],
        "宽松阈值2": th_loose[1],
        "严格阈值1": th_strict[0],
        "严格阈值2": th_strict[1],
    }


def flatten_stats_to_wide_row_cn(
    scope: str,
    model: str,
    stats: Dict[str, Any],
    total_samples: int,
    valid_samples: int,
    invalid_samples: int,
) -> Dict[str, Any]:
    """
    把嵌套统计拍平成一行（宽表，中文列名）。
    """
    row = {
        "范围": scope,
        "模型": model,
        **build_meta_for_scope(scope),
        "总样本数": total_samples,
        "可评估样本数": valid_samples,
        "不合法样本数": invalid_samples,
        "不合法样本占比": (invalid_samples / total_samples) if total_samples > 0 else 0.0,
    }

    # 宽松/严格：四类计数 + 失败内部占比 + 单属性成功时失败属性Δ统计
    for mode_cn in ("宽松", "严格"):
        cnt = stats.get(f"{mode_cn}_四类计数", {})
        row[f"{mode_cn}_两个属性均达标_数量"] = cnt.get("两个属性均达标", 0)
        row[f"{mode_cn}_两个属性均失败_数量"] = cnt.get("两个属性均失败", 0)
        row[f"{mode_cn}_仅属性1达标_数量"] = cnt.get("仅属性1达标", 0)
        row[f"{mode_cn}_仅属性2达标_数量"] = cnt.get("仅属性2达标", 0)

        ratios = stats.get(f"{mode_cn}_失败内部占比", {})
        row[f"{mode_cn}_失败样本内_两属性均失败_占比"] = ratios.get("两属性均失败占比", 0.0)
        row[f"{mode_cn}_失败样本内_仅属性1达标_占比"] = ratios.get("仅属性1达标占比", 0.0)
        row[f"{mode_cn}_失败样本内_仅属性2达标_占比"] = ratios.get("仅属性2达标占比", 0.0)

        case3 = stats.get(f"{mode_cn}_单属性成功时失败属性Δ统计", {})
        s1 = case3.get("属性1失败_Δ(out-in)", {})
        s2 = case3.get("属性2失败_Δ(out-in)", {})

        for label_cn, ss in [("属性1失败", s1), ("属性2失败", s2)]:
            row[f"{mode_cn}_单属性成功时_{label_cn}_Δ均值"] = ss.get("均值", 0.0)
            row[f"{mode_cn}_单属性成功时_{label_cn}_Δ标准差"] = ss.get("标准差", 0.0)
            row[f"{mode_cn}_单属性成功时_{label_cn}_Δ最小值"] = ss.get("最小值", 0.0)
            row[f"{mode_cn}_单属性成功时_{label_cn}_Δ最大值"] = ss.get("最大值", 0.0)
            row[f"{mode_cn}_单属性成功时_{label_cn}_样本数"] = ss.get("样本数", 0)

    # 方向：四类计数 + 失败内部占比
    dcnt = stats.get("方向四类计数", {})
    row["方向_两属性方向均正确_数量"] = dcnt.get("两属性方向均正确", 0)
    row["方向_两属性方向均错误_数量"] = dcnt.get("两属性方向均错误", 0)
    row["方向_仅属性1方向正确_数量"] = dcnt.get("仅属性1方向正确", 0)
    row["方向_仅属性2方向正确_数量"] = dcnt.get("仅属性2方向正确", 0)

    dr = stats.get("方向失败内部占比", {})
    row["方向失败样本内_两属性方向均错误_占比"] = dr.get("两属性方向均错误占比", 0.0)
    row["方向失败样本内_仅属性1方向正确_占比"] = dr.get("仅属性1方向正确占比", 0.0)
    row["方向失败样本内_仅属性2方向正确_占比"] = dr.get("仅属性2方向正确占比", 0.0)

    return row


def flatten_stats_to_long_rows_cn(
    scope: str,
    model: str,
    stats: Dict[str, Any],
    total_samples: int,
    valid_samples: int,
    invalid_samples: int,
) -> List[Dict[str, Any]]:
    """
    拍平为长表：每个指标一行（中文列名）。
    """
    rows = []
    meta = build_meta_for_scope(scope)

    def add(group: str, metric: str, value: Any):
        rows.append({
            "范围": scope,
            "模型": model,
            **meta,
            "指标组": group,
            "指标": metric,
            "值": value,
        })

    add("样本总览", "总样本数", total_samples)
    add("样本总览", "可评估样本数", valid_samples)
    add("样本总览", "不合法样本数", invalid_samples)
    add("样本总览", "不合法样本占比", (invalid_samples / total_samples) if total_samples > 0 else 0.0)

    for mode_cn in ("宽松", "严格"):
        cnt = stats.get(f"{mode_cn}_四类计数", {})
        for k, v in cnt.items():
            add(f"{mode_cn}_四类计数", k, v)

        ratios = stats.get(f"{mode_cn}_失败内部占比", {})
        for k, v in ratios.items():
            add(f"{mode_cn}_失败内部占比", k, v)

        case3 = stats.get(f"{mode_cn}_单属性成功时失败属性Δ统计", {})
        for which, ss in case3.items():
            for kk, vv in ss.items():
                add(f"{mode_cn}_{which}", kk, vv)

    dcnt = stats.get("方向四类计数", {})
    for k, v in dcnt.items():
        add("方向四类计数", k, v)

    dr = stats.get("方向失败内部占比", {})
    for k, v in dr.items():
        add("方向失败内部占比", k, v)

    return rows


# ========================
#  八、主入口：输出 Excel
# ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs",
        type=str,
        nargs="+",
        required=True,
        help='log 路径模式，代码内部 glob 展开，例如：--logs "chatdrug_20*_strict.log" "p1_20*_strict.log"',
    )
    ap.add_argument(
        "--out_excel",
        type=str,
        required=True,
        help="输出 Excel 路径，例如 results/multi_attr/双属性统计.xlsx",
    )
    args = ap.parse_args()

    # 展开 patterns
    log_files: List[str] = []
    for pattern in args.logs:
        log_files.extend(sorted(glob(pattern)))

    if not log_files:
        raise RuntimeError("没有匹配到任何 log 文件，请检查 --logs 模式与引号。")

    # 聚合结构：
    # records_by_task_model[taskid][model] -> list[Record]（可评估样本）
    records_by_task_model: Dict[int, Dict[str, List[Record]]] = defaultdict(lambda: defaultdict(list))

    # counts_by_task_model[taskid][model] -> dict(total/valid/invalid)
    counts_by_task_model: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {
        "总样本数": 0, "可评估样本数": 0, "不合法样本数": 0
    }))

    file_infos: List[Dict[str, Any]] = []

    for p in log_files:
        path = Path(p)
        if not path.is_file():
            continue
        recs, finfo = extract_records_from_log(path)
        file_infos.append(finfo)

        # 只统计 201~206（其它 task 的 finfo 已返回空 recs）
        tid = finfo.get("任务ID", None)
        model = finfo.get("模型", None)
        if tid in task2threshold_list and model in ("chatdrug", "ours"):
            counts_by_task_model[tid][model]["总样本数"] += int(finfo.get("样本块数量", 0))
            counts_by_task_model[tid][model]["可评估样本数"] += int(finfo.get("可评估样本数", 0))
            counts_by_task_model[tid][model]["不合法样本数"] += int(finfo.get("不合法样本数", 0))

        for r in recs:
            records_by_task_model[r.taskid][r.model].append(r)

    # 固定输出：6 个任务 × 2 个模型 = 12 行 + ALL × 2 行
    all_tasks = [201, 202, 203, 204, 205, 206]
    all_models = ["chatdrug", "ours"]

    # 先做 ALL 聚合计数与 records
    overall_records_by_model: Dict[str, List[Record]] = defaultdict(list)
    overall_counts_by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {"总样本数": 0, "可评估样本数": 0, "不合法样本数": 0})

    for tid in all_tasks:
        for model in all_models:
            overall_records_by_model[model].extend(records_by_task_model.get(tid, {}).get(model, []))
            overall_counts_by_model[model]["总样本数"] += counts_by_task_model.get(tid, {}).get(model, {}).get("总样本数", 0)
            overall_counts_by_model[model]["可评估样本数"] += counts_by_task_model.get(tid, {}).get(model, {}).get("可评估样本数", 0)
            overall_counts_by_model[model]["不合法样本数"] += counts_by_task_model.get(tid, {}).get(model, {}).get("不合法样本数", 0)

    wide_rows = []
    long_rows = []

    # ALL scope
    for model in all_models:
        recs = overall_records_by_model.get(model, [])
        st = run_for_model(recs)
        c = overall_counts_by_model.get(model, {"总样本数": 0, "可评估样本数": 0, "不合法样本数": 0})
        wide_rows.append(flatten_stats_to_wide_row_cn("ALL", model, st, c["总样本数"], c["可评估样本数"], c["不合法样本数"]))
        long_rows.extend(flatten_stats_to_long_rows_cn("ALL", model, st, c["总样本数"], c["可评估样本数"], c["不合法样本数"]))

    # per-task scope
    for tid in all_tasks:
        scope = f"task{tid}"
        for model in all_models:
            recs = records_by_task_model.get(tid, {}).get(model, [])
            st = run_for_model(recs)
            c = counts_by_task_model.get(tid, {}).get(model, {"总样本数": 0, "可评估样本数": 0, "不合法样本数": 0})
            wide_rows.append(flatten_stats_to_wide_row_cn(scope, model, st, c["总样本数"], c["可评估样本数"], c["不合法样本数"]))
            long_rows.extend(flatten_stats_to_long_rows_cn(scope, model, st, c["总样本数"], c["可评估样本数"], c["不合法样本数"]))

    df_wide = pd.DataFrame(wide_rows)
    df_long = pd.DataFrame(long_rows)
    df_files = pd.DataFrame(file_infos)

    out_excel = Path(args.out_excel)
    out_excel.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_wide.to_excel(writer, sheet_name="总览_宽表", index=False)
        df_long.to_excel(writer, sheet_name="总览_长表", index=False)
        df_files.to_excel(writer, sheet_name="文件级汇总", index=False)

    print("已输出 Excel：", out_excel)


if __name__ == "__main__":
    main()


'''
python utils/convenient_utils/muti_atrri_ana_bytask.py \
 --logs "results/opt_results/gpt_35/old/gpt/chatdrug_20*_strict.log" \
         "results/opt_results/gpt_35/new/withoutPrequery/1/p1_20*_strict.log" \
  --out_excel results/opt_results/compare/muti_prop_res/multi_attr_statsS.xlsx
'''