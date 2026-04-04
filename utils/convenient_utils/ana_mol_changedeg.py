#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同一份日志同时给两套口径结果：
A) scaffold 不变（Murcko scaffold 完全一致）口径
B) MCS 覆盖率 >= 阈值口径

并在两套口径内都统计：
- 可解析率（parse_fail）
- 可比对数
- changed_sites（位点变化数：MCS 骨架原子上外接取代基集合是否变化）
- 分布/均值/中位数/极值

输出 4 个文件：
1) detail.csv
2) summary.json
3) parse_fail.csv
4) invalid_for_sites.csv

新增：支持两种 Sample 行格式（自动识别）
1) >>>>> Sample X: input_drug, <SMILES>
2) >>>>> Sample X: <SMILES>            （或 >>>>> Sample X: 任意描述, <SMILES>）
规则：Sample 行冒号后面的内容，若包含逗号，取“最后一个逗号后”的字段作为 SMILES；否则整段作为 SMILES。
比较规则不变：每个 sample 内第一个 Generated Result 和 Sample 行里的 SMILES 比；之后每个 Generated Result 和上一个 Generated Result 比。
"""

import re
import json
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# 更宽松：只要匹配 “>>>>> Sample <id>: <rest>”
SAMPLE_RE = re.compile(r"^>>>>>+\s*Sample\s+(\d+)\s*:\s*(.*)\s*$")


def extract_smiles_from_sample_rest(rest: str) -> str:
    """
    支持：
      - "input_drug, <SMILES>"
      - "<SMILES>"
      - "xxx, yyy, <SMILES>"  -> 取最后一个逗号后的字段
    """
    if rest is None:
        return ""
    s = rest.strip()
    if not s:
        return ""
    if "," in s:
        # 取最后一个逗号后的字段
        s = s.rsplit(",", 1)[-1].strip()
    return s


def parse_log(text: str):
    """
    返回：
    [
      {"sample": int, "input": str, "generated": [str, ...]},
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
            rest = m.group(2)
            inp = extract_smiles_from_sample_rest(rest)
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
    """解析 SMILES；多片段取重原子数最大者；sanitize 失败返回 None。"""
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


def murcko_scaffold_smiles(mol: Chem.Mol):
    """返回 Murcko scaffold 的 canonical SMILES；失败返回 None。"""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaf = (scaf or "").strip()
        return scaf if scaf else ""  # 允许空 scaffold（链状分子）
    except Exception:
        return None


def substituent_signature(mol: Chem.Mol, scaffold_atom: int, ext_atom: int):
    """断开 scaffold_atom-ext_atom 的键，取 ext_atom 所在片段 SMILES 作为签名。"""
    bond = mol.GetBondBetweenAtoms(scaffold_atom, ext_atom)
    if bond is None:
        return None
    try:
        frag = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
        molfrags, mappings = Chem.GetMolFrags(
            frag, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=True
        )
    except Exception:
        return None
    for fm, mapping in zip(molfrags, mappings):
        if ext_atom in mapping:
            try:
                return Chem.MolToSmiles(fm, canonical=True)
            except Exception:
                return None
    return None


def mcs_info(m1: Chem.Mol, m2: Chem.Mol, timeout_s: int):
    """
    返回：
    - ok: bool
    - mcs_atoms: int
    - coverage: float (mcs_atoms/min(nAtoms1,nAtoms2))
    - patt: MolFromSmarts
    - match1/match2: tuple atom indices
    """
    try:
        res = rdFMCS.FindMCS(
            [m1, m2],
            timeout=timeout_s,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
    except Exception:
        return {"ok": False, "reason": "mcs_fail"}

    if getattr(res, "canceled", False) or res.numAtoms == 0:
        return {"ok": False, "reason": "no_mcs", "mcs_atoms": 0}

    patt = Chem.MolFromSmarts(res.smartsString)
    if patt is None:
        return {"ok": False, "reason": "mcs_fail", "mcs_atoms": res.numAtoms}

    match1 = m1.GetSubstructMatch(patt)
    match2 = m2.GetSubstructMatch(patt)
    if not match1 or not match2:
        return {"ok": False, "reason": "mcs_fail", "mcs_atoms": res.numAtoms}

    cov = res.numAtoms / min(m1.GetNumAtoms(), m2.GetNumAtoms())
    return {
        "ok": True,
        "mcs_atoms": res.numAtoms,
        "coverage": cov,
        "patt": patt,
        "match1": match1,
        "match2": match2,
    }


def changed_sites_from_mcs(m1: Chem.Mol, m2: Chem.Mol, match1, match2):
    """
    位点变化数定义：
    对 MCS 骨架上的每个对应原子，比较其“连到骨架外”的取代基签名集合是否一致；
    不一致则该骨架原子计 1 个位点变化。
    """
    set1, set2 = set(match1), set(match2)
    changed = 0

    for a1, a2 in zip(match1, match2):
        subs1 = []
        atom1 = m1.GetAtomWithIdx(a1)
        for nb in atom1.GetNeighbors():
            i = nb.GetIdx()
            if i not in set1:
                sig = substituent_signature(m1, a1, i)
                if sig is None:
                    bond = m1.GetBondBetweenAtoms(a1, i)
                    sig = f"atom{nb.GetAtomicNum()}_{bond.GetBondType() if bond else 'NA'}"
                subs1.append(sig)

        subs2 = []
        atom2 = m2.GetAtomWithIdx(a2)
        for nb in atom2.GetNeighbors():
            i = nb.GetIdx()
            if i not in set2:
                sig = substituent_signature(m2, a2, i)
                if sig is None:
                    bond = m2.GetBondBetweenAtoms(a2, i)
                    sig = f"atom{nb.GetAtomicNum()}_{bond.GetBondType() if bond else 'NA'}"
                subs2.append(sig)

        if sorted(subs1) != sorted(subs2):
            changed += 1

    return changed


def compute_one_pair(sm1: str, sm2: str, timeout_s: int, mcs_threshold: float):
    """
    返回一条记录字段：
    - parse_ok: bool
    - scaffold1/scaffold2/scaffold_same
    - mcs_ok/mcs_atoms/coverage/mcs_pass
    - changed_sites（仅当 parse_ok 且 mcs_ok 且 (scaffold_same or mcs_pass) 时计算；否则 None）
    - reason（parse_fail / no_mcs / mcs_fail / not_qualified）
    """
    m1 = largest_fragment_mol(sm1)
    m2 = largest_fragment_mol(sm2)
    if m1 is None or m2 is None:
        return {
            "parse_ok": False,
            "reason": "parse_fail",
            "scaffold1": None,
            "scaffold2": None,
            "scaffold_same": False,
            "mcs_ok": False,
            "mcs_atoms": None,
            "coverage": None,
            "mcs_pass": False,
            "changed_sites": None,
        }

    sc1 = murcko_scaffold_smiles(m1)
    sc2 = murcko_scaffold_smiles(m2)
    scaffold_same = (sc1 is not None) and (sc2 is not None) and (sc1 == sc2)

    mi = mcs_info(m1, m2, timeout_s=timeout_s)
    if not mi.get("ok", False):
        return {
            "parse_ok": True,
            "reason": mi.get("reason", "mcs_fail"),
            "scaffold1": sc1,
            "scaffold2": sc2,
            "scaffold_same": scaffold_same,
            "mcs_ok": False,
            "mcs_atoms": mi.get("mcs_atoms"),
            "coverage": None,
            "mcs_pass": False,
            "changed_sites": None,
        }

    cov = mi["coverage"]
    mcs_pass = cov >= mcs_threshold

    qualified_for_sites = scaffold_same or mcs_pass
    if not qualified_for_sites:
        return {
            "parse_ok": True,
            "reason": "not_qualified",
            "scaffold1": sc1,
            "scaffold2": sc2,
            "scaffold_same": scaffold_same,
            "mcs_ok": True,
            "mcs_atoms": mi["mcs_atoms"],
            "coverage": cov,
            "mcs_pass": mcs_pass,
            "changed_sites": None,
        }

    cs = changed_sites_from_mcs(m1, m2, mi["match1"], mi["match2"])
    return {
        "parse_ok": True,
        "reason": "ok",
        "scaffold1": sc1,
        "scaffold2": sc2,
        "scaffold_same": scaffold_same,
        "mcs_ok": True,
        "mcs_atoms": mi["mcs_atoms"],
        "coverage": cov,
        "mcs_pass": mcs_pass,
        "changed_sites": cs,
    }


def build_detail(samples, timeout_s: int, mcs_threshold: float):
    rows = []
    for s in samples:
        ref = s["input"]
        for step, gen in enumerate(s["generated"]):
            base = ref
            r = compute_one_pair(base, gen, timeout_s=timeout_s, mcs_threshold=mcs_threshold)
            rows.append(
                {
                    "sample": s["sample"],
                    "step": step,
                    "compare_from": "input" if step == 0 else "prev_gen",
                    "base_smiles": base,
                    "gen_smiles": gen,
                    **r,
                }
            )
            ref = gen
    return pd.DataFrame(rows)


def stats_block(df_ok: pd.DataFrame):
    if len(df_ok) == 0:
        return {
            "ok_count": 0,
            "avg_changed_sites": None,
            "median_changed_sites": None,
            "min_changed_sites": None,
            "max_changed_sites": None,
            "changed_sites_distribution": {},
        }
    dist = df_ok["changed_sites"].value_counts().sort_index().to_dict()
    return {
        "ok_count": int(len(df_ok)),
        "avg_changed_sites": float(df_ok["changed_sites"].mean()),
        "median_changed_sites": float(df_ok["changed_sites"].median()),
        "min_changed_sites": int(df_ok["changed_sites"].min()),
        "max_changed_sites": int(df_ok["changed_sites"].max()),
        "changed_sites_distribution": {str(k): int(v) for k, v in dist.items()},
    }


def summarize(df: pd.DataFrame, mcs_threshold: float):
    total = int(len(df))
    parse_fail = int((~df["parse_ok"]).sum())
    parse_ok = total - parse_fail

    # 口径 A：Murcko scaffold 完全一致
    a_candidates = df[df["parse_ok"] & df["scaffold_same"]]
    a_ok = a_candidates[a_candidates["changed_sites"].notna()]

    # 口径 B：MCS 覆盖率 >= 当前阈值
    b_candidates = df[df["parse_ok"] & df["mcs_ok"] & df["mcs_pass"]]
    b_ok = b_candidates[b_candidates["changed_sites"].notna()]

    # changed_sites 无法给出的一类
    invalid_for_sites = df[df["parse_ok"] & (df["changed_sites"].isna())]

    # 骨架口径失败：parse_ok 且 scaffold_same 为 False 或 None
    scaffold_fail_mask = df["parse_ok"] & (~df["scaffold_same"].fillna(False))

    # MCS 口径失败：parse_ok 且 (MCS 失败 或 coverage < 当前 mcs_threshold)
    mcs_fail_mask = df["parse_ok"] & (
        (~df["mcs_ok"]) | (~df["mcs_pass"].fillna(False))
    )

    # 不同 MCS 覆盖率区间的数量（区间宽度 0.1：0.0~0.1, 0.1~0.2, ..., 0.9~1.0）
    mcs_valid = df[df["parse_ok"] & df["mcs_ok"]].copy()
    pass_by_thr = {}

    if len(mcs_valid) > 0:
        cov = mcs_valid["coverage"]

        # 生成 0.0–0.1, 0.1–0.2, ..., 0.9–1.0
        for i in range(10):
            lo = round(i / 10, 1)
            hi = round((i + 1) / 10, 1)
            key = f"{lo:.1f}~{hi:.1f}"

            if i < 9:
                # 左闭右开 [lo, hi)
                cnt = int(((cov >= lo) & (cov < hi)).sum())
            else:
                # 最后一段包含 1.0： [0.9, 1.0]
                cnt = int(((cov >= lo) & (cov <= hi)).sum())

            pass_by_thr[key] = cnt

    summary = {
        "total_comparisons": total,
        "parse_fail_count": parse_fail,
        "parse_ok_count": parse_ok,
        "mcs_threshold": mcs_threshold,

        "scaffold_rule": {
            "candidate_count": int(len(a_candidates)),
            **stats_block(a_ok),
        },
        "mcs_rule": {
            "candidate_count": int(len(b_candidates)),
            **stats_block(b_ok),
            # 覆盖率区间统计
            "pass_counts_by_threshold": pass_by_thr,
        },
        "other_counts": {
            "invalid_for_sites_count": int(len(invalid_for_sites)),
            "no_mcs_or_mcs_fail_count": int(len(df[df["parse_ok"] & (~df["mcs_ok"])])),
            "not_qualified_count": int(
                len(df[df["parse_ok"] & df["mcs_ok"] & (~df["mcs_pass"]) & (~df["scaffold_same"])])
            ),
            "scaffold_fail_count": int(scaffold_fail_mask.sum()),
            "mcs_fail_count": int(mcs_fail_mask.sum()),
        },
        "status_counts_detail": {
            "reason_counts": df["reason"].value_counts().to_dict()
        },
    }
    return summary, invalid_for_sites


def main():
    ap = argparse.ArgumentParser()
    # 支持一次处理多个日志文件
    ap.add_argument("logfiles", type=str, nargs="+", help="一个或多个日志文件路径")

    ap.add_argument("--timeout", type=int, default=2, help="MCS 搜索超时（秒）")
    ap.add_argument("--mcs_threshold", type=float, default=0.8, help="MCS 覆盖率阈值（口径B）")

    # 统一输出目录，文件名根据输入日志文件名自动生成，每个任务生成detail、parse_fail、invalid_for_sites、summary四个文件
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")

    # 可选输出控制：summary 一定输出，其余三个可关闭
    ap.add_argument("--no_detail", action="store_true", help="不输出 detail.csv")
    ap.add_argument("--no_parse_fail", action="store_true", help="不输出 parse_fail.csv")
    ap.add_argument("--no_invalid", action="store_true", help="不输出 invalid_for_sites.csv")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for logfile in args.logfiles:     # 循环处理多个输入
        # 获取和定义路径
        log_path = Path(logfile)
        stem = log_path.stem

        detail_path = out_dir / f"{stem}_detail.csv"
        parse_fail_path = out_dir / f"{stem}_parse_fail.csv"
        invalid_path = out_dir / f"{stem}_invalid_for_sites.csv"
        summary_path = out_dir / f"{stem}_summary.json"

        text = log_path.read_text(errors="ignore")
        samples = parse_log(text)

        df = build_detail(samples, timeout_s=args.timeout, mcs_threshold=args.mcs_threshold)

        # 是否输出detail
        if not args.no_detail:
            df.to_csv(detail_path, index=False)

        # 是否输出parse_fail
        if not args.no_parse_fail:
            df[~df["parse_ok"]].to_csv(parse_fail_path, index=False)

        # 是否输出 invalid_for_sites
        if not args.no_invalid:
            invalid_for_sites.to_csv(invalid_path, index=False)

        # summary 一定输出
        summary, invalid_for_sites = summarize(df, mcs_threshold=args.mcs_threshold)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

        # 简单打印当前日志的输出情况
        print(f"[{logfile}] summary -> {summary_path}")
        if not args.no_detail:
            print(f"[{logfile}] detail -> {detail_path}")
        if not args.no_parse_fail:
            print(f"[{logfile}] parse_fail -> {parse_fail_path}")
        if not args.no_invalid:
            print(f"[{logfile}] invalid_for_sites -> {invalid_path}")
        print(summary)


if __name__ == "__main__":
    main()
   # 输出一个summary json