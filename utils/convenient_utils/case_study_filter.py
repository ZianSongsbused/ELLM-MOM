#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import shutil
from typing import Dict, Any, List, Optional, Tuple

# messages:兼容不同 langchain 版本
try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    from langchain.schema import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI


# -----------------------------
# LLM (不硬编码 key；用环境变量)
# -----------------------------
def complete_chatgpt_lc(temperature=0.2, top_p=1.0):
    llm = ChatOpenAI(
        api_key="sk-zUwWAxweB84ckK6uFaruILyZzUFwomf5YOwWr8sErhcLILAF",
        base_url="https://api.chatanywhere.tech",
        model="gpt-3.5-turbo",
        temperature=temperature,
        model_kwargs={"top_p": top_p},
    )
    return llm



# -----------------------------
# Prompt
# -----------------------------
def build_llm_prompt(prev_smiles: str, new_smiles: str, edit_op: str, rationale: str) -> str:
    return f"""
You are reviewing a medicinal chemistry case for a scientific paper.

Previous molecule (SMILES):
{prev_smiles}

New molecule after one optimization step (SMILES):
{new_smiles}

The claimed chemical edit operation:
"{edit_op}"

The stated rationale for this operation:
"{rationale}"

Tasks:
1. Determine whether the edit operation correctly describes the structural change between the molecules.
2. Determine whether the rationale logically and chemically explains the edit operation.
3. Decide whether this step is suitable to be reported as a case study in a paper.

Respond in JSON ONLY with the following fields:
- edit_op_correct: true or false
- rationale_consistent: true or false
- paper_worthy: true or false
- brief_reason: one concise sentence explaining your decision
""".strip()


def llm_check_step(llm, prev_smiles: str, new_smiles: str, edit_op: str, rationale: str) -> Dict[str, Any]:
    resp = llm.invoke([
        SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
        HumanMessage(content=build_llm_prompt(prev_smiles, new_smiles, edit_op, rationale)),
    ])
    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "edit_op_correct": False,
            "rationale_consistent": False,
            "paper_worthy": False,
            "brief_reason": "Failed to parse LLM output as JSON",
            "raw": getattr(resp, "content", None),
        }


# -----------------------------
# IO helpers
# -----------------------------
def list_selected_ops_jsons(root_dir: str) -> List[str]:
    paths = []
    for root, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith("_selected_ops.json"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def safe_basename_noext(path: str) -> str:
    b = os.path.basename(path)
    return b[:-5] if b.endswith(".json") else b


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def copy_any_matching_sample_dir(src_parent: str, sample_index: int, dst_parent: str) -> Tuple[bool, str]:
    """
    从 src_parent 下复制“该 sample 的分子图目录”到 dst_parent。
    兼容两种常见命名：
      - sample_0011/
      - sample_0011.svg (若你没分目录，此处不会复制到；建议你是目录)
    复制策略：
      1) 优先找 sample_{idx:04d} 目录
      2) 其次找 sample_{idx} 目录（非零填充）
      3) 若都没有，返回 False
    """
    cand_dirs = [
        os.path.join(src_parent, f"sample_{sample_index:04d}"),
        os.path.join(src_parent, f"sample_{sample_index}"),
    ]
    for d in cand_dirs:
        if os.path.isdir(d):
            dst = os.path.join(dst_parent, os.path.basename(d))
            # Python 3.8+ 支持 dirs_exist_ok
            shutil.copytree(d, dst, dirs_exist_ok=True)
            return True, dst
    return False, "sample dir not found"


# -----------------------------
# Core selection per JSON: collect up to K passed samples
# 约束：只接受 round0+round1（len==2 且 round_index==[0,1]）
# -----------------------------
def collect_passed_samples(llm, json_path: str, k: int) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    taskid = data.get("taskid")
    log_file = data.get("log_file")
    samples = data.get("samples", [])

    passed: List[Dict[str, Any]] = []

    for s in samples:
        if len(passed) >= k:
            break

        if not s.get("final_success"):
            continue

        rounds = s.get("rounds", [])
        # 只要 round0+round1
        if len(rounds) != 2:
            continue
        round_ids = sorted(r.get("round_index") for r in rounds)
        if round_ids != [0, 1]:
            continue

        prev_smiles = s.get("input_smiles")
        ok = True
        checked_rounds = []

        for r in rounds:
            sm = r.get("selected_match") or {}
            if sm.get("match_type") != "matched":
                ok = False
                break

            edit_op = sm.get("edit_op")
            rationale = sm.get("rationale")
            new_smiles = r.get("generated_result")

            if not edit_op or not rationale or not new_smiles:
                ok = False
                break

            judgement = llm_check_step(llm, prev_smiles, new_smiles, edit_op, rationale)

            if not (judgement.get("edit_op_correct") and judgement.get("rationale_consistent") and judgement.get("paper_worthy")):
                ok = False
                break

            checked_rounds.append({
                "round_index": r.get("round_index"),
                "prev_smiles": prev_smiles,
                "generated_result": new_smiles,
                "edit_op": edit_op,
                "rationale": rationale,
                "llm_judgement": judgement,
            })

            prev_smiles = new_smiles

        if ok and checked_rounds:
            passed.append({
                "sample_index": s.get("sample_index"),
                "rounds": checked_rounds,
            })

    return {
        "taskid": taskid,
        "log_file": log_file,
        "json_path": json_path,
        "k": k,
        "num_found": len(passed),
        "passed_samples": passed,
    }


# -----------------------------
# Orchestrate:
# output_root/task_<taskid>/<json_basename_noext>/...
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases_root_dir", help="Root dir containing per-log subfolders with *_selected_ops.json")
    ap.add_argument("out_root_dir", help="Output root dir")
    ap.add_argument("--k", type=int, default=5, help="Find at least/top K passed samples per log JSON (default 5)")
    args = ap.parse_args()

    llm = complete_chatgpt_lc()

    json_paths = list_selected_ops_jsons(args.cases_root_dir)
    print(f"Found {len(json_paths)} *_selected_ops.json")

    ensure_dir(args.out_root_dir)

    for jp in json_paths:
        res = collect_passed_samples(llm, jp, args.k)

        taskid = res.get("taskid")
        if taskid is None:
            print(f"Skip (missing taskid): {jp}")
            continue

        task_dir = os.path.join(args.out_root_dir, f"task_{taskid}")
        ensure_dir(task_dir)

        json_base = safe_basename_noext(jp)  # e.g. p1_202_strict_selected_ops
        log_out_dir = os.path.join(task_dir, json_base)
        ensure_dir(log_out_dir)

        # 写 summary
        summary_path = os.path.join(log_out_dir, "summary_topk.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        # 复制分子图信息（sample 目录）
        src_parent = os.path.dirname(jp)  # “读json的那个文件夹”
        copy_report = []
        for ps in res["passed_samples"]:
            idx = ps["sample_index"]
            ok, msg = copy_any_matching_sample_dir(src_parent, idx, log_out_dir)
            copy_report.append({
                "sample_index": idx,
                "copied": ok,
                "detail": msg,
            })

        # 追加一个复制报告文件，方便排查路径问题
        with open(os.path.join(log_out_dir, "copy_report.json"), "w", encoding="utf-8") as f:
            json.dump({
                "src_parent": src_parent,
                "dst_parent": log_out_dir,
                "report": copy_report,
            }, f, ensure_ascii=False, indent=2)

        print(f"{os.path.basename(jp)} -> {log_out_dir} (found {res['num_found']}/{args.k})")


if __name__ == "__main__":
    main()


"""
python utils/convenient_utils/case_study_filter.py \
  results/opt_results/gpt_35/new/cases/ \
  results/opt_results/gpt_35/new/cases/filtered/
"""
