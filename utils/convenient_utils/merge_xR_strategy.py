import re
import os
from typing import Dict, List, Tuple, Optional


SAMPLE_HEADER_RE = re.compile(r">+ *Sample\s+(\d+):")
METRIC_RE = re.compile(r"当前指标\s+num_correct\s*=\s*\d+,\s*acc\s*=\s*([0-9]*\.?[0-9]+)")

# 用于识别 Final Acc 块
FINAL_ACC_START_RE = re.compile(r"-+Final Acc-+")
FINAL_ACC_END_LINE_RE = re.compile(r"-{5,}")  # 例如 ----------------------------


def extract_sample_blocks(text: str) -> Dict[int, List[str]]:
    """
    拆成 {sample_idx: [lines]}，包含 header 行
    """
    blocks: Dict[int, List[str]] = {}
    cur_idx = None
    cur_lines: List[str] = []

    for line in text.splitlines():
        m = SAMPLE_HEADER_RE.search(line)
        if m:
            if cur_idx is not None:
                blocks[cur_idx] = cur_lines
            cur_idx = int(m.group(1))
            cur_lines = [line]
        else:
            if cur_idx is not None:
                cur_lines.append(line)

    if cur_idx is not None:
        blocks[cur_idx] = cur_lines

    return blocks


def strip_old_metric_lines(lines: List[str]) -> List[str]:
    """
    删除旧的 '当前指标 num_correct = ..., acc = ...' 行
    """
    return [l for l in lines if not METRIC_RE.search(l)]


def strip_final_acc_block(lines: List[str]) -> List[str]:
    """
    删除形如：
    ---------Final Acc---------
    Acc = ...
     num_correct = ...
    ----------------------------
    这一整块
    """
    out: List[str] = []
    in_block = False
    for l in lines:
        if not in_block and FINAL_ACC_START_RE.search(l):
            in_block = True
            continue
        if in_block:
            if FINAL_ACC_END_LINE_RE.fullmatch(l.strip()):
                in_block = False
            continue
        out.append(l)
    return out


def has_prequery_early_stopped(lines: List[str]) -> bool:
    return any(("Pre-query DB:" in l and "Task early stopped" in l) for l in lines)


def sample_is_correct(lines: List[str]) -> bool:
    """
    正确口径：
    1) 有 Evaluation result is True -> correct
    2) 没有任何 Evaluation result 行，但 Pre-query DB early stopped -> correct
    否则 -> incorrect
    """
    if has_prequery_early_stopped(lines):
        return True
    if any("Evaluation result is True" in l for l in lines):
        return True
    return False


def format_metric(num_correct: int, num_total: int) -> str:
    acc = (num_correct / num_total) if num_total > 0 else 0.0
    return f"当前指标 num_correct = {num_correct}, acc = {acc:.6f}"


def ensure_one_blank_line_end(lines: List[str]) -> List[str]:
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append("")
    return lines


INVALID_MARKERS = ["fail: Generation error", "fail:XR", "ReDF no XR"]

def sample_is_invalid(lines: List[str]) -> bool:
    return any(any(m in l for m in INVALID_MARKERS) for l in lines)

def merge_two_logs(p1_path: str, p2_path: str, p3_path: str) -> Tuple[int, int, float]:
    """
    合并 p1 + p2 -> p3
    重复 sample 以 p2 为准
    重新计算 num_correct/acc，并输出唯一 Final Acc（含 num_all_f）
    """
    with open(p1_path, "r", encoding="utf-8") as f:
        t1 = f.read()
    with open(p2_path, "r", encoding="utf-8") as f:
        t2 = f.read()

    b1 = extract_sample_blocks(t1)
    b2 = extract_sample_blocks(t2)

    merged = dict(b1)
    merged.update(b2)  # p2 覆盖 p1

    sample_indices = sorted(merged.keys())

    out_lines: List[str] = []
    num_correct = 0
    num_total = 0
    num_all_f = 0

    for idx in sample_indices:
        raw_lines = merged[idx]

        # ✅ 不管是不是最后一个 sample，都清掉旧 Final Acc 块（解决“两个 Final Acc”）
        raw_lines = strip_final_acc_block(raw_lines)

        # correct 判定（你的原逻辑）
        is_ok = sample_is_correct(raw_lines)

        # ✅ 统计 num_all_f（按不合法/失败标记口径）
        if sample_is_invalid(raw_lines):
            num_all_f += 1

        # 清理旧指标行
        lines = strip_old_metric_lines(raw_lines)

        # 更新累计
        num_total += 1
        if is_ok:
            num_correct += 1

        # 追加新指标行到 sample 末尾
        while lines and lines[-1].strip() == "":
            lines.pop()
        lines.append(format_metric(num_correct, num_total))
        lines = ensure_one_blank_line_end(lines)

        out_lines.extend(lines)

    acc = (num_correct / num_total) if num_total > 0 else 0.0

    # ✅ 文件末尾输出唯一 Final Acc，且强制带 num_all_f（固定格式）
    out_lines.append("---------Final Acc---------")
    out_lines.append(f"Acc = {num_correct}/{num_total}")
    out_lines.append(f" num_correct = {num_correct}, num_all ={num_total}, num_all_f ={num_all_f}")
    out_lines.append("----------------------------")

    os.makedirs(os.path.dirname(p3_path), exist_ok=True)
    with open(p3_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines).rstrip() + "\n")

    return num_total, num_correct, acc



# =========================
# 新增：批量处理文件夹
# =========================
def parse_case_from_filename(fname: str) -> Optional[Tuple[int, str]]:
    """
    从文件名解析 (number, mode)，支持：
      p1_105_loose.log / p1_105_strict.log
      p1_woE_105_loose.log / p1_woE_105_strict.log
    返回 (105, "loose") 或 (105, "strict")；解析失败返回 None
    """
    m = re.search(r"_(\d{3})_(loose|strict)\.log$", fname)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def build_p2_filename(case_num: int, mode: str) -> str:
    """
    p2 文件命名规则：p1_woE_{num}_{mode}.log
    """
    return f"p1_woP_{case_num:03d}_{mode}.log"


def build_p3_filename(case_num: int, mode: str) -> str:
    """
    p3 输出命名规则：p1_{num}_{mode}.log
    """
    return f"p1_{case_num:03d}_{mode}.log"


def merge_folder(
    p1_dir: str,
    p2_dir: str,
    p3_dir: str,
    only_ranges: bool = True,
) -> List[Dict[str, object]]:
    """
    遍历 p1_dir 下所有 p1_{num}_{loose|strict}.log
    对每个文件，去 p2_dir 找对应的 p1_woE_{num}_{loose|strict}.log
    若存在则合并生成到 p3_dir（同名 p1_{num}_{mode}.log）

    only_ranges=True 时，只处理 num 属于：
      101..108 或 201..206
    返回：每个任务的结果记录 list（便于打印/落盘）
    """
    results: List[Dict[str, object]] = []
    os.makedirs(p3_dir, exist_ok=True)

    p1_files = sorted([f for f in os.listdir(p1_dir) if f.endswith(".log")])

    for fname in p1_files:
        # 只处理形如 p1_105_loose.log / p1_105_strict.log
        if not fname.startswith("p1_"):
            continue
        if fname.startswith("p1_wo"):
            continue

        parsed = parse_case_from_filename(fname)
        if not parsed:
            continue
        case_num, mode = parsed

        if only_ranges:
            in_range = (101 <= case_num <= 108) or (201 <= case_num <= 206)
            if not in_range:
                continue

        p1_path = os.path.join(p1_dir, fname)
        p2_name = build_p2_filename(case_num, mode)
        p2_path = os.path.join(p2_dir, p2_name)
        p3_name = build_p3_filename(case_num, mode)
        p3_path = os.path.join(p3_dir, p3_name)

        if not os.path.exists(p2_path):
            results.append(
                {
                    "case_num": case_num,
                    "mode": mode,
                    "p1": p1_path,
                    "p2": p2_path,
                    "p3": p3_path,
                    "status": "missing_p2",
                }
            )
            continue

        total, correct, acc = merge_two_logs(p1_path, p2_path, p3_path)
        results.append(
            {
                "case_num": case_num,
                "mode": mode,
                "p1": p1_path,
                "p2": p2_path,
                "p3": p3_path,
                "status": "ok",
                "total": total,
                "correct": correct,
                "acc": acc,
            }
        )

    return results


if __name__ == "__main__":
    # p1 = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/p1_105_loose.log"
    # p2 = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/ablation/Tanimoto/p1_woE_105_loose.log"  # 优先的
    # p3 = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/p1_105_loose.log"  # 输出文件
    #
    # total, correct, acc = merge_two_logs(p1, p2, p3)
    # print(f"p3 written to: {p3}")
    # print(f"total samples: {total}, num_correct: {correct}, acc: {correct / total if total else 0.0:.6f}")

    p1_dir = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/2"
    p2_dir = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/ablation/P"
    p3_dir = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/2"

    """ 
    p1里有p2里没有的直接继承
    p1里有p2库也有的以p2未转
    num_correct都重新算
    p3是输出文件
    """
    results = merge_folder(p1_dir, p2_dir, p3_dir, only_ranges=True)

    # 打印汇总
    ok = [r for r in results if r["status"] == "ok"]
    miss = [r for r in results if r["status"] != "ok"]

    for r in ok:
        print(
            f"[OK] {r['case_num']:03d} {r['mode']}  acc={r['acc']:.6f}  "
            f"({r['correct']}/{r['total']})  -> {r['p3']}"
        )

    for r in miss:
        print(f"[SKIP:{r['status']}] {r['case_num']:03d} {r['mode']}  missing: {r['p2']}")
