import re
import os
from statistics import mean, stdev
from typing import Dict, List, Tuple
from openpyxl import Workbook

# ---------- 1. 按 Sample 切块 ----------

def extract_sample_blocks(text: str):
    """
    把整个 log 拆成 {sample_idx: [lines]} 这样的字典
    匹配形式：>>>>> Sample 0:
    """
    blocks = {}
    cur_idx = None
    cur_lines = []

    for line in text.splitlines():
        m = re.search(r">+ *Sample\s+(\d+):", line)
        if m:
            # 新的 Sample 开始
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


# ---------- 2. 判断是否包含 “Pre-query DB: ... Task early stopped.” ----------

def contains_pre_query_sample(lines):
    """
    整个 sample 里只要有一行同时包含 'Pre-query DB:' 和 'Task early stopped.' 就算 prequery 成功。
    """
    for line in lines:
        if "Pre-query DB:" in line and "Task early stopped" in line:
            return True
    return False


# ---------- 3. 主逻辑：统计每个轮次成功数量 ----------

def stat_success_rounds(path):
    """
    输入：
        path: 单个 .log 文件路径

    输出：
        success_round0: 第 0 轮就成功（第一个 Evaluation result is True 对应 round_index:0）的 sample 序号列表
        success_round1: 第 1 轮成功的 sample 序号列表
        success_round2: 第 2 轮成功的 sample 序号列表
        success_prequery: 没有任何 Evaluation result，但 Pre-query DB 提前停止 的 sample 序号列表
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    blocks = extract_sample_blocks(text)

    success_round0 = []
    success_round1 = []
    success_round2 = []
    success_prequery = []

    for idx, lines in blocks.items():
        # 1）找这个 sample 里的第一个 "Evaluation result is True"
        true_line_pos = None
        for i, line in enumerate(lines):
            if "Evaluation result is True" in line:
                true_line_pos = i
                break

        if true_line_pos is not None:
            # 2）从 True 那一行往上找最近的 "round_index:X"
            round_idx = None
            for j in range(true_line_pos - 1, -1, -1):
                if "round_index:" in lines[j]:
                    m = re.search(r"round_index:(\d+)", lines[j])
                    if m:
                        round_idx = int(m.group(1))
                    break

            # 3）根据轮次分类
            if round_idx == 0:
                success_round0.append(idx)
            elif round_idx == 1:
                success_round1.append(idx)
            elif round_idx == 2:
                success_round2.append(idx)
            # 如果有更高轮次，可以在这里继续加 elif
        else:
            # 4）没有任何 Evaluation result 的情况，判断是不是 prequery 成功
            if contains_pre_query_sample(lines):
                success_prequery.append(idx)

    success_round0.sort()
    success_round1.sort()
    success_round2.sort()
    success_prequery.sort()

    return success_round0, success_round1, success_round2, success_prequery


# ---------- 4. 批量处理文件夹下所有 .log 文件 ----------

def stat_success_rounds_folder(folder_path):
    """
    输入：
        folder_path: 含若干 .log 文件的文件夹

    输出：
        results: dict
            key: 文件名（不含路径）
            value: (success_round0, success_round1, success_round2, success_prequery)
    """
    results = {}

    for fname in sorted(os.listdir(folder_path)):   # 文件名升序
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(folder_path, fname)
        r0, r1, r2, preq = stat_success_rounds(fpath)
        results[fname] = (r0, r1, r2, preq)

    return results

#
# # ---------- 5. 简单命令行测试入口 ----------
#
# if __name__ == "__main__":
#     # 单文件测试
#     # path = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/p1_101_loose.log"
#     # r0, r1, r2, preq = stat_success_rounds(path)
#     # print("file:", path)
#     # print("round0 success:", r0, len(r0))
#     # print("round1 success:", r1, len(r1))
#     # print("round2 success:", r2, len(r2))
#     # print("prequery success:", preq, len(preq))
#
#     # 文件夹批量测试
#     folder = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new"
#     all_res = stat_success_rounds_folder(folder)
#     for fname, (rr0, rr1, rr2, pp) in all_res.items():
#         print(f"\n=== {fname} ===")
#         print("round0 success:", rr0, len(rr0))
#         print("round1 success:", rr1, len(rr1))
#         print("round2 success:", rr2, len(rr2))
#         print("prequery success:", pp, len(pp))
#         print(" ", len(rr0)/2, " ", len(rr1)/2, " ", len(rr2)/2, " ", len(pp)/2)


# ---------- 5. 汇总 1~5 五个子文件夹，并写入 Excel ----------

def collect_stats_across_subfolders(base_folder: str, subfolders=None):
    """
    base_folder 下有 1,2,3,4,5 五个子文件夹，每个结构和原来 new 一样。
    返回：
        all_stats[run_id][fname] = (n0, n1, n2, npre)
        run_id 是 '1'...'5'
    """
    if subfolders is None:
        subfolders = ["1", "2", "3", "4", "5"]

    all_stats: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}

    for run_id in subfolders:
        folder_path = os.path.join(base_folder, run_id)
        if not os.path.isdir(folder_path):
            # 跳过不存在的子文件夹
            continue
        res = stat_success_rounds_folder(folder_path)
        # 把列表长度变成数量
        all_stats[run_id] = {
            fname: (len(r0), len(r1), len(r2), len(preq))
            for fname, (r0, r1, r2, preq) in res.items()
        }

    return all_stats


def compute_mean_std(values: List[int]) -> str:
    if not values:
        return "0.00±0.00"
    if len(values) == 1:
        return f"{values[0]:.2f}±0.00"
    m = mean(values)
    s = stdev(values)
    return f"{m:.2f}±{s:.2f}"


def export_to_excel(all_stats, out_path: str):
    wb = Workbook()

    # ----- raw sheet -----
    ws_raw = wb.active
    ws_raw.title = "raw"
    ws_raw.append([
        "run_id", "file",
        "round0_count", "round1_count", "round2_count", "prequery_count",
        "round0_ratio(%)", "round1_ratio(%)", "round2_ratio(%)", "prequery_ratio(%)",
        "round0_share_in_correct(%)", "round1_share_in_correct(%)",
        "round2_share_in_correct(%)", "prequery_share_in_correct(%)",
    ])

    for run_id in sorted(all_stats.keys(), key=lambda x: int(x)):
        for fname in sorted(all_stats[run_id].keys()):
            n0, n1, n2, npre = all_stats[run_id][fname]
            # 总样本 200 → 百分比 = count/200*100 = count/2
            r0p = n0 / 2
            r1p = n1 / 2
            r2p = n2 / 2
            prep = npre / 2

            total_correct = n0 + n1 + n2 + npre
            if total_correct > 0:
                s0 = n0 / total_correct * 100
                s1 = n1 / total_correct * 100
                s2 = n2 / total_correct * 100
                spre = npre / total_correct * 100
            else:
                s0 = s1 = s2 = spre = 0.0

            ws_raw.append([
                run_id, fname,
                n0, n1, n2, npre,
                r0p, r1p, r2p, prep,
                s0, s1, s2, spre,
            ])

    # ----- summary sheet -----
    ws_sum = wb.create_sheet("summary")
    ws_sum.append([
        "file",
        # 数量 mean±std
        "round0 count mean±std", "round1 count mean±std",
        "round2 count mean±std", "prequery count mean±std",
        # 对 200 的比例 mean±std（百分比）
        "round0 ratio(%) mean±std", "round1 ratio(%) mean±std",
        "round2 ratio(%) mean±std", "prequery ratio(%) mean±std",
        # 在所有正确中的占比 mean±std（百分比）
        "round0 share_in_correct(%) mean±std",
        "round1 share_in_correct(%) mean±std",
        "round2 share_in_correct(%) mean±std",
        "prequery share_in_correct(%) mean±std",
    ])

    # 所有文件名并集
    all_files = set()
    for run_id in all_stats:
        all_files.update(all_stats[run_id].keys())

    for fname in sorted(all_files):
        # 数量
        v0, v1, v2, vpre = [], [], [], []
        # 对 200 的比例（百分比）
        vr0, vr1, vr2, vrpre = [], [], [], []
        # 在所有正确中的占比（百分比）
        vs0, vs1, vs2, vspre = [], [], [], []

        for run_id in all_stats:
            if fname in all_stats[run_id]:
                n0, n1, n2, npre = all_stats[run_id][fname]

                v0.append(n0)
                v1.append(n1)
                v2.append(n2)
                vpre.append(npre)

                vr0.append(n0 / 2)  # count/200*100
                vr1.append(n1 / 2)
                vr2.append(n2 / 2)
                vrpre.append(npre / 2)

                total_correct = n0 + n1 + n2 + npre
                if total_correct > 0:
                    vs0.append(n0 / total_correct * 100)
                    vs1.append(n1 / total_correct * 100)
                    vs2.append(n2 / total_correct * 100)
                    vspre.append(npre / total_correct * 100)
                else:
                    vs0.append(0.0)
                    vs1.append(0.0)
                    vs2.append(0.0)
                    vspre.append(0.0)

        ws_sum.append([
            fname,
            # 数量
            compute_mean_std(v0),
            compute_mean_std(v1),
            compute_mean_std(v2),
            compute_mean_std(vpre),
            # 对 200 的比例（百分比）
            compute_mean_std(vr0),
            compute_mean_std(vr1),
            compute_mean_std(vr2),
            compute_mean_std(vrpre),
            # 在所有正确中的占比（百分比）
            compute_mean_std(vs0),
            compute_mean_std(vs1),
            compute_mean_std(vs2),
            compute_mean_std(vspre),
        ])

    wb.save(out_path)


# ---------- 6. 主入口 ----------
if __name__ == "__main__":
    # path = r"results/opt_results/gpt_35/new/ablation/E/1/p1_woE_102_loose.log"
    # r0, r1, r2, preq = stat_success_rounds(path)
    # print("file:", path)
    # print("round0 success:", r0, len(r0))
    # print("round1 success:", r1, len(r1))
    # print("round2 success:", r2, len(r2))
    # print("prequery success:", preq, len(preq))


    # base_folder = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new"
    # base_folder = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/ablation/R"
    base_folder = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/ablation/T"
    all_stats = collect_stats_across_subfolders(base_folder, subfolders=["1", "2", "3", "4", "5"])

    out_excel = os.path.join(base_folder, "round_success_stats_5runs.xlsx")
    export_to_excel(all_stats, out_excel)
    print("Excel written:", out_excel)

    # 输出一个统计轮次占比的excel文件
