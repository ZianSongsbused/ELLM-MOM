import os
import re
import glob
from pathlib import Path
import pandas as pd

# ----------------------------
# 配置区：改成你的文件夹列表和输出路径
# ----------------------------
FOLDERS = [
    r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/1",
    r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/2",
    r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/3",
    r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/4",
    r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/5",
]
OUTPUT_XLSX = "/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/ratiolikechatdrug.xlsx"

TOTAL_SAMPLES_PER_FILE = 200
INVALID_MARKERS = ["fail: Generation error", "fail:XR", "ReDF no XR"]

# 文件名：p1_{taskid}_{constraint}.log
FILENAME_RE = re.compile(r"^p1_(\d+)_(loose|strict)\.log$", re.I)

# 只在 Final Acc 区块内抓 num_correct/num_all/num_all_f/Acc
FINAL_BLOCK_RE = re.compile(
    r"-{5,}\s*Final Acc\s*-{5,}\s*"
    r"Acc\s*=\s*(\d+)\s*/\s*(\d+)\s*"
    r"num_correct\s*=\s*(\d+)\s*,\s*num_all\s*=\s*(\d+)\s*,\s*num_all_f\s*=\s*(\d+)\s*"
    r"-{5,}",
    re.S
)

# sample 块
SAMPLE_BLOCK_RE = re.compile(
    r">>>>>\s*Sample\s+(\d+)\s*:(.*?)(?=(?:>>>>> Sample\s+\d+\s*:)|\Z)",
    re.S
)

VALID_TASKIDS = set(range(101, 109)) | set(range(201, 207))

def parse_one_log(log_path: str, total_samples: int = TOTAL_SAMPLES_PER_FILE):
    p = Path(log_path)
    name = p.name

    mname = FILENAME_RE.match(name)
    if not mname:
        return None

    taskid = int(mname.group(1))
    constraint = mname.group(2).lower()
    if taskid not in VALID_TASKIDS:
        return None

    text = p.read_text(encoding="utf-8", errors="ignore")

    # 统计 invalid sample：sample 块内出现任一 invalid marker
    blocks = [(int(m.group(1)), m.group(0)) for m in SAMPLE_BLOCK_RE.finditer(text)]
    invalid_samples = {idx for idx, blk in blocks if any(k in blk for k in INVALID_MARKERS)}

    invalid_count = len(invalid_samples)
    success_count = total_samples - invalid_count

    # 从 Final Acc 区块提取（避免匹配到“当前指标 num_correct = ...”）
    matches = list(FINAL_BLOCK_RE.finditer(text))
    mfinal = matches[-1] if matches else None
    print(matches)
    print("_____________________________________")
    if mfinal:
        acc_num = int(mfinal.group(1))
        acc_den = int(mfinal.group(2))
        num_correct = int(mfinal.group(3))
        num_all = int(mfinal.group(4))
        num_all_f = int(mfinal.group(5))
    else:
        acc_num = acc_den = num_correct = num_all = num_all_f = None

    # 你定义的新正确率：num_correct / (200 - invalid_samples_count)
    new_acc = (num_correct / success_count) if (num_correct is not None and success_count > 0) else None

    return {
        "folder": str(p.parent),
        "log_file": name,
        "taskid": taskid,
        "constraint": constraint,
        "total_samples": total_samples,
        "invalid_samples_count": invalid_count,
        "success_samples_count": success_count,
        "num_correct_from_final": num_correct,
        "num_all_from_final": num_all,
        "num_all_f_from_final": num_all_f,
        "acc_from_final_num": acc_num,
        "acc_from_final_den": acc_den,
        "new_accuracy": new_acc,
        "invalid_sample_indices": ",".join(map(str, sorted(invalid_samples))) if invalid_samples else "",
    }

def collect_all(folders):
    rows = []
    for folder in folders:
        for log_path in glob.glob(os.path.join(str(folder), "p1_*_*.log")):
            r = parse_one_log(log_path)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)

def make_mean_std_by_filename(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    跨多个文件夹，对同名 log_file 聚合：均值/标准差
    """
    if df_all.empty:
        return df_all

    numeric_cols = [
        "invalid_samples_count",
        "success_samples_count",
        "num_correct_from_final",
        "new_accuracy",
        "acc_from_final_num",
        "acc_from_final_den",
        "num_all_from_final",
        "num_all_f_from_final",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_all.columns]

    g = df_all.groupby("log_file", dropna=False)

    meta = g[["taskid", "constraint"]].first()
    n_folders = g["folder"].nunique().rename("n_folders")

    mean_df = g[numeric_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df  = g[numeric_cols].std(numeric_only=True, ddof=1).add_suffix("_std")

    out = pd.concat([meta, n_folders, mean_df, std_df], axis=1).reset_index()
    return out

def save_excel(df_all: pd.DataFrame, out_xlsx: str):
    mean_std = make_mean_std_by_filename(df_all)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # 1) 总表
        df_all.sort_values(["folder", "log_file"], inplace=True, ignore_index=True)
        df_all.to_excel(writer, index=False, sheet_name="all_results")

        # 2) 跨文件夹同名文件均值/标准差
        mean_std.sort_values(["taskid", "constraint", "log_file"], inplace=True, ignore_index=True)
        mean_std.to_excel(writer, index=False, sheet_name="mean_std_by_filename")

    return out_xlsx

if __name__ == "__main__":
    df = collect_all(FOLDERS)
    if df.empty:
        raise RuntimeError("未找到任何符合命名规则的 .log 文件：p1_{taskid}_{loose|strict}.log")

    out = save_excel(df, OUTPUT_XLSX)
    print(f"Saved: {out}")

