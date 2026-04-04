import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import csv
import json
import subprocess
import os
from utils.convenient_utils.suppress_useless_print import suppress_everything

#  python -m utils.process_drug_utils.mmpa_tool
# 从record中提取所有优化成功的输入分子与优化后分子对
def extract_success_and_failed_pairs(record):
    success_pairs, failed_records = [], []  # 每个元素是输入分子和成功优化分子的元组，每个元素是包含相关信息的字典
    for input_smi, record_info in record.items():  # 遍历每个smiles
        retrievals = record_info.get('retrieval_conversation', [])  # smiles下的每轮对话
        success_found = False
        for step in retrievals:  # 遍历smiles下的每轮对话
            result, generated, retrieval, answer \
                = step.get("result"), step.get("generated_drug"), step.get("retrieval_drug"), step.get("answer")

            if result == -1:  # 跳过直接查表的轮次
                continue
            if str(answer) == "True" and not success_found:  # 本轮是成功记录
                success_pairs.append((input_smi, generated))
                success_found = True
                continue  # 为true之后后续就没内容了，所以跳过即可
            elif generated:  # 本轮是失败伦茨
                failed_records.append({"input": input_smi, "generated": generated,
                                       "retrieval": retrieval, "reason": "未满足优化目标", "step": result})

    return success_pairs, failed_records


# 根据任务编号返回对应的属性计算函数
def get_target_prop_func(task_id):
    """
    :return: 属性函数，例如 Descriptors.MolLogP
    """
    if task_id in [101, 102, 201, 202, 203, 204, 205, 206]:
        return Descriptors.MolLogP
    elif task_id in [103, 104]:
        return QED.qed
    elif task_id in [105, 106, 205, 206]:
        return Descriptors.TPSA
    elif task_id == 107:
        return Descriptors.NumHAcceptors
    elif task_id == 108:
        return Descriptors.NumHDonors
    else:
        raise ValueError(f"当前 task_id = {task_id} 无对应属性函数")


# 将成功分子写入CSV文件，以供 mmpdb 处理。
def write_mmpdb_csv(pairs, task_id, prop_func, constraint):
    out_file = f"./results/mmpa/res_task{task_id}_{constraint}_succ.csv"
    with open(out_file, "w", newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["input_smiles", "generated_smiles", "property_input", "property_generated"])
        writer.writeheader()  # 写表头
        for input_smi, gen_smi in pairs:  # 遍历所有成成功的优化分子对
            input_mol = Chem.MolFromSmiles(input_smi)
            gen_mol = Chem.MolFromSmiles(gen_smi)
            if input_mol is None or gen_mol is None:
                continue
            try:  # 计算输入和输出的prop_func属性值
                input_value = prop_func(input_mol)
                gen_value = prop_func(gen_mol)
                writer.writerow({"input_smiles": input_smi, "generated_smiles": gen_smi,
                                 "property_input": input_value, "property_generated": gen_value})
            except Exception as e:
                print(f"[属性计算失败，跳过] {input_smi} -> {gen_smi}: {e}")
    print(f"[保存成功] 结果写入：{out_file}")


def write_failed_pairs_csv(failed_records, task_id, prop_func):
    out_file = f"./results/mmpa/res_task{task_id}_{constraint}_fail.csv"
    with open(out_file, "w", newline='') as f:
        fieldnames = ["input", "generated", "retrieval", "reason", "step", "prop_input", "prop_generated", "prop_diff"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # 写表头
        for item in failed_records:  # 字段名要和列名一致
            try:
                with suppress_everything():
                    mol_input = Chem.MolFromSmiles(item["input"])
                    mol_gen = Chem.MolFromSmiles(item["generated"])
                if mol_input is None or mol_gen is None:
                    continue  # 这种就不写进去文件了
                val_input = prop_func(mol_input)
                val_gen = prop_func(mol_gen)  # 额外添加这几个字段
                item["prop_input"], item["prop_generated"], item["prop_diff"] = val_input, val_gen, val_gen - val_input
            except:
                item["prop_input"] = item["prop_generated"] = item["prop_diff"] = "ERROR"
            writer.writerow(item)
    print(f"[失败记录] 写入 {out_file}, 总计 {len(failed_records)} 条")


# 从输入记录中提取成功和失败的分子对，计算属性并保存成 CSV 文件，供 MMP 分析使用。
def run_pipeline_for_task(record_path, task_id, constraint):
    """
    参数：
        record_path (str): 存储分子优化记录的 JSON 文件路径
        task_id (int): 任务编号（例如101、203等）
        constraint (str): "loose" 或 "strict"，影响属性阈值
    """
    print(f"\n[开始任务] Task ID: {task_id}，约束：{constraint}")
    with open(record_path, "r") as f:  # 1. 加载 JSON 格式的对话记录
        record = json.load(f)

    try:  # 2. 获取属性函数（根据任务编号）
        prop_func = get_target_prop_func(task_id)
    except Exception as e:
        print(f"[错误] 获取属性函数失败：{e}")
        return

    success_pairs, failed_records = extract_success_and_failed_pairs(record)    # 3. 提取成功分子对和失败分子记录
    print(f"[统计] 成功对: {len(success_pairs)}，失败记录: {len(failed_records)}")

    write_mmpdb_csv(success_pairs, task_id, prop_func, constraint)      # 4. 写入成功记录文件（包括属性值）
    write_failed_pairs_csv(failed_records, task_id, prop_func)          # 5. 写入失败记录文件（包括属性差值）

    print(f"[完成任务] Task {task_id} 已处理完毕。\n")


def run_mmpa_pipeline(task_id: int, constraint='loose'):
    print(f"\n[启动 MMPA 分析] Task ID: {task_id}, Constraint: {constraint}")

    base_dir = "./results/mmpa"
    input_csv = os.path.join(base_dir, f"res_task{task_id}_{constraint}_succ.csv")
    smiles_txt = os.path.join(base_dir, f"task{task_id}_{constraint}_smiles.txt")
    fragments_txt = os.path.join(base_dir, f"task{task_id}_{constraint}_succ_fragments.txt")
    db_file = os.path.join(base_dir, f"task{task_id}_{constraint}_succ.db")
    mmpa_output = os.path.join(base_dir, f"task{task_id}_{constraint}_succ_mma.csv")
    tmp_dir = os.path.join(base_dir, "tmp_transform")
    os.makedirs(tmp_dir, exist_ok=True)

    # Step 1: 提取 smiles（只需要 input_smiles 一列）
    generated_smiles = []
    with open(input_csv, "r") as fin, open(smiles_txt, "w") as fout:
        reader = csv.DictReader(fin)
        for idx, row in enumerate(reader):
            fout.write(f"{row['input_smiles']} mol{idx}\n")
            generated_smiles.append(row['generated_smiles'])
    print(f"[1/4] SMILES 提取完成：{smiles_txt}")

    # Step 2: 将每个分子切碎成 fragments
    cmd_frag = ["mmpdb", "fragment", smiles_txt, "--output", fragments_txt]
    print(f"[2/4] split fragment...")
    try:
        subprocess.run(cmd_frag, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[错误] fragment 失败：{e}")
        return

    # Step 3: index操作，将前一步得到的 fragment 信息存入一个 SQLite 数据库，供 transform 使用index
    cmd_index = ["mmpdb", "index", fragments_txt, "--output", db_file]
    print(f"[3/4] construct index...")
    try:
        subprocess.run(cmd_index, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[错误] index 失败：{e}")
        return

    # Step 4: transform，循环调用每个生成分子
    transform_files = []
    print(f"[4/4] transform (逐分子)...")
    for i, smi in enumerate(generated_smiles):
        if not smi:
            continue
        out_csv = os.path.join(tmp_dir, f"transform_{i}.csv")
        # cmd_transform = ["mmpdb", "transform", db_file, "--smiles", smi, "--output", out_csv]
        cmd_transform = ["mmpdb", "transform", db_file, "--smiles", smi, "--output", out_csv, "--min-pairs", "1"]
        try:
            subprocess.run(cmd_transform, check=True)
            transform_files.append(out_csv)
        except subprocess.CalledProcessError as e:
            print(f"[警告] transform失败，分子{i} ({smi}): {e}")

    # 合并所有transform结果
    if transform_files:
        df_list = [pd.read_csv(f) for f in transform_files if os.path.getsize(f) > 0]
        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
            df_all.to_csv(mmpa_output, index=False)
            print(f"\n✅ MMPA 分析完成，合并输出结果位于：{mmpa_output}")
        else:
            print("[错误] 没有有效的transform结果文件")
    else:
        print("[错误] 没有生成任何transform结果文件")


# 示例入口（如果作为主程序运行）
if __name__ == "__main__":
    # 假设你从某个文件加载record
    record_path = "/home/aita8180/data/mntdata/ziansong/p1/results/p1_102_loose_galactica-run1.json"
    # record_path = "/home/wangshuang/ziansong/p1/results/p1_102_loose_galactica-run1.json"
    # 替换为你想分析的任务编号
    task_id = 102
    constraint = 'loose'

    run_pipeline_for_task(record_path, task_id, constraint)
    run_mmpa_pipeline(task_id, constraint)
