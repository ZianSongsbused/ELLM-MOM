import re
import ast
import os  # >>>> 新增：遍历文件夹用

def strip_star(s: str) -> str:
    # 忽略星号
    return s.replace("*", "")

# ---------- 1. 按 Sample 切块 ----------

def extract_sample_blocks(text: str):
    """
    把整个 log 拆成 {sample_idx: [lines]} 这样的字典
    匹配形式：>>>> Sample 0:
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


# ---------- 2. 只取 round_index:0 那一段 ----------

def get_round0_segment(lines):
    """
    从一个 sample 的全部行中，截出 round_index:0 下的那一段：
    从包含 'round_index:0' 的行的下一行开始，
    到下一个 'round_index:' 出现的前一行为止。
    """
    start = None
    for i, line in enumerate(lines):
        if "round_index:0" in line:
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if "round_index:" in lines[j]:
            end = j
            break

    return lines[start + 1 : end]


# ---------- 3. 解析 round0 里 rulbank/rulebank_multi 的 new_smiles ----------

def parse_rulebank_smiles_from_segment(lines):
    """
    在 round0 这一小段中，解析【rulbank】 / 【rulebank】 / 【rulebank_multi】后面的 dict，
    把里面所有 new_smiles 抽出来（递归地找所有 new_smiles）。
    """
    smiles_set = set()

    for line in lines:
        if "【rulbank】" in line or "【rulebank_multi】" in line or "【rulebank】" in line:
            # 去掉前面的全角括号前缀
            try:
                payload = line.split("】", 1)[1].strip()
            except IndexError:
                continue

            # payload 是 Python 风格的 dict，用 literal_eval 直接解析
            try:
                obj = ast.literal_eval(payload)
            except Exception:
                # 如果末尾有日志噪声，尝试从最后一个 } 或 ] 截断后再解析
                last = max(payload.rfind("}"), payload.rfind("]"))
                if last == -1:
                    continue
                try:
                    obj = ast.literal_eval(payload[: last + 1])
                except Exception:
                    continue

            # 递归收集所有 new_smiles
            def traverse(x):
                if isinstance(x, dict):
                    if "new_smiles" in x and isinstance(x["new_smiles"], str):
                        smiles_set.add(strip_star(x["new_smiles"]))
                    for v in x.values():
                        traverse(v)
                elif isinstance(x, (list, tuple, set)):
                    for v in x:
                        traverse(v)

            traverse(obj)

    return sorted(smiles_set)


# ---------- 4. 解析 round0 里的 Generated Result ----------

def parse_generated_from_segment(lines):
    """
    在 round0 段中，找到第一条 'Generated Result:'，取后面的 SMILES。
    """
    for line in lines:
        if "Generated Result:" in line:
            s = line.split("Generated Result:", 1)[1].strip()
            if s:
                return strip_star(s)
    return None


# ---------- 5. 判断是否包含 “Pre-query DB: ... Task early stopped.” ----------

def contains_pre_query_sample(lines):
    """
    整个 sample 里只要有一行同时包含 'Pre-query DB:' 和 'Task early stopped.' 就计入 list2。
    """
    for line in lines:
        if "Pre-query DB:" in line and "Task early stopped" in line:
            return True
    return False


# ---------- 5b. Evaluation result 相关逻辑 ----------

def has_eval_true_any_round(lines):  # >>>> 修改：从只看 round0 改成看所有 round
    """
    任意 round_index 段中是否出现过 'Evaluation result is True'
    """
    return any("Evaluation result is True" in l for l in lines)


def all_rounds_eval_false(lines):  # >>>> 修改：无 Evaluation 行时也视为“全 False”
    """
    所有 round_index 段中的 Evaluation result 是否全为 False：
    - 只要有一行 'Evaluation result is True' 就返回 False
    - 若完全没有 Evaluation result 行，也视为 True（保证两类能覆盖 list1）
    """
    eval_lines = [l for l in lines if "Evaluation result" in l]
    if not eval_lines:
        return True
    return all("Evaluation result is False" in l for l in eval_lines)


# ---------- 6. 汇总：得到两个 list + 细分 ----------

def process_file(path):
    """
    返回：
    list1: Generated Result 的 SMILES（去掉*）不在 rulbank/rulebank_multi 任何 new_smiles 里的 sample 序号
    list2: sample 中出现 'Pre-query DB: ... Task early stopped.' 的 sample 序号

    list1_round_true_any: 属于 list1 且任意 round 出现过 'Evaluation result is True' 的样本
    list1_all_rounds_false: 属于 list1 且所有 round 的 Evaluation result 全为 False 的样本
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    blocks = extract_sample_blocks(text)
    list1 = []
    list2 = []
    list1_round_true_any = []      # >>>> 修改命名：更符合现在逻辑
    list1_all_rounds_false = []

    for idx, lines in blocks.items():
        # 规则 2：Pre-query DB 提前终止
        if contains_pre_query_sample(lines):
            list2.append(idx)

        # 规则 1：只看 round_index:0 来决定是否进 list1
        segment = get_round0_segment(lines)
        if segment is None:
            continue

        gen = parse_generated_from_segment(segment)
        if gen is None:
            continue

        rb_smiles = parse_rulebank_smiles_from_segment(segment)

        in_list1 = False
        # 找不到任何 rulebank 的 new_smiles，按“未出现在 rulebank”算
        if not rb_smiles:
            list1.append(idx)
            in_list1 = True
        else:
            if gen not in rb_smiles:
                list1.append(idx)
                in_list1 = True

        # 对已经进入 list1 的样本，进一步细分
        if in_list1:
            if has_eval_true_any_round(lines):          # >>>> 修改：使用新的“任意 round 有 True”
                list1_round_true_any.append(idx)
            elif all_rounds_eval_false(lines):
                list1_all_rounds_false.append(idx)
            else:
                # 理论上不会进这个分支，除非日志格式很奇怪
                # 可以留空，也可以按需打印调试
                pass

    list1.sort()
    list2.sort()
    list1_round_true_any.sort()
    list1_all_rounds_false.sort()
    return list1, list2, list1_round_true_any, list1_all_rounds_false


# ---------- 7. 遍历文件夹：返回每个文件的 4 个 list ----------

def process_folder(folder_path):
    """
    输入：包含若干 .log 文件的文件夹
    输出：dict，键为文件名，值为 (list1, list2, list1_round_true_any, list1_all_rounds_false)
    """
    results = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".log"):
            fpath = os.path.join(folder_path, fname)
            lists = process_file(fpath)
            results[fname] = lists
    return results



if __name__ == "__main__":
    '''
    path_101 = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/p1_101_loose.log"
    path_204 = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/p1_204_loose.log"

    list1_101, list2_101, l1_true_101, l1_all_false_101 = process_file(path_101)
    list1_204, list2_204, l1_true_204, l1_all_false_204 = process_file(path_204)

    print("p1_101 list1:", list1_101, len(list1_101))
    print("p1_101 list1_round_true_any:", l1_true_101, len(l1_true_101))
    print("p1_101 list1_all_rounds_false:", l1_all_false_101, len(l1_all_false_101))
    print("check union==list1:", sorted(set(l1_true_101) | set(l1_all_false_101)) == list1_101)

    print("p1_204 list1:", list1_204, len(list1_204))
    print("p1_204 list1_round_true_any:", l1_true_204, len(l1_true_204))
    print("p1_204 list1_all_rounds_false:", l1_all_false_204, len(l1_all_false_204))
    print("check union==list1:", sorted(set(l1_true_204) | set(l1_all_false_204)) == list1_204)
    '''
    folder = r"/home/aita8180/data/mntdata/ziansong/p1/results/opt_results/gpt_35/new/withoutPrequery/1/"
    all_results = process_folder(folder)
    for fname, (l1, l2, l1_true, l1_all_false) in all_results.items():
        print(f"\n=== {fname} ===")
        print("list1:", l1, len(l1))
        print("list2:", l2, len(l2))
        print("list1_round_true_any:", l1_true, len(l1_true))
        print("list1_all_rounds_false:", l1_all_false, len(l1_all_false))
        # 可以顺便检查一下：
        print("union==list1:", sorted(set(l1_true) | set(l1_all_false)) == l1)
