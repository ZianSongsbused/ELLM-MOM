from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import re

from utils.convenient_utils.wordart import ColorText
from utils.process_drug_utils.moledit_evaltool import evaluate, task2threshold_list


# 从llm返回的内容中解析出SMILES串（基于规则的方式）
def parse_molecule(input_sequence, raw_text, retrieval_sequence):
    pattern = re.compile(r'[0-9BCOHNSOPrIFlanocs@+\.\-\[\]\(\)\\\/%=#$]{6,}')
    output_sequence_list = pattern.findall(raw_text)  # 用正则匹配出回复里的所有的SMILES串
    while input_sequence in output_sequence_list:
        # 如果生成了和输入序列一样的，这是错误序列直接剃出
        output_sequence_list.remove(input_sequence)

    if retrieval_sequence != None:
        while retrieval_sequence in output_sequence_list:
            # 如果生成了和XR一样的，这是错误序列直接剃出
            output_sequence_list.remove(retrieval_sequence)

    if len(output_sequence_list) > 0:
        output_sequence = [output_sequence_list[0]]
    else:
        print("××××rawtext解析结果：LLM生成的全是相同分子")
        output_sequence = []
    return output_sequence


# 只有小分子就一个分支了，调用对应的解析分子(从生成的rawtext中提取出所有分子)函数
def parse(task, input_drug, generated_text, addition_drug=None):
    if task < 300:
        return parse_molecule(input_drug, generated_text, addition_drug)
    else:
        raise NotImplementedError

# mainnew里没用，不用管
def parse_molecule2(input_sequence, retrieval_sequence, raw_text, chatmol_lst,
                    fix_mol="false", task=None, constraint=None):
    """
    fix_mol:
    - 如果是 "false"（字符串），则返回第一个满足 evaluate 条件的分子；
    - 如果是整数 int，则直接返回 output_sequence_list[fix_mol]；
    不管fixmol是int还是false，最红返回的其实都是一个只有一个元素的list
    """
    # step.1 拿到返回的所有smiles
    pattern = re.compile(r'[0-9BCOHNSOPrIFlanocs@+\.\-\[\]\(\)\\\/%=#$]{6,}')
    output_sequence_list = pattern.findall(raw_text)  # 用正则匹配出回复里的所有的SMILES串
    if chatmol_lst is not None:
        output_sequence_list.extend(chatmol_lst)  # 把chatmol生成的分子列表拼到一块解析
    while input_sequence in output_sequence_list:     # 如果生成了和输入序列一样的smiles，直接剃出
        output_sequence_list.remove(input_sequence)
    if retrieval_sequence is not None:
        while retrieval_sequence in output_sequence_list:        # 如果生成了和XR一样的smiles，也直接剃出
            output_sequence_list.remove(retrieval_sequence)
    if not output_sequence_list:  # 没生成新分子
        return []

    def is_valid(smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return False
            return True
        except Exception:
            return False
    output_sequence_list = [s for s in output_sequence_list if is_valid(s)]
    if not output_sequence_list:
        return []
    # print(f"{' '*5}" + f"parse时提取LLM回复、拼接chatmol结果、去重后，匹配出了{len(output_sequence_list)}个smiles：\n{output_sequence_list}")

    # --- fix_mol 是 "false"，走 evaluate 判断路径
    if isinstance(fix_mol, str) and fix_mol.lower() == "false":
        print(f"{' '*5}" + "parse时，评估所有生成分子".center(28, '-'))
        errcnt = 0  # 用来计数不合规的smiles，如果全部不合规就返回None

        first_legal_but_not_satisfying = None  # 用于备选：合法但不满足 Δ 的
        for i, cand in enumerate(output_sequence_list):
            answer = evaluate(input_sequence, cand, task, constraint)
            if answer == 1:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})找到满足Δ的 SMILES".center(29)+"|", ColorText.GREEN)
                return [cand]
            elif answer == 0:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})评估不通过 SMILES".center(29)+"|", ColorText.PURPLE)
                if first_legal_but_not_satisfying is None:  # 记录下来合法但不满足Δ的分子，如果所有分子都没满足Δ就返回这个
                    first_legal_but_not_satisfying = cand
            else:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})无法评估 SMILES".center(30)+"|", ColorText.RED)
                errcnt += 1
        if errcnt == len(output_sequence_list) or first_legal_but_not_satisfying is None:  # 如果全部不合规就返回None
            ColorText.print(f"{' '*5}" + "failed".center(38, '-'), ColorText.RED)
            return None
        else:
            ColorText.print(f"{' '*5}" + "×没有SMILES满足Δ，默认返回首个合法分子".center(28, '-'), ColorText.PURPLE)
            return [first_legal_but_not_satisfying]  # 返回第一个合法分子
    # --- fix_mol 是 int时，直接返回int对应的位置的分子
    else:
        fix_mol = int(fix_mol)
        print(f"{' ' * 5}" + "parse时，选择固定位置分子".center(28, '-'))
        if 0 <= fix_mol < len(output_sequence_list):
            return [output_sequence_list[fix_mol]]
        else:
            ColorText.print(f"fix_mol 超出范围：{fix_mol}（最大 index: {len(output_sequence_list) - 1}），默认返回列表中首个smiles", ColorText.PURPLE)
            return [output_sequence_list[0]]


# 考虑chatmol生成分子的解析函数（主要是查重）
def parse2(input_drug, generated_text, chatmol_lst, retrieval_drug=None, fix_mol="false", task=None, constraint=None):
    return parse_molecule2(input_drug, retrieval_drug, generated_text, chatmol_lst, fix_mol, task, constraint)

def prop_direc_from_taskid(taskid):
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
def parse_and_eval(input_sequence, retrieval_sequence, gennerate_smiles,
                    fix_mol="false", task=None, constraint=None, showdetail=False, direction='increase'):
    """
    fix_mol:
    - 如果是 "false"（字符串），则返回第一个满足 evaluate 条件的分子；
    - 如果是整数 int，则直接返回 output_sequence_list[fix_mol]；
    不管fixmol是int还是false，最红返回的其实都是一个只有一个元素的list
    """
    # step.1 生成分子去重排查
    output_sequence_list = gennerate_smiles  # 手动编辑的就不用再正则匹配了

    len1 = len(output_sequence_list)
    while input_sequence in output_sequence_list:     # 如果生成了和输入序列一样的smiles，直接剃出
        output_sequence_list.remove(input_sequence)
    if retrieval_sequence is not None:
        while retrieval_sequence in output_sequence_list:        # 如果生成了和XR一样的smiles，也直接剃出
            output_sequence_list.remove(retrieval_sequence)
    if not output_sequence_list:  # 没生成新分子
        return []
    output_sequence_list = [s.replace('*', '') for s in output_sequence_list]
    len2 = len(output_sequence_list)
    if len1!=len2 and len2==0:
        ColorText.print("生成的全是重复分子！！",  ColorText.RED)

    print("【本轮生成分子】(去重后)")
    for i, cand in enumerate(output_sequence_list):
        print(f"（序号 {i}） {cand}")

    # step.2 评估分子
    # --- fix_mol 是 "false"，走 evaluate 判断路径
    if isinstance(fix_mol, str) and fix_mol.lower() == "false":
        print(f"{' '*5}" + "parse时，评估所有生成分子".center(28, '-'))
        errcnt = 0  # 用来计数不合规的smiles，如果全部不合法就返回None
        delta_records = []  # 记录每个分子的delta，存在合法但不合约束的就返回离目标最近的
        first_legal_but_not_satisfying = None  # 用于备选：合法但不满足 Δ 的

        for i, cand in enumerate(output_sequence_list):
            # answer = evaluate(input_sequence, cand, task, constraint, showdetail)
            answer, delta = evaluate(input_sequence, cand, task, constraint, showdetail)
            # answer是布尔值，delta在单属性的时候是单值、多属性时时元组
            if delta is not None and answer is not None:
                delta_records.append((cand, delta))  # 生成分子，实际属性变化量 的元组
            if answer == 1:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})找到满足Δ的 SMILES".center(29)+"|", ColorText.GREEN)
                return [cand]
            elif answer == 0:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})评估不通过 SMILES".center(29)+"|", ColorText.PURPLE)
                if first_legal_but_not_satisfying is None:  # 记录下来合法但不满足Δ的分子，如果所有分子都没满足Δ就返回这个
                    first_legal_but_not_satisfying = cand
            else:
                ColorText.print(f"{' '*5}" + "|"+f"(序号 {i})无法评估 SMILES".center(30)+"|", ColorText.RED)
                errcnt += 1

        # 根据评估结果返回结果
        if errcnt == len(output_sequence_list) or first_legal_but_not_satisfying is None:  # 如果全部不合规就返回None
            ColorText.print(f"{' '*5}" + "failed".center(38, '-'), ColorText.RED)
            return None
        else:  # KS  根据target_delta选择最接近的
            # # =============== [NEW] 单/多属性统一的“最近成功”选择逻辑 ==================
            # if delta_records:
            #     # 1) 取任务对应的属性列表和方向列表，比如 ["logp","tpsa"], ["decrease","decrease"]
            #     target_prop_lst, target_direc_lst = prop_direc_from_taskid(task)
            #
            #     # 2) 取当前约束对应的阈值列表（和 target_prop_lst 顺序一致）
            #     threshold_idx = 0 if constraint == "loose" else 1     # 下标
            #     target_deltas = task2threshold_list[task][threshold_idx]
            #     best_cand, best_score, eps = None, -1.0, 1e-8
            #
            #     for cand, d in delta_records:  # delta 统一视为 list
            #         if isinstance(d, (list, tuple)):
            #             deltas = list(d)               # 多属性时定成长度为K的list
            #         else:
            #             deltas = [d]                   # 单属性时定成长度为1的list
            #
            #         L = min(len(deltas), len(target_deltas), len(target_direc_lst))
            #
            #         score_sum = 0.0
            #         for k in range(L):                      # 遍历每个属性
            #             d_k = float(deltas[k])
            #             t_k = float(target_deltas[k])
            #             direc_k = str(target_direc_lst[k]).lower()
            #
            #             # increase -> sign = +1, decrease -> sign = -1  因为dk带符号，所以sign也带符号
            #             sign_k = 1.0 if direc_k == "increase" else -1.0   #
            #
            #             # norm_k表示结果正确与否
            #             if abs(t_k) == 0:           # loose约束
            #                 norm_k = 1.0 if sign_k * d_k > 0 else 0.0   # >0就是与预期方向一致，loose就满足了
            #             else:                       # strict约束
            #                 progress_k = sign_k * d_k / abs(t_k)  # pk=实际变化量/目标变化量，即为1的时候是刚好满足约束
            #                 if progress_k <= 0:                   # d_k【与预期方向不一致】就是负的了
            #                     norm_k = 0.0
            #                 elif progress_k >= 1:                 # d_k【与预期方向一致】且【变化量超过约束】就会大于1
            #                     norm_k = 1.0  # 已达到或超过目标
            #                 else:                                 # d_k【与预期方向一致】但【变化量没超约束】就会<1,>0
            #                     norm_k = progress_k
            #
            #             score_sum += norm_k
            #
            #         score = score_sum / L  # 多属性平均进展度，score_sum=L就是完成了，<L是没全完成
            #
            #         if score > best_score:
            #             best_score, best_cand = score, cand
            #
            #     # 3) 如果有至少一个属性朝正确方向推进，则返回多目标 score 最大的分子
            #     if best_cand is not None and best_score > 0:
            #         ColorText.print(
            #             f"{' ' * 5}" + f"×没有SMILES满足Δ，多目标进展度最大={best_score:.3f}，返回该分子".center(38, '-'),
            #             ColorText.PURPLE
            #         )
            #         return [best_cand]
            #     else:
            #         # 所有分子在所有属性上要么没动，要么反方向 → 保持旧逻辑：返回第一个合法分子
            #         ColorText.print(
            #             f"{' ' * 5}" + "×没有SMILES满足Δ，多目标进展度全为0，默认返回首个合法分子".center(38, '-'),
            #             ColorText.PURPLE
            #         )
            #         return [first_legal_but_not_satisfying]
            # else:
            #     ColorText.print(f"{' ' * 5}" + "×没有SMILES满足Δ，默认返回首个合法分子".center(28, '-'), ColorText.PURPLE)
            #     return [first_legal_but_not_satisfying]

            # ================== 分支1：单属性（保持原始逻辑） ==================
            # delta 是标量，仍然走“方向 + 绝对 Δ 最大/最小” 策略
            # ==================================================================
            # 这里通过“delta 是否为标量”来区分单属性 vs 多属性任务
            is_scalar_delta = False
            for cand, delta in delta_records:   # 拿delta_records的第1个合法元素的第2个值（即实际变化量）来判断
                if Chem.MolFromSmiles(cand) is not None:
                    is_scalar_delta = isinstance(delta, tuple)

            if not is_scalar_delta:
                # —— 单属性：保留原有逻辑不动 —— #
                target_sign = 1 if direction == 'increase' else -1  # 优化方向
                threshold_idx = 0 if constraint == "loose" else 1
                # target_delta = task2threshold_list[task][threshold_idx]  # 目标Δ
                best_cand, min_d, max_d = None, float('inf'), 0.0

                for cand, d in delta_records:
                    if d * target_sign <= 0:    # 暂时忽略方向完全相反的
                        continue
                    if abs(d) > max_d:          # 方向正确时，优先选“Δ 最大”的
                        max_d, best_cand = abs(d), cand

                # 如果一个“方向正确”的都没有，就选绝对变化最小的（尽量离原点近）
                if best_cand is None:
                    for cand, d in delta_records:
                        if abs(d) < min_d:
                            min_d, best_cand = abs(d), cand

                return [best_cand]

            # ================== 分支2：多属性（新的多目标 Δ 选择逻辑） ==================
            # 核心思想：
            #   1）先按“方向正确的属性个数占比”优先排序；
            #   2）在方向正确的属性上，看“朝着目标推进了多大比例”（forward_progress）；
            #   3）再用“反向偏离有多严重”（backward_penalty）微调；
            #   这样可以区分：
            #       - 两个属性都方向正确但幅度小；
            #       - 一个属性冲得很猛但另一个反向；
            #       - 两个都反向；
            #       - 全都几乎不动；
            #   并且在“所有候选都方向反了”的情况下，会自动选择“反得最轻”的那个。
            # ==================================================================
            # 1) 拿到任务对应的属性列表 & 期望方向
            target_prop_lst, target_direc_lst = prop_direc_from_taskid(task)
            L = len(target_prop_lst)

            threshold_idx = 0 if constraint == "loose" else 1
            # task2threshold_list[task][threshold_idx] 是一个 list，和 target_prop_lst 顺序一致
            target_delta_list = task2threshold_list[task][threshold_idx]

            best_cand = None
            best_key = None  # 排序 key: (direction_correct_ratio, forward_progress, -backward_penalty)

            for cand, delta_vec in delta_records:
                if Chem.MolFromSmiles(cand) is None:   # cand不合法也能评分，这里剔除掉
                    continue
                deltas = list(delta_vec[:L])           # 实际变化值转成list

                n_forward = 0  # 方向正确的属性个数
                forward_sum = 0.0  # 正向推进累积（只算方向正确的）
                backward_sum = 0.0  # 反向偏离累积（只算方向错误的）

                for idx, prop in enumerate(target_prop_lst):     # 遍历每个属性
                    # 实际变化量，要求变化方向，要求变化量
                    d, direc, t_delta = float(deltas[idx]), target_direc_lst[idx].lower(),float(target_delta_list[idx])

                    if t_delta == 0:    # 这里时为了避免除零：如果配置中给的是 0（loose 档），就只用 |Δ| 自身做尺度
                        t_delta = 1.0

                    if abs(d) < 1e-9:
                        # 变化几乎为 0：既不计入 forward，也不计入 backward
                        continue

                    sign = 1.0 if direc == "increase" else -1.0    # 因为dk带符号，所以sign也带符号
                    # Ⅰ.方向正确性
                    dir_correct = (sign * d) > 0      # sign * d > 0 即朝着正确方向走； <0 即反方向

                    # Ⅱ.修改满足比例
                    progress = abs(d) / t_delta    # 归一化到 [0,1]
                    if progress > 1.0:             # 满足约束
                        progress = 1.0             # 满足就行多了没用，因此1就行

                    if dir_correct:               # 方向正确
                        n_forward += 1
                        forward_sum += progress   # 只在方向正确时累计修改进度
                    else:                         # 方向错误  不用参考目标了，用实际变化量
                        penalty = abs(d) / t_delta  # 用目标Δ归一化，但不截断
                        backward_sum += penalty     # 这里后面要减掉，所以越大越坏
                # 遍历完所有属性了

                # Ⅰ. direction_correct_ratio：方向正确的属性比例
                direction_correct_ratio = n_forward / float(L)

                # Ⅱ. forward_progress：在“方向正确”的属性上，平均推进比例
                if n_forward > 0:     # 有方向正确的再说
                    forward_progress = forward_sum / float(n_forward)
                else:
                    forward_progress = 0.0

                # Ⅲ. backward_penalty：反向偏离强度（按属性数归一）
                backward_penalty = backward_sum / float(L)

                # 排序 key：
                #   先最大化 direction_correct_ratio（两正(包括改动很小的情况) > 一正一反 > 全错/不动）
                #   再最大化 forward_progress（正向走得多的排前）
                #   最后最小化 backward_penalty（反向偏离越少越好）
                key = (direction_correct_ratio, forward_progress, -backward_penalty)

                if (best_key is None) or (key > best_key):  # key > best_key按照key的顺序依次对比
                    best_key = key
                    best_cand = cand

            # 所有候选都评估完毕后，选择 key 最大的那个
            if best_cand is not None:
                return [best_cand]
            else:
                # 理论上很难到这一步，兜底：返回第一个合法但不满足 Δ 的分子
                ColorText.print(f"{' ' * 5}" + "×多属性Δ评分失败，默认返回首个合法分子".center(28, '-'), ColorText.PURPLE)
                return [first_legal_but_not_satisfying]



    # --- fix_mol 是 int时，直接返回int对应的位置的分子
    else:
        fix_mol = int(fix_mol)
        print(f"{' ' * 5}" + "parse时，选择固定位置分子".center(28, '-'))
        if 0 <= fix_mol < len(output_sequence_list):
            return [output_sequence_list[fix_mol]]
        else:
            ColorText.print(f"fix_mol 超出范围：{fix_mol}（最大 index: {len(output_sequence_list) - 1}），默认返回列表中首个smiles", ColorText.PURPLE)
            return [output_sequence_list[0]]



