from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import re

from utils.convenient_utils.wordart import ColorText

props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]  # 定义需要优化的属性名称
# Descriptors.descList是一个包含许多分子描述符名称及其计算函数的列表。n 是描述符的名称（字符串），func 是计算该描述符的函数。
prop_pred = []
for n, func in Descriptors.descList:
    # Descriptors.descList是RDKit.Chem里面的东西
    if n.split("_")[-1] in props:    # 检查提取的最后一部分是否存在于props列表中来筛选props里定义的描述符。
        prop_pred.append((n, func))
prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func
"""
最终，prop2func 字典会是这样的结构：
{
    "MolLogP": Descriptors.MolLogP,
    "qed": Descriptors.qed,
    "TPSA": Descriptors.TPSA,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors
}
"""

# 这俩函数只在205任务用一下
def _compute_prop_values(smiles, prop2func):
    """直接使用你给的 RDKit prop2func 字典计算性质"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    res = {}
    for pname, fn in prop2func.items():
        try:
            res[pname] = fn(mol)
        except:
            res[pname] = None
    return res
def _result_level_filter(S0_smiles, new_smiles_list, target_props, target_dirs,
                         prop2func, constraint_type):
    """
    基于“结果分子”的真实性质变化筛选规则：
    S0_smiles       : 原分子
    new_smiles_list : 规则执行后的生成分子   list[str]
    target_props    : ["MolLogP","TPSA"]
    target_dirs     : ["increase","increase"]
    prop2func       : 你的RDKit属性函数映射
    constraint_type : "loose" / "strict"

    返回: 过滤后仍保留的 new_smiles_list
    """

    # loose/strict 的阈值（你提供）
    delta_thr = {
        "MolLogP": (0.0, 0.5),   # (loose, strict)
        "TPSA":    (0.0, 10.0)
    }

    mol0_props = _compute_prop_values(S0_smiles, prop2func)
    if mol0_props is None:
        return []

    filtered = []
    for smi in new_smiles_list:
        new_props = _compute_prop_values(smi, prop2func)
        if new_props is None:
            continue

        ok = True
        # 对每个目标属性进行检查
        for prop, direc in zip(target_props, target_dirs):

            base = mol0_props[prop]
            newv = new_props[prop]
            if base is None or newv is None:
                ok = False
                break

            delta = newv - base
            loose_thr, strict_thr = delta_thr[prop]
            need_thr = loose_thr if constraint_type == "loose" else strict_thr

            if direc == "increase":
                if delta <= need_thr:     # 上升时delta是正值，即至少增加need_thr
                    ok = False
                    break                # 有一个属性不满足就是整体不满足
            else:  # decrease
                if delta >= -need_thr:    # 下降时delta是负值，即至少减少need_thr，所以不满足这个就筛掉
                    ok = False
                    break

        if ok:
            filtered.append(smi)

    return filtered

# 不同taskid的loose([0])和strict([1])下的Δ
task2threshold_list = {
    101: [[0], [0.5]],
    102: [[0], [0.5]],
    103: [[0], [0.1]],
    104: [[0], [0.1]],
    105: [[0], [10]],
    106: [[0], [10]],
    107: [[0], [1]],
    108: [[0], [1]],

    201: [[0, 0], [0.5, 1]],
    202: [[0, 0], [0.5, 1]],
    203: [[0, 0], [0.5, 1]],
    204: [[0, 0], [0.5, 1]],
    205: [[0, 0], [0.5, 10]],
    206: [[0, 0], [0.5, 10]],
}

def evaluate_molecule(input_SMILES, output_SMILES, task_id, threshold_list=[0], showdetail=False):
    """
    评估本轮生成的X浪是不是符合要求了
    :param input_SMILES:    本次对话的输入分子
    :param output_SMILES:   LLM本轮的生成分子
    :param task_id:
    :param threshold_list:
    :return:
    """
    input_mol = Chem.MolFromSmiles(input_SMILES)  # 将 input_SMILES 字符串转换为 RDKit 中的分子对象
    Chem.Kekulize(input_mol)  # 对分子进行 Kekulé 化，确保其化学结构是正确的
    try:
        # print(output_SMILES)
        output_mol = Chem.MolFromSmiles(output_SMILES)  # 将 output_SMILES 字符串转换为 RDKit 中的分子对象
        # print(output_mol)
        # Chem.Kekulize(output_mol)
    except Exception as e:
        ColorText.print(f"{' ' * 5}" + "|  " + f'生成分子似乎不合法:{e}'.center(29) + "  |", ColorText.RED)
        return None, None, -1

    if output_mol is None:
        ColorText.print(f"{' ' * 5}" + "|  " + '生成分子是None'.center(29) + "  |", ColorText.RED)
        return None, None, -1

    # 根据不同的taskid（即不同的优化任务）
    elif task_id == 101:
        prop = "MolLogP"
        threshold = threshold_list[0]
        # 调用prop2func里面定义的计算各个分子属性的指标
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value < input_value - threshold  # 输入的属性，输出属性，本输出是否满足了用户要求(即对应属性减少超过Δ)

    elif task_id == 102:
        prop = "MolLogP"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value > input_value + threshold  # 输入的属性，输出属性，本输出是否满足了用户要求(即对应属性增加超过Δ)

    elif task_id == 103:
        prop = "qed"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}|{' ' * 5}" + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}" + "  |")
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 104:
        prop = "qed"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value < input_value - threshold

    elif task_id == 105:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value < input_value - threshold

    elif task_id == 106:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 107:
        prop = "NumHAcceptors"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 108:
        prop = "NumHDonors"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp={input_value:.2f},  out={output_value:.2f},  Δ={output_value-input_value:.2f}".center(29) + "  |")
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 201:
        # 双属性优化问题就是把用元组的方式返回，然后最后一个返回值是两个任务的交集
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 101,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 107,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + f"inp=({input_value_01:.1f},{input_value_02:.1f}), out=({output_value_01:.1f},{output_value_02:.1f}), Δ=({output_value_01 - input_value_01:.1f},{output_value_02 - input_value_02:.1f})".center(29))
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 202:
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 102,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 107,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp=({input_value_01:.1f}, {input_value_02:.1f}),  out=({output_value_01:.1f}, {output_value_02:.2f}),  Δ=({output_value_01 - input_value_01:.2f}, {output_value_02 - input_value_02:.2f})".center(29) + "  |")
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 203:
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 101,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 108,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp=({input_value_01:.1f}, {input_value_02:.1f}),  out=({output_value_01:.1f}, {output_value_02:.2f}),  Δ=({output_value_01 - input_value_01:.2f}, {output_value_02 - input_value_02:.2f})".center(29) + "  |")
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 204:
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 102,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 108,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp=({input_value_01:.1f}, {input_value_02:.1f}),  out=({output_value_01:.1f}, {output_value_02:.2f}),  Δ=({output_value_01 - input_value_01:.2f}, {output_value_02 - input_value_02:.2f})".center(29) + "  |")
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 205:
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 101,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 105,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp=({input_value_01:.1f}, {input_value_02:.1f}),  out=({output_value_01:.1f}, {output_value_02:.2f}),  Δ=({output_value_01 - input_value_01:.2f}, {output_value_02 - input_value_02:.2f})".center(29) + "  |")
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 206:
        input_value_01, output_value_01, result_01 = evaluate_molecule(input_SMILES, output_SMILES, 101,
                                                                       [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_molecule(input_SMILES, output_SMILES, 106,
                                                                       [threshold_list[1]])
        if showdetail:
            print(f"{' ' * 5}" + "|  " + f"inp=({input_value_01:.1f}, {input_value_02:.1f}),  out=({output_value_01:.1f}, {output_value_02:.2f}),  Δ=({output_value_01 - input_value_01:.2f}, {output_value_02 - input_value_02:.2f})".center(29) + "  |")
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02


# 只有小分子就一个分支了，根据constraint评估是否达到了D(~,~;~)
def evaluate(input_drug, generated_drug, task, constraint, showdetail=False):
    if isinstance(task, str):
        print("tmd怎么变成str了")
        task = int(task)
    if constraint == 'loose':
        threshold_list = task2threshold_list[task][0]  # 找到loose约束下taskid对应的具体阈值
    else:
        threshold_list = task2threshold_list[task][1]
    # 评估generated_drug是否达到了用户要求(超过阈值)
    # _, _, answer = evaluate_molecule(input_drug, generated_drug, task, threshold_list, showdetail)
    #
    # return answer
    delta = 0
    input_v, output_v, answer = evaluate_molecule(input_drug, generated_drug, task, threshold_list, showdetail)
    if input_v is not None and isinstance(input_v, tuple):
        # 返回元组说明是双属性
        delta = (output_v[0] - input_v[0], output_v[1] - input_v[1])
    elif input_v is not None:
        # 返回单值说明是单属性任务
        delta = output_v - input_v   # 新旧分子的Δ

    return answer, delta