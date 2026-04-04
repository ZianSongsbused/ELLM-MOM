from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs, rdFMCS, rdRGroupDecomposition
from collections import defaultdict, Counter

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from utils.convenient_utils.wordart import ColorText


# -----------------------------
# Step 1: R-groups(替换基团)分析
# -----------------------------
def rgroup_difference(input_smi, gen_smi):
    mols = [Chem.MolFromSmiles(input_smi), Chem.MolFromSmiles(gen_smi)]

    # Step 1: 提取共同核心结构
    core = rdFMCS.FindMCS(mols)
    core_mol = Chem.MolFromSmarts(core.smartsString)
    if core_mol is None:
        return {"error": "未能找到公共核心结构"}

    # Step 2: 设置分解参数（可选）
    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    rgd = rdRGroupDecomposition.RGroupDecomposition([core_mol], params)

    # Step 3: 添加分子（新版 RDKit 需用 Add() 而不是 AddMolecule）
    rgd.Add(mols[0])
    rgd.Add(mols[1])
    rgd.Process()

    # Step 4: 获取替换基团差异
    result = rgd.GetRGroupsAsRows()
    if len(result) != 2:
        return {"error": "RGroup 分解失败"}

    diff_rgroups = {}
    for key in result[0].keys():
        smi1 = Chem.MolToSmiles(result[0][key]) if result[0][key] else None
        smi2 = Chem.MolToSmiles(result[1][key]) if result[1][key] else None
        if smi1 != smi2:
            diff_rgroups[key] = (smi1, smi2)

    # ColorText.print(diff_rgroups, ColorText.GREEN)
    return diff_rgroups


# 提取每对分子的 R-group 替换信息
def extract_rgroup_info(mol_pair):
    """
    输入: 一个成功优化的分子对 (input_smi, gen_smi)
    输出: (core_smarts, {Rn: (input_r, gen_r)})
    """
    input_smi, gen_smi = mol_pair
    input_mol, gen_mol = Chem.MolFromSmiles(input_smi), Chem.MolFromSmiles(gen_smi)
    if not input_mol or not gen_mol:
        return None

    try:
        # Step 1.1: 获得 R-group 替换详情
        rgroup_diff = rgroup_difference(input_smi, gen_smi)
    except Exception as e:
        ColorText.print(f"替换基团分析错误：{e}", ColorText.RED)
        return None
    if "Core" not in rgroup_diff or not isinstance(rgroup_diff["Core"], tuple):
        print(f"❌ 替换基团分析错误：缺失 Core，跳过该对")
        return None

    # Step 1.2: 提取 Core SMARTS 和 Rn 替换对
    core_smarts = rgroup_diff["Core"][0]  # 提取 Core 部分
    rgroup_dict = {}
    for k, v in rgroup_diff.items():  # 提取 R基 部分
        if k == "Core":
            continue
        rgroup_dict[k] = v

    return core_smarts, rgroup_dict

# -----------------------------
# Step 2: 将 SMARTS 解析为分子对象
# -----------------------------
def mol_from_smarts(smarts):
    try:
        return Chem.MolFromSmarts(smarts)
    except:
        return None


# -----------------------------
# Step 3: 生成指纹（Morgan Fingerprint），传入的需要是一个rdkit的分子对象
# -----------------------------
def fingerprint(mol):
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)   # 不做 Sanitize，防止 Kekulization 错误
        return GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    except Exception as e:
        print(f"[Fingerprint错误] {e}")
        return None

# -----------------------------
# Step 4: 计算 Tanimoto 相似度
# -----------------------------
def tanimoto_sim(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# -----------------------------
# Step 5: 对 Core SMARTS 聚类（基于tanimoto相似度）
# -----------------------------
def cluster_cores_tani(core_smarts_list, threshold=0.8):
    # thereshold表示两个core的相似度超过这个阈值会被归为一类
    # Step 5.1: 核心部分转为 Mol 对象并 生成指纹 （含smart的元组）
    cores_mol = [(smarts, mol_from_smarts(smarts)) for smarts in core_smarts_list]
    # cores_fp = [(smarts, fingerprint(mol)) for smarts, mol in cores_mol]
    cores_fp = []
    for smarts, mol in cores_mol:
        fp = fingerprint(mol)
        if fp is not None:
            cores_fp.append((smarts, fp))
        else:
            print(f"无法为 core {smarts} 计算指纹，已跳过", ColorText.RED)

    groups = []

    # Step 5.2: 遍历所有 core，按相似度分组
    for smarts, fp in cores_fp:
        placed = False  # 当前的这个smart是不是已经有分组了，用于提前退出（不在遍历其他分组）用
        for group in groups:   # 和已有group对比看属于的哪个组
            rep_smarts = next(iter(group))  # 取组中第一个 core 作为代表
            rep_fp = next((f for s, f in cores_fp if s == rep_smarts), None)
            if tanimoto_sim(fp, rep_fp) >= threshold:
                group.add(smarts)
                placed = True
                break
        if not placed:
            groups.append(set([smarts]))    # 首次/哪个组都不属于就建立新的组 (这里的key是smart)

    return groups
# -----------------------------
# 替代 Step 5: 用 Murcko Scaffold 聚类 Core
# -----------------------------
from rdkit.Chem.Scaffolds import MurckoScaffold

def cluster_cores(core_smarts_list, threshold=0.6):
    """
    输入：core_smarts_list 是 SMARTS 表示的 core
    输出：List[Set[原始 SMARTS]]，即每个分组包含原始 SMARTS
    """
    # 记录 SMARTS 到 scaffold FP 的映射
    smarts_to_fp = {}
    smarts_to_scaffold = {}

    for smarts in core_smarts_list:
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            continue
        try:
            Chem.GetSSSR(mol)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                continue
            scaffold_fp = fingerprint(scaffold)
            if scaffold_fp:
                smarts_to_fp[smarts] = scaffold_fp
                smarts_to_scaffold[smarts] = Chem.MolToSmiles(scaffold)
        except Exception as e:
            print(f"[跳过] Scaffold 提取失败: {e}")
            continue

    groups = []
    for smarts, fp in smarts_to_fp.items():
        placed = False
        for group in groups:
            rep_smarts = next(iter(group))
            rep_fp = smarts_to_fp.get(rep_smarts)
            if rep_fp and tanimoto_sim(fp, rep_fp) >= threshold:
                group.add(smarts)  # ✅ 添加的是原始 SMARTS
                placed = True
                break
        if not placed:
            groups.append(set([smarts]))

    return groups  # ✅ List[Set[SMARTS]]



# -----------------------------
# Step 6: 聚类 R-group 替换模式
# -----------------------------
# def aggregate_rgroup_patterns(rgroup_data_list):
#     """
#     输入: List[(core_smarts, {R1: (input, output), ...})]
#     输出: 每组 Core 的 R-group 替换频率统计
#     """
#     # Step 6.1: 所有 Core SMARTS
#     print("ggroup")
#     cores = [d[0] for d in rgroup_data_list]
#
#     # Step 6.2: 对 Core SMARTS 聚类
#     core_groups = cluster_cores(cores)
#     if not core_groups:
#         print("❌ 没有生成任何 core 分组，聚类失败")
#     print("聚类")
#
#     # Step 6.3: 反向索引样本至所属 Core 聚类组
#     core2samples = defaultdict(list)
#     for core, rgs in rgroup_data_list:  # 遍历所有core
#         for group in core_groups:       # 遍历所有基于core的聚类
#             if core in group:
#                 core2samples[frozenset(group)].append(rgs)  # 这句group就变成类似key的东西
#                 break
#     print("填充Rn")
#
#     # Step 6.4: 对每组统计每个 Rn 的替换频率
#     print("排序Rn")
#     results = {}
#     for core_group, samples in core2samples.items():
#         pattern_counter = defaultdict(Counter)
#
#         for rgs in samples:
#             for rname, rgpair in rgs.items():
#                 pattern_counter[rname][rgpair] += 1  # Rn和具体的基团串组成的二维数组
#
#         # Step 6.5: 转成替换频率排序
#         pattern_freq = {}
#         for rname, counter in pattern_counter.items():
#             total = sum(counter.values())
#             freq_list = [(k, v / total) for k, v in counter.items()]
#             freq_list.sort(key=lambda x: x[1], reverse=True)
#             pattern_freq[rname] = freq_list
#
#         results[tuple(core_group)] = pattern_freq
#
#     return results

# -----------------------------
# 运行全流程
# -----------------------------

from collections import defaultdict, Counter

def aggregate_rgroup_patterns(rgroup_data_list):
    """
    输入:
        rgroup_data_list: List of tuples
            (core_smarts, {Rn: (src, tgt), ...}, input_smi, output_smi)

    返回:
        - pattern_stats: {core_group_tuple: {R1: [((src, tgt, core), freq), ...]}}
    """
    print("正在聚合 R-group 替换模式...")

    # 所有 core 列表（用于聚类）
    cores = [d[0] for d in rgroup_data_list]

    # 聚类 core
    core_groups = cluster_cores(cores)
    if not core_groups:
        print("❌ 没有生成 core 分组，聚类失败")
        return {}

    # core 到聚类组的映射
    core2group = {}
    for group in core_groups:
        for c in group:
            core2group[c] = tuple(group)

    # 替换频率统计，包含 core
    pattern_stats = defaultdict(lambda: defaultdict(Counter))

    for core, rgs in rgroup_data_list:
        core_group = core2group.get(core)
        if core_group is None:
            continue

        for rname, (src, tgt) in rgs.items():
            # 加入 core（用于后续构造示例）
            pattern_stats[core_group][rname][(src, tgt, core)] += 1

    # 频率归一化、转 list
    final_pattern_stats = {}
    for core_group, rname_counters in pattern_stats.items():
        pattern_freq = {}
        for rname, counter in rname_counters.items():
            total = sum(counter.values())
            freq_list = [((src, tgt, core), count / total) for (src, tgt, core), count in counter.items()]
            freq_list.sort(key=lambda x: x[1], reverse=True)
            pattern_freq[rname] = freq_list
        final_pattern_stats[core_group] = pattern_freq

    print("✅ 替换模式聚合完成")
    return final_pattern_stats

def run_rgroup_pattern_mining(successful_pairs):
    """
    输入: 成功优化的分子对 [(input_smi, gen_smi), ...]
    输出: 打印每个 Core 群的代表 R-group 替换模式
    """
    # Step 7.1: 提取所有成功分子对的 R-group 信息
    rgroup_data = []
    for idx, pair in enumerate(successful_pairs):
        ColorText.print(f"【{idx}】inp：{pair[0]}，out：{pair[1]}", ColorText.BLUE)
        info = extract_rgroup_info(pair)
        if info:
            rgroup_data.append(info)

    # Step 7.2: 核心过程
    pattern_stats = aggregate_rgroup_patterns(rgroup_data)
    print(pattern_stats)
    # Step 7.3: 打印替换模式
    # for core_group, patterns in pattern_stats.items():
    #     ColorText.print("=" * 60, ColorText.BLUE)
    #     print(f"[Core Group] 共 {len(core_group)} 个 Core")
    #     for rname, freq_list in patterns.items():
    #         ColorText.print(f"  ➤ 替换位点 {rname}:", ColorText.GREEN)
    #         for (input_r, gen_r), freq in freq_list:
    #             print(f"    {input_r}  →  {gen_r}  （频率: {freq:.1%}）")


# -----------------------------
# Step 8: 从CSV中读取成功分子对并运行
# -----------------------------
def read_success_pairs_from_csv(csv_path):
    import csv
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["input_smiles"], row["generated_smiles"]))
    return pairs
if __name__ == "__main__":
    # Step 8.1: 从已有成功分子对CSV中加载
    csv_file = "./results/mmpa/res_task102_strict_succ.csv"
    successful_pairs = read_success_pairs_from_csv(csv_file)

    # Step 8.2: 执行 pipeline
    run_rgroup_pattern_mining(successful_pairs)
