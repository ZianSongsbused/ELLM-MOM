import csv

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdRGroupDecomposition, AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

from utils.convenient_utils.wordart import ColorText

# 简单规则库：结构变化 → 属性影响（暂时没用）
property_rules = {
    "LogP": {
        "OH": "降低 LogP（增加亲水性）",
        "COOH": "降低 LogP（极性增强）",
        "Cl": "升高 LogP（疏水性增强）",
        "alkyl": "升高 LogP（疏水性增强）",
        "ring_addition": "可能升高 LogP（增加疏水骨架）"
    },
    "TPSA": {
        "OH": "提高 TPSA（极性面积增大）",
        "NH2": "提高 TPSA（氢键供体）"
    }
}


# 基于最大公共子结构（MCSS）提取分子对的不同部分
def get_mcs_diff(input_smi, gen_smi):
    mol1, mol2 = Chem.MolFromSmiles(input_smi), Chem.MolFromSmiles(gen_smi)

    if mol1 is None or mol2 is None:
        return None, None, None

    # 1. 找最大公共子结构
    mcs_result = rdFMCS.FindMCS([mol1, mol2])
    mcs_smarts = mcs_result.smartsString    # SMART格式的两分子主干
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    # 2. 获取匹配上的子结构的index（公共部分）
    match1, match2 = mol1.GetSubstructMatch(mcs_mol), mol2.GetSubstructMatch(mcs_mol)

    # 3. 提取变化部分（非公共部分）
    def get_diff_atoms(mol, match):
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in match]

    diff1, diff2 = get_diff_atoms(mol1, match1), get_diff_atoms(mol2, match2)
    frag1 = Chem.PathToSubmol(mol1, diff1) if diff1 else None
    frag2 = Chem.PathToSubmol(mol2, diff2) if diff2 else None

    def safe_mol_to_smiles(mol):
        """
        将 Mol 安全地转为 SMILES，避免段错误。
        """
        _ = Chem.MolToMolBlock(mol)  # 不加这个就会段错误，不知道为什么
        if mol is None:     return None
        try:                return Chem.MolToSmiles(mol)
        except Exception:   return None

    return {
        "mcs": mcs_smarts,
        "unique_input": safe_mol_to_smiles(frag1),
        "unique_generated": safe_mol_to_smiles(frag2),
    }


def extract_mcs_and_unique_parts(input_smi: str, gen_smi: str):
    mol1 = Chem.MolFromSmiles(input_smi)
    mol2 = Chem.MolFromSmiles(gen_smi)

    if mol1 is None or mol2 is None:
        return None

    # 1. Find MCS
    mcs_result = rdFMCS.FindMCS(
        [mol1, mol2],
        completeRingsOnly=True,
        matchValences=True,
        ringMatchesRingOnly=True
    )

    mcs_query = Chem.MolFromSmarts(mcs_result.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_query)
    match2 = mol2.GetSubstructMatch(mcs_query)

    if not match1 or not match2:
        return None

    atoms_mcs_1 = set(match1)
    atoms_mcs_2 = set(match2)

    diff_atoms_1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetIdx() not in atoms_mcs_1]
    diff_atoms_2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetIdx() not in atoms_mcs_2]

    try:
        frag1 = Chem.PathToSubmol(mol1, diff_atoms_1) if diff_atoms_1 else None
        frag2 = Chem.PathToSubmol(mol2, diff_atoms_2) if diff_atoms_2 else None
        mcs_mol1 = Chem.PathToSubmol(mol1, list(atoms_mcs_1)) if atoms_mcs_1 else None

        return {
            "mcs": mcs_result.smartsString,
            "unique_input": Chem.MolToSmiles(frag1) if frag1 else None,
            "unique_generated": Chem.MolToSmiles(frag2) if frag2 else None,
        }
    except Exception as e:
        print(f"[提取失败] {input_smi} -> {gen_smi}: {e}")
        return None

# R-groups(替换基团)分析
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

    return diff_rgroups


# 使用 Morgan Fingerprint（ECFP） 进行位向量编码，比较两个分子的结构特征差异。
#     返回差异的位点数量、索引和 Tanimoto 相似度。
def fingerprint_difference(input_smi, gen_smi, radius=2, nBits=2048):
    mol1 = Chem.MolFromSmiles(input_smi)
    mol2 = Chem.MolFromSmiles(gen_smi)
    if mol1 is None or mol2 is None:
        return {"error": "rdkit解析出错"}
    # 生成圆形指纹（相当于 ECFP）
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nBits)

    # 记录不一致的位索引
    diff_bits = []
    for i in range(nBits):
        if fp1[i] != fp2[i]:
            diff_bits.append(i)

    # 计算 Tanimoto 相似度
    similarity = TanimotoSimilarity(fp1, fp2)

    return {
        "similarity": similarity,  # Tanimoto 相似度
        "n_diff_bits": len(diff_bits),  # 差异 bit 数
        "diff_bits_idx": diff_bits[:10]  # 展示前 10 个不同索引
    }


def analyze_difference(input_smi, gen_smi):
    return {
        "input": input_smi,
        "generated": gen_smi,
        "mcss": extract_mcs_and_unique_parts(input_smi, gen_smi),
        "rgroup": rgroup_difference(input_smi, gen_smi),
        "fingerprint": fingerprint_difference(input_smi, gen_smi)
    }


# === 主函数：从CSV读取并分析 ===
def run_diff_analysis_from_csv(csv_path, max_pairs=100):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            input_smi = row["input_smiles"]
            gen_smi = row["generated_smiles"]
            print(f"\n====== 分子对 {i + 1} ======")
            try:
                result = analyze_difference(input_smi, gen_smi)
                print("输入分子:", result["input"])
                print("生成分子:", result["generated"])
                ColorText.print("【MCSS差异】", result["mcss"], ColorText.GREEN)
                ColorText.print("【R-Group 替换】", result["rgroup"], ColorText.YELLOW)
                ColorText.print("【指纹差异】", result["fingerprint"], ColorText.BLUE)
            except Exception as e:
                ColorText.print(f"[错误] 第 {i + 1} 对分子处理失败: {e}", ColorText.RED)
                continue

            if i + 1 >= max_pairs:
                break


if __name__ == "__main__":
    csv_file = "./results/mmpa/res_task102_strict_succ.csv"
    run_diff_analysis_from_csv(csv_file)

# def frag_to_text(frag):
#     if frag is None:
#         return "无变化"
#     smi = Chem.MolToSmiles(frag)
#     # 简单规则映射
#     if "O" in smi and "C" not in smi:
#         return "引入了羟基"
#     elif "CO" in smi or "C(=O)O" in smi:
#         return "引入了羧基"
#     elif "Cl" in smi:
#         return "引入了氯原子"
#     elif Chem.MolFromSmiles(smi).GetRingInfo().NumRings() > 0:
#         return "引入了芳环或闭环结构"
#     else:
#         return f"引入了片段: {smi}"
#
#
# def explain_diff(input_smi, gen_smi, target_prop="LogP"):
#     frag1, frag2, mcs = get_mcs_diff(input_smi, gen_smi)
#     if frag1 is None and frag2 is None:
#         return "两个分子几乎没有差异"
#
#     change_text = f"{frag_to_text(frag2)}；取代了 {frag_to_text(frag1)}"
#
#     # 基于规则猜测属性变化方向（可用模型替代）
#     smi2 = Chem.MolToSmiles(frag2) if frag2 else ""
#     prop_change = "，可能对目标属性有影响"
#     for key, msg in property_rules.get(target_prop, {}).items():
#         if key in smi2:
#             prop_change = f"，这可能导致 {msg}"
#             break
#
#     return change_text + prop_change
