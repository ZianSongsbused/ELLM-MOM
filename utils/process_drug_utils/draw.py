import csv
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from math import ceil
import os

# 设置每张图显示的最大 SMILES 对数量（每对包含 2 或 4 个图）
N = 50

def get_murcko_scaffold(mol):
    """提取分子的 Murcko 骨架结构（失败返回 None）"""
    try:
        return MurckoScaffold.GetScaffoldForMol(mol)
    except:
        return None


def draw_smi_pairs(csv_path, out_dir="./figs", max_pairs=N):
    """
    读取 CSV 文件中的 input_smiles 和 generated_smiles 字段，
    将每一对分子画图对比（不含骨架），每张图显示不超过 max_pairs 对。
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 读取所有合法的分子对
    smiles_pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_smi = row["input_smiles"]
            gen_smi = row["generated_smiles"]
            mol1 = Chem.MolFromSmiles(input_smi)
            mol2 = Chem.MolFromSmiles(gen_smi)
            if mol1 and mol2:
                smiles_pairs.append((mol1, mol2))

    if not smiles_pairs:
        print("❌ 无有效 SMILES 可画图")
        return

    total = len(smiles_pairs)
    num_batches = ceil(total / max_pairs)

    for batch_idx in range(num_batches):
        mols = []
        legends = []

        # 每批次处理 max_pairs 对
        for i in range(batch_idx * max_pairs, min((batch_idx + 1) * max_pairs, total)):
            mol1, mol2 = smiles_pairs[i]
            mols.extend([mol1, mol2])
            legends.extend([f"Input {i+1}", f"Generated {i+1}"])

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=(300, 300),
            legends=legends,
            useSVG=False
        )
        out_path = os.path.join(out_dir, f"smi_compare_batch{batch_idx+1}.png")
        img.save(out_path)
        print(f"✅ SMILES 对比图保存：{out_path}")


def draw_smi_and_scaffold(csv_path, out_dir="./figs", max_pairs=N):
    """
    从 CSV 中读取 input 和 generated 的 SMILES，
    对每对 SMILES 提取骨架，并画出以下四个结构：
    - input
    - input scaffold
    - generated
    - generated scaffold
    每张图最多画 max_pairs 对（即 4 * max_pairs 个结构）
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_smi = row["input_smiles"]
            gen_smi = row["generated_smiles"]
            mol_input = Chem.MolFromSmiles(input_smi)
            mol_gen = Chem.MolFromSmiles(gen_smi)
            if mol_input and mol_gen:
                scaffold_input = get_murcko_scaffold(mol_input)
                scaffold_gen = get_murcko_scaffold(mol_gen)
                records.append((mol_input, scaffold_input, mol_gen, scaffold_gen))

    if not records:
        print("❌ 无有效分子用于绘图")
        return

    total = len(records)
    num_batches = ceil(total / max_pairs)

    for batch_idx in range(num_batches):
        mols = []
        legends = []

        for i in range(batch_idx * max_pairs, min((batch_idx + 1) * max_pairs, total)):
            mol_input, scf_input, mol_gen, scf_gen = records[i]

            # 只加入非空的分子或骨架（避免画图报错）
            sub_mols = [mol_input, scf_input, mol_gen, scf_gen]
            sub_legends = [
                f"Input {i+1}", f"Input Scaffold {i+1}",
                f"Generated {i+1}", f"Gen Scaffold {i+1}"
            ]
            for m, l in zip(sub_mols, sub_legends):
                if m:
                    mols.append(m)
                    legends.append(l)

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(250, 250),
            legends=legends,
            useSVG=False
        )
        out_path = os.path.join(out_dir, f"smi_scaffold_batch{batch_idx+1}.png")
        img.save(out_path)
        print(f"✅ 分子 + 骨架图保存：{out_path}")


# ✅ 使用示例（调用入口）
if __name__ == "__main__":
    input_csv = "./results/mmpa/res_task102_strict_succ.csv"
    output_folder = "./figs"

    draw_smi_pairs(input_csv, output_folder)
    draw_smi_and_scaffold(input_csv, output_folder)
