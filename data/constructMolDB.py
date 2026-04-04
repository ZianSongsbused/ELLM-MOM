from chembl_webresource_client.new_client import new_client
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski, rdMolDescriptors, SanitizeFlags
from rdkit import Chem
import pandas as pd
import gzip
from rdkit import RDLogger


# /||||||||||||||||||||||||||||||||||||||||||||||||  ChEMBL数据库  ||||||||||||||||||||||||||||||||||||||||||||||||\
# Step 1: 初始化 ChEMBL 的 molecule 客户端
molecules = new_client.molecule

# Step 2: 过滤出 max_phase=4 的药物(表示已批准使用的药物)
approved_drugs = molecules.filter(max_phase=4)
print(len(approved_drugs))

# Step 3: 遍历这些药物，提取 SMILES（结构字符串）
records = []
for mol in approved_drugs:
    # molecule_structures 里包含结构信息
    mol_struct = mol.get("molecule_structures")

    # 如果结构存在，且有标准 SMILES 表达式
    if mol_struct and mol_struct.get("canonical_smiles"):
        smiles = mol_struct["canonical_smiles"]
        mol_obj = Chem.MolFromSmiles(smiles)

        if mol_obj:
            # RDKit 解析成功就继续
            try:
                props = {
                    "chembl_id": mol["molecule_chembl_id"],     # ChEMBL ID
                    "name": mol.get("pref_name", "N/A"),        # 药物名称（可能为空）
                    "smiles": smiles,                           # SMILES 表达式
                    "MolLogP": Crippen.MolLogP(mol_obj),        # 脂溶性（分配系数）
                    "QED": QED.qed(mol_obj),                    # 药物相似性评分
                    "TPSA": rdMolDescriptors.CalcTPSA(mol_obj), # 极性表面积
                    "NumHAcceptors": Lipinski.NumHAcceptors(mol_obj),  # 氢键受体数量
                    "NumHDonors": Lipinski.NumHDonors(mol_obj),        # 氢键供体数量
                    "MolWt": Descriptors.MolWt(mol_obj),               # 分子量
                    "NumRotatableBonds": Lipinski.NumRotatableBonds(mol_obj),       # 可旋转键数
                    "RingCount": rdMolDescriptors.CalcNumRings(mol_obj),            # 环的数量
                    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol_obj),     # sp³碳占比
                    "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms(mol_obj)   # 重原子数量
                }
                records.append(props)       # 把这个分子的性质加入到 records 列表中
            except:
                continue  # 某些分子可能在计算时出错，直接跳过

# 对smiles去重后保存为文件
df = pd.DataFrame(records)
df = df.drop_duplicates(subset="smiles")
df.to_csv("chembl_approved_drugs.csv", index=False)

print(f"✅ 已成功保存 {len(df)} 个药物分子到 chembl_approved_drugs.csv")

# \||||||||||||||||||||||||||||||||||||||||||||||||  ChEMBL  ||||||||||||||||||||||||||||||||||||||||||||||||/

# /||||||||||||||||||||||||||||||||||||||||||||||||  PDB数据库  ||||||||||||||||||||||||||||||||||||||||||||||||\
# 在pdb的ligand数据库里匹配药物
# 排除非药物分子
def is_not_druglike(mol, name=None):
    """
    排除不是药物的分子，返回的是“不是药物”的bool值
    """
    # # 2. 分子太大（可能是多肽或聚合体）
    # if mol.GetNumAtoms() > 100:
    #     return True
    if Descriptors.MolWt(mol) > 1000:
        print("not drug: 分子量太大")
        return True
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols <= {"Na", "Cl", "K", "Mg", "Ca", "Zn", "Fe"}:
        print("not drug: 原子组成都是离子，基本可判断为无机物或缓冲离子")
        return True
    blacklist_ids = {"H2O", "HOH", "NA", "CL", "MG", "ZN", "CA", "K", "SO4", "MN", "CU", "CO", "CD", "NAG", "MAN", "GOL", "EDO", "DMS", "ACT", "PEG", "BME"}
    if name and name.upper() in blacklist_ids:
        print("not drug: 分子名字是已知的非药物小分子")
        return True

    return False
def compute_mol_properties(mol):
    """
    给定一个 RDKit 分子对象，计算一系列药物分子常用性质。
    如果分子无效则返回 None。
    """
    return {
        "smiles": Chem.MolToSmiles(mol),
        "MolLogP": Crippen.MolLogP(mol),  # 脂溶性（LogP）
        "QED": QED.qed(mol),  # 药物相似性分数（QED）
        "TPSA": Descriptors.TPSA(mol),                        # 极性表面积（Polar Surface Area）
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),      # 氢键受体数
        "NumHDonors": Descriptors.NumHDonors(mol),            # 氢键供体数
        "MolWt": Descriptors.MolWt(mol),  # 分子量
    }


# 读取本地SDF文件（.gz 文件）
sdf_path = "components-pub.sdf.gz"
molecule_data = []  # 存到csv里的数据

# 获取全局RDKit日志记录器
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)  # 设置日志级别为ERROR或CRITICAL，以屏蔽警告和错误信息

with gzip.open(sdf_path, 'rb') as f:
    # 定义从文件中迭代读取每个分子RDKit生成器
    suppl = Chem.ForwardSDMolSupplier(f)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None   # 获取化合物 ID（PDB ligand 名称）
        result = Chem.SanitizeMol(mol, catchErrors=True)
        # print(name, result)
        if result != SanitizeFlags.SANITIZE_NONE:
            print("mol fault: 分子sanitize失败")
            continue
        # 过滤不符合药物特征的分子
        if is_not_druglike(mol, name):
            continue

        # mol = Chem.RemoveHs(mol)
        props = compute_mol_properties(mol)  # 用RDKit计算相关性质存到props里面
        if props:
            molecule_data.append(props)
        if i % 5000 == 0:
            print(f"{i} mol has been processed")

# 转换为 DataFrame 并保存为 CSV
df = pd.DataFrame(molecule_data)
df.to_csv("pdb_ligand_molecules.csv", index=False)

print(f"\n✅ 已成功保存 {len(df)} 个分子到 pdb_ligand_molecules.csv")

# \||||||||||||||||||||||||||||||||||||||||||||||||  PDB  ||||||||||||||||||||||||||||||||||||||||||||||||/

# 合并和去重
df_chembl = pd.read_csv("chembl_approved_drugs.csv")
df_pdb = pd.read_csv("pdb_ligand_molecules.csv")
df_chembl["source"] = "ChEMBL"
df_pdb["source"] = "PDB"

combined_df = pd.concat([df_chembl, df_pdb], ignore_index=True)
combined_df_dedup = combined_df.drop_duplicates(subset="smiles", keep="first")

combined_df_dedup.to_csv("combined_unique_molecules.csv", index=False)

print(f"\n✅ 合并完成：共 {len(df_chembl)} + {len(df_pdb)} 条原始数据，去重后剩 {len(combined_df_dedup)} 条分子。")