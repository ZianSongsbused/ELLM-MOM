# /||||||||||||||||||||||||||||||||||||||||||||||||  启发式取代基统计与扩展  ||||||||||||||||||||||||||||||||||||||||||||||||\
import re
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, BRICS
from collections import Counter, defaultdict
import json

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm


# 尝试把 fragment smiles 规范化为 canonical smiles，失败返回 None
def _normalize_smiles(smi):
    """宽松构分子，防止非法价/金属等卡住"""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_CLEANUP)
        return mol
    except Exception:
        return None

# 方案一：用 BRICS 分解返回规范化后的片段 set（可能为空）
def _brics_fragments(mol, atom_limit=100):
    """限时 + 宽松 BRICS 分解"""
    out = set()
    try:
        if mol.GetNumAtoms() > atom_limit:
            return out  # 太大不跑BRICS
        frags = BRICS.BRICSDecompose(mol)
        for f in frags:
            try:
                m2 = Chem.MolFromSmiles(f)
                if m2:
                    out.add(Chem.MolToSmiles(m2, canonical=True))
            except Exception:
                continue
    except Exception:
        pass
    return out


# 方案二：基于 Murcko scaffold 的简易侧链提取，提取错误就返回空集合
def _murcko_sidechains(mol):
    """
    1. 取 scaffold 的 substructure match
    2. 找到 scaffold 与非-scaffold 之间的键，把这些键断开（FragmentOnBonds）
    3. 取断开后的片段（GetMolFrags），挑出非 scaffold 的小片段作为侧链
    """
    out = set()
    try:
        scaffold = GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return out
        # 1. 取 scaffold 的 substructure match
        match = mol.GetSubstructMatch(scaffold)
        if not match:
            return out
        scaffold_atoms = set(match)
        # 2. 找出连接 scaffold 与非-scaffold 的键索引
        bond_idxs = []
        for b in mol.GetBonds():
            a1 = b.GetBeginAtomIdx()
            a2 = b.GetEndAtomIdx()
            if (a1 in scaffold_atoms) ^ (a2 in scaffold_atoms):
                bond_idxs.append(b.GetIdx())  # 键连接的起止分子有一个不在scaffold就是连接侧链的键
        if not bond_idxs:
            return out
        fragged = Chem.FragmentOnBonds(mol, bond_idxs, addDummies=True)  # ※ 根据键分片段
        frags = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=True)  # 获取分子的所有碎片
        # 3. 去掉包含 scaffold 大部分原子或过小的片段
        parent_heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)  # 全分子重原子数
        for fm in frags:
            try:
                fsmi = Chem.MolToSmiles(fm, canonical=True)
            except Exception:
                continue
            # 规范化后判断是不是整分子，是就排除
            if fsmi == Chem.MolToSmiles(mol, canonical=True):
                continue
            heavy = rdMolDescriptors.CalcNumHeavyAtoms(fm)    # 片段重原子数
            if heavy < 1:
                continue
            # 排除过大的片段（例如占原分子的原子数 超过60%）
            if heavy > max(1, int(parent_heavy * 0.6)):
                continue
            out.add(fsmi)
    except Exception:
        pass
    return out


# 从 smiles_list 中提取常见取代基碎片并计数
def extract_substituents(smiles_list, min_heavy=0, top_n=300):
    """
    改进版：从 smiles_list 中提取候选替代基/片段并计数。
    优先用 BRICS，再回退到 Murcko+FragmentOnBonds。
    过滤规则：
      - 规范化 fragment smiles
      - heavy atom >= min_heavy
      - fragment 不能等于原分子 canonical smiles
      - fragment heavy atom <= 60% parent heavy atoms
    返回：Counter 形式的 [(fragment_smiles, count), ...]（按频次排序，最多 top_n）
    """
    counter = Counter()
    for smi in tqdm(smiles_list, desc="Extract Substituents from SMILES"):
        mol = _normalize_smiles(smi)
        if not mol:
            continue
        parent_heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)   # 重原子数量
        # 1) BRICS 分解
        frags = _brics_fragments(mol)
        # 2) 如果 BRICS 结果为空，尝试 Murcko-based 切割
        if not frags:
            frags = _murcko_sidechains(mol)
        # 3) 获取片段后，归一化与过滤
        for f in frags:
            fm = _normalize_smiles(f)
            if fm is None:
                continue
            fsmi = Chem.MolToSmiles(fm, canonical=True)
            fsmi = normalize_fragment_smi(fsmi)   # 把数字占位符去掉
            dummy_cnt = fsmi.count('[*]')
            if dummy_cnt != 1:   # ※ 跳过有多个虚拟原子的 ※
                continue
            if fsmi == Chem.MolToSmiles(mol, canonical=True):  # 跳过是整个分子的片段
                continue
            heavy = rdMolDescriptors.CalcNumHeavyAtoms(fm)
            if heavy < min_heavy:  # 重原子太少不认为是替换基团
                continue
            if heavy > max(1, int(parent_heavy * 0.6)):
                continue
            counter[fsmi] += 1
    return counter.most_common(top_n)    # 返回最常见的前 N 个碎片及其出现次数



def build_heuristic_families(top_substituents):
    """
    改进版：把高频 fragment 按简单启发规则分类，合并到手动定义族里。
    输入 top_substituents: list of (smiles, freq)
    返回 dict: {family_name: [smiles1, smiles2, ...], ...}
    说明：这里用简洁规则做初步分类（含O、含N、芳香、卤素、碳链、carbonyl等）。
    """
    # 手动基础族（可按需扩展）
    families = {
        "halogen": ["F", "Cl", "Br", "I", "CF3"],             # 卤素
        "common_alkyl": ["C", "CC", "CCC", "CC(C)C", "CH3"],  # 烷基、烃基
        "oxygenated": ["OH", "OMe", "OCF3", "COOH", "COO"],   # 氧化基
        "nitrogenous": ["NH2", "NMe2", "CN", "NO2"],          # 含氮基
        "aromatic_ring": ["c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1"],  # 芳香环
        "carbonyl": ["C=O", "C(=O)O", "CONH2"],               # 羰基
    }

    auto_groups = defaultdict(list)
    for smi, freq in top_substituents:
        m = Chem.MolFromSmiles(smi)
        # if not m:
        #     continue
        can = Chem.MolToSmiles(m, canonical=True)  # canonical化
        # counts
        atoms = [a.GetSymbol() for a in m.GetAtoms()]
        heavy = rdMolDescriptors.CalcNumHeavyAtoms(m)
        s = can

        if re.search(r"\bF\b|\[F\]|F", s) and heavy <= 2:
            # 如果 SMILES 字符串中包含氟（F）且重原子数量不超过2，归为卤素族。
            auto_groups["halogen"].append(s)
            continue
        if re.search(r"Cl|Br|I", s) and heavy <= 6:
            # 如果 SMILES 字符串中包含氯（Cl）、溴（Br）或碘（I）且重原子数量不超过6，也归为卤素族。
            auto_groups["halogen"].append(s)
            continue
        if "c1" in s:  # 含芳香环
            auto_groups["aromatic_ring"].append(s)
            continue
        if "N" in s and "O" not in s:
            auto_groups["nitrogenous"].append(s)
            continue
        if "O" in s and "N" not in s:
            auto_groups["oxygenated"].append(s)
            continue
        if re.search(r"C=O|C\(=O\)", s):
            # 如果 SMILES 字符串中包含羰基（如 C=O 或 C(=O)），归为羰基族
            auto_groups["carbonyl"].append(s)
            continue
        auto_groups["misc"].append(s)      # fallback 分到 miscellaneous

    # 合并手动族与自动族，去重并按 freq 不排序(可按需排序)
    merged = {}
    for k, v in families.items():
        merged[k] = sorted(list(dict.fromkeys(v + auto_groups.get(k, []))))

    # 添加自动发现的其他组（如 misc）
    for k, v in auto_groups.items():
        if k in merged:
            continue
        merged[k] = sorted(list(dict.fromkeys(v)))

    return merged


def normalize_fragment_smi(smi):
    """
    把带编号的dummy原子规范化成[*]
    例：[16*]c1c[nH]c2ccccc12 → [*]c1c[nH]c2ccccc12
    这里面的数字表示“断键的占位符”，只在原分子中有意义
    """
    # 通配更稳妥一点：
    smi = re.sub(r'\[\d+\*\]', '[*]', smi)
    return smi


if __name__ == "__main__":

    df = pd.read_csv("./data/small_molecule/combined_unique_molecules.csv")
    smiles_list = df["smiles"].dropna().tolist()
    print("文件读取完毕")
    top_subs = extract_substituents(smiles_list, min_heavy=1, top_n=300)
    print("基团提取完毕")
    heuristic_map = build_heuristic_families(top_subs)
    print("基团分析完毕")
    with open("./data/heuristic_substituent_families.json","w",encoding="utf-8") as wf:
        json.dump(heuristic_map, wf, ensure_ascii=False, indent=2)

    print(f"✅ 已生成启发式取代基映射，共 {len(heuristic_map)} 个族。")
    for family, members in heuristic_map.items():
        print(f"族名: {family}, 元素数量: {len(members)} \n元素: {members}")