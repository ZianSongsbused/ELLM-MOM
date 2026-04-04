import json
import re

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from langchain.schema import HumanMessage
from rdkit.Chem.rdmolops import GetShortestPath

from utils.convenient_utils.wordart import ColorText

GREEK_POS_MAP = {
        "alpha": 1, "beta": 2,  "gamma": 3, "delta": 4, "epsilon": 5, "zeta": 6,
        "eta": 7, "theta": 8, "iota": 9, "kappa": 10, "lambda": 11, "mu": 12,
        "nu": 13, "xi": 14, "omicron": 15, "pi": 16, "rho": 17, "sigma": 18,
        "tau": 19, "upsilon": 20, "phi": 21, "chi": 22, "psi": 23, "omega": 24,
    }

# func1. 获得锚点（用于后面找到位置信息对应的具体原子序号）
def pick_ring_and_anchor(mol, position=None):
    # step0: 尝试 kekulize，确保芳香环正确标记
    mol = Chem.Mol(mol)  # clone 一份，避免改原对象
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        print(f"    [pick_ring_and_anchor] Kekulize 失败: {e}")
    # step1 拿到分子照顾你所有的环
    rings = [tuple(r) for r in Chem.GetSymmSSSR(mol)]
    if not rings:
        ColorText.print("分子中没找到任何环", ColorText.RED)
        return None, None
    # step2 根据不同的position选择不同的分支
    pos = (position or "").lower().strip()
    # step2.1: ortho/meta/para/ipso -> simple benzene
    if pos in ("ipso", "o", "ortho", "m", "meta", "p", "para"):
        benzene_rings = [r for r in rings if is_simple_benzene_ring(mol, r)]
        if benzene_rings:  # 所有的苯环
            ring = benzene_rings[0]
            for aidx in ring:    # 找到连接侧链的原子作为 anchor
                a = mol.GetAtomWithIdx(aidx)
                for nb in a.GetNeighbors():
                    if nb.GetIdx() not in ring:   # 又不在环内地邻居及侧链
                        print("    [pick_ring_and_anchor] 选中苯环:")
                        print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
                        print("      锚点:", aidx, mol.GetAtomWithIdx(aidx).GetSymbol())
                        return ring, aidx
            return None, None         # 如果没有找到侧链连接的原子，返回 None
        else:
            return None, None    # fallback: 没有苯环 → 返回 None

    # step2.2: alpha/beta/gamma... -> 非 simple benzene
    if pos in GREEK_POS_MAP:
        non_benzene = [r for r in rings if not is_simple_benzene_ring(mol, r)]
        if non_benzene:
            # 优先带侧链的环
            for ring in non_benzene:
                ring_set = set(ring)
                for aidx in ring:
                    for nb in mol.GetAtomWithIdx(aidx).GetNeighbors():
                        if nb.GetIdx() not in ring_set:
                            print("    [pick_ring_and_anchor] 选中非苯环:")
                            print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
                            print("      锚点:", aidx, mol.GetAtomWithIdx(aidx).GetSymbol())
                            return ring, aidx
            # fallback: 没找到带侧链的环，就第一个非笨环的第一个点作为锚点
            print("    [pick_ring_and_anchor]fallback",non_benzene[0], non_benzene[0][0])
            return non_benzene[0], non_benzene[0][0]
        else:
            return None, None

    # case3: 其它或 None → fallback
    ring = rings[0]
    print("    [pick_ring_and_anchor] fallback 任意环:", ring)
    print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
    print("      锚点:", ring[0], mol.GetAtomWithIdx(ring[0]).GetSymbol())
    print("--------------------------------------------------------")
    return rings[0], rings[0][0]


# 支持找据具体的环（trigger）的版本
def pick_ring_and_anchor2(mol, position=None, trigger=None):
    # step0: 尝试 kekulize，确保芳香环正确标记
    mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        print(f"    [pick_ring_and_anchor] Kekulize 失败: {e}")

    # step1: 若提供 trigger，则优先匹配
    if trigger:
        for t in trigger:
            patt = Chem.MolFromSmarts(t)
            if patt is None:
                print(f"    [pick_ring_and_anchor] trigger SMARTS 无效，跳过: {t}")
                continue

            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue

            # 找到第一个匹配，定位其所在环
            matched_atoms = set(matches[0])
            rings = [tuple(r) for r in Chem.GetSymmSSSR(mol)]
            for ring in rings:
                ring_set = set(ring)
                if matched_atoms & ring_set:
                    # 命中 trigger 的环，按原逻辑选 anchor（优先有侧链的原子）
                    for aidx in ring:
                        a = mol.GetAtomWithIdx(aidx)
                        for nb in a.GetNeighbors():
                            if nb.GetIdx() not in ring_set:
                                print("    [pick_ring_and_anchor] 触发SMARTS匹配成功:")
                                print(f"      SMARTS={t}")
                                print("      环原子:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
                                print("      锚点:", aidx, mol.GetAtomWithIdx(aidx).GetSymbol())
                                return ring, aidx
                    # 若没找到带侧链原子，fallback 到环首原子
                    print("    [pick_ring_and_anchor] 触发SMARTS匹配成功但无侧链，使用环首原子作为锚点")
                    return ring, ring[0]
        # 若所有 trigger 都未匹配，继续原逻辑
        print("    [pick_ring_and_anchor] 所有 trigger 未匹配，使用默认逻辑")

    # step2: 原逻辑
    rings = [tuple(r) for r in Chem.GetSymmSSSR(mol)]
    if not rings:
        ColorText.print("分子中没找到任何环", ColorText.RED)
        return None, None

    pos = (position or "").lower().strip()

    if pos in ("ipso", "o", "ortho", "m", "meta", "p", "para"):
        benzene_rings = [r for r in rings if is_simple_benzene_ring(mol, r)]
        if benzene_rings:
            ring = benzene_rings[0]
            for aidx in ring:
                a = mol.GetAtomWithIdx(aidx)
                for nb in a.GetNeighbors():
                    if nb.GetIdx() not in ring:
                        print("    [pick_ring_and_anchor] 选中苯环:")
                        print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
                        print("      锚点:", aidx, mol.GetAtomWithIdx(aidx).GetSymbol())
                        return ring, aidx
            return None, None
        else:
            return None, None

    if pos in GREEK_POS_MAP:
        non_benzene = [r for r in rings if not is_simple_benzene_ring(mol, r)]
        if non_benzene:
            for ring in non_benzene:
                ring_set = set(ring)
                for aidx in ring:
                    for nb in mol.GetAtomWithIdx(aidx).GetNeighbors():
                        if nb.GetIdx() not in ring_set:
                            print("    [pick_ring_and_anchor] 选中非苯环:")
                            print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
                            print("      锚点:", aidx, mol.GetAtomWithIdx(aidx).GetSymbol())
                            return ring, aidx
            print("    [pick_ring_and_anchor] fallback", non_benzene[0], non_benzene[0][0])
            return non_benzene[0], non_benzene[0][0]
        else:
            return None, None

    ring = rings[0]
    print("    [pick_ring_and_anchor] fallback 任意环:", ring)
    print("      原子详情:", [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring])
    print("      锚点:", ring[0], mol.GetAtomWithIdx(ring[0]).GetSymbol())
    print("--------------------------------------------------------")
    return rings[0], rings[0][0]


# func2. 将 ortho/meta/para 映射为距离锚点的“步数”（环上）
def _ring_pos_offset(pos, ring_size):
    """
    锚点本身: 'ipso': 0（，常用于删除/替换“当前已有侧链”）
    ① ortho: 1  ②meta: 2  ③para: ring_size//2（仅在偶数环有严格定义；否则返回None）
    """
    pos = (pos or "").lower().strip()
    if pos in ("ipso", "self", "here"):
        return 0
    if pos in ("o", "ortho"):
        return 1
    if pos in ("m", "meta"):
        return 2
    if pos in ("p", "para"):
        return ring_size // 2 if ring_size % 2 == 0 else None

    # 非苯环情况
    greek_map = GREEK_POS_MAP
    if pos in greek_map:
        return greek_map[pos]

    # 如果输入是数字字符串
    if pos.isdigit():
        return int(pos)

    return None

# 获得@位置距离锚点的距离（苯环版本）
def get_atdist_to_anchor(pos, ring_size):
    """
      - 'ipso': 0（=锚点本身，常用于删除/替换“当前已有侧链”）
      - 'ortho': 1
      - 'meta' : 2
      - 'para' : ring_size//2（仅偶数环有严格定义；否则返回 None）
    """
    pos = (pos or "").lower().strip()
    if pos in ("ipso", "self", "here"):  # 锚点[本位]
        return 0
    if pos in ("o", "ortho"):  # 锚点[邻位]
        return 1
    if pos in ("m", "meta"):  # 锚点[间位]
        return 2            # 与锚点间隔一个碳原子
    if pos in ("p", "para"):   # 锚点[对位]
        return ring_size // 2 if ring_size % 2 == 0 else None
    return None


# 获得@位置距离锚点的距离（其他环版本）
def get_greek_dist(pos):
    """
    非苯环上的相对锚点的距离
    """
    pos = (pos or "").lower().strip()
    return GREEK_POS_MAP.get(pos, None)


# NEW: 判断选中的 ring 是否是“单环苯”  这个函数在mol进行kekule化了就不管用了（GetIsAromatic这一句），换下面的函数
def is_simple_benzene_ring2(mol, ring):
    if len(ring) != 6:  # ① 苯环长度为6
        return False
    ring_info = mol.GetRingInfo().AtomRings()
    shared_counts = []
    for idx in ring:
        a = mol.GetAtomWithIdx(idx)
        if not a.GetIsAromatic():   # ② 所有环内原子都为芳香
            return False
         # ③ 环内每个原子仅属于 1 个环（非并环/非桥连）
        count = sum(1 for r in ring_info if idx in r)
        if count > 1:  # 记录共享的原子数
            shared_counts.append(idx)
        # 如果有 ≥2 个共享原子，说明是 fused（萘型），排除
    print(shared_counts)
    if len(shared_counts) >= 2:
        return False
    return True


def is_simple_benzene_ring(mol, ring):
    """
    判断是否为 simple benzene:
      - 6元环
      - 所有原子是碳
      - 每个原子只属于 <=1 个其他环（即共享原子数 < 2）
      - 如果原子 aromatic flag 全为 True 则接受
      - 否则退化为检查环上键是否为单/双交替（Kekule 形式）
    """
    if len(ring) != 6:   # ① 必须是六元环
        return False

    # 所有原子必须是碳
    for idx in ring:     # ② 原子需要全是碳
        if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
            return False

    # ③ 统计每个原子属于多少个环；如果 >=2 个原子属于多个环，判定为 fused（排除）
    # (只是共享一个原子的话，可以看作是接了个环状基团，拿掉之后的环还是完整的)
    ring_info = mol.GetRingInfo().AtomRings()
    # shared_atoms = [idx for idx in ring if sum(1 for r in ring_info if idx in r) > 1]
    shared_atoms = []
    for idx in ring:
        count = sum(1 for r in ring_info if idx in r)
        if count > 1:
            shared_atoms.append(idx)

    if len(shared_atoms) >= 2:
        return False

    # 检查方式1 先用 aromatic flag 判定（如果分子未被 kekulize，这里常为 True）
    atoms = [mol.GetAtomWithIdx(i) for i in ring]
    if all(a.GetIsAromatic() for a in atoms):
        return True

    # 检查方式2  要求相邻键存在且为单/双/芳香，然后检验单双键交替（Kekule 形式）
    bond_types = []
    for i in range(6):
        a1 = ring[i]
        a2 = ring[(i + 1) % 6]
        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is None:  # 拿到相邻两原子的键，没有就是不相邻，不是环
            return False

        bt = b.GetBondType()
        # 只接受单键、双键或芳香键作为候选，以为苯环种只有这三种键
        if bt not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            return False
        bond_types.append(bt)

    # 将 bond_types 映射为 'S'/'D'（AROMATIC 当作 S 处理以便通过交替检查）
    s = "".join("D" if bt == Chem.BondType.DOUBLE else "S" for bt in bond_types)

    # 不允许出现连续两个单键或两个双键（要求交替）
    if "SS" in s or "DD" in s:
        return False

    return True


# func3. 返回@位置对应的具体原子序号（anchor_idx就来自func1，steps就来自func2）
def get_atatomidx_on_ring(ring, anchor_idx, steps):
    """
    给定环（原子idx元组）、锚点原子idx和步数，返回“沿环正向”走steps后的原子idx
    """
    n = len(ring)
    if steps is None:
        return None

    try:
        k = ring.index(anchor_idx)  # 在ring里的位置
    except ValueError:
        return None
    return ring[(k + steps) % n]


# func4. 找到添加片段接到目标分子上的“连接原子”
def find_attach_atom_on_newfrag(frag):
    """
    返回: (attach_idx, dummy_idx or None)
    """"""
    # 1 有[*]时优先用其邻居作为链接原子
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:  # 当前原子a为0，即为虚拟原子[*]
            nbs = a.GetNeighbors()
            if nbs:
                # 虚拟原子的首个邻居原子作为连接原子的索引，以及虚拟原子自身的索引
                return nbs[0].GetIdx(), a.GetIdx()
            else:
                return a.GetIdx(), a.GetIdx()  # 没有邻居就自己当连接点，稍后会被删
    # 2 无[*]时的启发式
    cand = []
    for a in frag.GetAtoms():
        sym = a.GetSymbol()
        if sym in ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I"):  # 常见的非金属或碳原子
            # 如果当前原子有[隐式化合价大于0/隐式氢键(说明能形成额外的键)/度大于1(连接时不会破坏原有键)]，将其索引添加到候选列表 cand
            if a.GetImplicitValence() > 0 or a.GetNumImplicitHs() > 0 or a.GetDegree() > 1:
                cand.append(a.GetIdx())
    if not cand:   # 没有合适候选 → fallback 到第 0 个原子
        # fallback: 取所有非氢原子里度最大的
        heavy_atoms = [a for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
        if heavy_atoms:
            best = max(heavy_atoms, key=lambda a: a.GetDegree())
            print(f"[Fallback] Using heavy atom idx={best.GetIdx()} ({best.GetSymbol()}) as attach")
            return best.GetIdx(), None
        else:
            print("[Warning] No obvious attach atom, fallback to idx=0")
            return 0, None
    if len(cand) > 1:
        syms = [frag.GetAtomWithIdx(i).GetSymbol() for i in cand]
        print(f"[Warning] Multiple candidate attach atoms {list(zip(cand, syms))}, "
              f"defaulting to idx={cand[0]} ({syms[0]})")

    return cand[0], None"""
    # 1) 虚拟原子优先
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:  # 当前原子a为0，即为虚拟原子[*]
            nbs = a.GetNeighbors()
            if nbs:  # 虚拟原子的首个邻居原子作为连接原子的索引，以及虚拟原子自身的索引
                return nbs[0].GetIdx(), a.GetIdx()
            else:    # 没有邻居就自己当连接点，稍后会被删
                return a.GetIdx(), a.GetIdx()

    # 2) 启发式方法
    # 连接点的优先级映射字典（数字越小优先级越高）
    priority = {'N': 0, 'O': 1, 'C': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    cand = []
    for a in frag.GetAtoms():
        idx, sym = a.GetIdx(), a.GetSymbol()
        # 异原子优先入候选（N,O,S,P）
        if sym in ('N', 'O', 'S', 'P'):
            cand.append(idx)
        # 有隐式价或隐式氢可能是可接位——说明能形成额外的键
        if a.GetImplicitValence() > 0 or a.GetNumImplicitHs() > 0:
            cand.append(idx)
        # 多键/多连的重原子也可以作为候选——连接时不会破坏原有键
        if a.GetAtomicNum() > 1 and a.GetDegree() > 0:
            cand.append(idx)

    if cand:  # 去重并按优先级排序
        seen, cand_u = set(), []
        for i in cand:
            if i not in seen:  # 去重
                seen.add(i)
                cand_u.append(i)

        # 排序：优先级小的在前；若未知则放到末尾
        cand_sorted = sorted(cand_u, key=lambda i: priority.get(frag.GetAtomWithIdx(i).GetSymbol(), 99))
        return cand_sorted[0], None              # 没在列表里的原子优先级最低99

    # 3) fallback（满足上面三种情况的原子都没有）：在重原子里面找一个
    heavy_atoms = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() > 1]  # GetAtomicNum()>1即非氢原子
    if heavy_atoms:
        heavy_sorted = sorted(heavy_atoms, key=lambda i: priority.get(frag.GetAtomWithIdx(i).GetSymbol(), 99))
        return heavy_sorted[0], None

    # 4) 最终fallback
    return 0, None


# func5. 正式的[+基团@位置]操作
def attach_fragment_on_ring(rw_host, host_idx, frag):
    """具体来讲，将基团 frag 接到目标分子 rw_host 的 host_idx 上："""
    # step1 拿到新片段的链接原子索引和虚拟原子索引
    attach_idx, dummy_idx = find_attach_atom_on_newfrag(frag)

    combo = Chem.CombineMols(rw_host, frag)
    rw2 = Chem.RWMol(combo)
    # step2 找到新添加片段的原子序号，并进行实际的添加
    offset = rw_host.GetNumAtoms()  # 目标原子原子数，用来添加新结构后定义原子序号
    rw2.AddBond(host_idx, offset + attach_idx, rdchem.BondType.SINGLE)  # 在目标原子的待链接位置 和 新片段的链接原子 之间链接一个单键

    # 删 dummy（如果有）
    if dummy_idx is not None:
        rw2.RemoveAtom(offset + dummy_idx)

    Chem.SanitizeMol(rw2)
    return rw2.GetMol()


# func6. 正式的[-基团@位置]操作
def cut_sidechain_at_atatom(mol, ring_atom_idx, ring_set):
    """
    从 ring_atom_idx 开始，删除“连接到该环原子、且不在该环里的整个侧链”（BFS）。
    保留环，不跨越到其他环（遇到 ring_set 则停止）。
    """
    rw = Chem.RWMol(mol)
    to_del = set()
    queue = []   # 用于BFS
    # 初始化queue，赋值为ring_atom_idx的非环内邻居原子
    a = rw.GetAtomWithIdx(ring_atom_idx)
    for nb in a.GetNeighbors():
        j = nb.GetIdx()
        if j not in ring_set:  # 删除不在环内的侧链（本步仅记录）
            queue.append(j)
            to_del.add(j)
    # BFS 扩展
    while queue:
        cur = queue.pop()
        cur_atom = rw.GetAtomWithIdx(cur)
        for nb in cur_atom.GetNeighbors():
            j = nb.GetIdx()
            # 遍历队列里所有原子的邻居原子，上一步存的和环内的分子不要
            if j in to_del:
                continue
            if j in ring_set:
                continue
            to_del.add(j)
            queue.append(j)
    # 实删（按 idx 降序）
    for idx in sorted(to_del, reverse=True):
        rw.RemoveAtom(idx)
    Chem.SanitizeMol(rw)
    return rw.GetMol()


def load_substituents(filepath="./rules/substituents.json"):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# func7. 找到分子骨架（路径上的原子）
def find_backbone_atoms(mol):
    """
    近似找 backbone：用最远的两点路径作为主链（多数情况下的主链）
    """
    atoms = list(mol.GetAtoms())  # 获取分子中的所有原子
    max_dist, backbone_path = -1,  []

    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            # 找到距离最远的两个原子当作主链
            path = GetShortestPath(mol, atoms[i].GetIdx(), atoms[j].GetIdx())
            if len(path) > max_dist:
                max_dist = len(path)
                backbone_path = path
    return set(backbone_path)


# ---------- 辅助函数 ----------
# 在 mol 的环列表里找出与 match（atom idx tuple）对应的环（返回环的循环顺序 tuple）
def _find_ring_tuple_for_match(mol, match):
    match_set = set(match)
    for r in Chem.GetSymmSSSR(mol):  # 在所有环中找到match这个环
        if set(r) == match_set:
            return tuple(r)  # 返回这个环里的原子索引

    return None


# 在片段 frag 里找出“要当作替换环”的那个 ring（循环序列）。
def _find_frag_ring_order(frag):
    """
    若 frag 有虚拟原子 [*]，优先找与[*]相连的那个环并返回该环（按环中 atom 索引顺序）
    否则返回 frag 的第一个 ring（如果有）
    """
    rings = [tuple(r) for r in Chem.GetSymmSSSR(frag)]
    if not rings:
        return None
    # 如果有 dummy [*], 找邻居并选包含该邻居的环
    dummy_idx = None
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:
            dummy_idx = a.GetIdx()
            break
    if dummy_idx is not None:
        for r in rings:
            if any(dummy_idx == idx for idx in r):
                # 虚拟原子直接位于环上（罕见） -> 返回该环
                return r
        # 否则找与 dummy 相邻的原子，并找包含该邻居的环
        for nb in frag.GetAtomWithIdx(dummy_idx).GetNeighbors():
            nb_idx = nb.GetIdx()
            for r in rings:
                if nb_idx in r:
                    return r
    # fallback: 返回第一环
    return rings[0]

# ---------- LLM输出的验证工具 ----------
# RDKit 严格校验 SMILES 是否可解析并能 sanitize
def validate_smiles_rdkit(smi: str) -> bool:

    if not isinstance(smi, str) or smi.strip() == "":
        return False
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return False
        Chem.SanitizeMol(m)  # 强校验
        return True
    except Exception:
        return False


#   尝试自动修复一些常见 LLM 输出错误
def attempt_auto_repair_smiles(smi):
    if not smi:
        return None
    s = smi.strip().strip("'\"`")

    # 去掉换行/多余符号
    s = re.sub(r'[\s\r\n]+', '', s)
    s = s.rstrip('.,;:')

    # 先直接验证
    if validate_smiles_rdkit(s):
        return s

    # 删除所有 ring digits（弱修复）
    no_digits = re.sub(r'\d', '', s)
    if validate_smiles_rdkit(no_digits):
        return no_digits

    return None



# 简单的“支化度量”：
def branch_metric(mol: Chem.Mol) -> int:
    """
    - 统计非环碳原子中度 >= 3 的原子数。
    用于评估 sidechain branch->linear 之类的变化方向。
    """
    if mol is None:
        return 0
    ring_atoms = set()
    for r in mol.GetRingInfo().AtomRings():
        ring_atoms.update(r)
    cnt = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() == "C" and a.GetIdx() not in ring_atoms and a.GetDegree() >= 3:
            cnt += 1
    return cnt

# 检查双键的立体化学信息（顺式或反式）
def get_doublebond_stereo_map(mol: Chem.Mol):
    """
    返回字典 { (atom_idx1, atom_idx2) : stereo }，键用有序元组形式(atom_min, atom_max)便于比较。
    stereo 值是 rdchem.BondStereo.STEREOZ / STEREOE / OTHER
    """
    res = {}
    for b in mol.GetBonds():
        if b.GetBondType() == rdchem.BondType.DOUBLE:
            a1 = b.GetBeginAtomIdx()
            a2 = b.GetEndAtomIdx()
            key = (min(a1, a2), max(a1, a2))
            st = b.GetStereo()
            res[key] = st
    return res


def get_chiral_map(mol: Chem.Mol):
    """返回 {atom_idx: 'R'/'S'}"""
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    except Exception:
        centers = []
    return {idx: cfg for idx, cfg in centers}

def ring_info_sizes(mol):
    rings = mol.GetRingInfo().AtomRings()
    return sorted([len(r) for r in rings])

def check_ring_change(before, after, op_type):
    # returns True if change matches expectation
    b_sizes = ring_info_sizes(before)
    a_sizes = ring_info_sizes(after)
    if op_type == "ring_fusion":
        return len(a_sizes) >= len(b_sizes) + 1 or any(a>b for a,b in zip(sorted(a_sizes, reverse=True), sorted(b_sizes, reverse=True)))
    if op_type == "ring_expansion":
        # expect at least one ring size increased by 1
        return any((s+1) in a_sizes for s in b_sizes)
    if op_type == "ring_contraction":
        return any((s-1) in a_sizes for s in b_sizes)


# ---------- 方向敏感的构象检查 ----------
def check_conformation_change(before: Chem.Mol, after: Chem.Mol, detail: str) -> bool:
    """
    direction-sensitive 检查：
    - sidechain: branch→linear 或 linear→branch（用 branch_metric）
    - doublebond: cis→trans 或 trans→cis（用 BondStereo Z/E）
    - stereocenter: R→S 或 S→R（用 FindMolChiralCenters）
    返回 True 表示 after 满足指定的方向性修改
    """
    if before is None or after is None:
        return False

    # sidechain 分支度变化
    if "target=sidechain" in detail:
        before_b = branch_metric(before)
        after_b = branch_metric(after)
        if "branch→linear" in detail:
            return after_b < before_b
        if "linear→branch" in detail:
            return after_b > before_b
        return False

    # double bond 方向（Z/E）比较
    if "target=doublebond" in detail:
        before_map = get_doublebond_stereo_map(before)
        after_map = get_doublebond_stereo_map(after)
        if not before_map or not after_map:
            return False
        # 尝试按相同原子对匹配并比较 stereo
        for key, b_st in before_map.items():
            a_st = after_map.get(key)
            if a_st is None:
                continue
            # cis->trans : before Z -> after E
            if "cis→trans" in detail:
                if b_st == rdchem.BondStereo.STEREOZ and a_st == rdchem.BondStereo.STEREOE:
                    return True
            if "trans→cis" in detail:
                if b_st == rdchem.BondStereo.STEREOE and a_st == rdchem.BondStereo.STEREOZ:
                    return True
        return False

    # 手性中心翻转 R<->S
    if "target=stereocenter" in detail:
        before_map = get_chiral_map(before)
        after_map = get_chiral_map(after)
        if "R→S" in detail:
            for idx, cfg in before_map.items():
                if cfg == "R" and after_map.get(idx) == "S":
                    return True
            return False
        if "S→R" in detail:
            for idx, cfg in before_map.items():
                if cfg == "S" and after_map.get(idx) == "R":
                    return True
            return False

    return False

# 检查双键的立体化学信息（顺式或反式）
def get_doublebond_stereo(mol):
    stereo_map = {}
    for bond in mol.GetBonds():   # 遍历所有键
        if bond.GetBondType() == rdchem.BondType.DOUBLE:  # 检查双键
            st = bond.GetStereo()   # 获取当前双键的立体化学状态
            if st in (rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE):
                stereo_map[bond.GetIdx()] = st  # 当前键: 顺式OZ/反式OE 的字典
    return stereo_map  # {bond_idx: stereo}


# --- 6 基团加法 ---
# def add_group(mol, substituent_smiles, position=None):
#     """在芳香环上按相对位点添加片段。片段推荐写成带 '[*]' 的锚定片段，如 '[*]Cl'、'[*]C(F)(F)F'。"""
#     # step 1. 返回待修改的环和参考锚点
#     ring, anchor = pick_ring_and_anchor(mol)
#     if not ring:    # 无芳环：退化为接在原子 0
#         rw = Chem.RWMol(mol)
#         frag = Chem.MolFromSmiles(substituent_smiles)
#         if frag is None:
#             return mol
#         # hostidx直接置0了，故不需要step2和3
#         return attach_fragment_on_ring(rw, 0, frag)
#
#     # step 2. 获得修改位置距离anchor的步长
#     steps = get_atdist_to_anchor(position or "ortho", len(ring))
#     if steps is None:
#         steps = 2  # 五元环没有 'para' 回退为 'meta'
#     # step3 拿到修改位置的实际原子序号
#     host_idx = get_atatomidx_on_ring(ring, anchor, steps)
#     if host_idx is None:
#         return mol
#
#     # step4 实际的加基团操作
#     frag = Chem.MolFromSmiles(substituent_smiles)
#     if frag is None:
#         return mol
#     rw = Chem.RWMol(mol)
#     return attach_fragment_on_ring(rw, host_idx, frag)


# 环操作相关
# 引入前面定义的函数
def _merge_atom_into(rw: Chem.RWMol, src: int, dst: int) -> None:
    """
    将 src 原子的所有邻居（除了 dst）重连到 dst，然后删除 src。
    等价于把 src “并入” dst，用于实现两环共享一条边（原子合并）。
    """
    src_atom = rw.GetAtomWithIdx(src)
    # 记录 src 的所有邻居索引；注意删除原子会改动索引，所以先拷贝列表
    nbs = [b.GetOtherAtomIdx(src) for b in src_atom.GetBonds()]
    for j in nbs:
        if j == dst:  # 避免自环
            continue
        bond = rw.GetBondBetweenAtoms(src, j)
        bt = bond.GetBondType()
        # 若 dst 与 j 之间尚无键，则添加保持原键型
        if rw.GetBondBetweenAtoms(dst, j) is None:
            rw.AddBond(dst, j, bt)
    # 合并完成后删除 src（索引将整体左移）
    rw.RemoveAtom(src)


def _pick_ring_to_fuse(mol, ring_smarts, mode="fusion", prefer_aromatic=True):
    """
    按操作类型选择要进行编辑的目标环。
    参数:
      mol: RDKit 分子对象
      ring_smarts: 想要命中的环 SMARTS（如 'c1ccccc1'）
      mode: 操作类型，可取 "fusion" / "expansion" / "contraction"
      prefer_aromatic: 是否优先芳香环（仅对 fusion 起作用）

    返回:
      (chosen_ring_atom_indices, all_ring_list)
    """
    ri = mol.GetRingInfo()
    all_rings = list(ri.AtomRings())
    if not all_rings:
        return [], all_rings

    # 1) 若提供 ring_smarts，则先按 SMARTS 命中挑环
    if ring_smarts:
        patt = Chem.MolFromSmarts(ring_smarts)
        if patt is not None:
            matches = mol.GetSubstructMatches(patt, uniquify=True)
            if matches:
                # 从命中的原子集中，挑选一个“刚好构成某个环子集”的环
                match_atom_set = set(matches[0])
                for r in all_rings:
                    if set(r).issubset(match_atom_set):
                        return list(r), all_rings
                # 若没有完整环子集命中，则选“覆盖度最高”的环
                best_r, best_overlap = None, -1
                for r in all_rings:
                    ov = len(set(r) & match_atom_set)
                    if ov > best_overlap:
                        best_r, best_overlap = r, ov
                if best_r is not None and best_overlap > 0:
                    return list(best_r), all_rings
        # SMARTS 无法解析或无命中则回退

    # 2) 默认选环策略按 mode 区分
    if mode == "fusion":
        # 优先芳香六元环
        if prefer_aromatic:
            aromatic_6 = [
                r for r in all_rings
                if len(r) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            ]
            if aromatic_6:
                return list(aromatic_6[0]), all_rings
        # 否则选任意 ≥5 元环
        fallback = [r for r in all_rings if len(r) >= 5]
        return (list(fallback[0]), all_rings) if fallback else (list(all_rings[0]), all_rings)

    elif mode == "expansion":
        # 优先五元环（扩张空间充足）
        target = [r for r in all_rings if len(r) == 5]
        if target:
            return list(target[0]), all_rings
        # 否则取 6–7 元环
        alt = [r for r in all_rings if 6 <= len(r) <= 7]
        if alt:
            return list(alt[0]), all_rings
        return list(all_rings[0]), all_rings

    elif mode == "contraction":
        # 优先 6 元及以上环（收缩空间足够）
        target = [r for r in all_rings if len(r) >= 6]
        if target:
            return list(target[0]), all_rings
        # 若全是 5 元环，也允许操作但风险高
        small = [r for r in all_rings if len(r) == 5]
        if small:
            return list(small[0]), all_rings
        return list(all_rings[0]), all_rings

# =====================================
# 在含挂载点的smiles上面直接加结构相关
# =====================================
# 在片段里用 [*] 替代 n 个氢原子，避免价度超限。   若氢不足或替代失败返回 None。
# def _add_dummies_by_replacing_H(frag_smiles: str, n: int):
#     mol = Chem.MolFromSmiles(frag_smiles)
#     if mol is None:
#         return None
#     rw = Chem.RWMol(mol)
#
#     hydrogens = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1]
#     if len(hydrogens) < n:
#         return None
#
#     # 从最末尾的氢开始替
#     for i in range(n):
#         h_idx = hydrogens[i]
#         nb = rw.GetAtomWithIdx(h_idx).GetNeighbors()[0]
#         nb_idx = nb.GetIdx()
#         rw.RemoveAtom(h_idx)
#         dummy = Chem.Atom(0)
#         new_idx = rw.AddAtom(dummy)
#         rw.AddBond(nb_idx, new_idx, rdchem.BondType.SINGLE)
#
#     try:
#         Chem.SanitizeMol(rw)
#         return Chem.MolToSmiles(rw)
#     except Exception:
#         return None
#
#
# # 将含 [*] 的基体 base 与片段 frag 自动挂接。
# def attach_fragment_auto(base_smiles: str, frag_smiles: str):
#     """
#     - 若 frag 缺少 [*]，自动补；补法仅替代氢，避免价度超限。
#     - 若 base 的 [*] 有多个邻居，则补足对应数量 dummy。
#     - 自动决定键方向（极性方向）。
#     """
#     base = Chem.MolFromSmiles(base_smiles)
#     frag0 = Chem.MolFromSmiles(frag_smiles)
#     if base is None or frag0 is None:
#         return None
#
#     base_dummies = [a.GetIdx() for a in base.GetAtoms() if a.GetAtomicNum() == 0]
#     if not base_dummies:
#         return None
#
#     d_base = base_dummies[0]
#     nbs_base = [n.GetIdx() for n in base.GetAtomWithIdx(d_base).GetNeighbors()]
#     deg = len(nbs_base)
#
#     frag_dummies_now = [a.GetIdx() for a in frag0.GetAtoms() if a.GetAtomicNum() == 0]
#     lack = max(0, deg - len(frag_dummies_now))
#     frag_smiles_fixed = Chem.MolToSmiles(frag0)
#
#     if lack > 0:
#         frag_smiles_fixed = _add_dummies_by_replacing_H(frag_smiles_fixed, lack)
#         if frag_smiles_fixed is None:
#             return None
#
#     frag = Chem.MolFromSmiles(frag_smiles_fixed)
#     if frag is None:
#         return None
#
#     frag_dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
#     frag_anchors_local = []
#     for d in frag_dummies[:deg]:
#         nbs = [n.GetIdx() for n in frag.GetAtomWithIdx(d).GetNeighbors() if n.GetAtomicNum() > 0]
#         if len(nbs) != 1:
#             return None
#         frag_anchors_local.append(nbs[0])
#
#     combo = Chem.CombineMols(base, frag)
#     rw = Chem.RWMol(combo)
#     offset = base.GetNumAtoms()
#
#     try:
#         for i in range(deg):
#             b = nbs_base[i]
#             f = offset + frag_anchors_local[i]
#             atom_b = rw.GetAtomWithIdx(b)
#             atom_f = rw.GetAtomWithIdx(f)
#             if atom_b.GetAtomicNum() in (7, 8, 16) and atom_f.GetAtomicNum() == 6:
#                 b, f = f, b
#             if rw.GetBondBetweenAtoms(b, f) is None:
#                 rw.AddBond(b, f, rdchem.BondType.SINGLE)
#
#         for g in sorted([offset + d for d in frag_dummies[:deg]], reverse=True):
#             if g < rw.GetNumAtoms():
#                 rw.RemoveAtom(g)
#         rw.RemoveAtom(d_base)
#
#         newmol = rw.GetMol()
#         Chem.SanitizeMol(newmol)
#         return newmol
#     except Exception:
#         return None

# 带*的分子直接加新片段
def _add_dummies_by_replacing_H(frag_smiles, need):
    """
    在片段中用“替代显式氢”的方式补充 need 个 [*]。
    规则：每次选一个带氢的重原子，删去该氢，添加一个 dummy 并与该重原子成单键。
    返回含 [*] 的片段 SMILES；若不足以补齐挂点返回 None。
    """
    m = Chem.MolFromSmiles(frag_smiles)
    if m is None:
        return None

    for _ in range(need):
        # 每次都显式化氢，查找一个可替代的氢
        mH = Chem.AddHs(m)
        done = False
        for h in [a.GetIdx() for a in mH.GetAtoms() if a.GetAtomicNum() == 1]:
            nb = mH.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()  # 氢只有一个邻居
            if mH.GetAtomWithIdx(nb).GetAtomicNum() > 1:
                rw = Chem.RWMol(mH)
                rw.RemoveAtom(h)                       # 删氢，释放一个价度
                d = rw.AddAtom(Chem.Atom(0))           # 加 dummy
                rw.AddBond(nb, d, rdchem.BondType.SINGLE)
                m = Chem.RemoveHs(rw.GetMol())         # 回到无显式氢表示
                done = True
                break
        if not done:
            return None  # 找不到可替代的氢，无法再补 dummy
    # 不在这里 Sanitize；交由上层合并后统一 Sanitize
    return Chem.MolToSmiles(m)

def attach_fragment_auto(base_smiles: str, frag_smiles: str):
    """
    将含 [*] 的基体 base 与片段 frag 自动挂接。
    - 若 frag 缺少 [*]，自动补；补法仅替代氢，避免价度超限。
    - 若 base 的该 [*] 度数=1，则 frag 需 1 个 dummy；=2 则需 2 个 dummy（两个不同锚点）。
    返回 RDKit Mol 或 None。
    """
    base = Chem.MolFromSmiles(base_smiles)
    frag0 = Chem.MolFromSmiles(frag_smiles)
    if base is None or frag0 is None:
        return None

    # 取 base 的第一个 dummy 及其邻居，确定所需挂点数
    base_dummies = [a.GetIdx() for a in base.GetAtoms() if a.GetAtomicNum() == 0]
    if not base_dummies:
        return None
    d_base = base_dummies[0]
    nbs_base = [n.GetIdx() for n in base.GetAtomWithIdx(d_base).GetNeighbors()]
    deg = len(nbs_base)  # 1 或 2

    # 统计 frag 已有 dummy 数
    frag_dummies_now = [a.GetIdx() for a in frag0.GetAtoms() if a.GetAtomicNum() == 0]
    lack = max(0, deg - len(frag_dummies_now))
    frag_smiles_fixed = Chem.MolToSmiles(frag0)

    # 不足则按需补 dummy（替代氢避免超价）
    if lack > 0:
        frag_smiles_fixed = _add_dummies_by_replacing_H(frag_smiles_fixed, lack)
        if frag_smiles_fixed is None:
            return None

    frag = Chem.MolFromSmiles(frag_smiles_fixed)
    if frag is None:
        return None

    # 取 frag 的 dummy 及各自锚点（dummy 的邻居）
    frag_dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(frag_dummies) < deg:
        return None

    frag_anchors_local = []
    for d in frag_dummies[:deg]:
        nbs = [n.GetIdx() for n in frag.GetAtomWithIdx(d).GetNeighbors() if n.GetAtomicNum() > 0]
        if len(nbs) != 1:
            return None
        frag_anchors_local.append(nbs[0])

    # 合并并成键；注意：先加键，后删 dummy，最后再 Sanitize，避免中间态超价触发异常
    combo = Chem.CombineMols(base, frag)
    rw = Chem.RWMol(combo)
    offset = base.GetNumAtoms()

    try:
        # 逐一配对 base 邻居 与 frag 锚点
        for i in range(deg):
            b = nbs_base[i]
            f = offset + frag_anchors_local[i]

            atom_b = rw.GetAtomWithIdx(b)
            atom_f = rw.GetAtomWithIdx(f)

            # === 检查“是否该反接” ===
            def has_carbonyl_like(atom):
                # 若该原子是C且有一个邻接O且键为双键
                for nb in atom.GetNeighbors():
                    if nb.GetAtomicNum() == 8:  # O
                        bnd = rw.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                        if bnd and bnd.GetBondTypeAsDouble() == 2.0:
                            return True
                return False

            flip = False
            # 若母体端是羰基碳，而片段锚点是N/O/S，则反接
            if atom_b.GetAtomicNum() == 6 and has_carbonyl_like(atom_b) and atom_f.GetAtomicNum() in (7, 8, 16):
                flip = True
            # 若母体端是N/O/S，片段端是羰基碳，也反接
            elif atom_f.GetAtomicNum() == 6 and has_carbonyl_like(atom_f) and atom_b.GetAtomicNum() in (7, 8, 16):
                flip = True

            if flip:
                b, f = f, b

            if rw.GetBondBetweenAtoms(b, f) is None:
                rw.AddBond(b, f, rdchem.BondType.SINGLE)

        # 删 frag 的 dummy（全局索引降序），再删 base 的 dummy
        for g in sorted([offset + d for d in frag_dummies[:deg]], reverse=True):
            if g < rw.GetNumAtoms():
                rw.RemoveAtom(g)
        rw.RemoveAtom(d_base)

        # 最终校验
        newmol = rw.GetMol()
        Chem.SanitizeMol(newmol)
        return newmol
    except Exception:
        return None

