import json
import logging
import re

from langchain_core.messages import AIMessage
from rdkit import Chem
from rdkit.Chem import rdchem, rdFMCS
from langchain.schema import HumanMessage
from rdkit.Chem.rdmolops import GetShortestPath

from utils.convenient_utils.suppress_useless_print import suppress_everything
from utils.convenient_utils.wordart import ColorText

from utils.process_drug_utils.edit_mol_tools import pick_ring_and_anchor, get_atatomidx_on_ring, cut_sidechain_at_atatom, attach_fragment_on_ring, validate_smiles_rdkit, check_conformation_change, \
    check_ring_change, get_atdist_to_anchor, GREEK_POS_MAP, get_greek_dist, find_backbone_atoms, _find_frag_ring_order, _find_ring_tuple_for_match, pick_ring_and_anchor2, _pick_ring_to_fuse, \
    _merge_atom_into, attach_fragment_auto
from utils.process_drug_utils.parse_chemform_tool import SubstituentResolver


# 从llm的返回结果里正则匹配出smiles
def extract_smiles_from_text(text):
    """
    从 LLM 输出文本中提取第一个合法 SMILES。
    """
    # cand = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)=#$\\/\.]+', text)
    cand = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)=#$\\/.\*:]+', text)  # 带*的也匹配出来
    for s in cand:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            continue
    return None


# ===========================================================
# 正则匹配出edit_op中的化学式或基团描述，方便后续处理
# ===========================================================
def parse_edit_op(edit_op, trgger, llm):
    resolver = SubstituentResolver()  # 定义分子式(->smiles)的解析器
    edit_op = edit_op.strip()  # 后面的思路都是1、先解析文本（拿关键词做json）。2、把分子式解析成smiles
    # # --- 5 构象调控 ---
    # if edit_op.startswith("conformation:"):
    #     return {"action": "conformation", "detail": edit_op}

    # --- 1 骨架替换 ---
    if edit_op.startswith("replace:"):  # (eg) replace: phenyl → furan
        src, tgt = edit_op.replace("replace:", "").split("→")
        src, tgt = src.strip(), tgt.strip()
        with suppress_everything():
            src_smiles, tgt_smiles \
                = resolver.resolve(src, llm) if resolver else src, resolver.resolve(tgt, llm) if resolver else tgt
        src, tgt = edit_op.replace("replace:", "").split("→")  # 拿出替换前后的两个表述

        ColorText.print(f"//////////replace {src_smiles} -> {tgt_smiles} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "replace",
            "src": src, "tgt": tgt,
            "src_smiles": src_smiles,
            "tgt_smiles": tgt_smiles,
        }
    # --- 2 功能团转化 ---
    if "→" in edit_op and not edit_op.startswith("replace:"):  # (eg)"transform: NH2 → NHCOCH3" 或 NH2 → NO2
        src, tgt = edit_op.split("→")  # 拿出替换前后的两个表述
        src, tgt = src.strip("+ "), tgt.strip()
        with suppress_everything():
            src_smiles, tgt_smiles \
                = resolver.resolve(src, llm) if resolver else src, resolver.resolve(tgt, llm) if resolver else tgt
        ColorText.print(f"//////////transform {src_smiles} -> {tgt_smiles} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "transform",
            "src": src, "tgt": tgt,
            "src_smiles": src_smiles,
            "tgt_smiles": tgt_smiles,
        }

    # --- 3 环操作 ---    ring_fusion|ring_expansion|ring_contraction
    if "ring_fusion" in edit_op:
        ex = edit_op.split(":")[1].strip()
        ex_smiles = resolver.resolve(ex, llm) if resolver else ex
        ColorText.print(f"//////////ring_fusion {ex_smiles} at {trgger} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "special",
            "type": "ring_fusion",
            "external": ex_smiles,
            "trigger": trgger
        }
    if "ring_expansion" in edit_op:
        ColorText.print(f"//////////ring_expansion at {trgger} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "special",
            "type": "ring_expansion",
            "trigger": trgger
        }
    if "ring_contraction" in edit_op:
        ColorText.print(f"//////////ring_contraction at {trgger} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "special",
            "type": "ring_contraction",
            "trigger": trgger
        }
    # --- 4 链长调节 ---
    if edit_op.startswith("chain_length:"):  # (eg) chain_length: (target=backbone) +CC / -CH2 / +CCC...
        inside = edit_op.replace("chain_length:", "").strip()
        target_match = re.search(r"target=(\w+)", inside)  # 提取出backbone或sidechain
        target = target_match.group(1) if target_match else "unknown"

        # delta_match = re.search(r"([+-]\s*[C,H,0-9]+)", inside)  # 找±开头，即匹配出后面加几个碳
        delta_match = re.search(r"([+-]\s*[A-Za-z0-9@+\-\[\]]+(?:\*[0-9]+)?)", inside)
        delta = delta_match.group(1).replace(" ", "") if delta_match else None

        mode = "extend" if delta and delta.startswith("+") else "shrink"
        ColorText.print(f"//////////chain_length {mode} {delta} at {target} ", ColorText.BLUE, ColorText.REVERSE)
        return {
            "action": "chain_length",
            "mode": mode,
            "target": target,
            "delta": delta
        }

    # --- 6 基团加法 ---
    if edit_op.startswith("+"):
        op = edit_op[1:]
        if "@" in op:
            group, pos = op.split("@", 1)
            with suppress_everything():
                smiles = resolver.resolve(group, llm) if resolver else group
            # return {"action": "add", "group": group.strip(), "position": pos.strip()}
            ColorText.print(f"//////////add {smiles} at {trgger} @{pos} ", ColorText.BLUE, ColorText.REVERSE)
            return {
                "action": "add",
                "group": group,
                "position": pos.strip(),
                "group_smiles": smiles,
                "trigger": trgger
            }
        else:  # 没加位置信息，暂时先不认为是错吧
            group = op.strip()
            with suppress_everything():
                smiles = resolver.resolve(group, llm) if resolver else group
            return {
                "action": "add",
                "group": group,
                "position": None,
                "group_smiles": smiles
            }
    # # --- 7 基团减法 ---
    # if edit_op.startswith("-"):
    #     op = edit_op[1:].strip()
    #     with suppress_everything():
    #         smiles = resolver.resolve(op, llm) if resolver else op
    #     # return {"action": "remove", "group": op.strip()}
    #     return {"action": "remove", "group": op, "group_smiles": smiles}

    raise ValueError(f"无法解析 edit_op: {edit_op}")


# ===========================================================
# 根据解析出的结果去做具体的分子编辑逻辑
# ===========================================================
def apply_edit_op(mol, parsed_op, llm):
    """
    根据 parse_edit_op 的结果调用对应的编辑函数
    - mol: RDKit Mol
    - parsed_op: parse_edit_op(...) 的返回 dict
    - llm: 用于需要 LLM 辅助（构象调控/环操作）
    """
    action = parsed_op["action"]

    # # --- 1 构象调控 ---
    # if action == "conformation":
    #     if llm is None:
    #         raise ValueError("Conformation editing requires an LLM instance")
    #     return llm_edit(mol, parsed_op, llm)

    # --- 2 骨架替换 ---
    if action == "replace":
        with suppress_everything():
            return replace_ring(mol, parsed_op["src_smiles"], "[*]" + parsed_op["tgt_smiles"],
                                False, True, 5, llm)

    # --- 3 功能团转化 ---
    if action == "transform":
        with suppress_everything():
            return replace_fg(mol, parsed_op["src_smiles"], '[*]' + parsed_op["tgt_smiles"], llm)

    # --- 4 环操作 ---
    if action == "special":
        with suppress_everything():
            return apply_ring_operation(mol, parsed_op)

    # --- 5 链长调节 ---
    if action == "chain_length":
        with suppress_everything():
            return chain_length_edit(
                mol,
                parsed_op["mode"],
                parsed_op["delta"],
                target=parsed_op.get("target", "backbone")
            )

    # --- 6 基团加法 ---
    if action == "add":
        with suppress_everything():
            return add_group(mol, parsed_op["group_smiles"], parsed_op.get("position"), parsed_op.get("trigger"), llm)

    # # --- 7 基团减法 ---
    # if action == "remove":
    #     return remove_group(mol, parsed_op.get("group"))

    raise ValueError(f"Unsupported action: {action}")


# ===========================================================
# -------------------- 具体的手动修改逻辑 --------------------
# ===========================================================
# 用 new_ring_smiles 替换分子中的 old_ring_smarts 环 (骨架替换单环版本)。
def replace_single_ring(mol, old_ring_smarts, new_ring_smiles, replace_all=True, allow_fused=False, llm=None):
    """
    - replace_all=True 则循环替换直到没有匹配（每次替换第一个匹配）,否则就替换一个old_ring_smarts的匹配
      allow_fused  是否匹配融合环中的old_ring_smarts,(为true时就会替换融合环里的匹配的单环)
    """
    ColorText.print("|||||||||||||||| replace_single_ring ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    patt = Chem.MolFromSmiles(old_ring_smarts)
    if patt is None:
        ColorText.print(f"//////////【F: replace_ring】old_ring_smarts不合法", ColorText.RED)
        return mol

    cur_mol = Chem.Mol(mol)  # 拷贝，避免修改原始对象
    replaced_atoms = set()  # 记录已替换过的原子，避免重复/重叠替换

    while True:  # 匹配原分子中的所有old_ring_smarts
        matches, match = cur_mol.GetSubstructMatches(patt), None

        if not matches:  # 没有找到匹配 1、确实没有匹配 2、分子中符合的部分都替换完了
            ColorText.print(f"//////////【F: replace_ring】本轮没在{Chem.MolToSmiles(cur_mol)}里找到{old_ring_smarts}，结束", ColorText.PURPLE)
            break
        ColorText.print(f"//////////【F: replace_ring】本轮{Chem.MolToSmiles(cur_mol)}里找到了{old_ring_smarts}，继续尝试", ColorText.GREEN)
        print(f"//////////【F: replace_ring】本轮新匹配出的matchs：{matches}")

        # 新增：去重并排序
        uniq, seen = [], set()
        for m in matches:
            key = tuple(sorted(m))
            if key not in seen:
                seen.add(key)
                uniq.append(m)
        uniq.sort(key=lambda m: max(m), reverse=True)  # 按最大索引降序（防止索引重排破坏后续匹配）

        for m in uniq:  # 遍历所有match的结构
            mset = set(m)  # 当前match的索引集合

            # --- 已替换过的原子不能再替 ---
            if replaced_atoms & mset:  # 交集不为空，即mset在replaced_atoms
                continue  # 就说明这个match(m)已经替换过了，跳过直到拿到没被替换m在进行下面的判断

            # --- 如果不允许融合环，跳过和其他环共享原子的匹配 ---
            if not allow_fused:
                rings = [set(r) for r in Chem.GetSymmSSSR(cur_mol)]  # 当前分子里面的所有环
                fused = any((mset & r) and (len(mset & r) < len(mset)) for r in rings)
                if fused:  # (mset & r): 当前match与所有环的交集
                    # len(mset & r) < len(mset)，说明当前匹配的环结构与另一个环共享了一些原子，【但不是完全重叠】
                    # any表示只要存在某个环与当前match【有部分共享】，说明这个匹配在一个融合环里
                    print("//////////跳过融合环匹配:", mset)
                    continue
            match = m  # 直到找到一个没操作过、不在融合环里的m这个循环就结束了
            ColorText.print(f"//////////【F: replace_ring】新的待处理match：{match}", ColorText.BLUE)
            break
        if match is None:  # 没有可替换的匹配
            break

        # === 正式修改逻辑 ===
        # step1 找到该 match 对应的“环的循环顺序”
        ring_tuple = _find_ring_tuple_for_match(cur_mol, match)
        ring_set = set(ring_tuple)

        # step2 为环上每个位置收集其外部邻居（sidechains）
        attach_map = {i: [] for i in range(len(ring_tuple))}
        for pos_idx, atom_idx in enumerate(ring_tuple):
            atom = cur_mol.GetAtomWithIdx(atom_idx)
            for nb in atom.GetNeighbors():
                if nb.GetIdx() not in ring_set:
                    attach_map[pos_idx].append(nb.GetIdx())

        # step3 在 rw 上把这些外部邻居打标（存 pos 编号），以便删除环之后还能找到对应的新索引
        rw = Chem.RWMol(cur_mol)
        for pos_idx, ext_list in attach_map.items():
            for ext_idx in ext_list:
                a = rw.GetAtomWithIdx(ext_idx)
                prev = a.GetProp("_to_attach_pos") if a.HasProp("_to_attach_pos") else ""
                if prev:
                    prev = prev + "," + str(pos_idx)
                else:
                    prev = str(pos_idx)
                a.SetProp("_to_attach_pos", prev)

        # step4 删除环上的原子（倒序以免索引错位）
        for idx in sorted(ring_tuple, reverse=True):  # 倒序删是因为删小的每删一次后续原子索引都会变
            if idx < rw.GetNumAtoms():  # 防止删的索引不在分子里
                rw.RemoveAtom(idx)  # KS  此时rw内的原子序号就已经变了，如果用原本ext_list记录的原子序号就对应错误了，这就体现了标记的作用
        print("//////////【F: replace_ring】删除旧环后（碎片）SMILES:", Chem.MolToSmiles(rw))

        # step4 删除旧环后重新获取留下的外部原子索引
        post_attach = {}
        for a in rw.GetAtoms():
            if a.HasProp("_to_attach_pos"):
                val = a.GetProp("_to_attach_pos")
                for tok in val.split(","):
                    pos = int(tok)
                    post_attach.setdefault(pos, []).append(a.GetIdx())
                a.ClearProp("_to_attach_pos")

        # step5 找到新环中的 dummy 和 dummy 的邻居 （这部分的）
        frag = Chem.MolFromSmiles(new_ring_smiles)
        if frag is None:
            return cur_mol

        frag_ring = _find_frag_ring_order(frag)
        if frag_ring is None:
            frag_attach_target_idxs = [0]
        else:
            frag_attach_target_idxs = list(frag_ring)

        # step6 合并并重连到新环上真正的挂点
        combo = Chem.CombineMols(rw, frag)  # 含有删除原片段的原分子、新片段的分子对象
        rw2 = Chem.RWMol(combo)
        offset = rw.GetNumAtoms()  # 计算删除环后的原分子(rw)中的原子数

        frag_ring_len = len(frag_attach_target_idxs)
        for pos_idx, ext_idxs in post_attach.items():
            mapped = frag_attach_target_idxs[pos_idx % frag_ring_len]
            mapped_global = offset + mapped
            for ext_global in ext_idxs:
                try:
                    rw2.AddBond(ext_global, mapped_global, rdchem.BondType.SINGLE)
                except Exception:
                    pass

        dummy_idx = None
        for a in frag.GetAtoms():
            if a.GetAtomicNum() == 0:
                dummy_idx = a.GetIdx()
                break
        if dummy_idx is not None:
            try:
                rw2.RemoveAtom(offset + dummy_idx)
            except Exception:
                pass

        try:
            Chem.SanitizeMol(rw2)
            cur_mol = rw2.GetMol()
        except Exception as e:
            print("//////////Sanitize 失败（replace_ring）：", e)
            return mol

        replaced_atoms.update(ring_tuple)  ### 新增：记录已替换过的环原子

        if not replace_all:
            break

    return replace_single_ring_with_llm(cur_mol, mol, old_ring_smarts, new_ring_smiles, llm)


# 用 new_ring_smiles 替换分子中的 old_ring_smarts 环 (骨架替换融合环版本)。
def replace_fused_ring(mol, old_ring_smiles, new_ring_smiles, replace_all=True, mcs_timeout=5, llm=None):
    ColorText.print("|||||||||||||||| replace_fused_ring ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    patt = Chem.MolFromSmiles(old_ring_smiles)
    if patt is None:
        return mol

    cur_mol = Chem.Mol(mol)
    replaced_atoms = set()

    while True:
        matches = cur_mol.GetSubstructMatches(patt)
        if not matches:
            break

        did_replace = False
        for m in matches:
            mset = set(m)
            if replaced_atoms & mset:
                continue

            frag_src = Chem.MolFromSmiles(old_ring_smiles)
            frag_tgt = Chem.MolFromSmiles(new_ring_smiles)
            if frag_src is None or frag_tgt is None:
                continue

            try:
                mcs = rdFMCS.FindMCS(
                    [frag_src, frag_tgt],
                    completeRingsOnly=True,
                    atomCompare=rdFMCS.AtomCompare.CompareAny,
                    bondCompare=rdFMCS.BondCompare.CompareAny,
                    timeout=mcs_timeout
                )
            except Exception:
                continue

            if not mcs or not getattr(mcs, "smartsString", None):
                continue

            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            src_match = frag_src.GetSubstructMatch(mcs_mol)
            tgt_match = frag_tgt.GetSubstructMatch(mcs_mol)
            if not src_match or not tgt_match:
                continue

            anchor_map = dict(zip(src_match, tgt_match))

            rw = Chem.RWMol(cur_mol)
            patt_idx_to_global = {pidx: gidx for pidx, gidx in enumerate(m)}
            for pidx, gidx in patt_idx_to_global.items():
                atom = rw.GetAtomWithIdx(gidx)
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() not in mset:
                        a = rw.GetAtomWithIdx(nb.GetIdx())
                        prev = a.GetProp("_to_attach_from_patt") if a.HasProp("_to_attach_from_patt") else ""
                        if prev:
                            prev = prev + "," + str(pidx)
                        else:
                            prev = str(pidx)
                        a.SetProp("_to_attach_from_patt", prev)

            for idx in sorted(m, reverse=True):
                rw.RemoveAtom(idx)

            attach_map_post = {}
            for a in rw.GetAtoms():
                if a.HasProp("_to_attach_from_patt"):
                    val = a.GetProp("_to_attach_from_patt")
                    for tok in val.split(","):
                        pidx = int(tok)
                        attach_map_post.setdefault(pidx, []).append(a.GetIdx())
                    a.ClearProp("_to_attach_from_patt")

            frag = Chem.MolFromSmiles(new_ring_smiles)
            if frag is None:
                return cur_mol

            combo = Chem.CombineMols(rw, frag)
            rw2 = Chem.RWMol(combo)
            offset = rw.GetNumAtoms()

            for src_pidx, ext_list in attach_map_post.items():
                if src_pidx not in anchor_map:
                    continue
                tgt_local = anchor_map[src_pidx]
                tgt_global = offset + tgt_local
                for ext_global in ext_list:
                    if rw2.GetBondBetweenAtoms(ext_global, tgt_global) is None:
                        rw2.AddBond(ext_global, tgt_global, rdchem.BondType.SINGLE)

            try:
                Chem.SanitizeMol(rw2)
                cur_mol = rw2.GetMol()
            except Exception:
                return mol

            replaced_atoms.update(mset)
            did_replace = True
            if not replace_all:
                break

        if not did_replace:
            break

    return replace_fused_ring_with_llm(cur_mol, mol, old_ring_smiles, new_ring_smiles, llm)


# --- 2 骨架替换 ---
def replace_ring(mol, old_ring_smiles, new_ring_smiles, replace_all=False, allow_fused=True, mcs_timeout=5, llm=None):
    """把前面两个环替换的情况加在一起"""
    patt = Chem.MolFromSmiles(old_ring_smiles)
    if patt is None:
        return mol

    patt_rings = list(Chem.GetSymmSSSR(patt))  # 在patt里面再获取环
    is_fused_pattern = len(patt_rings) > 1  # 旧环如果超过一个环说明是融合环

    if is_fused_pattern:  # 替换类型是融合环
        return replace_fused_ring(mol, old_ring_smiles, new_ring_smiles, replace_all, mcs_timeout, llm)
    else:  # 替换类型是单环（只替换第一个匹配环）
        return replace_single_ring(mol, old_ring_smiles, new_ring_smiles, False, False, llm)

    #
    # if is_fused_pattern and allow_fused:  # 存在融合环, 但是只替换其中一个环
    #     return replace_fused_ring(mol, old_ring_smiles, new_ring_smiles, replace_all, mcs_timeout, llm)
    # elif is_fused_pattern and not allow_fused:  # 存在融合环，但只想替换其中的某个环
    #     return replace_single_ring(mol, old_ring_smiles, new_ring_smiles, replace_all, allow_fused, llm)
    # else:
    #     return replace_single_ring(mol, old_ring_smiles, new_ring_smiles, replace_all, allow_fused, llm)


# --- 2 功能团转化 ---
def replace_fg(mol, src_smarts, new_frag_smiles, llm):
    """
    将分子中匹配 src_smarts 的第一个基团替换为 new_frag_smiles。
    - src_smarts 用 SMARTS 解析
    - new_frag_smiles 优先包含 [*] 作为挂点
    - 保守策略：要求每个“匹配位点”(pattern atom) 对应的外连原子数为 1，
      且 new_frag 的挂点数（dummy->neighbor slots）与匹配位点数一致。
      不满足则放弃替换并返回原分子（并输出调试信息）。
    调试输出使用 print（你的环境若有 ColorText 会尝试用它）。
    """
    ColorText.print("|||||||||||||||| replace_fg ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)

    def dbg(msg):
        try:
            ColorText.print(msg, ColorText.BLUE)
        except Exception:
            print(msg)

    # 1. pattern
    patt = Chem.MolFromSmarts(src_smarts)
    if patt is None:
        dbg(f"【F: replace_fg】 无效 src_smarts: {src_smarts}")
        return mol

    matches = mol.GetSubstructMatches(patt)
    if not matches:
        dbg(f"【F: replace_fg】 在 {Chem.MolToSmiles(mol)} 中未找到 pattern {src_smarts}")
        return mol
    dbg(f"【F: replace_fg】 在 {Chem.MolToSmiles(mol)} 中找到 matches: {matches}")

    # 只处理第一个 match（保持原行为）
    match = matches[0]
    match_atoms = set(match)

    # 2. 在删除前标记外部邻居（把匹配原子对应的 patt_idx 写到外部原子上）
    rw = Chem.RWMol(mol)
    for pidx, gidx in enumerate(match):
        atom = rw.GetAtomWithIdx(gidx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in match_atoms:
                a = rw.GetAtomWithIdx(nb_idx)
                prev = a.GetProp("_to_attach_from_patt") if a.HasProp("_to_attach_from_patt") else ""
                a.SetProp("_to_attach_from_patt", prev + ("," if prev else "") + str(pidx))

    # 3. 删除匹配原子（倒序）
    for idx in sorted(match, reverse=True):
        if idx < rw.GetNumAtoms():
            rw.RemoveAtom(idx)
    dbg(f"【F: replace_fg】 删除匹配原子后碎片 SMILES: {Chem.MolToSmiles(rw)}")

    # 4. 收集删除后被标记的外部原子，构造 attach_map_post: patt_idx -> [global_atom_idx]
    attach_map_post = {}
    for a in rw.GetAtoms():
        if a.HasProp("_to_attach_from_patt"):
            val = a.GetProp("_to_attach_from_patt")
            for tok in val.split(","):
                pidx = int(tok)
                attach_map_post.setdefault(pidx, []).append(a.GetIdx())
            a.ClearProp("_to_attach_from_patt")
    dbg(f"[replace_fg] attach_map_post (patt_idx -> new global atom idx list): {attach_map_post}")

    # 保守检查：要求每个 patt_idx 对应的外连原子数量为 1（单位连接位点）
    multi_attach = {p: lst for p, lst in attach_map_post.items() if len(lst) > 1}
    if multi_attach:
        dbg(f"[replace_fg] 检测到匹配位点存在多个外连原子，放弃替换。multi_attach={multi_attach}")
        return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)

    # 5. 解析新基团
    frag = Chem.MolFromSmiles(new_frag_smiles)
    if frag is None:
        dbg(f"【F: replace_fg】 new_frag_smiles 无法解析: {new_frag_smiles}")
        return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)

    # 6. 计算 frag 的挂点 slots（基于 dummy[*] 的邻居）
    frag_dummy_idxs = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    frag_slots = []  # 每个 slot 表示 frag 上可连到母体的真实原子索引（frag 局部 idx）
    frag_dummy_map = {}  # dummy -> neighbor list (局部 idx)
    for d in frag_dummy_idxs:
        neighs = [n.GetIdx() for n in frag.GetAtomWithIdx(d).GetNeighbors()]
        frag_dummy_map[d] = neighs
        for n in neighs:
            frag_slots.append(n)
    dbg(f"【F: replace_fg】 frag_dummy_idxs={frag_dummy_idxs}, frag_dummy_map={frag_dummy_map}, frag_slots={frag_slots}")

    patt_slots = sorted(attach_map_post.keys())  # patt_idx 列表（有顺序）
    dbg(f"【F: replace_fg】 patt_slots(sorted)={patt_slots}")

    # 7. 匹配数目检查
    if frag_slots:
        if len(frag_slots) != len(patt_slots):
            dbg(f"【F: replace_fg】 frag_slots 数量({len(frag_slots)}) != patt_slots 数量({len(patt_slots)}), 放弃替换")
            return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)
    else:
        # 没有 dummy 的情况下，仅允许单个 patt_slot 且该 slot 仅有单个外连原子（已保证），
        # 我们退化成把外连原子接到 frag 的第一个非氢原子
        if len(patt_slots) != 1:
            dbg(f"【F: replace_fg】 frag 没有 dummy，且 patt_slots 数 !=1，放弃替换")
            return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)

    # 8. 合并并重连
    combo = Chem.CombineMols(rw, frag)
    rw2 = Chem.RWMol(combo)
    offset = rw.GetNumAtoms()

    # 如果有 frag_slots，按顺序一一对应
    if frag_slots:
        for i, pidx in enumerate(patt_slots):
            frag_local = frag_slots[i]
            frag_global = offset + frag_local
            ext_atoms = attach_map_post.get(pidx, [])
            # ext_atoms 长度在前面已保证为 1
            for ext in ext_atoms:
                if rw2.GetBondBetweenAtoms(ext, frag_global) is None:
                    try:
                        rw2.AddBond(ext, frag_global, rdchem.BondType.SINGLE)
                    except Exception as e:
                        dbg(f"【F: replace_fg】 AddBond 失败 ext={ext} frag_global={frag_global} 异常={e}, 回退")
                        return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)
    else:
        # fallback: 没有 dummy，connect to first non-H atom in frag
        frag_attach_local = None
        for a in frag.GetAtoms():
            if a.GetAtomicNum() > 1:
                frag_attach_local = a.GetIdx()
                break
        if frag_attach_local is None:
            dbg("【F: replace_fg】 fallback 找不到 frag 的非氢原子，放弃")
            return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)
        frag_global = offset + frag_attach_local
        ext_atoms = []
        for lst in attach_map_post.values():
            ext_atoms.extend(lst)
        for ext in ext_atoms:
            if rw2.GetBondBetweenAtoms(ext, frag_global) is None:
                try:
                    rw2.AddBond(ext, frag_global, rdchem.BondType.SINGLE)
                except Exception as e:
                    dbg(f"【F: replace_fg】 fallback AddBond 失败 ext={ext} frag_global={frag_global} 异常={e}, 回退")
                    return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)

    # 9. 删除 frag 中的 dummy（按全局索引倒序删除）
    if frag_dummy_idxs:
        dummy_globals = sorted([offset + d for d in frag_dummy_idxs], reverse=True)
        for dg in dummy_globals:
            try:
                if dg < rw2.GetNumAtoms():
                    rw2.RemoveAtom(dg)
            except Exception as e:
                dbg(f"【F: replace_fg】 删除 dummy_global={dg} 失败: {e}, 回退")
                return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)

    # 10. Sanitize 并返回
    try:
        Chem.SanitizeMol(rw2)
        newmol = rw2.GetMol()
        dbg(f"【F: replace_fg】 替换成功, 新分子 SMILES: {Chem.MolToSmiles(newmol)}")
        return replace_fg_with_llm(newmol, mol, src_smarts, new_frag_smiles, llm)
    except Exception as e:
        dbg(f"【F: replace_fg】 Sanitize 失败: {e}, 放弃替换并返回原分子")
        return replace_fg_with_llm(mol, mol, src_smarts, new_frag_smiles, llm)



# --- 3 环操作 ---    用LLM做(这里还差个把smart加进去)
def build_ring_prompt(smi, edit_type):
    # 此处定义环操作的llm提示词
    examples = {  # 例子
        "ring_fusion": ('trigger:["c1ccccc1", "c1ccccc1"]', 'output:"c1ccc2ccccc2c1"'),  # benzene -> naphthalene
        "ring_expansion": ('trigger:["C1CCCCC1"]', 'output:"C1CCCCCC1"'),  # cyclohexane -> cycloheptane
        "ring_contraction": ('trigger:["C1CCCCC1"]', 'output:"C1CCCC1"'),  # cyclohexane -> cyclopentane
    }
    exemplar = examples.get(edit_type)
    gene = "Locate the 'trigger' part in the `Input` molecule and"
    task_map = {  # 任务解释
        # "ring_fusion": "Fuse an additional ring onto an existing aromatic ring",
        "ring_fusion": f"{gene} perform a fusion operation on these two parts. (e.g., 2 benzene->naphthalene).",
        "ring_expansion": f"{gene} expand `trigger`(n member ring) to a n+1 member ring (e.g., cyclohexane->cycloheptane).",
        "ring_contraction": f"{gene} contract `trigger`(n-member ring) to a n-1 member ring (e.g., cyclohexane->cyclopentane)."
    }
    task = task_map.get(edit_type, f"Please perform a ring operation: {edit_type}")

    example_text = ""
    if exemplar:
        example_text = f"\n: {exemplar[0]}\noutput: {exemplar[1]}\n"
    prompt = (
        f"You are a molecular editor.\n"
        f"Input: {smi}\n"
        f"Task: {task}\n"
        f"{example_text}"
        f"Output requirements:\n"
        f" - Only output the optimized molecule, no arrow, no prefix.\n"
    )
    return prompt


# 执行环类特殊编辑操作。 "ring_expansion" / "ring_contraction" / "ring_fusion",
def apply_ring_operation(mol, parsed_op):
    op_type, trigger = parsed_op.get("type"), parsed_op.get("trigger")

    try:
        if op_type == "ring_fusion":
            external = parsed_op.get("external")
            if not external:
                raise ValueError("ring_fusion 缺少 external 字段")
            new_mol = apply_ring_fusion(mol, external, trigger[0])

        elif op_type == "ring_expansion":
            new_mol = ring_expansion(mol, ring_smarts=trigger[0])

        elif op_type == "ring_contraction":
            new_mol = ring_contraction(mol, ring_smarts=trigger[0])

        else:
            raise ValueError(f"未知操作类型: {op_type}")

        # 校验生成的分子
        if new_mol is not None:
            Chem.SanitizeMol(new_mol)
            return new_mol
        else:
            print(f"//////////[WARN] 环操作 {op_type} 执行失败。返回原分子")
            return mol

    except Exception as e:
        print(f"//////////[ERROR] 环操作执行异常: {e}，返回原分子")
        return mol


# ringfusion
def apply_ring_fusion(mol, external_ring="c1ccccc1", target_ring_smarts=None):
    """
    将分子 mol 与 external_ring 进行“共边环融合（fused）”。
    关键实现：通过“原子合并”共享一条边，避免出现两环之间多余的矩形“四方环”。

    参数：
    - mol: 目标分子（RDKit Mol）
    - external_ring: 外部环的 SMILES，默认苯 'c1ccccc1'
    - target_ring_smarts: 指定希望融合的目标环 SMARTS；若未命中则回退到默认选环策略

    实现步骤：
    1) 选择目标分子中的一个环（按 target_ring_smarts 命中，否则默认策略）。
    2) 选择外部环的一个环（默认取第一个环）。
    3) 遍历目标环与外部环的所有相邻原子对（候选共享边）。
    4) 对于每一对候选边，执行：
        a) CombineMols 组合分子；
        b) 将外部环边 b1–b2 的两个端点原子“合并”到目标环边 a1–a2（b1→a1，b2→a2）；
        c) 保留 a1–a2 原有边，不再新增跨接键；
        d) Sanitize 检查结构合法性；
       若成功，返回该融合产物。
    5) 若所有边组合均失败，返回 None。
    """
    ColorText.print("|||||||||||||||| apply_ring_fusion ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    if mol is None:
        return None

    ext = Chem.MolFromSmiles(external_ring)
    if ext is None:
        return None

    # 1) 选取目标环
    tgt_ring, tgt_ring_list = _pick_ring_to_fuse(mol, target_ring_smarts, prefer_aromatic=True)
    if not tgt_ring:
        return None

    # 2) 外部环信息
    ri_e = ext.GetRingInfo()
    ext_rings = list(ri_e.AtomRings())
    if not ext_rings:
        return None
    ext_ring = ext_rings[0]  # 简化：外部只取第一个环；如需更复杂策略可改

    # 预先组合一次，用于拷贝基体以节省重复构造
    base_combo = Chem.CombineMols(mol, ext)
    offset = mol.GetNumAtoms()

    # 3) 遍历所有相邻原子对（候选共享边）
    def _adjacent_pairs(ring_atoms):
        n = len(ring_atoms)
        return [(ring_atoms[i], ring_atoms[(i + 1) % n]) for i in range(n)]

    tgt_edges = _adjacent_pairs(tgt_ring)
    ext_edges = _adjacent_pairs(list(ext_ring))

    for (a1, a2) in tgt_edges:
        for (b1, b2) in ext_edges:
            try:
                # 每次尝试都从 base_combo 复制新的 RWMol
                rw = Chem.RWMol(base_combo)

                # 将外部环两个端点索引平移到组合分子的编号空间
                b1_shift = b1 + offset
                b2_shift = b2 + offset

                # === 核心：原子合并以共享边 ===
                # 先把 b1 并入 a1
                _merge_atom_into(rw, b1_shift, a1)
                # 删除 b1 后，若 b2 的索引在 b1 之后，需要 -1 修正
                if b2_shift > b1_shift:
                    b2_shift -= 1
                # 再把 b2 并入 a2
                _merge_atom_into(rw, b2_shift, a2)

                # 共享边即为 a1–a2 原有的环内边，不再添加任何新键

                fused = rw.GetMol()
                Chem.SanitizeMol(fused)  # 价态、芳香性等一致性检查

                return fused
            except Exception:
                # 本对边不合法（价态/拓扑失败），试下一个组合
                continue

    # 所有组合尝试失败
    return None


# 对目标环执行环扩张操作：在选定环的一条单键上插入 -CH2-
def ring_expansion(mol, ring_smarts=None):
    """
       在选中的目标环的一条边上插入 -CH2-，完成环扩张。
       - ring_smarts: 指定优先编辑的环 SMARTS；若未命中，内部按规则选择环（优先五元环）。
       返回：新分子；若所有边尝试失败，返回 None。
       """
    ColorText.print("|||||||||||||||| ring_expansion ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    if mol is None:
        return None

    # 正确解包：得到“被选中的那个环”的原子索引列表
    chosen_ring, _ = _pick_ring_to_fuse(mol, ring_smarts=ring_smarts, mode="expansion")
    if not chosen_ring:
        return None

    # 尝试该环的每一条相邻边
    n = len(chosen_ring)
    for i in range(n):
        a1 = int(chosen_ring[i])
        a2 = int(chosen_ring[(i + 1) % n])

        # 每次尝试都从原分子重新拷贝，避免上一次失败的编辑残留
        rw = Chem.RWMol(mol)

        # 允许芳香键或单键：只要这两个原子之间有键即可
        bond = rw.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue

        try:
            # 1) 在边 a1–a2 上插入一个碳
            new_idx = rw.AddAtom(Chem.Atom("C"))
            # 删除原边（可能为单键或芳香键）
            rw.RemoveBond(a1, a2)
            # 新增两条单键 a1–C 和 C–a2
            rw.AddBond(a1, new_idx, Chem.BondType.SINGLE)
            rw.AddBond(new_idx, a2, Chem.BondType.SINGLE)

            # 2) 构分子并校验；若不合法会抛异常
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol  # 第一处成功即返回
        except Exception:
            # 该边插入失败（价态/芳香性/张力等），尝试下一条边
            continue

    # 所有边都失败
    return None


#  对目标环执行环收缩操作：删除一个碳原子并重连两端邻居。
def ring_contraction(mol, ring_smarts=None):
    """
    在选定环上执行“缩环”：删除环中一个原子，并将该原子的环外取代基转接到相邻环原子，
    同时把两个环邻居直接成键，使环长 -1（6→5，7→6 …）。
    若所有候选位点均失败，返回 None。
    """
    ColorText.print("|||||||||||||||| ring_contraction ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    if mol is None:
        return None

    # 尝试去芳香化（可选，失败不致命），减少芳香性导致的 kekulize 失败
    m0 = Chem.Mol(mol)
    try:
        Chem.Kekulize(m0, clearAromaticFlags=True)
    except Exception:
        m0 = Chem.Mol(mol)  # 保留原始形式继续

    # 选环：优先 ≥6 元环；若 ring_smarts 命中则用命中环
    chosen_ring, _ = _pick_ring_to_fuse(m0, ring_smarts, mode="contraction")
    print(chosen_ring)
    if not chosen_ring or len(chosen_ring) < 5:
        return None

    ring_set = set(chosen_ring)

    # 遍历环上每一个原子，尝试把它作为“被删原子”
    for rm in chosen_ring:
        rw = Chem.RWMol(m0)
        try:
            atom = rw.GetAtomWithIdx(rm)
        except Exception:
            continue

        # 环邻居（应恰好 2 个）与环外邻居
        nbs_all = [n.GetIdx() for n in atom.GetNeighbors()]
        nbs_ring = [k for k in nbs_all if k in ring_set]
        nbs_ext = [k for k in nbs_all if k not in ring_set]

        if len(nbs_ring) != 2:
            continue
        n1, n2 = nbs_ring[0], nbs_ring[1]

        try:
            # 1) 将 rm 的所有环外邻居转接到 n1，保持原键型
            for j in nbs_ext:
                b = rw.GetBondBetweenAtoms(rm, j)
                bt = b.GetBondType() if b is not None else Chem.BondType.SINGLE
                if rw.GetBondBetweenAtoms(n1, j) is None:
                    rw.AddBond(n1, j, bt)

            # 2) 两个环邻居直接成键（若还未相连）
            if rw.GetBondBetweenAtoms(n1, n2) is None:
                rw.AddBond(n1, n2, Chem.BondType.SINGLE)

            # 3) 删除 rm（索引整体左移，但后续不再用到 rm）
            rw.RemoveAtom(rm)

            # 4) 构分子并校验；若失败则换下一个 rm
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            # 此位点缩环失败，尝试下一个原子
            continue

    # 所有位点都失败
    return None


# --- 4 链长调节 ---   （这个先不用加llm了，基本不错）
def chain_length_edit(mol, mode, delta, target="backbone"):
    """
    - mode: "extend" / "shrink"
    - delta: +CH2 / +CC / -CH2 / +C*3  （加几个碳）
    - target: backbone / sidechain
    """
    ColorText.print("|||||||||||||||| chain_length_edit ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    # --- 解析 delta ---
    # e.g. "+C*3" → sign="+" , unit="C", count=3
    rw = Chem.RWMol(mol)
    backbone_atoms = find_backbone_atoms(mol)

    m = re.match(r"^([+-])([A-Za-z0-9@+\-\[\]]+)(?:\*([0-9]+))?$", delta.strip())
    if not m:
        return mol
    sign, unit, count = m.groups()
    count = int(count) if count else 1
    delta_smiles = unit * count  # 简单拼接：C*3 → CCC, CH2*2 → CH2CH2

    # anchor 选择
    if target == "backbone":
        anchor_idx = next((i for i in backbone_atoms if rw.GetAtomWithIdx(i).GetSymbol() == "C"), None)
    else:
        anchor_idx = next(
            (a.GetIdx() for a in rw.GetAtoms()
             if a.GetSymbol() == "C" and a.GetIdx() not in backbone_atoms), None)

    if anchor_idx is None:
        return mol

    if mode == "extend" and sign == "+":
        frag = Chem.MolFromSmiles(delta_smiles)
        if frag is None:
            return mol
        combo = Chem.CombineMols(rw, frag)
        rw2 = Chem.RWMol(combo)
        rw2.AddBond(anchor_idx, rw.GetNumAtoms(), rdchem.BondType.SINGLE)
        Chem.SanitizeMol(rw2)
        return rw2.GetMol()

    elif mode == "shrink" and sign == "-":
        n_remove = count
        if target == "backbone":
            path = sorted(backbone_atoms)
            for idx in reversed(path):
                if n_remove <= 0:
                    break
                atom = rw.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "C" and atom.GetDegree() == 1:
                    rw.RemoveAtom(idx)
                    n_remove -= 1
        else:
            for atom in list(rw.GetAtoms()):
                if n_remove <= 0:
                    break
                if atom.GetSymbol() == "C" and atom.GetDegree() == 1 and atom.GetIdx() not in backbone_atoms:
                    rw.RemoveAtom(atom.GetIdx())
                    n_remove -= 1
        Chem.SanitizeMol(rw)
        return rw.GetMol()

    return mol


# --- 5 构象调控 ---  用LLM做
def build_conformation_prompt(smi, edit_type):
    task, example = "Perform conformational regulation.", None
    task = "Perform conformational regulation."

    if "target=sidechain" in edit_type:
        if "branch→linear" in edit_type:  # conformation: (target=sidechain) → (state=branch→linear)
            task = "Modify the molecule so that the side chain changes from branched to linear."
            example = ("CC(C)C", "CCCC")  # isobutane -> n-butane
        elif "linear→branch" in edit_type:
            task = "Modify the molecule so that the side chain changes from linear to branched."
            example = ("CCCC", "CC(C)C")

    elif "target=doublebond" in edit_type:
        if "cis→trans" in edit_type:  # conformation: (target=doublebond) → (state=cis→trans)
            task = "Change the double bond stereochemistry from cis (Z) to trans (E)."
            example = ("C/C=C\\C", "C/C=C/C")
        elif "trans→cis" in edit_type:
            task = "Change the double bond stereochemistry from trans (E) to cis (Z)."
            example = ("C/C=C/C", "C/C=C\\C")

    elif "target=stereocenter" in edit_type:
        if "R→S" in edit_type:  # conformation: (target=stereocenter) → (state=R→S)
            task = "Invert the stereochemistry of the stereocenter from R to S."
            example = ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O")
        elif "S→R" in edit_type:
            task = "Invert the stereochemistry of the stereocenter from S to R."
            example = ("C[C@@H](O)C(=O)O", "C[C@H](O)C(=O)O")

        # 加上示例
    example_text = f"Example input: {example[0]}\nExample output: {example[1]}"

    # 最终提示词
    prompt = (
        f"You are a chemistry molecular editor.\n Task: {task}\n"
        f"{example_text}\n Input: {smi}\n"
        f"Output requirements:\n"
        f" - Only output the optimized molecule, no arrow, no prefix.\n"
        f" - Do not provide explanations.\n"
    )
    return prompt


# --- 3 环操作 & 5 构象调控：调用 LLM ---
def llm_edit(mol, parsed_op, llm, max_trials=3):
    """
    parsed_op: parse_edit_op(...) 的结果 dict，支持 action == "conformation" 或 "special"（ring）
    返回修改后 mol（若失败返回原 mol）
    """

    def parse_smiles_from_text(text):
        """从 LLM 输出中提取候选 SMILES，防止因为内容格式"""
        candidates = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("input:"):
                line = line.split(":", 1)[1].strip()
            if "->" in line:
                parts = [p.strip() for p in line.split("->")]
                line = parts[-1]
            candidates.append(line)

        # 倒序尝试，优先返回合法的
        for cand in reversed(candidates):
            mol = Chem.MolFromSmiles(cand)
            if mol is not None:
                return cand

        # 如果没有合法的，也返回最后一个候选字符串，而不是 None
        return candidates[-1] if candidates else None

    smi0 = Chem.MolToSmiles(mol)  # 输入分子
    # step1 创建 构象调整/环操作 的提示词
    if parsed_op["action"] == "conformation":
        prompt = build_conformation_prompt(smi0, parsed_op["detail"])
    elif parsed_op["action"] == "special":
        prompt = build_ring_prompt(smi0, parsed_op["type"])
    else:
        return mol

    messages = [HumanMessage(content=prompt)]
    # step2 多次尝试获得结果
    for attempt in range(1, max_trials + 1):
        with suppress_everything():
            resp = llm.invoke(messages)
        origin_out = resp.content.strip()

        messages.append(AIMessage(content=origin_out))
        out = parse_smiles_from_text(origin_out)

        # br1 分子本身不合法的fallback
        if not validate_smiles_rdkit(out):
            # ColorText.print('分子不合法', ColorText.RED)
            feedback = (f"Your previous output \"{out}\" is not a valid RDKit SMILES.\n"
                        f"Please try again and ensure output is a single valid SMILES string.\n")
            messages.append(HumanMessage(content=feedback))
            continue

        new_m = Chem.MolFromSmiles(out)  # 如果通过了validate_smiles_rdkit，new_m至少是合法的

        # br2 分子不符合修改要求的fallback
        ok = True
        if parsed_op["action"] == "conformation":
            ok = check_conformation_change(mol, new_m, parsed_op["detail"])
        elif parsed_op["action"] == "special":
            ok = check_ring_change(mol, new_m, parsed_op["type"])
        if ok:
            # ColorText.print('检验通过', ColorText.BLUE)
            return new_m

        # ColorText.print('分子不合规', ColorText.RED)
        if parsed_op["action"] == "conformation":
            messages.append(HumanMessage(content=f"The modification did not satisfy  {parsed_op['detail']}. "
                                                 f" Make sure to modify the GIVEN molecule ({smi0}), not replace it with an unrelated molecule."
                                                 f" Please adjust accordingly."))
        else:
            messages.append(HumanMessage(
                content=f"The modification did not apply {parsed_op['type']}. "
                        f" Make sure to modify the GIVEN molecule ({smi0}), not replace it with an unrelated molecule."
                        f"Please adjust accordingly."))

    # 全部尝试失败，返回原分子
    return mol


# --- 6 基团加法 ---
def add_group(mol, substituent_smiles, position=None, trigger=None, llm=None):
    """
    在一个芳香环上“相对某个锚点”添加基团（+Group@ortho/meta/para）
    """
    ColorText.print("|||||||||||||||| add_group ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    #  step.1 从分子中选择出要操作的 环 和 锚点原子
    ring, anchor = pick_ring_and_anchor2(mol, position, trigger)
    if not ring:    # 没找到环的fallback
        rw = Chem.RWMol(mol)
        frag = Chem.MolFromSmiles(substituent_smiles)
        # if frag is None:
        #     return add_group_with_llm(mol, mol, substituent_smiles, position, trigger, llm)
        out = attach_fragment_on_ring(rw, 0, frag)     # 在默认位置（原子 0）尝试直接把片段接到分子上
        return out

    # step.2 获得pos在分子中的实际位点
    steps,  pos = None, (position or "").lower().strip()
    if pos in ("ipso", "o", "ortho", "m", "meta", "p", "para"):   # 仅限苯环
        steps = get_atdist_to_anchor(pos, len(ring))  # 根据环大小计算相对锚点的步数
    elif pos in GREEK_POS_MAP:                                    # 其他环
        steps = get_greek_dist(pos)                   # 根据环大小计算相对锚点的步数
    if steps is None:      # step=None说明输入格式就错了，不走llm直接返回
        return mol

    target = get_atatomidx_on_ring(ring, anchor, steps)   # 需要编辑的目标位点索引
    if target is None:    # 正常情况下不会说位点都不存在，不走llm直接返回
        return mol

    # step.3 正式编辑
    rw = Chem.RWMol(mol)
    frag = Chem.MolFromSmiles(substituent_smiles)
    if frag is None:     # 可能要加的基团解析出错了，用llm解析一下试试
        return add_group_with_llm(mol, mol, substituent_smiles, position, trigger, llm)

    combo = Chem.CombineMols(rw, frag)
    rw2 = Chem.RWMol(combo)
    offset = rw.GetNumAtoms()

    frag_attach = None
    # 在取代基中寻找第一个非氢原子，作为连接点
    for a in frag.GetAtoms():
        if a.GetAtomicNum() > 1:  # 第一个非氢原子
            frag_attach = a.GetIdx()
            break
    if frag_attach is None:    # 如果整个取代基只有氢（不合理），直接返回原分子
        return mol
    frag_global = offset + frag_attach    # 将取代基原子索引转换为“全局索引”
    tgt_atom = rw2.GetAtomWithIdx(target) # 融合原子中的实际锚点位置

    # --- 分情况处理连接逻辑 ---
    if tgt_atom.GetIsAromatic():
        # 芳香碳（默认带一个隐式H）：删一个氢，强制加键
        for nb in tgt_atom.GetNeighbors():
            if nb.GetAtomicNum() == 1:  # 找到一个氢删掉
                rw2.RemoveAtom(nb.GetIdx())
                break
        if rw2.GetBondBetweenAtoms(target, frag_global) is None:  # 如果目标原子和取代基之间还没键，添加单键
            rw2.AddBond(target, frag_global, rdchem.BondType.SINGLE)

    else:
        # 非芳香碳：严格检查价态看是否允许再接一个键
        pt = Chem.GetPeriodicTable()
        cur_valence = sum(bond.GetBondTypeAsDouble() for bond in tgt_atom.GetBonds())   # 当前价态 = 所有键的键级之和
        max_valence = pt.GetDefaultValence(tgt_atom.GetAtomicNum())                     # 元素的默认最大价态
        # 如果加一个单键不会超价， 且该键不存在，则添加单键
        if cur_valence + 1 <= max_valence:
            if rw2.GetBondBetweenAtoms(target, frag_global) is None:
                rw2.AddBond(target, frag_global, rdchem.BondType.SINGLE)
        else:  # 超价
            print(f"//////////[add_group] 放弃更改: atom {target} 超价 ({cur_valence + 1}>{max_valence})")
            return add_group_with_llm(mol, mol, substituent_smiles, position, trigger, llm)

    # step.4 检查编辑后的分子
    try:
        Chem.SanitizeMol(rw2)       # 通过检查
        return rw2.GetMol()
    except Exception as e:  #fallback 去芳香性（全改单键）再试一次
        print("//////////[add_group] Sanitize失败:", e)

        for a in rw2.GetAtoms():   # 将所有原子标记为非芳香
            a.SetIsAromatic(False)
        for b in rw2.GetBonds():   # 将所有芳香键改成单键
            if b.GetBondType() == Chem.BondType.AROMATIC:
                b.SetBondType(Chem.BondType.SINGLE)

        try:   # 再次 sanitize检查，但跳过 kekulize检查
            Chem.SanitizeMol(rw2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            print("//////////[add_group] Fallback: 取消芳香性后成功")
            return rw2.GetMol()
        except Exception as e2:
            # 取消芳香性还失败，让LLM试试还能怎么操作
            print("//////////[add_group] fallback 也失败:", e2)
            return add_group_with_llm(mol, mol, substituent_smiles, position, trigger, llm)


# --- 7 基团减法 ---
def remove_group(mol, position=None):
    """
    删除芳香环上某相对位点连到环外的整条侧链。
    - 若要“删当前已有侧链”，请用 position='ipso'
    - 不校验具体基团是什么（只要是连到环外的都删）
    """
    ring, anchor = pick_ring_and_anchor(mol)
    if not ring:
        return mol
    ring_set = set(ring)

    steps = get_atdist_to_anchor(position or "ipso", len(ring))
    if steps is None:
        steps = 2
    host_idx = get_atatomidx_on_ring(ring, anchor, steps)
    if host_idx is None:
        return mol

    return cut_sidechain_at_atatom(mol, host_idx, ring_set)



# ===========================================================
# llm兜底逻辑
# ===========================================================
# --- 添加基团 ---
def add_group_with_llm(result, mol, substituent_smiles, position=None, trigger=None, llm=None):
    ColorText.print("|||||||||||||||| add_group_with_llm ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    try:
        if Chem.MolToSmiles(result) != Chem.MolToSmiles(mol):
            return result
    except Exception:
        print("//////////[add_group_with_llm] result 无法比较 SMILES，走 LLM 模式")

    base_smiles = Chem.MolToSmiles(mol)
    pos = position or "unspecified"
    trigger_text = ", ".join(trigger) if trigger else "none"

    prompt = f"""
You are a chemist. Modify the given molecule by adding a substituent group.

- Base molecule (SMILES): {base_smiles}
- Substituent group (SMILES): {substituent_smiles}
- Target position: {trigger} {pos}
- Prefer adding on ring structures matching any of these SMARTS: {trigger_text}

Requirements:
1. Add the substituent logically on the target ring if present.
2. Ensure the result is chemically valid and syntactically correct.
3. Output a single SMILES string only.
4. Do not output fragment mixtures ("A.B").
5. No explanation, only SMILES.
"""

    try:
        ColorText.print("//////////【add_group】替换无效，调用LLM生成新分子", ColorText.YELLOW)
        with suppress_everything():
            response = llm.invoke([HumanMessage(content=prompt)])
        ColorText.print(f"//////////【add_group】llm answer {response}", ColorText.BLUE)
        new_smiles = extract_smiles_from_text(response.content)
        if not new_smiles:
            return mol
        # 情况1 输出的分子是断开的（即输出里带星号）
        if "*" in new_smiles:
            ColorText.print("//////////[LLM✳️AUTO_ATTACH] 检测到断点结构，自动挂接取代基", ColorText.YELLOW)
            newmol = attach_fragment_auto(new_smiles, substituent_smiles)
            if newmol:
                frags = Chem.GetMolFrags(newmol, asMols=False)
                if len(frags) == 1:
                    print(f"//////////[add_group_with_llm] LLM✳️ attach 后 SMILES 通过单组分检查")
                    return newmol
                else:
                    ColorText.print("//////////[LLM MIXTURE_AFTER_ATTACH] attach 后仍为多组分，丢弃", ColorText.RED)
                return mol  # 返回原分子
        # 情况2 输出了分子碎片(即输出分子里带点)
        if "." in new_smiles:
            ColorText.print("//////////[LLM MIXTURE] 检测到包含 '.' 的多组分 SMILES，丢弃", ColorText.RED)
            return mol

        # 情况3 输出的是一个正常分子
        newmol = Chem.MolFromSmiles(new_smiles)
        if newmol:
            frags = Chem.GetMolFrags(newmol, asMols=False)
            if len(frags) > 1:
                ColorText.print("//////////[LLM MIXTURE] RDKit 解析为多组分分子，丢弃", ColorText.RED)
                return mol
            print(f"//////////[add_group_with_llm] LLM 生成替代 SMILES: {new_smiles}")
            return newmol
        else:
            print("//////////[add_group_with_llm] LLM 生成的 SMILES 无法解析")
            return mol
    except Exception as e:
        print(f"//////////[add_group_with_llm] LLM 备份生成失败: {e}")
        return mol


# --- 2(llm) 功能团转化 ---
def replace_fg_with_llm(replaced, mol, src_smarts, new_frag_smiles, llm):
    ColorText.print("|||||||||||||||| replace_fg_with_llm ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    # 若替换后分子与输入分子完全一致，则调用 LLM 保底生成
    if Chem.MolToSmiles(replaced) == Chem.MolToSmiles(mol):
        prompt = (
            "You are a molecular design assistant.\n"
            "Task: Replace a functional group in the given molecule.\n"
            "Requirements:\n"
            "- Output **only one** valid SMILES string.\n"
            "- Do not include any explanations or comments.\n"
            "- The molecule must be a single compound, not a mixture like 'frag1.frag2'.\n"
            "- Ensure correct valence and valid chemical structure.\n"

            f"- Input molecule: {Chem.MolToSmiles(mol)}\n"
            f"- Replace functional group matching pattern: {src_smarts}\n"
            f"- New fragment: {new_frag_smiles}\n"
            "Return only the final SMILES."
        )

        ColorText.print("//////////【replace_fg】替换无效，调用LLM生成新分子", ColorText.YELLOW)
        with suppress_everything():
            response = llm.invoke([HumanMessage(content=prompt)])

        ColorText.print(f"//////////【replace_fg】llm answer {response}", ColorText.BLUE)
        smiles = extract_smiles_from_text(response.content)
        print(smiles)

        if smiles:
            # 情况1 允许带 * 的中间结构交给 attach_fragment_auto 处理，即使包含 '.'
            if "*" in smiles:
                ColorText.print("//////////[LLM✳️AUTO_ATTACH] 检测到断点结构，自动挂接新基团", ColorText.YELLOW)
                new_mol = attach_fragment_auto(smiles, new_frag_smiles)
                if new_mol is not None:
                    # 再次确保不是混合物
                    frags = Chem.GetMolFrags(new_mol, asMols=False)  # new_mol有n个., frags就是n+1个元素
                    if len(frags) == 1:
                        return new_mol
                    else:
                        ColorText.print("//////////[LLM MIXTURE_AFTER_ATTACH] attach 后仍为多组分，丢弃", ColorText.RED)
                return mol  # 返回原分子

            # 情况2 普通 SMILES：禁止混合物形式 "frag1.frag2"
            if "." in smiles:
                ColorText.print("//////////[LLM MIXTURE] 检测到包含 '.' 的多组分 SMILES，丢弃", ColorText.RED)
                return mol  # 返回原分子

            # 情况3 直接返回了合法的分子
            try:
                new_mol = Chem.MolFromSmiles(smiles)
                if new_mol is not None:
                    frags = Chem.GetMolFrags(new_mol, asMols=False)  # 再以防万一RDKit 检查是否为多片段
                    if len(frags) > 1:
                        ColorText.print("//////////[LLM MIXTURE] RDKit 解析为多组分分子，丢弃", ColorText.RED)
                    else:
                        ColorText.print("//////////[LLM] 直接返回了合法分子", ColorText.GREEN)
                        return new_mol
            except Exception:
                pass

        return mol  # LLM 结果不可用，回退到原始分子

    return replaced


# (llm)用 new_ring_smiles 替换分子中的 old_ring_smarts 环 (骨架替换融合环版本)。
def replace_fused_ring_with_llm(replaced, mol, old_ring_smarts, new_ring_smiles, llm):
    ColorText.print("|||||||||||||||| replace_fused_ring_with_llm ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    if Chem.MolToSmiles(replaced) == Chem.MolToSmiles(mol):
        prompt = (
            "You are a molecular design assistant.\n"
            "Task: Generate a new molecule by replacing a fused ring system in the given molecule.\n"
            "Requirements:\n"
            "- Output **only one** valid SMILES string.\n"
            "- Do not include any text explanation.\n"
            "- The molecule must be a single compound, not a mixture like 'fragment1.fragment2'.\n"
            "- Preserve chemical validity and proper valence.\n"
            f"- Input molecule: {Chem.MolToSmiles(mol)}\n"
            f"- Target replacement: replace fused ring {old_ring_smarts} with {new_ring_smiles}.\n"
            "Return only the final SMILES."
        )
        ColorText.print("//////////【replace_fused_ring】替换无效，调用LLM生成新分子", ColorText.YELLOW)
        with suppress_everything():
            response = llm.invoke([HumanMessage(content=prompt)])
        ColorText.print(f"//////////【replace_fused_ring】llm answer {response}", ColorText.BLUE)
        smiles = extract_smiles_from_text(response.content)
        if smiles:
            # 情况1 输出断开的分子
            if "*" in smiles:
                ColorText.print("//////////[LLM✳️AUTO_ATTACH] 检测到断点结构，自动挂接新环", ColorText.YELLOW)
                newmol = attach_fragment_auto(smiles, new_ring_smiles)
                if newmol is not None:
                    frags = Chem.GetMolFrags(newmol, asMols=False)
                    if len(frags) == 1:
                        return newmol
                    else:
                        ColorText.print("//////////[LLM❌MIXTURE_AFTER_ATTACH] attach 后仍为多组分，丢弃", ColorText.RED)
                return mol

            # 情况2 输出分子片段
            if "." in smiles:
                ColorText.print("//////////[LLM❌MIXTURE] 检测到包含 '.' 的多组分 SMILES，丢弃", ColorText.RED)
                return mol

            # 情况3 输出正常的分子
            try:
                new_mol = Chem.MolFromSmiles(smiles)
                if new_mol is not None:
                    frags = Chem.GetMolFrags(new_mol, asMols=False)
                    if len(frags) > 1:
                        ColorText.print("//////////[LLM❌MIXTURE] RDKit 解析为多组分分子，丢弃", ColorText.RED)
                        return mol
                    return new_mol
            except Exception:
                pass
        return mol
    return replaced


# (llm)用 new_ring_smiles 替换分子中的 old_ring_smarts 环 (骨架替换单环版本)。
def replace_single_ring_with_llm(result_mol, mol, old_ring_smarts, new_ring_smiles, llm):
    ColorText.print("|||||||||||||||| replace_single_ring_with_llm ||||||||||||||||", ColorText.PURPLE, ColorText.REVERSE)
    # 判断替换前后分子是否相同
    if Chem.MolToSmiles(result_mol) == Chem.MolToSmiles(mol):
        # 相同的话用llm再试试
        prompt = (
            "You are a molecular design assistant.\n"
            "Task: Generate a new molecule by replacing a ring in the given molecule.\n"
            "Requirements:\n"
            "- Output **only one** valid SMILES string.\n"
            "- Do not include any text explanation.\n"
            "- The molecule must be a single compound, not a mixture like 'fragment1.fragment2'.\n"
            "- Preserve chemical validity and proper valence.\n"
            f"- Input molecule: {Chem.MolToSmiles(mol)}\n"
            f"- Target replacement: replace ring pattern {old_ring_smarts} with {new_ring_smiles}.\n"
            "Return only the final SMILES."
        )
        ColorText.print("//////////【replace_single_ring】替换无效，调用LLM生成新分子", ColorText.YELLOW)
        with suppress_everything():
            gen_resp = llm.invoke([HumanMessage(content=prompt)])
        ColorText.print(f"//////////【replace_single_ring】llm answer {gen_resp}", ColorText.BLUE)
        try:
            new_smi = extract_smiles_from_text(gen_resp.content)
            if new_smi:
                # 情况1 输出是断开的分子
                if "*" in new_smi:
                    ColorText.print("//////////[LLM✳️AUTO_ATTACH] 检测到断点结构，自动挂接新环", ColorText.YELLOW)
                    newmol = attach_fragment_auto(new_smi, new_ring_smiles)
                    if newmol is not None:
                        frags = Chem.GetMolFrags(newmol, asMols=False)
                        if len(frags) == 1:
                            return newmol
                        else:
                            ColorText.print("//////////[LLM MIXTURE_AFTER_ATTACH] attach 后仍为多组分，丢弃", ColorText.RED)
                        return mol

                # 情况2 输出是分子片段
                if "." in new_smi:
                    ColorText.print("//////////[LLM MIXTURE] 检测到包含 '.' 的多组分 SMILES，丢弃", ColorText.RED)
                    return mol

                # 情况3 输出正常的分子
                new_mol = Chem.MolFromSmiles(new_smi)
                if new_mol is not None:
                    frags = Chem.GetMolFrags(new_mol, asMols=False)
                    if len(frags) > 1:
                        ColorText.print("//////////[LLM MIXTURE] RDKit 解析为多组分分子，丢弃", ColorText.RED)
                        return mol
                    result_mol = new_mol
        except Exception:
            ColorText.print("//////////LLM 生成分子解析失败", ColorText.RED)
    # 不一样的话直接返回行了
    return result_mol


