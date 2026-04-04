# ===== Step 0 helpers: 规则库 =====
import copy
import json
import os
import pathlib
import random
import re

import numpy as np
from typing import List, Dict, Any

from rdkit import Chem
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.convenient_utils.suppress_useless_print import suppress_everything
from utils.convenient_utils.wordart import ColorText
from utils.process_drug_utils.parse_chemform_tool import SubstituentResolver


# 检查和创建规则文件保存路径
def ensure_rules_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# 会检查文件 rules/logp.json 是否存在，如果存在就 load 出来。
def load_rulebank(rules_dir: str, prop: str) -> Dict[str, Any]:
    fp = os.path.join(rules_dir, f"{prop}.json")
    print(f"rule_dir:{fp}")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as rf:
            return json.load(rf)
    return {}

# =============================================================
# 先输出edit_op和smarts_trigger和direction字段（本函数只构建提示词）
def build_rule_prompt(prop, batch_size, existing_ops=None, target_direction=None):
    """
    target_direction=None意思是两个方向都生成，否则只生成一个方向
    """
    avoid_part = ""  # 设默认值，防止首轮报错
    if existing_ops:
        # ======================================================================
        #            改成基于聚类的逻辑了，防止在生成100条规则后频繁重复的问题
        # # 传已有规则，提醒模型不要重复  截断到前100条避免 prompt 爆炸（太多会浪费token）
        # existing_json = json.dumps(existing_ops[:100], ensure_ascii=False)
        n_clusters = len(existing_ops) // 10
        if isinstance(existing_ops[0], dict):   # 从 existing_ops 提取 edit_op 文本（去重）
            ops_texts = [x.get("edit_op", "") for x in existing_ops if x.get("edit_op")]
        else:
            ops_texts = list(set(existing_ops))  # 若已是字符串列表

        ops_texts = list(set(ops_texts))  # 再去重，防止聚类时重复样本

        # 若数量少于 n_clusters，则直接使用全部
        if len(ops_texts) <= n_clusters:
            selected_ops = ops_texts
        else:
            # 计算文本 embedding
            model = SentenceTransformer("/home/aita8180/data/mntdata/ziansong/p1/pretrain_models/all-MiniLM-L6-v2")
            embeddings = model.encode(ops_texts, convert_to_numpy=True)

            # 聚类
            k = min(n_clusters, len(ops_texts))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # 从每个簇中挑距离中心最近的 1 条作为代表
            selected_ops = []
            for cluster_id in range(k):
                cluster_idx = np.where(labels == cluster_id)[0]
                cluster_embeds = embeddings[cluster_idx]
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_embeds - center, axis=1)
                closest_idx = cluster_idx[np.argmin(distances)]
                selected_ops.append(ops_texts[closest_idx])

        existing_json = json.dumps(selected_ops, ensure_ascii=False, indent=2)
        # avoid_part = f"""Already generated edit_ops (do NOT repeat any of these):
        #   {existing_json}
        #   """
        avoid_part = f"""The following edit_ops are representative examples of previously generated categories. 
        Avoid generating any operations that are semantically similar to these (same chemical transformation pattern, same substitution type or same structural motif).
        Representative edit_ops (avoid similar ones):
        {existing_json}
        """
    # direction 约束段
    if target_direction is None:
        direction_guidance = f"""
        For each operation, you must also decide the "direction" field for property "{prop}":
          - The direction can only be "increase" or "decrease".
              - "increase": this edit_op tends to increase {prop} on average
              - "decrease": this edit_op tends to decrease {prop} on average
          - Output as much as possible in an equal amount of "increase" and "decrease"
        """
    else:
        direction_guidance = f"""
        For EVERY operation, set the "direction" field to exactly "{target_direction}" 
        for property "{prop}". Only choose a different direction if it is 
        chemically impossible for this edit_op to have that trend.
        """
    return f"""
    Generate {batch_size} diverse molecular editing operations (edit_op) for property "{prop}".
    Output a JSON array of objects. Each object must contain:
      - "edit_op": the operation string, e.g. "+OMe@para" or "replace: phenyl → pyridine" or "transform: aldehyde → acid"
      - "smarts_trigger": up to 3 SMARTS (RDKit-valid) for the EXACT scaffold/local motif BEING MODIFIED.
      - "direction": "increase" or "decrease" indicating how this edit_op tends to change property "{prop}".
    
    Allowed edit_op formats ONLY:
      1) +Substituent@pos
      2) replace: X → Y          (X, Y are named rings/scaffolds, e.g., phenyl, indole, furan)
      3) transform: W → N        (W, N are functional groups, e.g., NH2, COOH, CH3)
      4) ring_fusion: RingName      (RingName = concrete ring by SMILES or standard ring name)
      5) ring_expansion                 (NO payload after the colon; must be exactly "ring_expansion")
      6) ring_contraction               (NO payload after the colon; must be exactly "ring_contraction")
      7) chain_length: (target=sidechain/backbone) ±nC    (smarts_trigger must be [])

    Position rules:
      - If pos ∈ ortho/meta/para → smarts_trigger must be aromatic ring SMARTS (e.g., "c1ccccc1").
      - If pos ∈ alpha/beta/gamma... → smarts_trigger must be NON-aromatic ring SMARTS (e.g., "C1CCCCC1").

    Trigger–operation consistency rules (HARD):
      - For "+Substituent@pos": smarts_trigger must be the ring where substitution occurs. The substituent itself MUST NOT appear inside smarts_trigger.
      - For "replace: X → Y": smarts_trigger must be X (the part to be replaced), not Y.  X and Y MUST use ring/scaffold names.
      - For "transform: W → N": smarts_trigger must be W (the group being transformed). W and N MUST be given as formulas.
      - For "ring_fusion: RingName": smarts_trigger MUST be SMARTS of the ring to be operated (1–3 items allowed), RingName is the ring name or SMILES that needs to be fusion with smarts_trigger.
      - For ring_expansion / ring_contraction: smarts_trigger MUST be SMARTS of the ring to be operated (1–3 items allowed).
      - For chain_length: smarts_trigger must be []. 
    
    {direction_guidance}
    
    Chemistry validity constraints:
      - DO NOT use overly general names such as “amine”, “acid”, “hydrocarbon”, “alkyl”, and etc.
      - Every substituent or scaffold must correspond to a concrete, resolvable structure (e.g., “methylamine” or “ethylamine”, not “amine”).
      - Do NOT output SMARTS of single atoms ("C","N","O",...). Use chemically meaningful fragments or rings.
      - Do NOT use meta/ortho/para on non-aromatic rings, and do NOT use alpha/beta on aromatic rings.
    
    Example output:
    [
      {{"edit_op":"+Cl@ortho","smarts_trigger":["c1ccccc1"],"direction":"increase"}},
      {{"edit_op":"+CF3@alpha","smarts_trigger":["C1CCCCC1"],"direction":"increase"}},
      {{"edit_op":"replace: phenyl → pyridine","smarts_trigger":["c1ccccc1"],"direction":"increase"}},
      {{"edit_op":"chain_length: (target=backbone) +C*2","smarts_trigger":[],"direction":"decrease"}},
      {{"edit_op":"ring_fusion: c1ccccc1","smarts_trigger":["C1CCCCC1"],"direction":"decrease"}},
      {{"edit_op":"ring_expansion","smarts_trigger":["C1CCCCC1"],"direction":"decrease"}} 
    ]
    These examples are merely for formatting reference only. The content does not necessarily conform to chemical knowledge. You need to generate rules that are in line with chemical preconceptions.
    
    Incorrect examples (never output forms like these):
       {{"edit_op":"replace: C → N","smarts_trigger":["C"],"direction":"increase"}} Reason: It is not allowed to generate a single molecule as the replacement structure
       {{"edit_op":"+F@meta","smarts_trigger":["C1CCCCC1"],"direction":"increase"}} Reason: For non-benzene rings, "meta" is used; only benzene rings allow the use of ortho/meta/para".
       {{"edit_op":"transform: ketone → alcohol","smarts_trigger":["C(=O)=O"],"direction":"decrease"}} Reason: The structure being replaced in smarts_trigger and editop is inconsistent.
       {{"edit_op":"transform: amine → amide","smarts_trigger":["N"],"direction":"decrease"}} Reason: “amine” is too general
       {{"edit_op":"ring_fusion: c1ccccc1","smarts_trigger":["N"],"direction":"increase"}} — trigger must represent a full ring, not an atom.
   Each smarts_trigger must correspond exactly to the structure being modified,
   and must be chemically meaningful (no single atoms, no mismatched rings).
    
    {avoid_part}
    Return ONLY a JSON array.
    """.strip()


# 询问llm后，仅解析 edit_op 列表，而不是完整JSON（对应上面的ask_llm_for_rules实现函数）
def ask_llm_for_editops(llm, prompt):
    with suppress_everything():
        resp = llm.invoke([SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
                           HumanMessage(content=prompt)])
    txt = resp.content.strip()
    ColorText.print(f"用户提示(edit生成)\n{prompt}", ColorText.YELLOW)
    ColorText.print(f"模型回答(edit生成)\n{txt}", ColorText.BLUE)

    def remove_json_markers(text):
        if text.startswith('```json'):
            text = text[len('```json'):].strip()
        if text.endswith('```'):
            text = text[:-len('```')].strip()
        return text

    def fix_trailing_commas(json_str):  # edit列表最后一个元素带逗号时jsonload解析不了，这里把最后一个逗号去掉
        # 移除数组中多余的逗号（如 [1, 2,] → [1, 2]）
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json_str

    fixed_txt = fix_trailing_commas(txt.strip())
    data = json.loads(remove_json_markers(fixed_txt))

    # 合法性检查
    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "edit_op" in item:
                out.append({
                    "edit_op": item["edit_op"],
                    "smarts_trigger": item.get("smarts_trigger", []) or [],
                    "direction": item.get("direction")
                })
    return out


# ===========================
# 补全相关（edit_op以外的字段）
# ===========================
# 批量补全
def batch_complete_rules(llm, edit_items, prop) -> List[Dict[str, Any]]:
    """
    给定一批 edit_ops+trigger，一次性让模型补全为完整规则。
    """
    prompt22222222 = f"""
    Given the following list of molecular editing operations (edit_op),
    complete them into full rule entries with fields:
    - edit_op: keep the same string
    - smarts_trigger: keep the same list
    - "direction": "increase" or "decrease" for property "{prop}"
    - "rationale": give a *mechanistic explanation* (1–2 sentences) based on medicinal chemistry knowledge.
       Requirements:
          * Mention the EXACT group/ring named in edit_op (e.g., NO2, OMe, phenyl→pyridine, aldehyde→acid).
          * Include at least ONE concrete factor: electronic (donation/withdrawal/induction/resonance),
            steric (size, bulk), polarity/TPSA, H-bonding (HBD/HBA), basicity/acidity (pKa),
            lipophilicity (AlogP), conformation/rigidity, halogen-bonding.
          * Avoid vague claims like "improves lipophilicity" alone; tie the claim to the specific change.
          * If halogenation/alkylation/ring edit, specify HOW that change shifts {prop} through these factors.
          * The functional group mentioned MUST be detectable on the product of the edit_op.
    - "confidence": integer 1–5. It indicates the frequency of the current edit_op application in real scenarios. The larger the number, the more commonly it is used.

    Validation you MUST satisfy in the text(rationale):
        - The rationale must explicitly name the same group(s) as in edit_op.
        - Do NOT claim π–π enhancement unless an added aryl ring is present; for halogens refer to halogen bonding or polarizability instead.
        - Keep to 1–2 sentences. No hedging words unless marking "context-dependent".

    Input items:
    {edit_items}

    Output JSON array only.
    """.strip()

    prompt = f"""
      Given the following list of molecular editing operations (edit_op),
      complete them into full rule entries with fields:

      Your task is to complete each item into a full rule entry with fields:
        - "edit_op": keep exactly the same string as input
        - "smarts_trigger": keep exactly the same list as input
        - "direction": keep exactly the same string as input
        - "rationale": give a mechanistic explanation (1–2 sentences) consistent with BOTH:
             Requirements:
              * Mention the EXACT group/ring named in edit_op (e.g., NO2, OMe, phenyl→pyridine, aldehyde→acid).
              * Include at least ONE concrete factor: electronic (donation/withdrawal/induction/resonance),
                steric (size, bulk), polarity/TPSA, H-bonding (HBD/HBA), basicity/acidity (pKa),
                lipophilicity (AlogP), conformation/rigidity, halogen-bonding.
              * Avoid vague claims like "improves lipophilicity" alone; tie the claim to the specific change.
              * If halogenation/alkylation/ring edit, specify HOW that change shifts {prop} through these factors.
              * The functional group mentioned MUST be detectable on the product of the edit_op.
              * The final effect on "{prop}" must match the given "direction".
        - "confidence": integer 1–5 indicating how frequently this edit_op is used in real medicinal chemistry.

     Validation you MUST satisfy in the text(rationale):
        - The rationale must explicitly name the same group(s) as in edit_op.
        - Do NOT claim π–π enhancement unless an added aryl ring is present; for halogens refer to halogen bonding or polarizability instead.
        - Keep to 1–2 sentences. No hedging words unless marking "context-dependent".

      If the given "direction" is chemically dubious, still KEEP the original "direction" field unchanged,
      but in the rationale you may briefly note that the effect can be context-dependent.

      Input items:
      {edit_items}

      Output a JSON array of objects with fields:
        "edit_op", "smarts_trigger", "direction", "rationale", "confidence".
      """.strip()

    with suppress_everything():
        resp = llm.invoke([
            SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
            HumanMessage(content=prompt)
        ])
    txt = resp.content.strip()
    ColorText.print(f"用户提示(一次性补全所有字段)\n{prompt}", ColorText.YELLOW)
    ColorText.print(f"模型回答(一次性补全所有字段)\n{txt}", ColorText.BLUE)

    def remove_json_markers(text):
        if text.startswith('```json'):
            text = text[len('```json'):].strip()
        if text.endswith('```'):
            text = text[:-len('```')].strip()
        return text

    data = json.loads(remove_json_markers(txt))
    if isinstance(data, list):
        return data
    return []


# ===========================
# 检查与修复相关
# ===========================
# 对生成的edit_op，smarts_trigger进行一个简单筛选，（即第一阶段的筛选）
# 并对replace、transformer类模型做X/W和smart_trigger的检查和修复
def quick_filter_editops(items, llm2):
    """
       对 LLM 生成的 edit_op + smarts_trigger 对进行快速筛查。
       过滤掉：
         1. smarts_trigger 过小（单原子、单键）；
         2. o/m/p 仅允许苯环；希腊字母位点禁止苯环；
         3. replace/transform：触发必须“覆盖”左半边（trigger ⊇ left）。
       """
    GREEK_POS_SET = {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"
    }

    def _is_benzene(smarts: str) -> bool:
        _BENZENE = Chem.MolFromSmiles("c1ccccc1")
        s_mol = Chem.MolFromSmarts(smarts)
        if not s_mol:
            return False
        # 等价判定：互为子图（避免书写差异）
        return _BENZENE.HasSubstructMatch(s_mol) and s_mol.HasSubstructMatch(_BENZENE)

    substituent_resolver, filtered = SubstituentResolver(), []
    for it in items:
        op, trig_raw = it["edit_op"], it.get("smarts_trigger", [])
        # —— 规范化触发：字符串→列表 ——
        if isinstance(trig_raw, str):
            triggers = [trig_raw.strip()] if trig_raw.strip() else []
        else:
            triggers = [s.strip() for s in trig_raw if s and s.strip()]

        # 1. 过滤单原子的 smarts_trigger
        if any(re.match(r"^[A-Z][a-z]?$", s) for s in triggers):
            continue
        # 2. 芳环与位置词对应检查
        if "@" in op:
            pos = op.split("@")[-1]
            if pos in ["ortho", "meta", "para"]:
                # 至少一个触发必须是苯环，且所有触发都必须是苯环
                if not triggers or not all(_is_benzene(s) for s in triggers):
                    continue
            elif pos in GREEK_POS_SET:
                # 只要求“不是苯环”（非芳香其他环你后续执行时再找锚点）
                if any(_is_benzene(s) for s in triggers):
                    continue
            else:        # 其他未知位置词直接丢弃
                continue
        # 3. replace/transform 自动验证(replace偏骨架，transform偏基团)
        # if op.startswith("replace:") or op.startswith("transform:"):
        #     try:
        #         left, right = op.split(":")[1].split("→")  # left为被替换掉的
        #         left, right = left.strip(), right.strip()  # left为要替换成的
        #     except Exception:
        #         continue  # 无法解析 → 格式
        #     # 调用 SubstituentResolver 解析文字为 SMILES
        #     left_smiles = substituent_resolver.resolve(left, llm2)
        #     right_smiles = substituent_resolver.resolve(right, llm2)
        #     if not left_smiles or not right_smiles:
        #         continue
        #
        #     # 检查 smarts_trigger 是否与左半部分匹配
        #     # 判定逻辑：smarts_trigger 至少一个包含左半部分的主骨架
        #     valid_trigger = False
        #     left_mol = Chem.MolFromSmiles(left_smiles)
        #     if left_mol:
        #         for s in triggers:
        #             s_mol = Chem.MolFromSmarts(s)
        #             if s_mol and left_mol.HasSubstructMatch(s_mol):  # trigger 至少“覆盖”左半边
        #                 break
        #         valid_trigger = True  # 每个trigger都包含左半部分才行
        #     if not valid_trigger:
        #         continue
        if op.startswith("replace:") or op.startswith("transform:"):
            # 把editop里的俩子结构拆出来
            try:
                left, right = op.split(":")[1].split("→")  # left为被替换掉的
                left, right = left.strip(), right.strip()  # left为要替换成的
            except Exception:
                continue

            # 检查 left/right 是否是 smiles
            is_left_smiles = bool(re.search(r"[#=\[\]@]", left))
            is_right_smiles = bool(re.search(r"[#=\[\]@]", right))

            # 若不是SMILES，用substituent_resolver转换为 SMILES
            if is_left_smiles:
                left_smiles = left
            else:
                # print(f"{op} 里的 {left} 需要转化")
                # with suppress_everything():
                left_smiles = substituent_resolver.resolve(left, llm2)

            # 丢弃模糊/无效解析(多出现在模糊描述上)
            def invalid(smiles: str) -> bool:
                # 含多个分子、原子太多、解析失败
                if not smiles or "." in smiles:
                    return True
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    return True
                if mol.GetNumAtoms() > 60:  # 太大可能是拼错或过于笼统
                    return True
                return False

            if invalid(left_smiles):  # 暂时先不考虑右半边
                 continue
            # 检查 trigger 是否与左半部分匹配和修补逻辑
            left_mol = Chem.MolFromSmiles(left_smiles)
            valid_trigger = False
            for t in triggers:
                t_mol = Chem.MolFromSmarts(t)
                # trigger 和 editop 是否一样
                if t_mol and left_mol and (t_mol.HasSubstructMatch(left_mol) or left_mol.HasSubstructMatch(t_mol)):
                    valid_trigger = True
                    break
            # 若 smarts_trigger 不匹配左半结构，则用 left_smiles 替换（不替换了，而是加在后面，防止生成的也错了）
            if not valid_trigger:
                print(f"{trig_raw}修改成{left_smiles}")
                it["smarts_trigger"].append(left_smiles)

        filtered.append(it)
    return filtered


# 发现llm生成的规则库存在direction和rationale字段的描述不一致的问题，保存之前加个检查
def correct_rule_directions(rules, prop, llm):
    """
    用 LLM 检查 direction（increase/decrease）与 rationale 的逻辑一致性。
    - 仅对 replace/transform 类型规则加入 smarts_trigger 辅助判断；
    - 以 id 作为唯一标识；
    - 节省 token，仅传必要字段；
    """
    def remove_json_markers(text: str) -> str:
        """移除```json```或```包裹，返回干净的 JSON 文本或原始文本"""
        if not isinstance(text, str):
            return text
        t = text.strip()
        if t.startswith("```json"):
            t = t[len("```json"):].strip()
        if t.startswith("```"):
            t = t[3:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
        return t

    def extract_json_from_llm_output(text):
        # 尝试直接解析整个文本
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 从后向前扫描寻找有效JSON
        for i in range(len(text) - 1, -1, -1):
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                continue

        # 寻找最长的有效JSON数组
        max_valid_length = 0
        result = []
        for start in range(len(text)):
            for end in range(start + 1, len(text) + 1):
                try:
                    candidate = json.loads(text[start:end])
                    if isinstance(candidate, list):
                        if (end - start) > max_valid_length:
                            max_valid_length = end - start
                            result = candidate
                except:
                    continue
        return result

    corrected, total, batch_size = [], len(rules), 30
    for i in range(0, total, batch_size):
        print(i)
        chunk = rules[i:i + batch_size]

        # ====== 构造输入 ======
        minimal = [
            {
                "id": r["id"],
                "edit_op": r["edit_op"],
                "direction": r.get("direction", ""),
                "rationale": r.get("rationale", ""),
                "smarts_trigger": r.get("smarts_trigger")
            }
            for r in chunk
        ]

        prompt = f"""
    You are a medicinal chemistry consistency checker. 
    For each rule below, check if its `direction` ("increase"/"decrease") matches its `rationale`.
    The `rationale` describes how the edit_op affects property "{prop}".
    If inconsistent and rationale is clear, fix the direction accordingly.
    If inconsistent and rationale is unclear, mark consistent=false.

    Return strictly a JSON array:
    [{{"id": "...", "consistent": true/false, "fixed_direction": "increase"/"decrease"}}...]

    Notes:
    - Do not modify any fields except for "direction".
    - "increase" means the described modification improves the property.
    - "decrease" means the described modification worsens the property.
    - Consider chemistry terms like "enhance", "boost", "reduce", "diminish", "lower", "decrease" in rationale.

    Input rules:
    {json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))}
    """.strip()

        with suppress_everything():
            resp = llm.invoke([
                SystemMessage(content="You are a strict chemistry rule checker. Output JSON only."),
                HumanMessage(content=prompt)
            ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        ColorText.print(f"用户提问(一致性检查){prompt}", ColorText.YELLOW)
        ColorText.print(f"模型回答(一致性检查){prompt}", ColorText.BLUE)
        cleaned = remove_json_markers(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = extract_json_from_llm_output(cleaned)

        # 根据大模型结果选择扔掉还是修改
        if isinstance(parsed, list):
            fix_map = {obj["id"]: obj for obj in parsed if "id" in obj}
            for r in chunk:
                fid = r["id"]          # 原始规则库的id
                f = fix_map.get(fid)   # 检查修改的规则库内容
                if not f:
                    continue
                if not f.get("consistent", True):
                    continue  # rationale太模糊的直接丢弃
                if f.get("fixed_direction") and f["fixed_direction"] != r.get("direction"):
                    r["direction"] = f["fixed_direction"]   # 方向不一致就换一下
                corrected.append(r)
        else:
            # LLM出错则直接保留原规则
            corrected.extend(chunk)

    print(f"[Direction修正] 输入 {len(rules)} 条 → 输出 {len(corrected)} 条")
    return corrected


# （最后调用的）批量补充版本（外加基于替代基库的edit替换，以补充不够total_rules的部分）
def build_large_rulebank_batch(prop, llm1, llm2, total_rules=300, batch_size=50) -> Dict[str, Any]:
    seen, all_rules, all_editops, rule_idx = set(), [], [], 1

    for _ in range(0, total_rules, batch_size):
        # step1：生成 edit_ops + smarts_trigger + direction
        prompt = build_rule_prompt(prop, batch_size, all_editops)   # ①定义提示词
        items = ask_llm_for_editops(llm1, prompt)  # ②[{edit_op, smarts_trigger，direction}, ...]
        print(1)
        print(items)
        items = quick_filter_editops(items, llm2)  # ③对单原子和位置进行简单过滤
        print(2)
        print(items)
        # editop+trigger联合去重   (这个函数里面没用上下文的方式，而是直接把前面生成的规则一起丢给llm让他别生成相同的)
        new_items = []
        for it in items:
            key = (it["edit_op"], tuple(sorted(it["smarts_trigger"])))
            if key not in seen:
                seen.add(key)
                new_items.append(it)
        all_editops.extend(new_items)

        # step2：一次性补全新规则(当前这轮生成的editop和trigger)
        completed = batch_complete_rules(llm2, new_items, prop)

        # step3：最后再补全id
        for r in completed:
            r_final = {
                "id": f"R{rule_idx}",
                "edit_op": r["edit_op"],
                "smarts_trigger": r["smarts_trigger"],
                "direction": r["direction"],
                "rationale": r["rationale"],
                "confidence": r["confidence"]
            }
            all_rules.append(r_final)
            rule_idx += 1
            if len(all_rules) >= total_rules:
                break
        if len(all_rules) >= total_rules:
            break


    # KS  后处理：不足 total_rules 时自动扩增 ======================================================================※
    if len(all_rules) < total_rules:
        ColorText.print(f"当前生成规则数 {len(all_rules)} < 目标 {total_rules}，开始进行替代基扩增...", ColorText.RED)
        # step1 调用扩充操作（只有editop和smarttrigger）
        expanded = expand_rulebank_with_substituents(all_rules, "./data/heuristic_substituent_families.json")

        # step2 扩充完再去重
        # seen_op是(edit_op，(smarts_trigger[0],[1]……))
        seen_ops = {(r["edit_op"], tuple(sorted(r["smarts_trigger"]))) for r in all_rules}
        filtered = []  # 对expanded去重之后的结构
        for r in expanded:
            key = (r["edit_op"], tuple(sorted(r["smarts_trigger"])))  # 和seen_op保持相同结构
            if key not in seen_ops:
                seen_ops.add(key)
                filtered.append(r)

        # step3 随机抽取补足条数
        needed = total_rules - len(all_rules)  # 还需要补充的条数
        # ① 选择要补充的部分
        if len(filtered) > needed:  # 如果上一步筛出来的“补充规则”超了需要的数量，就随机出来需要的条数
            filtered = random.sample(filtered, needed)
        # else直接用所有的

        # ② 用llm2给抽样后的新规则批量补 direction（覆盖旧的 / None）
        filtered = assign_direction_for_new_rules(llm2, filtered, prop)
        filtered = [r for r in filtered if r.get("direction") in ("increase", "decrease")]   # 丢掉没有打上方向的

        # ③ 用 llm2 补全这些新规则字段
        completed_new = []
        batch_size_fill = 20
        for i in range(0, len(filtered), batch_size_fill):  # 防止超上下文，还是分批次生成
            sub_batch = filtered[i:i + batch_size_fill]
            completed_sub = batch_complete_rules(llm2, sub_batch, prop)
            completed_new.extend(completed_sub)
        for r in completed_new:
            r_final = {
                "id": f"R{rule_idx}",
                "edit_op": r["edit_op"],
                "smarts_trigger": r["smarts_trigger"],
                "direction": r["direction"],
                "rationale": r["rationale"],
                "confidence": r["confidence"]
            }
            all_rules.append(r_final)
            rule_idx += 1

        ColorText.print(f"替代基扩增完成，共补充 {len(completed_new)} 条规则，最终规则总数 {len(all_rules)}", ColorText.GREEN)

    # 再过一个direction与retionale的检查
    all_rules = correct_rule_directions(all_rules, prop, llm2)
    ColorText.print(f"方向性检查完成，还剩规则总数 {len(all_rules)}", ColorText.GREEN, ColorText.REVERSE)
    return {"property": prop, "rules": all_rules}


# KS  实际的替代基扩增逻辑
def expand_rulebank_with_substituents(rules, substituent_path):
    """rules是已生成的规则，  sub是替代基库"""
    with open(substituent_path, "r", encoding="utf-8") as f:
        families = json.load(f)   # 读替代基文件

    # 建立 “子结构 → 同族列表” 的反向索引
    sub_map = {}  # families是一个“族”下面多个“子结构”，submap给他改成“子结构”->“同族下的所有其他子结构”
    for fam, members in families.items():
        for m in members:
            sub_map[m] = members

    expanded = []
    for rule in rules:
        op = rule["edit_op"]
        # ------------------ 取代类 (±substituent@pos) ------------------
        if op.startswith("+") or op.startswith("-"):
            sign, rest = op[0], op[1:]  # sign是"+"或者"-"
            if "@" in rest:
                group, pos = rest.split("@", 1)  # 在第一个@处分割，就是 基团+操作
                for key, subs in sub_map.items():
                    if group == key:   # 如果已有操作的”group“在则库里
                        for new_g in subs:
                            if new_g != group:  # 替换出库里当前分类的其他子结构
                                new_rule = copy.deepcopy(rule)
                                new_rule["edit_op"] = f"{sign}{new_g}@{pos}"
                                expanded.append(new_rule)  # 形成editop后加到expanded里面
                        break  # 找到对应族后退出

        # ------------------ 替换(骨架)类 replace: X → Y ------------------
        elif op.startswith("replace:"):
            try:
                src, tgt = [x.strip() for x in op.replace("replace:", "").split("→")]
            except ValueError:
                continue
            for key, subs in sub_map.items():
                if tgt == key:
                    for new_tgt in subs:
                        if new_tgt != tgt:
                            new_rule = copy.deepcopy(rule)
                            new_rule["edit_op"] = f"replace: {src} → {new_tgt}"
                            new_rule["smarts_trigger"] = [src]  # 被替换的结构
                            expanded.append(new_rule)
                    break

        # ------------------ 转化(基团)类 transform: A → B ------------------
        elif op.startswith("transform:"):
            try:
                src, tgt = [x.strip() for x in op.replace("transform:", "").split("→")]
            except ValueError:
                continue
            for key, subs in sub_map.items():
                if tgt == key:
                    for new_tgt in subs:
                        if new_tgt != tgt:
                            new_rule = copy.deepcopy(rule)
                            new_rule["edit_op"] = f"transform: {src} → {new_tgt}"
                            new_rule["smarts_trigger"] = [src]  # 被转换的结构
                            expanded.append(new_rule)
                    break

    return expanded


# 替代基扩充逻辑这里面不含direction信息，这里用LLM补充
def assign_direction_for_new_rules(llm, rules, prop):
    """
    给一批“结构扩增后”的规则重新分配 direction。
    输入的 rules 至少要有:
      - edit_op
      - smarts_trigger
    返回时为每条规则写入新的 "direction" 字段。
    """

    # 只抽必要字段进 prompt
    items_for_llm = [
        {
            "idx": i,
            "edit_op": r["edit_op"],
            "smarts_trigger": r.get("smarts_trigger", []),
        }
        for i, r in enumerate(rules)
    ]

    # 用llm填充direction字段
    prompt = f"""
    For the property "{prop}", you are given a list of molecular edit operations.
    For each item, decide whether the operation tends to INCREASE or DECREASE "{prop}" on average.

    Input format (Python-like list of dicts):
    {items_for_llm}

    For each item, output a JSON array of objects with:
      - "idx": the same index as in the input
      - "direction": either "increase" or "decrease" for property "{prop}"

    Rules:
      - Consider the effect of the structural change described in "edit_op" on "{prop}".
      - If the trend is ambiguous or strongly context-dependent, choose the direction that is more common in typical medicinal chemistry.

    Output ONLY the JSON array.
    """.strip()

    with suppress_everything():
        resp = llm.invoke([
            SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
            HumanMessage(content=prompt),
        ])
    txt = resp.content.strip()
    ColorText.print(f"[方向分配提示]\n{prompt}", ColorText.YELLOW)
    ColorText.print(f"[方向分配回答]\n{txt}", ColorText.BLUE)

    def remove_json_markers(text: str) -> str:
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-len("```")].strip()
        return text

    data = json.loads(remove_json_markers(txt))
    # 建一个 idx -> direction 映射
    idx2dir = {item["idx"]: item["direction"] for item in data if "idx" in item and "direction" in item}

    # 写回原 rules（覆盖旧 direction）
    for i, r in enumerate(rules):
        if i in idx2dir:
            r["direction"] = idx2dir[i]
        else:  # 如果没返回该条，保底策略：可以丢弃或默认用原 direction，这里暂时选保留原有 direction
            r["direction"] = r.get("direction", None)

    return rules

# 保存生成的规则库
def save_rulebank(rules_dir, prop, data) -> str:
    ensure_rules_dir(rules_dir)
    fp = os.path.join(rules_dir, f"{prop}.json")
    with open(fp, "w", encoding="utf-8") as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)
    return fp
