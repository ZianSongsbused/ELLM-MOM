import json
import os
import pickle
import random
import signal
import re

import numpy as np
import pandas as pd

from rdkit.Chem import AllChem, DataStructs, rdFMCS
from rdkit import Chem, DataStructs
from tqdm import tqdm

from utils.convenient_utils.aggre_rules import _aggregate_multiobjective_rules
from utils.convenient_utils.suppress_useless_print import suppress_everything
from utils.convenient_utils.wordart import ColorText
from utils.edit_by_rule_tool import parse_edit_op, apply_edit_op
from utils.llm_tool import complete_chatgpt_lc, complete_deepseek_lc, complete_localllm_lc, complete_localllm_lc_muti
from utils.process_drug_utils.moledit_prompt import get_task_specification_dict, test_prompt_from_api
from utils.process_drug_utils.moledit_evaltool import evaluate
from utils.retrievingDB_tool import task2prop, compute_final_embedding_batch, compute_final_embedding
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
import pandas as pd
import os
import pickle

from utils.process_drug_utils.parse_chemform_tool import SubstituentResolver
from utils.rulebook_tool import load_rulebank

TIMEOUT_SECONDS = 600  # 10分钟

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("计算超时")


# 根据args["conversational_LLM"]选择核心LLM
def choose_llm(llm_type, temperature=0.2, top_p=1.0):
    if llm_type == 'gpt':
        llm = complete_chatgpt_lc(temperature, top_p)
    elif llm_type == 'deepseek':
        llm = complete_deepseek_lc()
    elif llm_type == 'galactica':
        llm = complete_localllm_lc('galactica')
    else:
        llm = complete_localllm_lc('llama2')
    return llm


# 批量做多个任务时，为了只定义一次llm传进来模型参数和分词器
def choose_llm_muti(llm_type, local_llm, local_tokenizer):
    if llm_type == 'gpt':
        llm = complete_chatgpt_lc()
    elif llm_type == 'deepseek':
        llm = complete_deepseek_lc()
    elif llm_type == 'galactica':
        llm = complete_localllm_lc_muti('galactica', local_llm, local_tokenizer)
    else:
        llm = complete_localllm_lc_muti('llama2', local_llm, local_tokenizer)
    return llm

# =========================================================
# 提示词设置部分
# =========================================================
# 调用chatmol时，把prompt截断在512以内
def truncate_prompt(prompt: str, mode: str = "smart", max_len: int = 512) -> str:
    """把chatmol控制在512之内"""
    if len(prompt) <= max_len:
        return prompt  # 上下文再512之内，不用处理
    if mode == "hard":
        return prompt[-max_len:]  # 直接从512的位置截断
    elif mode == "smart":
        sub_prompt = prompt[:max_len]  # 在512处，从后往前找第一个句号，后面的截断
        split_index = sub_prompt.rfind(".")  # 找到最右边的句号
        if split_index != -1:  # 如果找到了句号就截到句号；没找到句号就按hard 模式
            return prompt[:split_index + 1]
        else:
            return prompt[:max_len]


# 定义每次对话，用户的首个提示（调用下面的两个函数）
def init_prompt_in_a_conversation(llm, task_specific_prompt_dict, input_drug, drug_type, task):
    if llm == "galactica":
        prompt = construct_PDDS_prompt_galactica(task_specific_prompt_dict, input_drug, drug_type, task)
    else:
        prompt = construct_PDDS_prompt(task_specific_prompt_dict, input_drug, drug_type, task)
    return prompt


# 构造提示词时，参考SME给出的分子归隐构造“带有方向信息”的提示词
def init_prompt_in_a_conversation_with_direc(llm, task_specific_prompt_dict, input_drug, drug_type, task):
    if llm == "galactica":
        prompt = construct_PDDS_prompt_galactica(task_specific_prompt_dict, input_drug, drug_type, task)
    else:
        prompt = construct_PDDS_prompt_with_direc(task_specific_prompt_dict, input_drug, drug_type, task)
    return prompt


# 构造PDDS(llama/gpt)
def construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, task):
    if drug_type == 'molecule':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)  # 把模板里面的分子占位符换成实际的
        prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points. No explanation is needed."  # 再提示后面再加一个固定的后缀“生成5个分子，不要解释”
    return prompt

# 同样，带方向的pdds
def construct_PDDS_prompt_with_direc(task_specification_dict, input_drug, drug_type, task):
    if drug_type == 'molecule':
        with suppress_everything():
            direction = test_prompt_from_api(input_drug, "brics", task, True)  # 第一轮用来构建提示的，所以是true
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)  # 把模板里面的分子占位符换成实际的
        prompt = prompt + direction +"\nGive me five molecules in SMILES only and list them using bullet points. No explanation is needed."  # 再提示后面再加一个固定的后缀“生成5个分子，不要解释”
    return prompt


# 构造PDDS(galactica)
def construct_PDDS_prompt_galactica(task_specification_dict, input_drug, drug_type, task):
    if drug_type == 'molecule':
        # 思路和construct_PDDS_prompt一样，不一样的只有prompt的格式
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', "[START_I_SMILES]"+input_drug+"[END_I_SMILES]")
        prompt = "Question: " + prompt + "\nAnswer:"
    return prompt


# =========================================================
# 根据规则库手动修改分子、
# =========================================================
def apply_rules_from_cot(mol_smiles, rulebank, llm):
    # Step 1. 获取输入分子
    # mol_smiles = rulebank.get("mol")
    mol = Chem.MolFromSmiles(mol_smiles)

    # Step 2. 遍历规则
    new_selected = []   # cot筛选完的规则
    selected_items = rulebank.get("selected", [])
    if not selected_items:  # selected_items是[]说明是【第二轮】的传，走下面的逻辑提取规则
        strict_val, partial_val = rulebank.get("strict"), rulebank.get("partial")
        processed_item = []

        if strict_val:
            print("检测到严格匹配规则，已加载。")
            processed_item.extend(strict_val)
        if partial_val:
            print("检测到部分匹配规则，已加载。")
            processed_item.extend(partial_val)
        selected_items.extend(processed_item)
    # print(f"[test] selected items:\n {selected_items}")

    # selected_items不[]说明是【第一轮】的传入，直接遍历selected_items
    for item in selected_items:  # 遍历与操作规则
        rid, rule = item.get("id"), item.get("rule", {})
        if rule == {}:  # 第一轮和其他轮的selected_items结构不一样,selected_items里面你直接就是规则
            edit_op, rationale, trigger = item.get("edit_op"), item.get("rationale"), item.get("smarts_trigger")
        else:           # 非第一轮的selected_items里边还套了一层rule
            edit_op, rationale, trigger = rule.get("edit_op"), rule.get("rationale"), rule.get("smarts_trigger")
        try:
            # Step 2.1 解析 edit_op  ***再加个smart
            with suppress_everything():
                parsed_op = parse_edit_op(edit_op, trigger, llm)
                print(f"parsed_op {parsed_op}")
            # Step 2.2 应用规则
            new_mol = apply_edit_op(mol, parsed_op, llm)
            # print(f"new_mol {new_mol}")
            # Step 2.3 转 SMILES
            new_smiles = Chem.MolToSmiles(new_mol) if new_mol else None
            # Step 2.4 保存
            item_with_result = dict()
            item_with_result["new_smiles"] = new_smiles
            item_with_result["success"] = len(new_smiles) > 0
            item_with_result["edit_op"] = edit_op
            item_with_result["smarts_trigger"] = trigger
            item_with_result["rationale"] = rationale    # 这里返回的是规则库里面的理由
            new_selected.append(item_with_result)

        except Exception as e:
            item_with_result = dict()
            item_with_result["new_smiles"] = ""
            item_with_result["success"] = False
            item_with_result["error"] = str(e)
            item_with_result["edit_op"] = edit_op
            item_with_result["smarts_trigger"] = trigger
            item_with_result["rationale"] = rationale
            new_selected.append(item_with_result)

    # Step 3. 返回结果
    return {
        "property": rulebank.get("property"),
        "mol": mol_smiles,
        "selected": new_selected
    }


# =========================================================
# retrieval相关的
# =========================================================
# 找到S0和D的差异子结构（这个方法采用从原分子里删除公共部分的方式）
def extract_diff_substructures(S0, D):
    """
    找到 S0 和 D 的差异子结构。
    返回:
      common (str): 最大公共子结构SMARTS
      diff_S0 (list[str]): S0中特有的片段smiles
      diff_D (list[str]): D中特有的片段smiles
    """
    mcs = rdFMCS.FindMCS([S0, D])   # S0, D的最大公共子结构
    patt = Chem.MolFromSmarts(mcs.smartsString)
    common_atoms_S0 = S0.GetSubstructMatch(patt)    # S0中公共部分索引
    common_atoms_D = D.GetSubstructMatch(patt)      # D中公共部分索引
    common = mcs.smartsString

    def get_diff_fragments(mol, common_atoms):
        diff_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in common_atoms]
        if not diff_atoms:   # mol中不不属于公共部分的索引，这里额外判断一次是为了在没有公共部分时提前返回
            return []
        # 用RWMol删除不相关原子
        rw = Chem.RWMol(mol)
        for idx in sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() in common_atoms], reverse=True):
            rw.RemoveAtom(idx)
        frag = rw.GetMol()
        try:
            Chem.SanitizeMol(frag)  # 删完的分子再标准化
            return [Chem.MolToSmiles(frag)]
        except:
            return []

    diff_S0 = get_diff_fragments(S0, common_atoms_S0)
    diff_D = get_diff_fragments(D, common_atoms_D)

    return common, diff_S0, diff_D


# 根据S0和D的差异筛选规则库
def map_diff_to_rules(S0, D, rules, target_direction):
    # 原分子S0和新分子D的独有结构
    _, diff_S0, diff_D = extract_diff_substructures(S0, D)
    print("diff_S0: ", diff_S0, " diff_D: ",  diff_D)
    matched, partial = [], []

    for rule in rules:
        if target_direction and rule.get("direction") != target_direction:  # 和要求修改方向不一样的先排除
            continue
        edit, smarts_list = rule.get("edit_op", ""),  rule.get("smarts_trigger", []) or []

        smarts_mols = []     # 解析 SMARTS_trigger
        for s in smarts_list:
            try:
                mol = Chem.MolFromSmarts(s)
                if mol:
                    smarts_mols.append(mol)
            except Exception:
                pass

        # KS  1 链长类操作中没有可以匹配的基团故直接跳过
        if "chain_length" in edit:
            continue

        # KS  2 replace / transform类规则，左侧被替换掉，故应该匹配S0，右侧新增，故要匹配D
        if edit.startswith("replace:") or edit.startswith("transform:"):
            try:
                left, right = re.split(r"→|->", edit.split(":", 1)[1])
                left, right = left.strip(), right.strip()
                left_mol, right_mol = Chem.MolFromSmiles(left),  Chem.MolFromSmiles(right)
            except Exception:
                left_mol = right_mol = None

            if left_mol and any(Chem.MolFromSmiles(f).HasSubstructMatch(left_mol) for f in diff_S0):
                if right_mol and any(Chem.MolFromSmiles(f).HasSubstructMatch(right_mol) for f in diff_D):
                    # 左边右边都能匹配上就进[完全匹配的规则库]
                    matched.append(rule)
                else:
                    # 只有左边都能匹配上就进[部分匹配的规则库]，（只有右边能匹配上没有意义，因为S0没有的结构不知道拿什么换）
                    partial.append(rule)
            continue

        # KS  3 +Substituent 类规则描述了要替换成的部分，trigger里不是被替换的具体结构，故只用sub匹配D
        if edit.startswith("+"):
            # 1) 提取加成取代基（去掉位置标记）
            m = re.search(r"\+([A-Za-z0-9\*\[\]\(\)=#\+\-]+)(?:@[\w\d]+)?", edit)
            if not m:
                continue

            sub = m.group(1)
            try:
                sub_mol = Chem.MolFromSmiles(sub)
            except Exception:
                sub_mol = None
            if sub_mol is None:
                continue

            # 2) sub 必须在 diff_D 中出现
            sub_hit = False
            for f in diff_D:
                try:
                    fm = Chem.MolFromSmiles(f)
                except Exception:
                    fm = None
                if fm is None:
                    continue
                if fm.HasSubstructMatch(sub_mol):
                    sub_hit = True
                    break
            if not sub_hit:
                continue

            # 3) smarts_trigger 至少一个能在 S0 上触发（保证可用性）
            smarts_list = rule.get("smarts_trigger", []) or []

            trigger_ok = False
            for s in smarts_list:
                try:
                    patt = Chem.MolFromSmarts(s)
                except Exception:
                    patt = None
                if patt is None:
                    continue
                if S0.HasSubstructMatch(patt):
                    trigger_ok = True
                    break

            if trigger_ok:
                matched.append(rule)

        # KS  ring_* 操作，smarttigger要被换掉，所以用S0匹配trigger
        if edit.startswith("ring_"):
            if any(Chem.MolFromSmiles(f).HasSubstructMatch(patt) for patt in smarts_mols for f in diff_S0):
                matched.append(rule)
            continue

    # KS  用分子独有部分没匹配到时，用整体分子做匹配
    if not matched:
        # for rule in rules:
        #     if target_direction and rule.get("direction") != target_direction:
        #         continue
        #     edit = rule.get("edit_op", "")
        #     smarts_list = rule.get("smarts_trigger", [])
        #     for s in smarts_list:
        #         try:
        #             patt = Chem.MolFromSmarts(s)
        #         except Exception:
        #             patt = None
        #         if not patt:
        #             continue
        #         # 加基团 和 环操作 --> D里有smart_trigger
        #         if D.HasSubstructMatch(patt) and (edit.startswith("+") or edit.startswith("ring")):
        #             partial.append(rule)
        #         # 替换基团 和 替换环 --> S里有smart_trigger
        #         if S0.HasSubstructMatch(patt) and (edit.startswith("replace") or edit.startswith("transform") or edit.startswith("-")):
        #             partial.append(rule)
        for rule in rules:
            if target_direction and rule.get("direction") != target_direction:
                continue

            edit = rule.get("edit_op", "") or ""
            smarts_list = rule.get("smarts_trigger", []) or []

            if "chain_length" in edit:
                continue

            if not smarts_list:
                continue

            # 将 trigger 编译成 patt
            patts = []
            for s in smarts_list:
                try:
                    patt = Chem.MolFromSmarts(s)
                except Exception:
                    patt = None
                if patt:
                    patts.append(patt)
            if not patts:
                continue

            # replace/transform/ring_/- : trigger 应该能在 S0 上触发（可用性）
            if edit.startswith("replace:") or edit.startswith("transform:") or edit.startswith("ring_") or edit.startswith("-"):
                ok = False
                for patt in patts:
                    if S0.HasSubstructMatch(patt):
                        ok = True
                        break
                if ok:
                    partial.append(rule)
                continue

            # +Substituent: 如果 trigger 是位点/骨架提示，也要求能在 S0 上触发
            if edit.startswith("+"):
                ok = False
                for patt in patts:
                    if S0.HasSubstructMatch(patt):
                        ok = True
                        break
                if ok:
                    partial.append(rule)
                continue
        # KS  整个分子还匹配不出来，则优先考虑能用就行
        if not partial:
            for rule in rules:
                if target_direction and rule.get("direction") != target_direction:
                    continue

                edit = rule.get("edit_op", "") or ""
                smarts_list = rule.get("smarts_trigger", []) or []

                if "chain_length" in edit:
                    continue
                if not smarts_list:
                    continue

                usable_on_s0 = False
                for s in smarts_list:
                    try:
                        patt = Chem.MolFromSmarts(s)
                    except Exception:
                        patt = None
                    if patt and S0.HasSubstructMatch(patt):  # 能用就行
                        usable_on_s0 = True
                        break

                if usable_on_s0:
                    partial.append(rule)

    return matched, partial


# S0→D映射出的候选规则丢给——llm叫他排序
def llm_score_rules(S0, D, target_property, target_direction, candidate_rules, llm, topk=5, last=False):
    """
    参数:
      S0, D: rdkit Mol
      target_property: 如 "logp"
      target_direction: "increase" 或 "decrease"
      candidate_rules: list of rule dicts
      llm: 一个封装好的 LLM 接口，要求 .run(prompt) -> str

    返回:
      list of rule dicts，按相关性排序
    """
    print(f"次轮初筛出的规则库数量 {len(candidate_rules)}")
    S0_smiles, D_smiles = Chem.MolToSmiles(S0), Chem.MolToSmiles(D)

    rules_text = "\n".join(
        [f"- ID: {r['id']}: Edit_op: {r['edit_op']}. SMARTS_trigger: {r['smarts_trigger']}Rationale: {r['rationale']}"
         for r in candidate_rules]
    )

    prompt = f"""
Task: Molecular optimization
Input Molecular (S0): {S0_smiles}
Reference molecule (D): {D_smiles}
Optimization objective: {target_direction} the {target_property} of the input molecular

The following are the candidate rules. Please sort them according to their relevance to the optimization from S0 to D and the satisfaction of the target attributes (the most relevant ones come first):
{rules_text}

You must internally do the following reasoning steps, but DO NOT output them:
1) Infer key deltas Δ(S0→D): what structural/features in D should S0 gain/lose to better match the objective? Focus on ring system, (hetero)aromaticity/conjugation, polarity/TPSA, HBD/HBA, steric bulk/rigidity, and lipophilicity cues evident in D vs S0.
2) For each rule, evaluate:
   - Applicability to S0:
     • replace:/transform:/ring_*: smarts_trigger must exist in S0; if absent → strongly downrank.
     • +Substituent@pos: intended substituent should be directionally consistent with Δ(S0→D); if clearly irrelevant → downrank.
   - Directional consistency with the objective ({target_direction} {target_property}) and with Δ(S0→D).
   - Minimality/plausibility: fewer, more local edits that directly move S0 toward Δ(S0→D) outrank broad/indirect edits.
   - Conflict penalty: if a rule likely moves S0 opposite to the objective or opposite to Δ(S0→D) → rank to the bottom.

Ranking rules:
- First filter by Applicability (must match S0 for trigger-based ops).
- Then sort by how directly the edit realizes Δ(S0→D) while satisfying the objective.
- Tie-breakers, in order: (i) higher plausibility/minimality, (ii) better symmetry/positional match, (iii) rationale consistency.

Output:
Return a JSON array of rule IDs in descending relevance only, e.g.:
["R2","R5","R1"]

Constraints:
- Output JSON array only, no comments, no extra fields.

"""
    with suppress_everything():
        response = llm.invoke(prompt)
    #####################################################################
    # ColorText.print(f"用户提示(次轮CoT)\n{prompt}", ColorText.YELLOW)
    # ColorText.print(f"模型回答(次轮CoT)\n{response}", ColorText.BLUE)
    #####################################################################
    try:
        ids_sorted = json.loads(response)
    except Exception:        # fallback：如果 LLM 没返回标准 JSON，做个简单兜底返回全部
        ids_sorted = [r["id"] for r in candidate_rules]

    # 重新排序
    id2rule = {r["id"]: r for r in candidate_rules}  # if是防止llm生成的id没在原来的候选列表的id范围里面
    sorted_rules = [id2rule[rid] for rid in ids_sorted if rid in id2rule]

    if last:
        return sorted_rules[-topk:]
    else:
        return sorted_rules[:topk]


# 第二轮及之后，单属性的规则筛选
def select_rules_for_prompt(S0, D, rules, target_property, target_direction, llm, topk=5, last=False):
    # 基于[分子独有部分]的规则预筛选
    matched, partial = map_diff_to_rules(S0, D, rules, target_direction)
    final_rules = {"strict": [], "partial": []}
    # 对筛选的这些规则进行打分与排序
    if matched:  # 严格匹配（如果严格和部分都有就优先严格匹配）
        print("有严格匹配的规则")
        if last:
            sorted_rules = llm_score_rules(S0, D, target_property, target_direction, matched, llm, topk,True)
        else:
            sorted_rules = llm_score_rules(S0, D, target_property, target_direction, matched, llm, topk)
        final_rules = {"strict": sorted_rules, "partial": []}

    if partial:  # 部分匹配
        if last:
            sorted_rules = llm_score_rules(S0, D, target_property, target_direction, partial, llm, topk,True)
        else:
            sorted_rules = llm_score_rules(S0, D, target_property, target_direction, partial, llm, topk)
        final_rules["partial"] = sorted_rules

    return final_rules


# 第二轮及之后，单属性的规则筛选消融版__随机
def select_rules_for_prompt_random(S0, D, rules, target_property, target_direction, llm, topk=5, last=False, seed=None):
    if seed is not None:
        random.seed(seed)
    # 基于[分子独有部分]的规则预筛选（和原函数一致）
    matched, partial = map_diff_to_rules(S0, D, rules, target_direction)
    final_rules = {"strict": [], "partial": []}

    # 严格匹配：只在 matched 中随机
    if matched:
        if len(matched) <= topk:
            final_rules["strict"] = matched                 # 防止筛选的规则数量还没有topk多
        else:
            final_rules["strict"] = random.sample(matched, topk)   # 随机取

    # 部分匹配：只在 partial 中随机
    if partial:
        if len(partial) <= topk:
            final_rules["partial"] = partial
        else:
            final_rules["partial"] = random.sample(partial, topk)
    ColorText.print("后续轮次随机选规则", ColorText.GREEN)
    # 注意：last 参数仅为兼容保留，在随机版中不改变行为
    return final_rules


# 二轮及之后的多属性（S0 -> XR）规则筛选
def build_multiobjective_rulebank_for_round(
    llm,
    S0_smiles,
    XR_smiles,
    per_prop_rulebanks,
    target_prop_lst,
    target_direc_lst,
    per_prop_topk=5,
    topk_multi=5,
    use_main_aux_score=False,  # [NEW] 是否启用“主属性+辅属性”的 u_main + u_aux 评分
    neutral_score=0.5          # [NEW] 中性分数，这里 rank utility ∈ (0,1]，中点设为 0.5
):
    """
    二轮及以后多属性规则选择 (S0 -> XR)：

    步骤：
    1) 每个属性独立调用 select_rules_for_prompt(S0, XR, ...) 得到排序后的规则列表；
       - 排序语义：在该属性、该方向下，越靠前认为“越有利”；
       - 用名次归一化成 score ∈ (0,1]，score 越大越好。
    2) 在规则层面按 edit_op 做交集/并集；
    3) 对同一 edit_op，在所有属性上做 smarts_trigger 交集（链长类空 trigger 额外放行）；
    4) 使用 _aggregate_multiobjective_rules 进行多属性汇总打分：
       - 默认模式：平均(score) × 覆盖率；
       - use_main_aux_score=True 时：
           main_prop = target_prop_lst[0] 视为主属性；
           u_main = |s_main - neutral|
           u_aux  = - Σ |s_aux - neutral|
           score_multi = u_main + u_aux
         即：主属性越偏离中性越好；辅属性越接近中性越好。
    5) 打包成 apply_rules_from_cot 可用的“伪规则库”。

    返回结构：
      {
        "property": "prop1+prop2+...",
        "mol": S0_smiles,
        "candidate_count": N,
        "selected": [
          {"id": i, "rule": rule_dict, "score": score_multi, "enhanced_rationale": "..."},
          ...
        ]
      }
    """

    S0 = Chem.MolFromSmiles(S0_smiles)
    XR = Chem.MolFromSmiles(XR_smiles) if XR_smiles else None

    def _rule_key(rule_dict):
        # 二轮及之后仍然用 edit_op 作为跨属性对齐 key
        return rule_dict.get("edit_op")

    props = list(target_prop_lst)
    per_prop_selected, per_prop_keys = {}, {}
    index_per_prop = {}   # {prop: {edit_op: {"rule": rule_dict, "score": u, "rationale": str}}}
    # 主属性定义：列表第2个
    main_prop = target_prop_lst[1] if target_prop_lst else None

    # 1) 每个属性单独跑 select_rules_for_prompt，取前 per_prop_topk
    for prop, direc in zip(target_prop_lst, target_direc_lst):
        rb_prop = per_prop_rulebanks[prop]      # {"rules":[...]}
        rules_all = rb_prop["rules"]

        if use_main_aux_score and (prop != main_prop) and (len(target_prop_lst) > 1):
            use_last = True
        else:
            use_last = False

        # 取出各个属性的topk个规则
        sel = select_rules_for_prompt(S0, XR, rules_all, prop, direc, llm, per_prop_topk, last=use_last)

        strict_list = sel.get("strict", []) or []
        partial_list = sel.get("partial", []) or []
        ranked = (strict_list + partial_list)[:per_prop_topk]

        per_prop_selected[prop] = ranked
        per_prop_keys[prop] = {_rule_key(r) for r in ranked}

        # 2) 将名次归一化为 score ∈ (0,1]，构建 index_per_prop
        N = max(1, len(ranked))
        idx_map = {}
        for rank_idx, rule in enumerate(ranked):
            if N == 1:
                u = 1.0
            else:
                # 名次归一化：rank 越靠前，u 越接近 1
                u = (N - rank_idx) / N

            key = _rule_key(rule)
            rec = idx_map.get(key)
            # 同一属性同一 edit_op，只保留 u 更高的那条
            if (rec is None) or (u > rec.get("score", 0.0)):
                idx_map[key] = {
                    "rule": rule,
                    "score": u,                          # 这里的 score = rank utility（越大越好）
                    "rationale": rule.get("rationale", "")
                }
        index_per_prop[prop] = idx_map

    # 3) 调用公共多属性聚合逻辑
    #    注意：这里 use_direction_filter=False，不再做 score>thr 的方向阈值过滤，
    #          因为这一轮的 score 是相对排序 utility，本身已经按目标方向排过序。
    selected = _aggregate_multiobjective_rules(
        props=props,
        per_prop_keys=per_prop_keys,
        index_per_prop=index_per_prop,
        topk_multi=topk_multi,
        use_direction_filter=False,    # 二轮及之后不做阈值过滤
        prop2dir=None,                 # 兼容参数，内部不会再用来翻转符号
        thr=neutral_score,             # 传入但不会被使用（因为 use_direction_filter=False）
        use_main_aux_score=use_main_aux_score,
        neutral_score=neutral_score,
        stage_label="多属性后续轮"
    )

    return {
        "property": "+".join(target_prop_lst),
        "mol": S0_smiles,
        "candidate_count": len(selected),
        "selected": selected
    }

# 二轮及之后的多属性（S0 -> XR）规则筛选消融版__随机
def build_multiobjective_rulebank_for_round_random(
    llm,
    S0_smiles,
    XR_smiles,
    per_prop_rulebanks,
    target_prop_lst,
    target_direc_lst,
    per_prop_topk=5,
    topk_multi=5,
    use_main_aux_score=False,  # 与原函数保持一致，传给 _aggregate_multiobjective_rules
    neutral_score=0.5,         # 随机时给每条规则的中性得分
    seed=None                  # 便于复现
):
    """
    二轮及以后多属性规则选择 (S0 -> XR) 的随机消融版本：

    区别于原版：
      - 每个属性仍然用 map_diff_to_rules(S0, XR, rules, dir) 做结构预筛（通过
        select_rules_for_prompt_random 内部调用）；
      - 不再调用 llm_score_rules，不做 LLM 排序，在 matched / partial 各自集合中随机选
        per_prop_topk 条规则；
      - 每条规则 score 统一设置为 neutral_score，不再反映 LLM 排名。

    其它流程（按 edit_op 对齐、多属性聚合）与 build_multiobjective_rulebank_for_round 保持一致。
    """

    if seed is not None:
        import random
        random.seed(seed)

    S0 = Chem.MolFromSmiles(S0_smiles)
    XR = Chem.MolFromSmiles(XR_smiles) if XR_smiles else None

    def _rule_key(rule_dict):
        # 二轮及之后仍然用 edit_op 作为跨属性对齐 key
        return rule_dict.get("edit_op")

    props = list(target_prop_lst)
    per_prop_selected, per_prop_keys = {}, {}
    index_per_prop = {}   # {prop: {edit_op: {"rule": rule_dict, "score": u, "rationale": str}}}
    # 主属性定义：列表第2个（保持与原函数一致）
    main_prop = target_prop_lst[1] if target_prop_lst else None

    # 1) 每个属性单独跑“随机版选择”，得到候选集合
    for prop, direc in zip(target_prop_lst, target_direc_lst):
        rb_prop = per_prop_rulebanks[prop]      # {"rules":[...]}
        rules_all = rb_prop["rules"]

        # 主属性：last=False
        # 副属性：在 use_main_aux_score=True 且有多个属性时，last=True
        if use_main_aux_score and (prop != main_prop) and (len(target_prop_lst) > 1):
            use_last = True
        else:
            use_last = False
        # =============================== 这往上都一样 =================================
        # 随机版单属性筛选，不再走 llm_score_rules
        sel = select_rules_for_prompt_random(S0, XR, rules_all, prop, direc, llm,
            topk=per_prop_topk, last=use_last, seed=None if seed is None else seed + hash(prop) % 10000)

        # 这后面除了 u = neutral_score 这一句以外也都一样了
        strict_list = sel.get("strict", []) or []
        partial_list = sel.get("partial", []) or []

        # 这里 candidates 的顺序已经是“随机抽样”的结果，直接拼接即可
        ranked = (strict_list + partial_list)[:per_prop_topk]

        per_prop_selected[prop] = ranked
        per_prop_keys[prop] = {_rule_key(r) for r in ranked}

        # 2) 不再用名次归一化，所有规则统一给 neutral_score
        idx_map = {}
        for rule in ranked:
            u = neutral_score
            key = _rule_key(rule)
            rec = idx_map.get(key)
            # 同一属性同一 edit_op，只保留 u 更高的那条（这里其实都一样）
            if (rec is None) or (u > rec.get("score", 0.0)):
                idx_map[key] = {
                    "rule": rule,
                    "score": u,                          # 中性 utility
                    "rationale": rule.get("rationale", "")
                }
        index_per_prop[prop] = idx_map

    # 3) 调用公共多属性聚合逻辑（和原函数保持一致）
    selected = _aggregate_multiobjective_rules(
        props=props,
        per_prop_keys=per_prop_keys,
        index_per_prop=index_per_prop,
        topk_multi=topk_multi,
        use_direction_filter=False,    # 与原函数一致
        prop2dir=None,
        thr=neutral_score,
        use_main_aux_score=use_main_aux_score,
        neutral_score=neutral_score,
        stage_label="多属性后续轮-随机消融"
    )

    return {
        "property": "+".join(target_prop_lst),
        "mol": S0_smiles,
        "candidate_count": len(selected),
        "selected": selected
    }





# =========================================================
# 加载数据、计算嵌入相关
# =========================================================
# 加载所有的输入分子(即测试集)
def load_dataset(drug_type, task, task_specification_dict):
    if drug_type == 'molecule':
        with open('data/small_molecule/small_molecule_editing.txt') as f:
            test_data = f.read().splitlines()
    else:
        raise NotImplementedError
    return test_data


# 读取数据库，可选返回原始数据库还是返回数据库的嵌入向量
def load_retrieval_DB(task, seed, return_embedding=False):
    """
        :param task: 任务名称
        :param seed: 随机种子
        :param return_embedding: 是否返回embedding（True则返回带embedding的DB）
        :return: input_drug_list, DB 或 (input_drug_list, DB_embeddings)
    """
    drug_type = 'molecule'
    # DBfile = './data/small_molecule/combined_unique_molecules.csv'       # 数据库文件的路径
    DBfile = './data/small_molecule/250k_rndm_zinc_drugs_clean_3.csv'    # zinc数据库
    task_specification_dict = get_task_specification_dict(task)  # 获得PDDS模板
    input_drug_list = load_dataset(drug_type, task, task_specification_dict)
    input_drug_list = list(set(input_drug_list))  # 去重
    prop = task2prop(task)

    # 加载原始DB
    DB = pd.read_csv(DBfile)
    DB = DB[['smiles']]
    DB = DB.rename(columns={"smiles": "sequence"})
    for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
        DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED) < 0]    # 剔除和input重复的分子
    DB = DB.sample(10000, random_state=seed)  # 随机抽取1w条

    # === 如果要直接返回embedding ===
    if return_embedding:
        print("DB分子的嵌入生成中...")
        smiles_list = DB['sequence'].tolist()
        DB_embeddings = compute_final_embedding_batch(smiles_list, prop, 32)
        print("完成！")
        return input_drug_list, DB, DB_embeddings  # 多返回一个DB_embeddings

    # === 否则，正常返回DB本身 ===
    return input_drug_list, DB


# 多轮运行时、加载全部数据库时使用这个代码仅加载一次数据库嵌入
def load_retrieval_DB2(task, return_embedding=False, cache_path='./embed_caches/embedding_cache_optiDB.pkl'):
    """
        :param task: 任务名称
        :param seed: 随机种子
        :param return_embedding: 是否返回embedding（True则返回带embedding的DB）
        :return: input_drug_list, DB 或 (input_drug_list, DB_embeddings)
    """
    drug_type = 'molecule'
    DBfile = './data/small_molecule/combined_unique_molecules.csv'       # 数据库文件的路径
    task_specification_dict = get_task_specification_dict(task)  # 获得PDDS模板
    input_drug_list = load_dataset(drug_type, task, task_specification_dict)
    input_drug_list = list(set(input_drug_list))  # 去重
    prop = task2prop(task)

    # 加载原始DB
    DB = pd.read_csv(DBfile)
    DB = DB[['smiles']]
    DB = DB.rename(columns={"smiles": "sequence"})
    for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
        DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED) < 0]    # 剔除和input重复的分子
    DB = DB.sample(48134)  # 随机抽取1w条

    if return_embedding:
        print("DB分子的嵌入生成中...")
        # 加载缓存（如果有）
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                embedding_cache = pickle.load(f)
                print(f"缓存里有{len(embedding_cache)}条数据")
        else:
            embedding_cache = {}

        # 把没在缓存里的拿出来继续处理
        smiles_list = DB['sequence'].tolist()
        new_smiles = [s for s in smiles_list if s not in embedding_cache]

        batch_size = 32
        # 单独处理后3条防绷
        last_batches_to_skip = 3   # 还剩num_batches个batch没处理
        num_batches = (len(new_smiles) + batch_size - 1) // batch_size
        # 选出剩下的batch中剔除后三个batch的下标，这个下标之前的都正常处理（所以小于3给batch就是0了）
        last_batch_start_idx = max(0, (num_batches - last_batches_to_skip) * batch_size)

        # for i in tqdm(range(0, len(new_smiles), batch_size)):
        # compute_final_embedding_batch会直接返回整个输入的向量中间没有保存手段
        # 所以这里要把输入截断成一个个batch才能在这个文件里加保存逻辑
        for i in tqdm(range(0, last_batch_start_idx, batch_size), desc="批量嵌入"):
            batch = new_smiles[i:i + batch_size]
            # with suppress_everything():
            embeddings = compute_final_embedding_batch(batch, prop, batch_size)  # 返回 shape=(batch_size, dim)
            for smile, emb in zip(batch, embeddings):
                embedding_cache[smile] = emb
            with open(cache_path, 'wb') as f:    # 每一批都存一次，避免中途崩溃白算
                pickle.dump(embedding_cache, f)

        # === 剩余最后几个 batch：逐个处理（防崩） ===

        signal.signal(signal.SIGALRM, handler)  # 设置 signal handler
        last_smiles = new_smiles[last_batch_start_idx:]
        print(f"开始逐个处理最后 {len(last_smiles)} 个 SMILES（共 {num_batches} 个 batch）")
        for smile in tqdm(last_smiles, desc="稳妥处理"):
            if smile in embedding_cache: # 防止中间再
                continue
            try:
                signal.alarm(TIMEOUT_SECONDS)  # 启动计时器
                emb = compute_final_embedding(smile, prop)
                signal.alarm(0)  # 取消定时器
            except TimeoutException:
                print(f"[超时] SMILES: {smile}，赋零向量")
                emb = np.zeros(1580, dtype=np.float32)
            except Exception as e:
                print(f"[失败] SMILES: {smile}，原因: {e}")
                emb = np.zeros(1580, dtype=np.float32)
            finally:
                embedding_cache[smile] = emb
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding_cache, f)
        print("所有嵌入已生成并缓存！")

        # 保证顺序一致
        DB_embeddings = np.array([embedding_cache[s] for s in smiles_list])
        return input_drug_list, DB, DB_embeddings

    return input_drug_list, DB



# 计算Tanimoto相似度
def sim_molecule(smile0, smile1):
    mol0 = Chem.MolFromSmiles(smile0)  # 转换成RDKit分子表示
    fp0 = AllChem.GetMorganFingerprint(mol0, 2)  # 计算分子的Morgan Fp，2表示指纹半径是2

    mol1 = Chem.MolFromSmiles(smile1)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    
    sim = DataStructs.TanimotoSimilarity(fp0, fp1)  # 计算两个分子的Tanimoto相似度
    return sim


def sim_sequence(task, SEQ1, SEQ2):
    sim = sim_molecule(SEQ1,SEQ2)
    return sim


# # 找到XR，XR = arg max<x~，xR'> ∩ D(Xin,XR';Xt)
# def retrieve_and_feedback(task, DB, input_drug, generated_drug, constraint, threshold_dict):
#     """
#     :param task:
#     :param DB:
#     :param input_drug:
#     :param generated_drug:     llm当前轮次生成的药物，即x浪
#     :param constraint:         loose/strict
#     :param threshold_dict:
#     :return:
#     """
#     # 》》 arg max<x~，xR'> 《《
#     sim_DB = DB.copy()  # 创建数据库 DB 的副本，避免修改原始数据。
#     # sim_DB['sim']=[0]*len(DB)
#     sim_list = []
#     for index, row in sim_DB.iterrows():
#         # 遍历数据库中每个分子
#         smiles = row['sequence'].replace('\n','')
#         # 评估当前分子(DB中，即XR')与generated_drug的相似度
#         sim = sim_sequence(task, smiles, generated_drug)
#         sim_list.append(sim)
#         # sim_DB['sim'][index] = sim
#         # sim_DB.at[index,'sim']=sim
#     sim_DB['sim'] = sim_list
#     sim_DB = sim_DB.sort_values(by=['sim'], ascending=False)  # 降序排序，这样下面从第一个分子开始匹配，第一个匹配到的就是相似度最大的
#
#     # 》》 D(Xin,XR';Xt) 《《
#     ColorText.print("开始搜索满足Δ的XR‘", ColorText.PURPLE, ColorText.REVERSE)
#     for index, row in sim_DB.iterrows():
#         # 实参row['sequence'].replace('\n','')对应形参generated_drug，这里评估的实际是每个XR'与input是不是满足①合规②属性阈值
#         answer = evaluate(input_drug, row['sequence'].replace('\n',''), task, constraint, threshold_dict=threshold_dict)
#         # answer这里代表的就是D(~,~;~)的结果
#         if answer:
#             # 满足了constraint
#             ColorText.print("当前XR'满足Δ【D(~,~;~)=True】", ColorText.GREEN, ColorText.REVERSE)
#             return row['sequence'].replace('\n', '')  # 找到第一个符合D(~,~;~)的就返回
#         elif answer == 0:
#             pass
#             # ColorText.print("当前XR'不满足Δ【D(~,~;~)=false】", ColorText.RED)
#         else:
#             # answer=-1
#             ColorText.print("当前XR'有问题或为空(无法计算D【~,~;~)】", ColorText.RED)
#     raise Exception("Sorry, Cannot fined a good one")












if __name__ == '__main__':

    rules_db = load_rulebank("rules", "logp")
    S0 = Chem.MolFromSmiles("c1ccccc1")  # benzene
    D = Chem.MolFromSmiles("c1cc(Cl)ccc1")  # chlorobenzene (XR分子)

    # feedback = build_feedback_from_diff(S0, D, rules_db["rules"], "increase")
    # print(feedback)
    with suppress_everything():
        llm = choose_llm("gpt")
    sel_rules = select_rules_for_prompt(S0, D, rules_db["rules"], "logp", "increase", llm, 5)
    print(sel_rules)