import hashlib
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from rdkit import Chem
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

from utils.convenient_utils.aggre_rules import _aggregate_multiobjective_rules
from utils.convenient_utils.parse_json import safe_parse_incomplete_json, safe_parse_trend_json, remove_json_markers, extract_json_from_llm_output
from utils.convenient_utils.safe_llm import safe_llm_invoke_retry_500
from utils.convenient_utils.suppress_useless_print import suppress_everything
from utils.convenient_utils.wordart import ColorText
from utils.main_utils import choose_llm
from utils.rulebook_tool import load_rulebank
import random

# ======================================================
# 标准化相关
# ======================================================
# 把外部输入的规则结构标准化为 list[rule_dict]
def normalize_rules_input(rules_input: Any) -> List[Dict[str, Any]]:
    """
    支持两种输入:
    ① 一个 dict, 包含 keys: "property","rules" -> 返回 rules 列表
    ②- 直接传 rules 列表 -> 直接返回
    """
    if isinstance(rules_input, dict) and "rules" in rules_input:
        return rules_input["rules"]
    if isinstance(rules_input, list):
        return rules_input
    raise ValueError("rules_input must be dict-with-rules or a list of rule dicts")


# ======================================================
# LoC部分
# ======================================================
# ---------- 步骤1: RDKit 预筛选（基于smarts_trigger的子结构匹配） ----------
def prefilter_rules_by_smarts(rules, mol_smiles: str) -> List[Tuple[int, Dict[str, Any]]]:
    """
    用规则库的 smarts_trigger 在目标分子上做子结构匹配，只有能匹配到的规则（或没有 trigger 的泛化规则）才作为候选返回。
    返回值：[(原始规则索引, rule_dict), ...]
    """
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        raise ValueError("Invalid mol_smiles: cannot parse with RDKit")

    candidates = []
    for idx, rule in enumerate(rules):
        triggers = rule.get("smarts_trigger") or []
        if not triggers:   # 无 trigger 的规则视为通用规则
            candidates.append((idx, rule))  # 故直接保留
            continue

        matched = False
        for t in triggers:  # 遍历smart_trigger字段里的所有smart
            try:
                patt = Chem.MolFromSmarts(t)  # trigger转成分子对象
            except Exception:
                patt = None
            if patt is None:  # 如果模型提供的不是合法 SMARTS，尝试把它当 SMILES 再解析一次
                try:
                    patt = Chem.MolFromSmiles(t)
                except Exception:
                    patt = None
            if patt is None:  # 不能解析则跳过该 trigger
                continue

            # 上述分子转换都合法了，接下来就在原分子mol中匹配当前trigger patt
            try:
                if mol.HasSubstructMatch(patt):
                    matched = True
                    break
            except Exception:
                continue
        if matched:   # 只要有一个能匹配上的smart_trigger就纳入可用规则
            candidates.append((idx, rule))
    return candidates


# 步骤2prompt： 规则映射表的提示词
def build_trend_prompt(prop, direc, filtered_trend_tags):
    trend_prompt = f"""
       You are a medicinal chemist. For the molecular optimization target "{direc} {prop}",
       determine whether each of the following structural trends generally increases or decreases the property.
       Return valid JSON: each key is a trend, value is +1 (positive effect) or -1 (negative effect).

       Trends:
       {json.dumps(filtered_trend_tags, ensure_ascii=False, indent=2)}

       Example output:
       {{"increases aromaticity":+1,"decreases aromaticity":-1,...}}
       Return JSON only.
       """
    return trend_prompt


# 步骤2prompt：llm打分的提示词
def build_score_prompt(mol_smiles, prop, direction, trend_tags, minimal):
    # KS  用llm对规则逐批评估每条 trend 是否发生 (0/1)
    return f"""
You are a medicinal chemistry expert. For each candidate rule, you need to determine which structural trends it is related to.
Molecule to be optimized):
{mol_smiles}

Optimization target:
{prop} — the goal is to **{direction}** it.
Return a JSON array where each element has:
  - "id": same as input
  - Each trend name as key with value 0 or 1 (1 = related, 0 = not relevant)

Trends:
{json.dumps(trend_tags, ensure_ascii=False)}

Input:
{json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))} 

Output example:
[{{"id": same as inpt,"increases aromaticity":1,"decreases aromaticity":0,...}},...]

Note:
1. If the trend even weakly applies (minor or secondary effect), still mark as 1.
2. Return valid JSON only, no extra explanation.
Return only JSON."""


# ---------- 步骤2: 用 LLM 对候选规则在做评分/筛选（相当于对筛选后的规则库再思考一遍） ----------
def score_candidate_rules_with_llm(llm, mol_smiles, prop, target_direction, candidates, chunk_size=15, score_scale=(0, 5)):
    # 这个是对初次筛选完的规则(硬性匹配，方向、触发)打分，对于某分子指定方向下筛选逻辑都是相同的，所以单双属性都可以通用，因为这步还没涉及到排序逻辑
    def smooth_trend_output(parsed_list):
        # LLM对trendmap的判断倾向于判断成0，这里做一些trend之间的联动，总体提升分数让筛出的规则多一点
        for obj in parsed_list:
            # ① 极性->亲脂性-疏水性联动
            if obj.get("increases polarity", 0) == 1:
                obj["decreases lipophilicity"] = 1
                obj["decreases hydrophobicity"] = 1
            if obj.get("decreases polarity", 0) == 1:
                obj["increases lipophilicity"] = 1
                obj["increases hydrophobicity"] = 1

            # ② 氢键供体/受体->极性联动
            if obj.get("adds hydrogen bond donor", 0) == 1 or obj.get("adds hydrogen bond acceptor", 0) == 1:
                obj["increases polarity"] = 1
            if obj.get("removes hydrogen bond donor", 0) == 1 or obj.get("removes hydrogen bond acceptor", 0) == 1:
                obj["decreases polarity"] = 1

            # ③ 芳香性->共轭-刚性联动
            if obj.get("increases aromaticity", 0) == 1:
                obj["increases conjugation"] = 1
            if obj.get("increases conjugation", 0) == 1:
                obj["increases rigidity"] = 1

            # ④ 体积->亲脂性弱联动
            if obj.get("increases steric bulk", 0) == 1:
                obj["increases lipophilicity"] = 1
            if obj.get("decreases steric bulk", 0) == 1:
                obj["decreases lipophilicity"] = 1

        return parsed_list

    results, total = [], len(candidates)
    low, high = score_scale  # result是跨chunk用于记录最终结果的变量, total是需要评分的规则总数，lh是评分区间

    # 有评分结果的缓存就读取，没有就创建
    cache_dir, map_dir = "./rules/logs", "./rules"

    # KS  一次性生成或读取 trend→property 映射表(内容：trend_tags对当前prop又正向影响还是负向影响，后面打分阶段通用)
    # 每两行是一类，包括电子结构、极性、疏水性、体积和拓扑
    # [NEW] 扩充趋势集合，加入 HBD/HBA 显式条目；保留其它结构趋势
    trend_tags = [
        "increases aromaticity", "decreases aromaticity",     # “增强芳香性”，“降低芳香性”，
        "increases conjugation", "decreases conjugation",     # “增加共轭结构”，“减少共轭结构”，
        "increases polarity", "decreases polarity",           # “增强极性”，“降低极性”，
        "adds hydrogen bond donor", "removes hydrogen bond donor",       # “添加氢键供体”，“移除氢键供体”，
        "adds hydrogen bond acceptor", "removes hydrogen bond acceptor", # “添加氢键受体”，“移除氢键受体”
        "increases steric bulk", "decreases steric bulk",     # “增加空间体积”，“减少空间体积”，
        "increases rigidity", "decreases rigidity",           # “增加刚性”，“减少刚性”，
        "increases lipophilicity", "decreases lipophilicity", # “增加亲脂性”，“降低亲脂性”，
        "increases hydrophobicity", "decreases hydrophobicity"# “增加疏水性”，“降低疏水性”
    ]
    # === 根据属性过滤无关 trends ===
    if prop.lower() == "logp":    # 亲脂性、疏水性和logp直接对应，故忽略
        ignore_keys = ["lipophilicity", "hydrophobicity"]
    elif prop.lower() == "qed":   # 综合指标，不忽略任何趋势
        ignore_keys = []
    elif prop.lower() == "tpsa":  # 极性表面积与方向性、共轭结构、刚性、空间体积无关，故忽略
        ignore_keys = ["aromaticity", "conjugation", "rigidity", "steric bulk", "lipophilicity", "hydrophobicity"]
    elif prop.lower() in ["hbd", "hydrogen bond donor"]:   # 氢键供体数就直接反应了氢键供体这个指标，故忽略
        ignore_keys = ["hydrogen bonding", "hydrogen bond donor"]
    elif prop.lower() in ["hba", "hydrogen bond acceptor"]:  # 同上
        ignore_keys = ["hydrogen bonding", "hydrogen bond acceptor"]
    else:
        ignore_keys = []

    filtered_trend_tags = [t for t in trend_tags if not any(key in t for key in ignore_keys)]

    trend_map_file = os.path.join(map_dir, f"{prop}_{target_direction}_trend_map.json")
    if os.path.exists(trend_map_file):      # 有缓存直接加载
        trend_impact = json.load(open(trend_map_file, "r", encoding="utf-8"))
    else:
        # 构建映射表的提示词
        trend_prompt =build_trend_prompt(prop, target_direction, filtered_trend_tags)
        llm.temperature = 0
        resp = llm.invoke([
            SystemMessage(content="You are a precise medicinal chemistry assistant."),
            HumanMessage(content=trend_prompt)
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        ColorText.print(f"用户输入(trend→property 映射表): {trend_prompt}", ColorText.YELLOW)
        ColorText.print(f"模型回答(trend→property 映射表): {raw}", ColorText.BLUE)
        trend_impact = safe_parse_trend_json(raw)
        print(trend_impact)
        # 写到缓存里
        json.dump(trend_impact, open(trend_map_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ks  开始评分
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{prop}_rule_scores_{target_direction}.json")
    print(f"一轮评分cache位置：{cache_file}")
    # 读取缓存
    cache_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            cache_data = {}
    # 初始化该分子的缓存（最外层）
    if mol_smiles not in cache_data:
        cache_data[mol_smiles] = {}

    # =====================================================

    # 设置步长为chunk_size是为了防止total太长导致的爆上下文
    # for batch_id, i in enumerate(tqdm(range(0, total, chunk_size), desc=f"(首轮)规则评分 {prop}", unit="chunk")):
    i, batch_id = 0, 0   # i是当前cand读到哪了，batchid表示是第几个循环
    while i < total:    # 改for为while以支持动态chunk缩减
        chunk = candidates[i:i + chunk_size]   # chunk参数里传进来的时候暂定为整个规则库长度

        batch_key = f"batch_{batch_id}"   # ======================== 如果该batch已存在缓存，则直接复用当前batch ===
        if batch_key in cache_data[mol_smiles]:
            print(f"[cache hit] (SMILESKEY){mol_smiles}  (BATCHKEY){batch_key}")
            results.extend(cache_data[mol_smiles][batch_key])
            cached_len = len(cache_data[mol_smiles][batch_key])
            i, batch_id = i + cached_len, batch_id + 1    # i和batch_id的自增
            continue
        print(f"[cache MISS] (SMILESKEY){mol_smiles}  (BATCHKEY){batch_key}")
        # ============================================================================== 否则在进行实际的处理 ===
        minimal = []             # 构造输入（格式化）
        for idx, rule in chunk:
            minimal.append({
                "id": idx,
                "edit_op": rule.get("edit_op"),
                "smarts_trigger": rule.get("smarts_trigger", []),
                # "confidence": rule.get("confidence", None),
                "rationale_preview": rule.get("rationale") or ""
            })

        # ================================================================ 评分的提示词（加入输入分子作为额外信息）===
        # KS  用llm对规则逐批评估每条 trend 是否发生 (0/1)
        prompt = build_score_prompt(mol_smiles, prop, target_direction, trend_tags, minimal)
        # === 修改2：动态检测上下文token并递减chunk_size ===
        est_output_tokens, est_input_tokens = len(chunk) * 80, len(prompt) // 4
        while est_input_tokens + est_output_tokens > 4096 and chunk_size > 5:   # ==== 超上下文的话缩短chunk_size ===
        # while est_input_tokens + est_output_tokens > 16000 and est_output_tokens <4096 and chunk_size > 5:
            ColorText.print(f"当前chunk(chunksize={chunk_size})过长({est_input_tokens + est_output_tokens} tokens)，自动收缩一半", ColorText.YELLOW)
            chunk_size //= 2
            chunk = candidates[i:i + chunk_size]  # 重新计算新chunksize下的，输出估计长度
            est_output_tokens = len(chunk) * 80
            minimal = [{                          # 构造输入（格式化）
                "id": idx,
                "edit_op": rule.get("edit_op"),
                "smarts_trigger": rule.get("smarts_trigger", []),
                "confidence": rule.get("confidence", None),
                "rationale_preview": (rule.get("rationale") or "")[:240]
            } for idx, rule in chunk]

            # 根据新输入构造提示词，重新估计输入的token
            prompt = build_score_prompt(mol_smiles, prop, target_direction, trend_tags, minimal)
            est_input_tokens = len(prompt) // 4   # 看满足while要求了吗，满足了就退出，chunksize就定下来了
        print("chunk ok，chunksize=", chunk_size)
        llm.temperature, llm.max_tokens = 0, 4096         # KS  强制温度为0，确保确定性
        # 让llm对候选库里的规则评分
        ColorText.print("###########################################################################################",ColorText.CYAN)
        # with suppress_everything():
        #     resp = llm.invoke([
        #         SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
        #         HumanMessage(content=prompt)
        #     ])
        resp = safe_llm_invoke_retry_500(llm, [
            SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
            HumanMessage(content=prompt)
        ],  tag=f"score_batch_{batch_id}", max_attempts=2)
        ColorText.print("###########################################################################################",ColorText.CYAN)
        ###########################################################################################
        # ColorText.print(f"用户提示(CoT)\n{prompt}", ColorText.YELLOW)
        # ColorText.print(f"模型回答(CoT)\n{resp}", ColorText.BLUE)
        ###########################################################################################
        raw = resp.content if hasattr(resp, "content") else str(resp)
        cleaned = remove_json_markers(raw)
        parsed = safe_parse_incomplete_json(cleaned)

        # 如果返回条数 < 输入条数，则只重试缺失部分（改成最多重试3次）
        if isinstance(parsed, list):
            # 累积已解析结果并去重
            agg, seen_ids = [], set()
            for obj in parsed:    # 解析成功的部分
                if isinstance(obj, dict) and "id" in obj and obj["id"] not in seen_ids:
                    agg.append(obj)
                    seen_ids.add(obj["id"])

            max_retries, retry_cnt = 3, 0
            while retry_cnt < max_retries and len(seen_ids) < len(minimal):
                missing_ids = [m["id"] for m in minimal if m["id"] not in seen_ids]
                if not missing_ids:
                    break
                retry_chunk = [m for m in minimal if m["id"] in missing_ids]

                # 构造重试提示词
                retry_prompt = build_score_prompt(mol_smiles, prop, target_direction, trend_tags, retry_chunk)
                ColorText.print(f"检测到丢失{len(missing_ids)}条记录，进行第{retry_cnt+1}次补全", ColorText.YELLOW)
                ColorText.print("###########################################################################################", ColorText.BLUE)
                # with suppress_everything():
                #     resp2 = llm.invoke([
                #         SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
                #         HumanMessage(content=retry_prompt)
                #     ])
                resp2 = safe_llm_invoke_retry_500(llm, [
                    SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
                    HumanMessage(content=retry_prompt)
                ], tag=f"score_batch_{batch_id}", max_attempts=2)
                ColorText.print("###########################################################################################", ColorText.BLUE)
                ###########################################################################################
                # ColorText.print(f"用户提示(CoT补充缺失 第{retry_cnt+1}次)\n{retry_prompt}", ColorText.YELLOW)
                # ColorText.print(f"模型回答(CoT补充缺失 第{retry_cnt+1}次)\n{resp2}", ColorText.BLUE)
                ###########################################################################################

                raw2 = resp2.content if hasattr(resp2, "content") else str(resp2)
                parsed2 = safe_parse_incomplete_json(remove_json_markers(raw2))
                # 合并重试结果并去重
                if isinstance(parsed2, list):
                    if len(missing_ids) == 1:    # 只缺一条：不要相信模型返回的 id，直接用我们本地知道的那一个
                        expected_id = missing_ids[0]
                        for obj in parsed2:
                            if not isinstance(obj, dict):
                                continue
                            obj["id"] = expected_id  # 强制覆盖
                            if expected_id not in seen_ids:
                                agg.append(obj)
                                seen_ids.add(expected_id)
                    for obj in parsed2:    # 记录已经生成的id，后续将继续生成没生成的（如果有的话）
                        if isinstance(obj, dict) and "id" in obj and obj["id"] not in seen_ids:
                            agg.append(obj)
                            seen_ids.add(obj["id"])

                retry_cnt += 1

            parsed = agg  # 用累积的完整结果覆盖

        # KS  用 trend_impact 与本批激活趋势（parse）做加权，得到最终分数
        parsed = smooth_trend_output(parsed)
        batch_results = []
        if isinstance(parsed, list):
            # print("parsed: \n", parsed)
            # 如果某些 trend 被提前过滤（filtered_trend_tags），trend_impact 只会包含保留的 trend
            K = sum(abs(int(v)) for v in trend_impact.values())  # trend_impact中的有效趋势（不为0）总数
            K = max(1, K)  # 防止除零

            for obj in parsed:                             # 遍历每个分子对有效规则的映射情况
                rid = int(obj.get("id"))
                # 原始得分 = Σ(激活trend 0/1 × 对应影响 +1/-1)
                raw_sum = 0
                active_trends = []
                for trend, impact in trend_impact.items():  # 遍历所有有效规则
                    v = int(obj.get(trend, 0))              # v是该规则对该trend有无影响(0/1)
                    raw_sum += v * int(impact)              # impact是该属性下，该trend对该属性的影响(±1)
                    if v == 1:
                        active_trends.append(f"{trend}({impact:+d})")

                # 线性映射到 [low, high]
                score_f = (raw_sum / K) * (high - low) / 2 + (high + low) / 2
                score = max(low, min(high, score_f))

                # 记录结果
                # KS  score已经通过相反方向采用相反trendmap的方式，不管是inc或dec都是2.5以上是对的了
                batch_results.append({
                    "id": rid,
                    "relevance": score > (low + high) // 2,   # 简单把最终分数大于一半的认为是有效
                    "score": score,
                    "enhanced_rationale": (
                        "Trend-based score: "
                        + (", ".join(active_trends) if active_trends else "no strong trends")
                    )   # enhanced_rationale是各个Trend的得分情况
                })
        else:  # 解析失败：用原规则(本步llm的输入)的confidence作为评分，且reason置为fallback
            print("解析失败")
            for item in minimal:
                rid = item["id"]
                conf = item.get("confidence")
                batch_results.append({
                    "id": rid,
                    "relevance": True,
                    "score": conf,
                    "enhanced_rationale": "fallback: 无法生成增强型解释"
                })

        # === 将该batch评分结果写入缓存 ===
        cache_data[mol_smiles][batch_key] = batch_results
        results.extend(batch_results)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        i += chunk_size
        batch_id += 1

    return results

# ----------- 步骤2消融版，不用映射表而是直接打分 -----------
def score_candidate_rules_with_llm_wo_table(llm, mol_smiles, prop, target_direction, candidates, chunk_size=15, score_scale=(0, 5)):
    """
    对候选规则分块发送给 LLM，让 LLM 给出每条规则的：
      - id (原规则索引)
      - relevance (true/false)
      - score (整数，score_scale范围)
      - enhanced_rationale (1-2句解释；作为“思维链”式的短推理)

    返回：list of dicts, 每个 dict 至少含 keys: id, relevance, score, enhanced_rationale

    约定（按你的要求不改）：
      - cache 仍按 cache_data[mol_smiles]["batch_{batch_id}"] 存
      - candidates / chunk_size / total 运行中不改变（这里只会在“prompt过长”时临时缩小本次chunk重试，并在成功后恢复回初始chunk_size）
    """
    results, total = [], len(candidates)
    low, high = score_scale

    # ====== 缓存路径：按属性分文件 ======
    cache_dir = "./rules/logs/abl"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{prop}_rule_scores_{target_direction}_abl.json")

    # 读取缓存
    cache_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            cache_data = {}

    # 初始化该分子的缓存（最外层）
    if mol_smiles not in cache_data:
        cache_data[mol_smiles] = {}

    # =====================================================
    # 关键修复点：
    # - 不用 range(..., chunk_size) 这种固定步长的 for
    # - 改为 while 手动推进 i，才支持“过长就缩小并重试同一段”
    # - 解析后补齐缺失 id，避免后续 score_map.get(idx) 变 None
    # =====================================================

    base_chunk_size = max(1, int(chunk_size))
    i = 0
    batch_id = 0

    # tqdm 进度条（按 chunk 起始位置推进）
    pbar = tqdm(total=total, desc=f"(首轮)规则评分 {prop}", unit="rule")

    while i < total:
        # === 如果该batch已存在缓存，则直接复用当前batch ===
        batch_key = f"batch_{batch_id}"
        if batch_key in cache_data[mol_smiles]:
            cached = cache_data[mol_smiles][batch_key]
            print(f"[cache hit] {mol_smiles} - {batch_key}")
            results.extend(cached)

            # 按缓存长度推进（假设你缓存时每条对应一个规则；若缓存为空也至少推进 base_chunk_size 防死循环）
            adv = len(cached) if isinstance(cached, list) and len(cached) > 0 else min(base_chunk_size, total - i)
            i += adv
            pbar.update(adv)
            batch_id += 1
            continue

        # 默认使用 base_chunk_size；若 prompt 过长则临时减小重试
        cur_chunk_size = min(base_chunk_size, total - i)

        # 这里用一个内层循环来做“缩小后重试同一段”
        while True:
            chunk = candidates[i:i + cur_chunk_size]

            minimal = []
            for idx, rule in chunk:
                minimal.append({
                    "id": idx,
                    "edit_op": rule.get("edit_op"),
                    "smarts_trigger": rule.get("smarts_trigger", []),
                    "confidence": rule.get("confidence", None),
                    "rationale_preview": (rule.get("rationale") or "")[:240]
                })

            prompt = f"""
You are a medicinal chemistry assistant. For the given molecule and property, evaluate the relevance of candidate molecular-edit rules.
Molecule SMILES: {mol_smiles}
Target property: {prop}

For each candidate rule provide:
- "id": the id field (same as input)
- "relevance": true/false (whether applying this rule has plausible impact on the property for this molecule)
- "score": integer between {low} and {high} indicating expected usefulness (higher = more useful)
- "enhanced_rationale": 1–2 sentences, explain *why this rule is especially credible for this specific molecule and property*, 
   not just general chemistry. (mention site accessibility, scaffold context, binding pocket fit, polarity/sterics compatibility, etc.)

Input candidates (JSON array of minimal items):
{json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))}

Output a JSON array of objects like:
[{{"id":0,"relevance":true,"score":4,"enhanced_rationale":"..."}}, ...]
Return JSON only.
""".strip()

            # ==== 动态长度保护（真正可用：过长则缩小 chunk 并重试同一 i） ====
            # 这只是估算，目的仅是避免明显爆上下文；不改你的 cache key 设计
            est_output_tokens = len(chunk) * 80  # 比原来50保守一点，减少截断风险
            if len(prompt) // 4 > 4096 - est_output_tokens:
                if cur_chunk_size <= 5:
                    # 已经缩到最小仍过长：退化策略——不再缩小，直接走 fallback（不调用LLM）
                    ColorText.print("当前chunk仍过长且已到最小，跳过LLM评分，直接fallback", ColorText.YELLOW)
                    parsed = None
                    raw = ""
                    break
                ColorText.print("当前chunk过长，自动收缩一半并重试同一段", ColorText.YELLOW)
                cur_chunk_size = max(5, cur_chunk_size // 2)
                continue

            # ==== 正常调用 LLM ====
            llm.temperature = 0
            llm.max_tokens = 4096
            with suppress_everything():
                resp = llm.invoke([
                    SystemMessage(content="You are a medicinal chemistry assistant. Output JSON only."),
                    HumanMessage(content=prompt)
                ])

            #############################################################################
            ColorText.print(f"用户提示(CoT)\n{prompt}", ColorText.YELLOW)
            ColorText.print(f"模型回答(CoT)\n{resp}", ColorText.BLUE)
            #############################################################################

            raw = resp.content if hasattr(resp, "content") else str(resp)
            cleaned = remove_json_markers(raw)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = extract_json_from_llm_output(cleaned)

            break  # 成功调用（或已决定fallback）则退出内层重试循环

        # ===== 解析与兜底 =====
        batch_results = []
        if isinstance(parsed, list):
            for obj in parsed:
                try:
                    rid = int(obj.get("id"))
                except Exception:
                    continue
                relevance = bool(obj.get("relevance", True))
                try:
                    score = int(obj.get("score"))
                except Exception:
                    score = low
                # clamp
                if score < low:
                    score = low
                if score > high:
                    score = high
                enhanced = str(obj.get("enhanced_rationale", "")).strip()
                batch_results.append({
                    "id": rid,
                    "relevance": relevance,
                    "score": score,
                    "enhanced_rationale": enhanced
                })
        else:
            # 解析失败或主动fallback：用 confidence 兜底
            print("解析失败或fallback")
            for item in minimal:
                rid = item["id"]
                conf = item.get("confidence")
                try:
                    score = int(conf) if conf is not None else low
                except Exception:
                    score = low
                if score < low:
                    score = low
                if score > high:
                    score = high
                batch_results.append({
                    "id": rid,
                    "relevance": True,
                    "score": score,
                    "enhanced_rationale": "fallback: 无法生成增强型解释"
                })

        # ===== 关键补齐：确保每个 chunk 的 id 都有结果，避免后续缺项 =====
        chunk_ids = [idx for idx, _ in candidates[i:i + cur_chunk_size]]
        got_ids = {x["id"] for x in batch_results}
        if len(got_ids) < len(chunk_ids):
            # 用原规则 confidence 补齐缺失项
            id2rule = {idx: rule for idx, rule in candidates[i:i + cur_chunk_size]}
            for rid in chunk_ids:
                if rid in got_ids:
                    continue
                rule = id2rule.get(rid, {})
                conf = rule.get("confidence", None)
                try:
                    score = int(conf) if conf is not None else low
                except Exception:
                    score = low
                if score < low:
                    score = low
                if score > high:
                    score = high
                batch_results.append({
                    "id": rid,
                    "relevance": True,
                    "score": score,
                    "enhanced_rationale": "fallback: missing item from LLM output"
                })

        # === 将该batch结果写入缓存 ===
        cache_data[mol_smiles][batch_key] = batch_results
        results.extend(batch_results)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[cache write] {mol_smiles} - {batch_key} 已写入缓存 ({len(batch_results)} 条)")

        # 推进到下一个 chunk：按本次实际 chunk 大小推进
        adv = min(cur_chunk_size, total - i)
        i += adv
        pbar.update(adv)
        batch_id += 1

    pbar.close()
    return results


# ---------- 整个思维链筛选规则库的流程 ----------
# 单属性版本
def filter_rules_with_chain_of_thought(rules_input, mol_smiles, prop, llm,
        top_k=10, max_cand_count=300, chunk_size=15, target_direction=None, min_confidence=2, last=False):
    """
    主流程：
      1) normalize rules input -> list[rules]
      2) RDKit 预筛选（smarts_trigger 匹配或者空 trigger 保留）
      3) 分块调用 LLM 做“思维链式”评分（relevance/score/reason）
      4) 依据 relevance + score 排序返回 top_k（并附带 reason）
    返回：
      {
        "property": prop,
        "mol": mol_smiles,
        "candidate_count": N,
        "selected": [ { "id":R_idx, "rule":rule_dict, "score":int, "reason":str }, ... ]
      }
    说明：函数不会修改原始规则库；所有 id 指的是原始 rules 列表的索引（0-based）。
    """
    # 1) 标准化输入为单纯的rule列表
    rules = normalize_rules_input(rules_input)
    # =================================
    # 2.1) 按方向过滤
    if target_direction:
        before = len(rules)
        rules = [r for r in rules if r.get("direction") in (target_direction, None)]
        ColorText.print(f"[方向过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # 2.2) 按置信度过滤
    before = len(rules)
    rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]
    ColorText.print(f"[置信度过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # # 2.3) 骨架类型标签过滤（可选）
    # 2) step1 基于smart_trigger的RDKit预筛选
    candidates = prefilter_rules_by_smarts(rules, mol_smiles)  # 返回list of (orig_idx, rule_dict)
    candidate_count = len(candidates)
    ColorText.print(f"[trigger过滤]匹配smart_trigger的预筛选{candidate_count}条", ColorText.CYAN)

    # 4) step2 筛选完的规则，进 LLM 批量评分
    llm_scores = score_candidate_rules_with_llm(llm, mol_smiles, prop, target_direction, candidates, chunk_size=len(candidates))

    # 5) 取llm评分的top_k个规则最终返回
    score_map = {s["id"]: s for s in llm_scores}  # 将 llm_scores 转为 dict by id
    merged = []
    """candidates(index, rule_dict)、
    llm_scores(id,relevance,score,enhanced_rationale)都涉及键值对、
    score_map(index2:value)
    index1，id，index2这里要求得一致的，如果出现s为nonetype问题重点检查这几个
    之前的错误是因为llm_scores缺了项，有可能是超上下文的原因"""
    for idx, rule in candidates:    # cand是一个(index, rule_dict)的格式
        # print(idx,rule)
        s = score_map.get(idx)
        if not s:
            ColorText.print(f"{idx}没有打分，跳过了", ColorText.RED)
            continue
        # 只保留 relevance 为 True 的
        if s.get("relevance", True):
            merged.append({
                    "id": idx,
                    "rule": rule,
                    "score": s.get("score", 0),
                    "enhanced_rationale": s.get("enhanced_rationale", "")
                })
    # 排序并取 top_k
    merged.sort(reverse=True, key=lambda x: x["score"])
    if last:    # 取lastk个
        print(f"{len(merged)} 里取 LAST K")
        selected = merged[-top_k:]
    else:
        print(f"{len(merged)} 里取 TOP K")
        selected = merged[:top_k]
    ColorText.print(f"首轮筛选出的规则情况，匹配smart_trigger的预筛选{candidate_count}条，llm评分结束之后的{len(llm_scores)}条，relevance字段筛选之后{len(merged)}条", ColorText.GREEN)

    return {
        "property": prop,
        "mol": mol_smiles,
        "candidate_count": candidate_count,
        "selected": selected  # 最终规则库
    }


# 单属性消融版本__随机消融
def filter_rules_random(rules_input, mol_smiles, prop,
                        top_k=10, target_direction=None, min_confidence=2, seed=None):
    if seed is not None:    # 随机种子确保可复现性
        random.seed(seed)

    rules = normalize_rules_input(rules_input)

    # 按方向过滤
    if target_direction:
        before = len(rules)
        rules = [r for r in rules if r.get("direction") in (target_direction, None)]
        ColorText.print(f"[方向过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # 按置信度过滤
    before = len(rules)
    rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]
    ColorText.print(f"[置信度过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # SMARTS trigger 预筛
    candidates = prefilter_rules_by_smarts(rules, mol_smiles)  # list[(orig_idx, rule_dict)]
    candidate_count = len(candidates)
    ColorText.print(f"[trigger过滤] 匹配smart_trigger的预筛选 {candidate_count} 条", ColorText.CYAN)
    # =================这上面都不变================

    k = min(top_k, candidate_count)   # 防止candi数不够topk报错
    sampled = random.sample(candidates, k)  # 随机取k个规则

    merged = [{"id": idx, "rule": rule,
                "score": 0,                   # 消融，故不打分
                "enhanced_rationale": ""      # 不需要 LLM rationale
            } for idx, rule in sampled]

    ColorText.print(f"[随机首轮] trigger 预筛 {candidate_count} 条，随机取 {len(merged)} 条作为规则库", ColorText.GREEN)

    return {
        "property": prop,
        "mol": mol_smiles,
        "candidate_count": candidate_count,
        "selected": merged
    }

# 单属性消融版本__不用映射表打分
def filter_rules_wo_table(rules_input, mol_smiles, prop, llm,
        top_k=10, max_cand_count=300, chunk_size=15, target_direction=None, min_confidence=2, last=False):
    # 1) 标准化输入为单纯的rule列表
    rules = normalize_rules_input(rules_input)
    # =================================
    # 2.1) 按方向过滤
    if target_direction:
        before = len(rules)
        rules = [r for r in rules if r.get("direction") in (target_direction, None)]
        ColorText.print(f"[方向过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # 2.2) 按置信度过滤
    before = len(rules)
    rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]
    ColorText.print(f"[置信度过滤] 保留 {len(rules)}/{before} 条规则", ColorText.CYAN)

    # # 2.3) 骨架类型标签过滤（可选）
    # 2) step1 基于smart_trigger的RDKit预筛选
    candidates = prefilter_rules_by_smarts(rules, mol_smiles)  # 返回list of (orig_idx, rule_dict)
    candidate_count = len(candidates)
    ColorText.print(f"[trigger过滤]匹配smart_trigger的预筛选{candidate_count}条", ColorText.CYAN)

    # 4) step2 筛选完的规则，进 LLM 批量评分
    # llm_scores = score_candidate_rules_with_llm(llm, mol_smiles, prop, target_direction, candidates, chunk_size=len(candidates))
    llm_scores = score_candidate_rules_with_llm_wo_table(llm, mol_smiles, prop, target_direction, candidates, chunk_size=len(candidates))

    # 5) 取llm评分的top_k个规则最终返回
    score_map = {s["id"]: s for s in llm_scores}  # 将 llm_scores 转为 dict by id
    merged = []
    """candidates(index, rule_dict)、
    llm_scores(id,relevance,score,enhanced_rationale)都涉及键值对、
    score_map(index2:value)
    index1，id，index2这里要求得一致的，如果出现s为nonetype问题重点检查这几个
    之前的错误是因为llm_scores缺了项，有可能是超上下文的原因"""
    for idx, rule in candidates:    # cand是一个(index, rule_dict)的格式
        # print(idx,rule)
        s = score_map.get(idx)
        if not s:
            ColorText.print(f"{idx}没有打分，跳过了", ColorText.RED)
            continue
        # 只保留 relevance 为 True 的
        if s.get("relevance", True):
            merged.append({
                    "id": idx,
                    "rule": rule,
                    "score": s.get("score", 0),
                    "enhanced_rationale": s.get("enhanced_rationale", "")
                })
    # 排序并取 top_k
    merged.sort(reverse=True, key=lambda x: x["score"])
    if last:    # 取lastk个
        print(f"{len(merged)} 里取 LAST K")
        selected = merged[-top_k:]
    else:
        print(f"{len(merged)} 里取 TOP K")
        selected = merged[:top_k]
    ColorText.print(f"首轮筛选出的规则情况，匹配smart_trigger的预筛选{candidate_count}条，llm评分结束之后的{len(llm_scores)}条，relevance字段筛选之后{len(merged)}条", ColorText.GREEN)

    return {
        "property": prop,
        "mol": mol_smiles,
        "candidate_count": candidate_count,
        "selected": selected  # 最终规则库
    }


# 多属性版本
# 【首轮多属性规则筛选】先对每个属性独立做单属性筛选，再在规则层面做交集/并集 + 多属性聚合。
def build_multiobjective_rulebank_for_mol(
    llm,
    mol_smiles,
    per_prop_rulebanks,
    target_prop_lst,
    target_direc_lst,
    topk_single=30,
    max_cand_count=300,
    topk_multi=15,
    use_main_aux_score=False,  # [NEW] 是否启用主属性+辅属性的 u_main + u_aux
    neutral_score=2.5          # [NEW] 中性分数，对应 score_scale=(0,5) 的中点
):
    """
    一轮多属性规则筛选（S0 -> 多目标）：
    返回:
      {
        "property": "prop1+prop2+...",
        "mol": mol_smiles,
        "candidate_count": N,
        "selected": [
          {"id": i, "rule": rule_dict, "score": score_multi, "enhanced_rationale": "..."},
          ...
        ]
      }
    """

    def _rule_key(rule_dict):
        # 用 edit_op 对齐多属性规则
        return rule_dict.get("edit_op")

    props = list(target_prop_lst)

    # 1) 每个属性单独跑单属性规则筛选
    per_prop_selected = {}   # {prop: [ {id, rule, score, enhanced_rationale}, ... ]}
    per_prop_keys = {}       # {prop: set(edit_op)}
    index_per_prop = {}      # {prop: {edit_op: {"rule":..., "score":..., "rationale":...}}}
    prop2dir = {p: d.lower() for p, d in zip(target_prop_lst, target_direc_lst)}  # 仅用于追踪方向

    # -------- 这里加 enumerate，用 idx 区分主/辅属性 --------
    for idx, (prop, direc) in enumerate(zip(target_prop_lst, target_direc_lst)):
        rb_prop = per_prop_rulebanks[prop]          # {"rules":[...]}
        rules_all = rb_prop["rules"]

        # 单属性首轮筛选
        if idx == 1:
            # 主属性：保持原来的调用方式
            res = filter_rules_with_chain_of_thought(
                rb_prop,               # 支持 dict / list，内部 normalize
                mol_smiles,
                prop,
                llm,
                top_k=topk_single,
                max_cand_count=max_cand_count,
                chunk_size=len(rules_all),
                target_direction=direc,
                min_confidence=2
            )
        else:
            # 辅属性：根据 use_main_aux_score 决定是否启用“主/辅分支”
            if use_main_aux_score:
                res = filter_rules_with_chain_of_thought(      # <<<
                    rb_prop,
                    mol_smiles,
                    prop,
                    llm,
                    top_k=topk_single,
                    max_cand_count=max_cand_count,
                    chunk_size=len(rules_all),
                    target_direction=direc,
                    min_confidence=2,
                    last=True                   # <<< 关键：只在辅属性上传 True
                )
            else:
                # 兼容旧逻辑，不传该参数
                res = filter_rules_with_chain_of_thought(
                    rb_prop,
                    mol_smiles,
                    prop,
                    llm,
                    top_k=topk_single,
                    max_cand_count=max_cand_count,
                    chunk_size=len(rules_all),
                    target_direction=direc,
                    min_confidence=2
                )

        sel = res.get("selected", [])   # [ {id, rule, score, enhanced_rationale}, ... ]
        per_prop_selected[prop] = sel
        per_prop_keys[prop] = {_rule_key(item["rule"]) for item in sel}

        # 构建 index_per_prop
        idx_map = {}
        for item in sel:
            rule_dict = item["rule"]
            k = _rule_key(rule_dict)
            sc = float(item.get("score", 0.0))
            rec = idx_map.get(k)
            # 同一属性同一 edit_op，只保留 score 更高的那条
            if (rec is None) or (sc > rec.get("score", 0.0)):
                idx_map[k] = {
                    "rule": rule_dict,
                    "score": sc,   # 这里的 score 已经是“越大越好”
                    "rationale": item.get("enhanced_rationale", "") or rule_dict.get("rationale", "")
                }
        index_per_prop[prop] = idx_map

    # 2) 聚合多属性规则
    selected = _aggregate_multiobjective_rules(
        props=props,
        per_prop_keys=per_prop_keys,
        index_per_prop=index_per_prop,
        topk_multi=topk_multi,
        use_direction_filter=True,
        prop2dir=prop2dir,
        thr=neutral_score,
        use_main_aux_score=use_main_aux_score,
        neutral_score=neutral_score,
        stage_label="多属性首轮"
    )

    return {
        "property": "+".join(target_prop_lst),
        "mol": mol_smiles,
        "candidate_count": len(selected),
        "selected": selected
    }


# 多属性消融版本___随机排序
def build_multiobjective_rulebank_for_mol_random(
    llm,                      # 保留签名一致，实际不用
    mol_smiles,
    per_prop_rulebanks,       # {prop: {"rules": [...]}}
    target_prop_lst,
    target_direc_lst,
    topk_single=30,           # 每个属性单属性随机选多少条
    max_cand_count=300,       # 保留参数，不用
    topk_multi=15,            # 聚合后最多保留多少条多属性规则
    use_main_aux_score=False, # 保留参数，传给 _aggregate_multiobjective_rules
    neutral_score=2.5,        # 给所有候选规则的“中性分数”
    seed=None                 # 随机种子，便于复现
):
    """
  只消融：
    不再调用 filter_rules_with_chain_of_thought 做 LLM 打分排序
    在每个属性的“候选规则集合”里均匀随机抽 topk_single 条，score 统一设成一个中性分数（比如 neutral_score），让 _aggregate_multiobjective_rules 不再受到 LLM 打分影响。
    """

    def _rule_key(rule_dict):
        # 用 edit_op 对齐多属性规则（与原函数保持一致）
        return rule_dict.get("edit_op")

    if seed is not None:
        random.seed(seed)    # 可复现

    props = list(target_prop_lst)

    per_prop_selected = {}   # {prop: [ {id, rule, score, enhanced_rationale}, ... ]}
    per_prop_keys = {}       # {prop: set(edit_op)}
    index_per_prop = {}      # {prop: {edit_op: {"rule":..., "score":..., "rationale":...}}}
    prop2dir = {p: d.lower() for p, d in zip(target_prop_lst, target_direc_lst)}

    for idx, (prop, direc) in enumerate(zip(target_prop_lst, target_direc_lst)):
        rb_prop = per_prop_rulebanks[prop]          # {"rules":[...]}
        rules_all = rb_prop["rules"]
        # KS  ===================  这往上都不用变 ==================

        if seed is not None:
            random.seed(seed + idx)          # 每个属性单独设一个子种子，保证可复现又不完全一样

        # 1) direction / confidence 过滤
        rules = rules_all
        if direc:
            before = len(rules)
            rules = [r for r in rules if r.get("direction") in (direc, None)]
            # print("rule count: ", len(rules))   # 这里也没发现
            ColorText.print(f"[多属性-随机-{prop}] 方向过滤 {len(rules)}/{before}", ColorText.CYAN)

        before = len(rules)
        rules = [r for r in rules if r.get("confidence", 0) >= 2]
        ColorText.print(f"[多属性-随机-{prop}] 置信度过滤 {len(rules)}/{before}", ColorText.CYAN)

        # 2) SMARTS trigger 预筛
        candidates = prefilter_rules_by_smarts(rules, mol_smiles)  # list[(orig_idx, rule_dict)]
        # print("candidates count:", len(candidates))   # 这里没问题
        cand_cnt = len(candidates)
        ColorText.print(f"[多属性-随机-{prop}] trigger 预筛 {cand_cnt} 条", ColorText.CYAN)

        # 3) 在候选规则中随机采样 topk_single 条
        if cand_cnt == 0:
            sel = []
        else:
            k = min(topk_single, cand_cnt)
            sampled = random.sample(candidates, k)
            sel = [{
                "id": ridx,
                "rule": rdict,
                "score": neutral_score,
                "enhanced_rationale": ""      # 不要 LLM rationale
            } for ridx, rdict in sampled]

        ColorText.print(
            f"[多属性-随机-{prop}] 候选 {cand_cnt} 条，随机取 {len(sel)} 条",
            ColorText.GREEN
        )

        # KS  ===================  这往下都不用变 ==================
        per_prop_selected[prop] = sel
        per_prop_keys[prop] = {_rule_key(item["rule"]) for item in sel}

        # 4) 构建 index_per_prop（与原函数一致）
        idx_map = {}
        for item in sel:
            rule_dict = item["rule"]
            k_edit = _rule_key(rule_dict)
            sc = float(item.get("score", 0.0))
            rec = idx_map.get(k_edit)
            if (rec is None) or (sc > rec.get("score", 0.0)):
                idx_map[k_edit] = {
                    "rule": rule_dict,
                    "score": sc,
                    "rationale": item.get("enhanced_rationale", "") or rule_dict.get("rationale", "")
                }
        index_per_prop[prop] = idx_map

    # 5) 多属性聚合：完全沿用原来的 _aggregate_multiobjective_rules
    selected = _aggregate_multiobjective_rules(
        props=props,
        per_prop_keys=per_prop_keys,
        index_per_prop=index_per_prop,
        topk_multi=topk_multi,
        use_direction_filter=True,
        prop2dir=prop2dir,
        thr=neutral_score,
        use_main_aux_score=use_main_aux_score,
        neutral_score=neutral_score,
        stage_label="多属性首轮-随机消融"
    )

    return {
        "property": "+".join(target_prop_lst),
        "mol": mol_smiles,
        "candidate_count": len(selected),
        "selected": selected
    }


# 多属性消融版本___不用映射表排序
def build_multiobjective_rulebank_for_mol_wotab(
    llm,
    mol_smiles,
    per_prop_rulebanks,
    target_prop_lst,
    target_direc_lst,
    topk_single=30,
    max_cand_count=300,
    topk_multi=15,
    use_main_aux_score=False,  # [NEW] 是否启用主属性+辅属性的 u_main + u_aux
    neutral_score=2.5          # [NEW] 中性分数，对应 score_scale=(0,5) 的中点
):
    """
    一轮多属性规则筛选（S0 -> 多目标）：
    返回:
      {
        "property": "prop1+prop2+...",
        "mol": mol_smiles,
        "candidate_count": N,
        "selected": [
          {"id": i, "rule": rule_dict, "score": score_multi, "enhanced_rationale": "..."},
          ...
        ]
      }
    """

    def _rule_key(rule_dict):
        # 用 edit_op 对齐多属性规则
        return rule_dict.get("edit_op")

    props = list(target_prop_lst)

    # 1) 每个属性单独跑单属性规则筛选
    per_prop_selected = {}   # {prop: [ {id, rule, score, enhanced_rationale}, ... ]}
    per_prop_keys = {}       # {prop: set(edit_op)}
    index_per_prop = {}      # {prop: {edit_op: {"rule":..., "score":..., "rationale":...}}}
    prop2dir = {p: d.lower() for p, d in zip(target_prop_lst, target_direc_lst)}  # 仅用于追踪方向

    # -------- 这里加 enumerate，用 idx 区分主/辅属性 --------
    for idx, (prop, direc) in enumerate(zip(target_prop_lst, target_direc_lst)):
        rb_prop = per_prop_rulebanks[prop]          # {"rules":[...]}
        rules_all = rb_prop["rules"]

        # 单属性首轮筛选
        if idx == 1:
            # 主属性：保持原来的调用方式

            res = filter_rules_wo_table(
                rb_prop,               # 支持 dict / list，内部 normalize
                mol_smiles,
                prop,
                llm,
                top_k=topk_single,
                max_cand_count=max_cand_count,
                chunk_size=len(rules_all),
                target_direction=direc,
                min_confidence=2
            )
        else:
            # 辅属性：根据 use_main_aux_score 决定是否启用“主/辅分支”
            if use_main_aux_score:
                res = filter_rules_wo_table(      # <<<
                    rb_prop,
                    mol_smiles,
                    prop,
                    llm,
                    top_k=topk_single,
                    max_cand_count=max_cand_count,
                    chunk_size=len(rules_all),
                    target_direction=direc,
                    min_confidence=2,
                    last=True                   # <<< 关键：只在辅属性上传 True
                )
            else:
                # 兼容旧逻辑，不传该参数
                res = filter_rules_wo_table(
                    rb_prop,
                    mol_smiles,
                    prop,
                    llm,
                    top_k=topk_single,
                    max_cand_count=max_cand_count,
                    chunk_size=len(rules_all),
                    target_direction=direc,
                    min_confidence=2
                )

        sel = res.get("selected", [])   # [ {id, rule, score, enhanced_rationale}, ... ]
        per_prop_selected[prop] = sel
        per_prop_keys[prop] = {_rule_key(item["rule"]) for item in sel}

        # 构建 index_per_prop
        idx_map = {}
        for item in sel:
            rule_dict = item["rule"]
            k = _rule_key(rule_dict)
            sc = float(item.get("score", 0.0))
            rec = idx_map.get(k)
            # 同一属性同一 edit_op，只保留 score 更高的那条
            if (rec is None) or (sc > rec.get("score", 0.0)):
                idx_map[k] = {
                    "rule": rule_dict,
                    "score": sc,   # 这里的 score 已经是“越大越好”
                    "rationale": item.get("enhanced_rationale", "") or rule_dict.get("rationale", "")
                }
        index_per_prop[prop] = idx_map

    # 2) 聚合多属性规则
    selected = _aggregate_multiobjective_rules(
        props=props,
        per_prop_keys=per_prop_keys,
        index_per_prop=index_per_prop,
        topk_multi=topk_multi,
        use_direction_filter=True,
        prop2dir=prop2dir,
        thr=neutral_score,
        use_main_aux_score=use_main_aux_score,
        neutral_score=neutral_score,
        stage_label="多属性首轮"
    )

    return {
        "property": "+".join(target_prop_lst),
        "mol": mol_smiles,
        "candidate_count": len(selected),
        "selected": selected
    }
# if __name__ == "__main__":
#     # 假设 llm 已在外部初始化且支持 .invoke(messages)
#     llm = choose_llm("gpt")
#     rules_input = load_rulebank("rules", "logp")  # 你的规则库
#     res = filter_rules_with_chain_of_thought(rules_input, "O=C(NC[C@H]1CCCO1)c1ccccc1N1CCCC1=O", "logp", llm, top_k=5)
#     print(json.dumps(res, indent=2, ensure_ascii=False))
#     pass
