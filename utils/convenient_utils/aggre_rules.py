from utils.convenient_utils.wordart import ColorText


###############################################
# 公共工具函数：多属性规则聚合（交集/并集 + trigger 交集）
###############################################
# def _aggregate_multiobjective_rules(props, per_prop_keys, index_per_prop,
#                                     topk_multi=15, use_direction_filter=False, prop2dir=None, thr=2.5,
#                                     use_main_aux_score=False,  # [NEW] 是否启用 “主属性+辅属性” 的 u_main+u_aux 评分
#                                     neutral_score=2.5,  # [NEW] 中性分数（首轮是 2.5）
#                                     stage_label="multiobj"
# ):
#     """
#     @param props:                       list[str]，属性列表
#     @param per_prop_keys:               {prop: set(edit_op)}，每个属性下出现过的 edit_op
#     @param index_per_prop:              {prop: {edit_op: {"rule": rule_dict, "score": float, "rationale": str}}}
#     @param topk_multi:                  最后返回的条数
#     @param use_direction_filter:        是否按方向阈值过滤（首轮 True，后续轮 False）
#     @param prop2dir:                    {prop: "increase"/"decrease"}, 首轮方向过滤用
#     @param thr:                         单属性阈值（首轮用）
#     @param use_main_aux_score:          [NEW] 为 True 时，使用 “主属性+辅属性” u_main + u_aux 的评分方案
#                                         约定 props[0] 是主属性，其余属性都是辅属性
#     @param neutral_score:               [NEW] 中性分数（例如首轮 0~5 打分时是 2.5）
#     @param stage_label:                 日志前缀（例如 "多属性首轮" / "多属性后续轮"）
#
#     返回:
#       selected: [
#         {"id": i, "rule": rule_dict, "score": score_multi, "enhanced_rationale": "..."},
#         ...
#       ]
#     """
#     # [NEW] 主属性：默认取 props[0]，其余都是辅属性
#     main_prop = props[0] if (use_main_aux_score and props) else None
#
#     # 1) 先用 edit_op 的【交集】 keys
#     inter_keys = set(per_prop_keys[props[0]])
#     for p in props[1:]:
#         inter_keys &= per_prop_keys[p]      # &=交集
#
#     # 2) 再准备【并集】 keys 备用
#     union_keys = set()
#     for p in props:
#         union_keys |= per_prop_keys[p]      # |=并集
#
#     def merge_once(keys_to_use, use_union: bool):
#         merged_local = []
#
#         for key in keys_to_use:
#             scores, rationales = {}, {}        # {prop: score 或 None}, {prop: rationale 或 ""} 这两个字段后续单独处理
#
#             smarts_sets = {}                   # {prop: set(smarts_trigger)}
#             all_empty_smarts = True            # smarts_trigger是否全空
#             has_rule_for_all = True            # 交集模式下要求所有属性都有这条 edit_op，并集下为false
#
#             # ① 收集各属性下该 edit_op 的信息
#             for prop in props:
#                 rec = index_per_prop[prop].get(key)   # 这里的key就是editop，函数外定义
#                 if rec is None:
#                     has_rule_for_all = False          # 只要有一个属性没有这个key就不能用交集了
#                     scores[prop], rationales[prop] = None, ""    # 这俩字段也得对应置空
#                     continue
#
#                 # 当前字段能找到交集
#                 rule_dict = rec["rule"]
#                 sc = float(rec.get("score", 0.0))
#                 scores[prop] = sc
#                 rationales[prop] = rec.get("rationale", "") or rule_dict.get("rationale", "")
#
#                 s_list = rule_dict.get("smarts_trigger") or []
#                 s_set = set(s_list)
#                 smarts_sets[prop] = s_set
#                 if s_set:
#                     all_empty_smarts = False
#
#             # 交集模式：某个属性缺这条 edit_op，直接丢弃
#             if (not use_union) and (not has_rule_for_all):
#                 continue
#
#             # ② smarts_trigger 交集
#             if smarts_sets:
#                 common_smarts = None
#                 for sc in smarts_sets.values():
#                     common_smarts = set(sc) if common_smarts is None else (common_smarts & sc)
#             else:
#                 common_smarts = set()
#
#             # common_smarts 为空时，仅在“所有属性都有该规则且所有 trigger 全空”时视为链长类规则，允许保留
#             if not common_smarts:
#                 if has_rule_for_all and all_empty_smarts:
#                     pass
#                 else:
#                     continue
#
#             # ③ 首轮可选：按方向阈值过滤
#             if use_direction_filter:
#                 all_good = True
#                 for prop in props:
#                     rec = index_per_prop[prop].get(key)
#                     if rec is None:
#                         continue
#                     sc = float(rec.get("score", 0.0))
#                     direc = (prop2dir or {}).get(prop, "increase")
#                     if direc == "increase":
#                         if sc <= thr:
#                             all_good = False
#                     elif direc == "decrease":
#                         if sc >= thr:
#                             all_good = False
#                     else:
#                         all_good = False
#                 if not all_good:
#                     continue
#
#             # ④ 多属性聚合打分：对有效属性取均值 × 覆盖率
#             if use_main_aux_score and (main_prop is not None):    # 使用”主属性+副属性“的模式
#                 # 属性 + 辅属性的 u_main + u_aux 评分方案
#                 sc_main = scores.get(main_prop)
#                 if sc_main is None:
#                     continue
#
#                 # 主属性：越【偏离】中性越好  u_main = |s_main - neutral|
#                 u_main = abs(sc_main - neutral_score)
#
#                 # 辅属性：越【接近】中性越好  u_aux = -|s_aux - neutral|
#                 u_aux_total = 0.0
#                 for prop in props:
#                     if prop == main_prop:
#                         continue
#                     sc = scores.get(prop)
#                     if sc is None:
#                         continue
#                     u_aux_total += -abs(sc - neutral_score)   # 因为副属性是越接近越好，所以用负值表示
#
#                 score_multi = u_main + u_aux_total    #
#             else:
#                 valid_scores = [v for v in scores.values() if v is not None]
#                 if not valid_scores:
#                     continue
#                 avg_score = float(sum(valid_scores)) / len(valid_scores)
#                 coverage = len(valid_scores) / float(len(props)) if props else 1.0
#                 score_multi = avg_score * coverage
#
#             # ⑤ rationale 拼接
#             parts = []
#             for prop in props:
#                 rt = (rationales.get(prop) or "").strip()
#                 if rt:
#                     parts.append(f"{prop}: {rt}")
#             combined_rationale = " | ".join(parts)
#
#             # ⑥ 从任一属性拿 rule_dict 副本，并覆盖 smarts_trigger 为交集
#             rule_dict = None
#             for prop in props:
#                 rec = index_per_prop[prop].get(key)
#                 if rec is not None:
#                     rule_dict = rec["rule"].copy()
#                     break
#             if rule_dict is None:
#                 continue
#
#             if common_smarts:
#                 rule_dict["smarts_trigger"] = sorted(common_smarts)
#             else:
#                 rule_dict["smarts_trigger"] = []
#
#             merged_local.append({
#                 "rule": rule_dict,
#                 "score_multi": score_multi,
#                 "score_per_prop": scores,
#                 "enhanced_rationale": combined_rationale
#             })
#
#         return merged_local
#
#     # 3) 优先走交集模式
#     merged = merge_once(inter_keys, use_union=False)
#
#     # 如果交集模式在 trigger/阈值之后一个规则都没有，再退化到并集模式
#     if len(merged) < 1:
#         merged = merge_once(union_keys, use_union=True)
#         ColorText.print(f"{stage_label}使用【并集模式】，最终规则数={len(merged)}", ColorText.PURPLE)
#     else:
#         ColorText.print(f"{stage_label}使用【交集模式】，最终规则数={len(merged)}", ColorText.GREEN)
#
#     # 4) 排序截断并包装 selected
#     merged.sort(key=lambda x: x["score_multi"], reverse=True)
#     merged = merged[:topk_multi]
#
#     selected = []
#     for i, item in enumerate(merged):
#         selected.append({
#             "id": i,
#             "rule": item["rule"],
#             "score": item["score_multi"],
#             "enhanced_rationale": item["enhanced_rationale"]
#         })
#     return selected


def _aggregate_multiobjective_rules(
    props,
    per_prop_keys,
    index_per_prop,
    topk_multi=15,
    use_direction_filter=False,
    prop2dir=None,            # [保留参数，但不再用来翻转阈值，仅兼容旧接口]
    thr=2.5,
    use_main_aux_score=False, # 主属性+辅属性 u_main + u_aux
    neutral_score=2.5,
    stage_label="multiobj"
):
    """
    @param props:             list[str]，属性列表
    @param per_prop_keys:     {prop: set(edit_op)}，每个属性下出现过的 edit_op
    @param index_per_prop:    {prop: {edit_op: {"rule": rule_dict, "score": float, "rationale": str}}}
    @param topk_multi:        最后返回的条数
    @param use_direction_filter:
                              是否做 score 阈值过滤（首轮 True，后续轮 False）
                              新语义：score 已经编码了 increase/decrease 方向，
                              统一用 score > thr 判定“对该属性有利”
    @param prop2dir:          兼容参数，不再用于逻辑
    @param thr:               单属性“中性”阈值（例如 score_scale=(0,5) 时的 2.5）
    @param use_main_aux_score:
                              为 True 时，使用“主属性+辅属性”的 u_main + u_aux 评分方案；
                              约定 props[0] 是主属性，其余为辅属性。
    @param neutral_score:     中性分数（例如 2.5）
    @param stage_label:       日志前缀

    返回:
      selected: [
        {"id": i, "rule": rule_dict, "score": score_multi, "enhanced_rationale": "..."},
        ...
      ]
    """
    # [NEW] 主属性：约定 props[0] 是主属性（仅在 use_main_aux_score=True 时生效）
    main_prop = props[0] if (use_main_aux_score and props) else None

    # 1) 先用 edit_op 的【交集】 keys
    inter_keys = set(per_prop_keys[props[0]])
    for p in props[1:]:
        inter_keys &= per_prop_keys[p]      # &= 交集

    # 2) 再准备【并集】 keys 备用
    union_keys = set()
    for p in props:
        union_keys |= per_prop_keys[p]      # |= 并集

    def merge_once(keys_to_use, use_union: bool):
        merged_local = []

        for key in keys_to_use:
            scores, rationales = {}, {}        # {prop: score 或 None}, {prop: rationale 或 ""}
            smarts_sets = {}                   # {prop: set(smarts_trigger)}
            all_empty_smarts = True            # 是否所有 trigger 都为空
            has_rule_for_all = True            # 交集模式下要求所有属性都有该 edit_op

            # ① 收集各属性下该 edit_op 的信息
            for prop in props:
                rec = index_per_prop[prop].get(key)
                if rec is None:
                    has_rule_for_all = False
                    scores[prop], rationales[prop] = None, ""
                    continue

                rule_dict = rec["rule"]
                sc = float(rec.get("score", 0.0))
                scores[prop] = sc
                rationales[prop] = rec.get("rationale", "") or rule_dict.get("rationale", "")

                s_list = rule_dict.get("smarts_trigger") or []
                s_set = set(s_list)
                smarts_sets[prop] = s_set
                if s_set:
                    all_empty_smarts = False

            # 交集模式：有属性缺这条 edit_op，则直接丢弃
            if (not use_union) and (not has_rule_for_all):
                continue

            # ② smarts_trigger 交集
            if smarts_sets:
                common_smarts = None
                for s in smarts_sets.values():
                    common_smarts = set(s) if common_smarts is None else (common_smarts & s)
            else:
                common_smarts = set()

            # common_smarts 为空时，仅在“所有属性都有该规则且所有 trigger 全空”时视为链长类规则
            if not common_smarts:
                if has_rule_for_all and all_empty_smarts:
                    pass
                else:
                    continue

            # ③ 可选：按 score 阈值过滤
            if use_direction_filter:
                # 统一语义：score 已含方向信息，无论 increase / decrease，
                #           “score > thr” 才视为对该属性有利。
                all_good = True
                for prop in props:
                    sc = scores.get(prop, None)
                    if sc is None:
                        # 并集模式下允许某些属性缺失
                        continue
                    if sc <= thr:          # 有一个不满足阈值的就剔除这条规则
                        all_good = False
                        break
                if not all_good:
                    continue

            # ④ 多属性聚合打分
            if use_main_aux_score and (main_prop is not None):
                # 主属性 + 辅属性模式：
                #   u_main = |s_main - neutral|           （越偏离中性越好）
                #   u_aux  = - Σ |s_aux - neutral|        （越接近中性越好）
                sc_main = scores.get(main_prop)
                if sc_main is None:
                    # 没有主属性评分，直接丢弃
                    continue

                u_main = abs(sc_main - neutral_score)

                u_aux_total = 0.0
                for prop in props:
                    if prop == main_prop:
                        continue
                    sc = scores.get(prop)
                    if sc is None:
                        continue
                    u_aux_total += -abs(sc - neutral_score)

                score_multi = u_main + u_aux_total
            else:
                # 旧逻辑：平均分 × 覆盖率
                valid_scores = [v for v in scores.values() if v is not None]
                if not valid_scores:
                    continue
                avg_score = float(sum(valid_scores)) / len(valid_scores)
                coverage = len(valid_scores) / float(len(props)) if props else 1.0
                score_multi = avg_score * coverage

            # ⑤ rationale 拼接
            parts = []
            for prop in props:
                rt = (rationales.get(prop) or "").strip()
                if rt:
                    parts.append(f"{prop}: {rt}")
            combined_rationale = " | ".join(parts)

            # ⑥ 从任一属性拿 rule_dict 副本，并覆盖 smarts_trigger 为交集
            rule_dict = None
            for prop in props:
                rec = index_per_prop[prop].get(key)
                if rec is not None:
                    rule_dict = rec["rule"].copy()
                    break
            if rule_dict is None:
                continue

            if common_smarts:
                rule_dict["smarts_trigger"] = sorted(common_smarts)
            else:
                rule_dict["smarts_trigger"] = []

            merged_local.append({
                "rule": rule_dict,
                "score_multi": score_multi,
                "score_per_prop": scores,
                "enhanced_rationale": combined_rationale
            })

        return merged_local

    # 3) 优先走交集模式
    merged = merge_once(inter_keys, use_union=False)

    # 如果交集模式在 trigger/阈值之后一个规则都没有，再退化到并集模式
    if len(merged) < 1:
        merged = merge_once(union_keys, use_union=True)
        ColorText.print(f"{stage_label}使用【并集模式】，最终规则数={len(merged)}", ColorText.PURPLE)
    else:
        ColorText.print(f"{stage_label}使用【交集模式】，最终规则数={len(merged)}", ColorText.GREEN)

    # 4) 排序截断并包装 selected
    merged.sort(key=lambda x: x["score_multi"], reverse=True)
    merged = merged[:topk_multi]

    selected = []
    for i, item in enumerate(merged):
        selected.append({
            "id": i,
            "rule": item["rule"],
            "score": item["score_multi"],
            "enhanced_rationale": item["enhanced_rationale"]
        })
    return selected
