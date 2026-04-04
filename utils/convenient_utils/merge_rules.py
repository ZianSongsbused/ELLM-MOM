import os
import json
from collections import defaultdict


def load_and_union_rulebanks(rules_dir="./rules", props=None):
    """
    从 rules_dir 中读取多个属性的规则库，按下面规则做并集合并：

    合并条件：
      - edit_op 完全相同
      - direction 完全相同
      - smarts_trigger 至少有一个 SMARTS 一样，或者双方 smarts_trigger 都为空（链长类）

    合并后规则：
      - smarts_trigger: 只保留在 >=2 条规则中出现过的 SMARTS；
                        如果簇里全是空，就保持 []。
      - rationale: 按 “属性名：原rationale” 的形式拼接，多条之间用换行
      - confidence: 所有被合并规则的 confidence 取平均（无则按 3）
      - id: 重新编号 R1, R2, ...
      - properties: 该合并规则来源的属性列表（如 ["logp","qed"]）
    """
    # ---------- 1. 决定要加载哪些属性 ----------
    if props is None:
        props = []
        for fname in os.listdir(rules_dir):
            if fname.endswith(".json"):
                prop_name = os.path.splitext(fname)[0]  # "logp.json" -> "logp"
                props.append(prop_name)

    print(f"[union] 将合并属性：{props}")

    # ---------- 2. 按属性读取规则，记录来源 ----------
    all_rules = []  # 每项: dict(rule) + "_prop": 属性名
    for prop in props:
        path = os.path.join(rules_dir, f"{prop}.json")
        if not os.path.exists(path):
            print(f"[warn] 规则文件不存在: {path}，跳过")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "rules" in data:
            rules = data["rules"]
        elif isinstance(data, list):
            rules = data
        else:
            print(f"[warn] 文件结构异常(既不是dict也不是list): {path}，跳过")
            continue

        print(f"[load] {prop}: 读取到 {len(rules)} 条规则")

        for r in rules:
            r = dict(r)  # 拷贝
            r["_prop"] = prop
            # 统一 smarts_trigger 类型为 list[str]
            smt = r.get("smarts_trigger") or []
            if isinstance(smt, str):
                smt = [smt]
            r["smarts_trigger"] = [s for s in smt if s]  # 去掉空字符串
            all_rules.append(r)

    print(f"[union] 合并前总规则数: {len(all_rules)}")

    # ---------- 3. 先按 (edit_op, direction) 分组 ----------
    groups = defaultdict(list)  # key: (edit_op, direction) -> list[rule]
    for r in all_rules:
        key = (r.get("edit_op", ""), r.get("direction", None))
        groups[key].append(r)

    merged_rules = []
    new_id_counter = 1

    # ---------- 4. 在每个 (edit_op, direction) 组内按照 smarts 交集做“连通分量”合并 ----------
    for (edit_op, direction), rules in groups.items():
        if not rules:
            continue

        n = len(rules)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # 每条规则的 smarts 集合
        smarts_sets = [set(r["smarts_trigger"]) for r in rules]

        # ====== 这里是关键修改点 ======
        # 有共同 SMARTS，或者两边都为空（链长类规则），就union到同一簇
        for i in range(n):
            for j in range(i + 1, n):
                inter = smarts_sets[i] & smarts_sets[j]
                if inter or (not smarts_sets[i] and not smarts_sets[j]):
                    union(i, j)
        # =================================

        # 收集连通分量
        comp_index = defaultdict(list)
        for i in range(n):
            root = find(i)
            comp_index[root].append(i)

        # 每个簇生成一条合并规则
        for comp_ids in comp_index.values():
            cluster = [rules[i] for i in comp_ids]

            # === 新增：只保留“跨属性”的簇（至少两个不同属性）===
            properties_set = {r.get("_prop") or r.get("property") or "unknown_prop"
                              for r in cluster}
            if len(properties_set) < 2:
                # 如果你只想要严格“交集规则”，单属性簇直接丢弃
                continue
            # ===============================================

            # ---- smarts_trigger: 取在 >=2 条规则中出现过的 SMARTS ----
            from collections import defaultdict as ddict
            smarts_count = ddict(int)
            for r in cluster:
                for s in set(r["smarts_trigger"]):
                    smarts_count[s] += 1
            merged_smarts = [s for s, cnt in smarts_count.items() if cnt >= 2]

            # 如果簇里都是空 SMARTS，merged_smarts 就保持空；否则退回第一条
            if not merged_smarts:
                if any(r["smarts_trigger"] for r in cluster):
                    merged_smarts = cluster[0]["smarts_trigger"]
                else:
                    merged_smarts = []  # 纯链长簇，保持 []

            # ---- rationale: “prop: rationale” 拼接 ----
            rationale_lines = []
            for r in cluster:
                prop_name = r.get("_prop") or r.get("property") or "unknown_prop"
                rat = (r.get("rationale") or "").strip()
                if rat:
                    rationale_lines.append(f"{prop_name}: {rat}")
                else:
                    rationale_lines.append(f"{prop_name}: (no rationale)")
            merged_rationale = "\n".join(rationale_lines)

            # ---- confidence: 取均值 ----
            conf_vals = []
            for r in cluster:
                conf = r.get("confidence")
                try:
                    conf_vals.append(float(conf))
                except (TypeError, ValueError):
                    conf_vals.append(3.0)
            avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 3.0

            new_rule = {
                "id": f"R{new_id_counter}",
                "edit_op": edit_op,
                "smarts_trigger": merged_smarts,
                "direction": direction,
                "rationale": merged_rationale,
                "confidence": avg_conf,
                "properties": sorted(properties_set),
            }
            new_id_counter += 1
            merged_rules.append(new_rule)

    print(f"[union] 合并后规则数: {len(merged_rules)}")

    return {"property": "multi", "rules": merged_rules}


# # 只合并 logp + qed
# rb_lq = load_and_union_rulebanks("./rules", props=["logp", "NumHAcceptors"])
# print(len(rb_lq["rules"]))
# rb_lq = load_and_union_rulebanks("./rules", props=["logp", "NumHDonors"])
# print(len(rb_lq["rules"]))
# rb_lq = load_and_union_rulebanks("./rules", props=["logp", "tspa"])
# print(len(rb_lq["rules"]))


