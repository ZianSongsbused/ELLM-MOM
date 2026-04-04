import json

import requests

from utils.convenient_utils.suppress_useless_print import suppress_everything

# from utils.convenient_utils.wordart import ColorText

# 不同taskid的PDDS模板，替换占位符后直接作为第一轮的prompt
task_specification_dict_molecule = {
    101: "Can you make molecule SMILES_PLACEHOLDER more soluble in water? The output molecule should be similar to the input molecule.",
    102: "Can you make molecule SMILES_PLACEHOLDER less soluble in water? The output molecule should be similar to the input molecule.",
    103: "Can you make molecule SMILES_PLACEHOLDER more like a drug? The output molecule should be similar to the input molecule.",
    104: "Can you make molecule SMILES_PLACEHOLDER less like a drug? The output molecule should be similar to the input molecule.",
    105: "Can you make molecule SMILES_PLACEHOLDER higher permeability? The output molecule should be similar to the input molecule.",
    106: "Can you make molecule SMILES_PLACEHOLDER lower permeability? The output molecule should be similar to the input molecule.",
    107: "Can you make molecule SMILES_PLACEHOLDER with more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    108: "Can you make molecule SMILES_PLACEHOLDER with more hydrogen bond donors? The output molecule should be similar to the input molecule.",

    201: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    202: "Can you make molecule SMILES_PLACEHOLDER less soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    203: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    204: "Can you make molecule SMILES_PLACEHOLDER less soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    205: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and higher permeability? The output molecule should be similar to the input molecule.",
    206: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and lower permeability? The output molecule should be similar to the input molecule.",
}
# redf时最后一句再强调一下优化目标，防止llm不知道要干什么
task_specification_dict_molecule2 = {
    101: "which is more soluble in water? The output molecule should be similar to the input molecule.",
    102: "which is less soluble in water? The output molecule should be similar to the input molecule.",
    103: "which is more like a drug? The output molecule should be similar to the input molecule.",
    104: "which is less like a drug? The output molecule should be similar to the input molecule.",
    105: "which is higher permeability? The output molecule should be similar to the input molecule.",
    106: "which is lower permeability? The output molecule should be similar to the input molecule.",
    107: "with more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    108: "with more hydrogen bond donors? The output molecule should be similar to the input molecule.",

    201: "which is more soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    202: "which is less soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    203: "which is more soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    204: "which is less soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    205: "which is more soluble in water and higher permeability? The output molecule should be similar to the input molecule.",
    206: "which is more soluble in water and lower permeability? The output molecule should be similar to the input molecule.",
}

def generate_redf_prompt_with_direc(task, conversational_LLM, drug_type, generated_drug, closest_drug, expl_xr=None):
    """
    根据不同的 LLM 类型和药物类型生成 ReDF 提示。
    :param drug_type: 药物类型，可以是 'molecule', 'peptide', 或 'protein'。
    :param generated_drug: 当前生成的药物序列。
    :param closest_drug: 最接近的正确药物序列。
    :param expl_xr: 对 closest_drug 的解释（仅在 LLM 不是 Galactica 时使用）。
    :return: 生成的 ReDF 提示。
    """
    with suppress_everything():
        direction = test_prompt_from_api(closest_drug, "brics", task, False)  # redf的情况下那就不是第一轮了
    if conversational_LLM == "galactica":
        if drug_type == "molecule":
            if expl_xr is not None:
                prompt_ReDF = (
                    f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. '
                    f'We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. '
                    f'For the sequence [START_I_SMILES]{closest_drug}[END_I_SMILES], we know that {expl_xr}'
                    + direction +
                    f'Can you give me a new {drug_type}{task_specification_dict_molecule2[task]}\n\nAnswer:'
                )
            else:
                prompt_ReDF = (
                    f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. '
                    f'We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. '
                    + direction +
                    f'Can you give me a new {drug_type} {task_specification_dict_molecule2[task]}\n\nAnswer:'
                )
    else:
        if expl_xr is not None:
            prompt_ReDF = (
                f'The smiles you provided is not correct. We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. '
                f'For {closest_drug}, {expl_xr}' + direction + f'Can you give me a new {drug_type}  {task_specification_dict_molecule2[task]} The generated SMILES must be a valid chemical structure. And in your reply, there should only be the smiles themselves , without any explanations.'
            )
        else:
            prompt_ReDF = (
                f'The smiles you provided is not correct. We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. '
                + direction + f'Can you give me a new {drug_type}  {task_specification_dict_molecule2[task]} The generated SMILES must be a valid chemical structure. And in your reply, there should only be the smiles themselves , without any explanations.'
            )
    # print(prompt_ReDF)
    return prompt_ReDF

def generate_redf_prompt(task, conversational_LLM, drug_type, generated_drug, closest_drug, expl_xr=None):
    """
    根据不同的 LLM 类型和药物类型生成 ReDF 提示。
    :param drug_type: 药物类型，可以是 'molecule', 'peptide', 或 'protein'。
    :param generated_drug: 当前生成的药物序列。
    :param closest_drug: 最接近的正确药物序列。
    :param expl_xr: 对 closest_drug 的解释（仅在 LLM 不是 Galactica 时使用）。
    :return: 生成的 ReDF 提示。
    """
    if conversational_LLM == "galactica":
        if drug_type == "molecule":
            if expl_xr is not None:
                prompt_ReDF = (
                    f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. '
                    f'We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. '
                    f'For the sequence [START_I_SMILES]{closest_drug}[END_I_SMILES], we know that {expl_xr}'
                    f'Can you give me a new {drug_type} {task_specification_dict_molecule2[task]}\n\nAnswer:'
                )
            else:
                prompt_ReDF = (
                    f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. '
                    f'We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. '
                    f'Can you give me a new {drug_type} {task_specification_dict_molecule2[task]}\n\nAnswer:'
                )
    else:
        if expl_xr is not None:
            prompt_ReDF = (
                f'Your provided sequence {generated_drug} is not correct. '
                f'We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. '
                f'For {closest_drug}, {expl_xr} Can you give me a new {drug_type}  {task_specification_dict_molecule2[task]}'
            )
        else:
            prompt_ReDF = (
                f'Your provided sequence {generated_drug} is not correct. '
                f'We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. '
                f'Can you give me a new {drug_type}  {task_specification_dict_molecule2[task]}'
            )
    # print(prompt_ReDF)
    return prompt_ReDF


# 这个函数也类似，只有小分子就一个分支了，目的是加载PDDS模板
def get_task_specification_dict(task):
    return task_specification_dict_molecule

# 根据sme返回的rusult构建提示
def get_direction_prompt_sme(smiles, results, ratio=0.3, is_first_round=False, task_type=102):
    """
    构造关于子结构贡献的英文自然语言 Prompt。
    参数:
    - smiles: SMILES 字符串
    - results: 子结构归因分析结果 (List[Dict])
    - sub_type: 子结构划分方式
    - ratio: 选取的前百分比（按绝对值归一化贡献排序）

    返回:
    - prompt 字符串（英文）
    双属性通过拼接单属性实现，所以没有单独写双属性的分支
    """

    # === 任务目标的描述 ===
    target_map = {101: "more soluble in water", 102: "less soluble in water", 103: "more like a drug",
                  104: "less like a drug", 105: "higher permeability", 106: "lower permeability",
                  107: "more hydrogen bond acceptors", 108: "more hydrogen bond donors"}
    target_text = target_map.get(task_type, "the overall property")

    total_attr = sum(abs(r["attribution"]) for r in results) or 1e-6  # 防止除0
    # 正向 / 负向归因划分
    pos = [r for r in results if r["attribution"] > 0]
    neg = [r for r in results if r["attribution"] < 0]

    # 选取前 top-k 个（按绝对值排序）
    k_pos = max(1, int(len(pos) * ratio))
    k_neg = max(1, int(len(neg) * ratio))

    top_pos = sorted(pos, key=lambda r: abs(r["attribution"]), reverse=True)[:k_pos]
    top_neg = sorted(neg, key=lambda r: abs(r["attribution"]), reverse=True)[:k_neg]
    canon_smiles = results[0]["canonical_smiles"]

    prompt_parts = []
    if task_type == 101:   # 更低的水溶性
        if not is_first_round:
            print(101)
            for r in top_neg:
                ratio_display = f"{abs(r['attribution']) / total_attr:.2%}"
                line = (f"- The substructure '{r['sub_name']}' contributes positively towards the goal of {target_text}, "
                        f"with an attribution value of {r['attribution']:.4f}({ratio_display} of total attribution)")
                prompt_parts.append(line)   # 分析的是参考分子，所以是添加结构
            prompt_parts.append("You are not required to keep the fragments above, but you may optionally take them into consideration. ")

    if task_type == 102:   # 更低的水溶性
        if is_first_round:  # 第一轮：选负值向归因(因为logP越低水溶性越高)，这里挑出让原分子水溶性更高的部分
            for r in top_neg:
                ratio_display = f"{abs(r['attribution']) / total_attr:.2%}"
                line = (f"- The substructure '{r['sub_name']}' contributes negatively towards the goal of {target_text}, "
                        f"with an attribution value of {r['attribution']:.4f}({ratio_display} of total attribution)")
                prompt_parts.append(line)   # 分析的是原始分子，所以是替换结构
            prompt_parts.append("You may consider modifying or replacing these unfavorable substructures.")
        else:
            print(102)
            # 后续轮次：选正值归因
            for r in top_pos:
                ratio_display = f"{abs(r['attribution']) / total_attr:.2%}"
                line = (f"- The substructure '{r['sub_name']}' contributes positively towards the goal of {target_text}, "
                        f"with an attribution value of {r['attribution']:.4f}({ratio_display} of total attribution)")
                prompt_parts.append(line)   # 分析的是参考分子，所以是添加结构
            prompt_parts.append("You are not required to keep the fragments above, but you may optionally take them into consideration. ")
    elif task_type == 103:   # 更高的类药性
        if not is_first_round:  # 非第一轮
            print(103)
            for r in top_pos:  # 选正归因
                ratio_display = f"{abs(r['attribution']) / total_attr:.2%}"
                line = (f"- The substructure '{r['sub_name']}' contributes positively towards the goal of {target_text}, "
                        f"with an attribution value of {r['attribution']:.4f}({ratio_display} of total attribution)")
                prompt_parts.append(line)  # 分析的是参考分子，所以是添加结构
            prompt_parts.append("You are not required to keep the fragments above, but you may optionally take them into consideration. ")
    elif task_type == 105:   # 更高的渗透性
        if not is_first_round:
            print(105)
            for r in top_neg:
                ratio_display = f"{abs(r['attribution']) / total_attr:.2%}"
                line = (f"- The substructure '{r['sub_name']}' contributes positively towards the goal of {target_text}, "
                        f"with an attribution value of {r['attribution']:.4f}({ratio_display} of total attribution)")
                prompt_parts.append(line)   # 分析的是参考分子，所以是添加结构
            prompt_parts.append("You are not required to keep the fragments above, but you may optionally take them into consideration. ")
    # 拼接总提示词
    summary = (
        f"\nAnd after converting the input molecule {smiles} to its canonical SMILES representation: {canon_smiles}, "
        "we performed a substructure-based attribution analysis. The most influential substructures are listed below. "
        f"These substructure may be relevant for {target_text}, but you are encouraged to use your own creativity to redesign the molecule"
    )
    full_prompt = summary + "\n" + "\n".join(prompt_parts)

    return full_prompt




def test_prompt_from_api(smiles, sub_type, task_type, is_first_round):
    model_type, model_type1, model_type2 = "", "", "",
    if task_type == 102 or task_type == 101:
        model_type="MolLogP"
    elif task_type == 103 or task_type == 104:
        model_type="QED"
    elif task_type == 105 or task_type == 106:
        model_type = "TPSA"
    elif task_type == 107 :
        model_type = "NumHAcceptors"
    elif task_type == 108:
        model_type = "NumHDonors"
    elif task_type == 205:
        model_type1, model_type2 = "MolLogP", "TPSA"


    url = "http://localhost:7891/explain"
    if task_type // 100 == 1:
        payload = {"smiles": smiles, "sub_type": sub_type, "task_name": model_type}

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = json.loads(response.text)

        if data.get("success") and "results" in data:
            results = data["results"]
            prompt = get_direction_prompt_sme(smiles, results, 0.1, is_first_round, task_type)  # 根据返回值构建提示文本
            print(prompt)
            return prompt
        else:
            print("❌ 接口返回失败:", data)
            return None
    if task_type // 100 == 2:    # 双属性优化时
        task_type_single1, task_type_single1 = None, None
        if task_type == 205:
            task_type_single1, task_type_single2 = 101, 105

        payload1 = {"smiles": smiles, "sub_type": sub_type, "task_name": model_type1}
        payload2 = {"smiles": smiles, "sub_type": sub_type, "task_name": model_type2}

        response1 = requests.post(url, json=payload1)
        response1.raise_for_status()
        data1 = json.loads(response1.text)

        response2 = requests.post(url, json=payload2)
        response2.raise_for_status()
        data2 = json.loads(response2.text)

        if data1.get("success") and "results" in data1:
            results = data1["results"]
            prompt1 = get_direction_prompt_sme(smiles, results, 0.1, is_first_round, task_type_single1)  # 根据返回值构建提示文本
            print(prompt1)
        else:
            print("❌ 接口返回失败:", data1)
            return None

        if data2.get("success") and "results" in data2:
            results = data2["results"]
            prompt2 = get_direction_prompt_sme(smiles, results, 0.1, is_first_round, task_type_single2)  # 根据返回值构建提示文本
            print(prompt2)
        else:
            print("❌ 接口返回失败:", data2)
            return None
        return prompt1 + prompt2

if __name__ == '__main__':
    test_prompt_from_api("CC(=O)OC1=CC=CC=C1C(=O)O", "brics", 102, False)