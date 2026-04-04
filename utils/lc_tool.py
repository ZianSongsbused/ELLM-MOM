import json
import time

import requests
from sklearn.metrics.pairwise import cosine_similarity
from utils.convenient_utils.test_timeout import run_with_timeout
from utils.convenient_utils.wordart import ColorText

from utils.main_utils import sim_sequence
from utils.process_drug_utils.mol_parsetool import parse
from utils.process_drug_utils.moledit_evaltool import evaluate
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from langchain.schema import HumanMessage, AIMessage
from typing import List, Union

from utils.retrievingDB_tool import task2prop, compute_final_embedding


def is_valid_smiles(smiles, max_atoms=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= max_atoms

# 1 DB排序仅依赖tanimoto相似度的ReDF
def retrieve_and_feedback(task, DB, input_drug, generated_drug, constraint):
    """
    :param task:               实验的任务序号
    :param DB:
    :param input_drug:         本轮的输入药物
    :param generated_drug:     llm当前轮次生成的药物，即x浪
    :param constraint:         loose/strict
    :param threshold_dict:     好像没用到
    :return:                   retrieval的分子，与input的相似度
    """
    # KS  arg max<x~，xR'>
    sim_DB = DB.copy()  # 创建数据库 DB 的副本，避免修改原始数据
    sim_list = []
    for index, row in sim_DB.iterrows():
        smiles = row['sequence'].replace('\n', '')           # 遍历数据库中每个分子
        sim = sim_sequence(task, smiles, generated_drug)    # 评估当前分子(DB中，即XR')与generated_drug的相似度
        sim_list.append(sim)                                # DB中的分子与generated_drug的相似度列表

    sim_DB['sim'] = sim_list                                  # 把相似度列表赋给sim_DB的sim列
    sim_DB = sim_DB.sort_values(by=['sim'], ascending=False)  # 降序排序，这样下面从第一个分子开始匹配，第一个匹配到的就是相似度最大的

    # KS  D(Xin,XR';Xt)
    ColorText.print("开始搜索满足Δ的XR‘", ColorText.PURPLE, ColorText.REVERSE)
    for index, row in sim_DB.iterrows():
        # 实参row['sequence'].replace('\n','')对应形参generated_drug，这里评估的实际是每个XR'与input是不是满足①合规②属性阈值
        answer, delta = evaluate(input_drug, row['sequence'].replace('\n', ''), task, constraint)
        sim = row['sim']
        # answer这里代表的就是D(~,~;~)的结果
        if answer:  # 满足了constraint
            ColorText.print(f"当前XR'满足Δ【D(~,~;~)=True】,Δ={delta}", ColorText.GREEN)
            return row['sequence'].replace('\n', ''), sim  # 找到第一个符合D(~,~;~)的就返回
        elif answer == 0:
            pass                                            # 不满足D，继续看下一个DB分子
        else:                                               # answer=-1，DB分子有问题
            ColorText.print("当前XR'有问题或为空(无法计算D【~,~;~)】", ColorText.RED)
    return None, 0  # 一个符合D(~,~;~)的都没有


# 1 DB排序依赖特征向量的ReDF
def retrieve_and_feedback_vemb(task, DB, DB_embeddings, input_drug, generated_drug, constraint):
    """"
    和上面的函数比就多了这一个形参，返回值也没变
    :param DB_embeddings: DB中所有分子的embedding矩阵 (n_samples, feature_dim)  # 多出来的参数
    """
    print(f"{' '*5}" + "执行retrieval".center(28, '-'))
    # KS  arg max<x~，xR'>
    # ①： 先计算 generated_drug 的embedding （这里的prop只和算embedding有关，但吧dimenet删掉了其实有没有都没用了）
    prop = task2prop(task)

    # 有时候生成的rdkit有问题会卡死，这里加一个100s超时就跳过的逻辑。
    embedding = run_with_timeout(compute_final_embedding, args=(generated_drug, prop), timeout=100)
    # embedding = compute_final_embedding(generated_drug, prop)
    if embedding is None:
        ColorText.print("计算嵌入超时，跳过该分子", ColorText.RED)
        return None, 0
    generated_embedding = embedding.reshape(1, -1)

    # ②: 计算 generated_drug 与 DB 中所有分子的余弦相似度
    ColorText.print(f"{' ' * 5}|" + "查询分子嵌入计算完成，开始比对生成分子与DB分子的嵌入相似度".center(28, '-'), ColorText.BLUE)
    sim_list = cosine_similarity(generated_embedding, DB_embeddings)[0]  # shape (n_samples, )
    # ③: 把相似度加到DB上，并排序
    sim_DB = DB.copy()
    sim_DB['sim'] = sim_list
    sim_DB = sim_DB.sort_values(by=['sim'], ascending=False)  # 降序排列，相似度高的排前面

    # KS  D(Xin,XR'; Xt) 这后边就和tanimoto版本一样了
    ColorText.print(f"{' ' * 5}|" + "开始搜索满足Δ的XR‘".center(28, '-'), ColorText.CYAN)
    for index, row in sim_DB.iterrows():   # 遍历检查DB分子是否满足input_drug的约束条件
        answer, delta = evaluate(input_drug, row['sequence'].replace('\n', ''), task, constraint)
        sim = row['sim']
        if answer:  # 满足约束
            ColorText.print(f"{' ' * 5}|" + f"当前XR'满足Δ【D(~,~;~)=True】, Δ={delta}".center(28, '-'), ColorText.GREEN)
            return row['sequence'].replace('\n', ''), sim
        elif answer == 0:  # 不满足约束
            # ColorText.print(f"{' ' * 5}" + "当前XR'不满足D【~,~;~)】".center(28, '-'), ColorText.PURPLE)
            pass       # 直接查数据库也会吊这个，所以就不输出了
        else:       # 分子异常
            ColorText.print(f"{' ' * 5}|" + "当前XR'有问题或为空(无法计算D【~,~;~)】".center(28, '-'), ColorText.RED)
    print(f"{' ' * 5}|" + "end retrieval".center(28, '-'))
    return None, 0  # 没找到符合条件的


# 1 返回langchain的Tool需要的func形式
def make_retrieve_tool_lc(task, DB, DB_embeddings=None, constraint='loose', threshold_dict=None):
    def retrieve_tool_func(drug_json_str: str) -> str:
        drug_json = json.loads(drug_json_str)   # 解析json字符串
        input_drug, generated_drug = drug_json.get("input_drug"), drug_json.get("generated_drug")

        # 把retrieve_and_feedback封装成langchain.agents.Tool，langchain才能调用这个工具
        if DB_embeddings is None:     # DB_embeddings是none用tanimoto比
            result, similarity \
                    = retrieve_and_feedback(task, DB, input_drug, generated_drug, constraint)
        else:
            result, similarity \
                = retrieve_and_feedback_vemb(task, DB, DB_embeddings, input_drug, generated_drug, constraint)

        return json.dumps({"retrieved": result, "similarity": similarity})  # 错误时返回的是none和0

    # constraint是list的版本
    def retrieve_muti_tool_func(drug_json_str: str) -> str:
        drug_json = json.loads(drug_json_str)  # 解析json字符串
        input_drug, generated_drug = drug_json.get("input_drug"), drug_json.get("generated_drug")

        results = {}   # 把retrieve_and_feedback封装成langchain.agents.Tool，langchain才能调用这个工具
        for constr in constraint:
            if DB_embeddings is None:
                result, similarity = retrieve_and_feedback(task, DB, input_drug, generated_drug, constr)
            else:
                result, similarity = retrieve_and_feedback_vemb(task, DB, DB_embeddings, input_drug, generated_drug, constr)

            results[constr] = {"retrieved": result, "similarity": similarity}

        return json.dumps(results, ensure_ascii=False)

    if isinstance(constraint, str):
        return retrieve_tool_func
    else:
        return retrieve_muti_tool_func




# 2 在response中提取smiles串的工具
def make_parse_molecular_tool_lc(task, trial_index):
    def _tool_func(input_json: str) -> str:
        try:
            drug_json = json.loads(input_json) # 传入的json中应该包含input_drug、generated_text、closest_drug三个字段
            input_drug, generated_text, closest_drug \
                = drug_json.get("input_drug"), drug_json.get("generated_text"), drug_json.get("closest_drug")

            generated_drug_list = parse(
                task=task, input_drug=input_drug, generated_text=generated_text, addition_drug=closest_drug
            )  # closest_drug用来去重，因为X~和XR相同的情况也不正确

            if generated_drug_list is None:
                ColorText.print("---parse result: 失败，终止对话", ColorText.PURPLE)
                # record[input_drug]['skip_round'] = round_index
                return json.dumps({"status": -1, "generated_drug": None})
            elif len(generated_drug_list) == 0:
                ColorText.print("---parse result: 生成药物，但可能与之前的分子重复，继续下一轮", ColorText.PURPLE,
                                ColorText.UNDERLINE)
                # record[input_drug]['retrieval_conversation'][round_index]['answer'] = 'False'
                return json.dumps({"status": 0, "generated_drug": None})
            else:
                ColorText.print("---parse result: 成功，选择第trial_index个作为本轮结果", ColorText.PURPLE,
                                ColorText.ITALIC)
                # 从生成的药物列表 generated_drug_list 中最多取前5个，然后选出第 trial_index 个药物作为x~
                generated_drug = generated_drug_list[:min(len(generated_drug_list), 5)][trial_index]
                # print("Generated Result:" + str(generated_drug), file=f)
                # ColorText.print("Generated Result(第trial_index个):" + str(generated_drug), ColorText.GREEN)
                # record[input_drug]['retrieval_conversation'][round_index]['generated_drug'] = generated_drug
                return json.dumps({"status": 1, "generated_drug": generated_drug})
        except Exception as e:
            return json.dumps({"status": -2, "error": str(e)})


# 3 /|||||||||||||||||||||||||||| 0轮前用到的函数 ||||||||||||||||||||||||||||\
# 3 获取分子骨架
def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:   # 解析不了smile
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


# 3 判断两个分子骨架是否相同
def is_same_scaffold(smiles1: str, smiles2: str) -> bool:
    scaffold1 = get_scaffold(smiles1)
    scaffold2 = get_scaffold(smiles2)
    return scaffold1 == scaffold2


# 3 chatmol通过fastAPI调用，这是调用chatmol的方法
def molecule_understanding_tool_func(json_text: str) -> str:
    """
    :param json_text: JSON 字符串，格式如 {"input": "...", "task": "..."}
    :return: JSON 字符串，格式如 {"status": "ok", "result": [...]}
    """
    max_retry, delay = 3, 1
    url = "http://localhost:8001/run"
    for attempt in range(1, max_retry + 1):
        try:
            payload = json.loads(json_text)  # 确保是合法 JSON
            task = payload.get("task")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()["result"]
            # 处理返回类型
            if task == "generate":
                if isinstance(result, str):  # 如果chatmol返回了一个字符串
                    smiles_list = [s.strip() for s in result.replace("\n", ",").split(",") if s.strip()]
                    return json.dumps({"status": "1", "result": smiles_list})
                elif isinstance(result, list):  # 如果chatmol返回了一个列表
                    smiles_list = result
                    return json.dumps({"status": "1", "result": smiles_list})
                else:
                    if attempt < max_retry:
                        time.sleep(delay)
                    else:
                        smiles_list = []
                        return json.dumps({"status": "0", "result": smiles_list})
            elif task == "describe":
                return json.dumps({"status": "1", "result": str(result)})

        except Exception as e:
            if attempt < max_retry:
                time.sleep(delay)
            else:
                ColorText.print(f"[ChatMol 调用错误]: {str(e)}", ColorText.RED, ColorText.REVERSE)
                return json.dumps({"status": "-1", "message": str(e)})


# 3 从messages列表中提取出用户问题+大模型回答，构建成chatmol进行【分子生成】使用的输入
def extract_qa_from_messages(messages: List[Union[HumanMessage, AIMessage]]) -> str:
    """
    将交替的 HumanMessage 和 AIMessage 格式化为一问一答的字符串。
    """
    print(messages)
    result = []
    i = 0
    while i < len(messages) - 1:
        human = messages[i]
        ai = messages[i + 1]
        result.append(f"question: {human.content.strip()}\tanswer: {ai.content.strip()}")
    print(" ".join(result))
    return " ".join(result)






# 执行测试
if __name__ == "__main__":
    # 注意需要使用双引号才能被识别成json
    input_str = '{"input": "It derives from a tryptamine", "task": "generate"}'
    print(molecule_understanding_tool_func(input_str))
    input_str = '{"input": "CNCCC1=CNC2=CC=CC=C21", "task": "describe"}'
    print(molecule_understanding_tool_func(input_str))




