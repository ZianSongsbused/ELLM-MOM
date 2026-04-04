import argparse
import json
import os
import time
from rdkit import Chem

from utils.lc_tool import make_retrieve_tool_lc, is_same_scaffold, molecule_understanding_tool_func
from utils.convenient_utils.wordart import ColorText
from utils.main_utils import (load_retrieval_DB, choose_llm,
                              truncate_prompt, init_prompt_in_a_conversation, load_retrieval_DB2, load_dataset, init_prompt_in_a_conversation_with_direc,
                              apply_rules_from_cot, select_rules_for_prompt, build_multiobjective_rulebank_for_round)

from utils.process_drug_utils.moledit_prompt import get_task_specification_dict
from utils.process_drug_utils.mol_parsetool import parse_and_eval
from utils.process_drug_utils.moledit_evaltool import evaluate
from utils.convenient_utils.suppress_useless_print import suppress_everything
# langchain
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import re
from utils.rulebook_cot_tool import filter_rules_with_chain_of_thought, build_multiobjective_rulebank_for_mol
from utils.rulebook_tool import load_rulebank, save_rulebank, build_large_rulebank_batch


# log里已有的内容就先跳过了，省token。把最后一个部分（防止他没完成）去掉再写回文件
def process_log_file(file_path):
    # 检查文件是否存在/是否为空
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"文件 {file_path} 不存在/空。")
        return 0, 0
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()

    # 1. 匹配 Sample 的数量
    numall = len(re.findall(r">>>>> Sample", log_content))
    print(numall)
    # 2. 去掉最后一个 Sample 后面的内容
    last_sample_index = log_content.rfind(">>>>> Sample")
    if last_sample_index != -1:
        log_content = log_content[:last_sample_index]

    # 3. 提取倒数第二个 Sample 下的 num_correct
    second_last_sample_index = log_content.rfind(">>>>> Sample", 0, last_sample_index)
    if second_last_sample_index != -1:
        second_last_sample_content = log_content[second_last_sample_index:last_sample_index]
        num_correct_match = re.search(r"num_correct = (\d+)", second_last_sample_content)
        if num_correct_match:
            num_correct = int(num_correct_match.group(1))
        else:
            num_correct = None
    else:
        num_correct = None

    # 将处理后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(log_content)

    # 返回结果
    return numall, num_correct


# 根据taskid返回要处理的属性以及对应的修改方向
def prop_direc_from_taskid(taskid):
    if taskid == 101:
        return ["logp"], ["decrease"]
    elif taskid == 102:
        return ["logp"], ["increase"]
    elif taskid == 103:
        return ["qed"], ["increase"]
    elif taskid == 104:
        return ["qed"], ["decrease"]
    elif taskid == 105:
        return ["tpsa"], ["decrease"]
    elif taskid == 106:
        return ["tpsa"], ["increase"]
    elif taskid == 107:
        return ["NumHAcceptors"], ["increase"]
    elif taskid == 108:
        return ["NumHDonors"], ["increase"]
    elif taskid == 201:
        return ["logp", "NumHAcceptors"], ["decrease", "increase"]
    elif taskid == 202:
        return ["logp", "NumHAcceptors"], ["increase", "increase"]
    elif taskid == 203:
        return ["logp", "NumHDonors"], ["decrease", "increase"]
    elif taskid == 204:
        return ["logp", "NumHDonors"], ["increase", "increase"]
    elif taskid == 205:
        return ["logp", "tpsa"], ["decrease", "decrease"]
    elif taskid == 206:
        return ["logp", "tpsa"], ["increase", "increase"]


# 跳过结果记录文件已有的内容
def extract_input_drugs_from_file(filepath=None):
    """ 从文件中提取所有 'Sample X: input_drug' 后面的内容，返回一个列表，并删除最后一个 sample 的内容。"""
    if filepath is None:
        return []
    else:
        smiles_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r">>>>> Sample \d+:\s*(.+)", line.strip())
                if match:
                    smiles = match.group(1).strip()
                    smiles_list.append(smiles)

        # 删除最后一个（如果有），防止最后一个没生成完
        if smiles_list:
            smiles_list = smiles_list[:-1]

        return smiles_list


def main_lc(args, DB=None, DB_emb=None):
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 解析，并返回最好的一个结果 ||||||
    def parse_molecular_in_response(input_drug, generated_drug, closest_drug, record, constraint=None, f=None, direction=None):
        # 解析出LLM回答中的所有分子  # closest_drug用来去重，因为X~和XR相同的情况也不正确
        generated_drug_list = parse_and_eval(input_drug, closest_drug, generated_drug, fix_mol, task, constraint, True, direction)
        # KS  3 检查解析结果
        if generated_drug_list is None:
            ColorText.print("---parse result: 失败，终止对话", ColorText.RED, ColorText.REVERSE)
            record[input_drug]['skip_round'] = round_index  # skip_round对应的是result的值，所以不用改
            return -1, None
        elif len(generated_drug_list) == 0:
            ColorText.print("---parse result: 生成药物，但没有生成可用分子（可能是重复分子）", ColorText.PURPLE)
            record[input_drug]['retrieval_conversation'][round_index + 1]['answer'] = 'False'  # 下标round_index加了1，因为正式对话从下标1开始，此时round_index=0
            return 0, None
        else:
            ColorText.print("---parse result: 成功", ColorText.GREEN, ColorText.ITALIC)
            print(generated_drug_list)
            generated_drug = generated_drug_list[0]
            print("Generated Result:" + str(generated_drug), file=f)
            ColorText.print("Generated Result:" + str(generated_drug), ColorText.GREEN)
            record[input_drug]['retrieval_conversation'][round_index + 1]['generated_drug'] = generated_drug  # 也是下标问题，加了1
            return 1, generated_drug

    # |||||||||||||||||||||||||||||||||||||||||| 再过一下这个函数主要是为了补充record，懒得改了 ||||||
    def evaluate_generated_drug(input_drug, generated_drug, task, constraint, record, f=None):
        # KS  评估生成的药物：获得评估结果
        answer, delta = evaluate(input_drug, generated_drug, task, constraint)

        # KS  理评估结果：根据评估结果决定是否进入下一轮或终止
        if answer == -1:
            ColorText.print("---evaluation result: 失败，LLM生成分子不合法，无法进行后续retrieval，故终止对话", ColorText.PURPLE)
            record[input_drug]['skip_round'] = round_index  # skip_round对应的是result的值，所以不用改
            return -1, None
        print('Evaluation result is ' + str(answer), file=f)
        ColorText.print("---evaluation result(是否生成满足要求的药物)：" + str(answer), ColorText.GREEN)
        record[input_drug]['retrieval_conversation'][round_index + 1]['answer'] = str(answer)  # 下标问题，+1
        if answer:
            ColorText.print("---评估结果满足Δ约束，正常结束" + str(answer), ColorText.GREEN, ColorText.REVERSE)
            return 1, generated_drug
        else:  # answer==0
            ColorText.print(f"---评估不通过Δ约束，尝试使用ReDF再找新的药物" + str(answer), ColorText.PURPLE)
            return 0, generated_drug


    # ===============================================================================================================
    print(f"********************当前进程 PID: {os.getpid()}********************")
    # 加载预定义的参数
    constraint = ["loose", "strict"]
    drug_type, task, seed, llm_type, fix_mol, C = \
        'molecule', args['task'], args['seed'], args['conversational_LLM'], args['fix_mol'], args['C']
    is_emb_retri, is_chatmol_und, is_chatmol_gen, is_prequery = \
        args['is_emb_retri'], args['is_chatmol_und'], args['is_chatmol_gen'], str2bool(args['is_prequery'])
    ColorText.print(f"可选参数情况：\nLLM：{llm_type}，约束：{constraint}，是否用嵌入retrieval：{is_emb_retri}，"
                    f"是否加chatmol分子理解：{is_chatmol_und}，是否加chatmol分子生成：{is_chatmol_gen}，对话前是否查数据库：{is_prequery}",
                    ColorText.WHITE, ColorText.REVERSE)
    base_log, base_rec = os.path.splitext(args['log_file'])[0], os.path.splitext(args['record_file'])[0]
    log_loose, log_strict = f"{base_log}_loose.log", f"{base_log}_strict.log"
    rec_loose, rec_strict = f"{base_rec}_loose.json", f"{base_rec}_strict.json"
    task_specific_prompt_dict = get_task_specification_dict(task)  # 获取任务对应的提示模板


    # KS  加载输入药物列表和检索数据库
    if str2bool(is_emb_retri) and DB_emb is None:  # 按原逻辑随机选1w条（原始数据库总共是249455条）
        print("DB使用原逻辑")
        input_drug_list, retrieval_DB, DB_embeddings = load_retrieval_DB(task, seed, True)
    elif str2bool(is_emb_retri) and DB_emb is not None:  # 此时用的是整个数据库，所以数据库的嵌入就是固定的了，多次任务运行直接算一遍就可以
        print("DB全量嵌入数据库")
        task_specification_dict = get_task_specification_dict(task)  # 获得PDDS模板
        input_drug_list = load_dataset(drug_type, task, task_specification_dict)
        retrieval_DB, DB_embeddings = DB, DB_emb     # 这俩参数是main外边传进来的
        print(f"{len(input_drug_list)} {len(retrieval_DB)} {len(DB_embeddings)}")
    elif not str2bool(is_emb_retri):
        print("DB全量原始数据库")
        input_drug_list, retrieval_DB = load_retrieval_DB(task, seed)
        DB_embeddings = None  # 后面根据这个是不是None选择retrieval时用嵌入/tanimoto

    # /||||||||||||||||||||||||||||||||||||||||||||||||  初始化  ||||||||||||||||||||||||||||||||||||||||||||||||\
    # KS  定义核心大模型  (两个llm分别是editop生成阶段和rationale补全阶段的不同llm)
    llm, llm2 = choose_llm(llm_type, 0.9, 0.9), choose_llm(llm_type, 0.2, 1.0)

    # KS  定义工具 （DB_embeddings=None的时候就用tanimoto相似度匹配，否则基于嵌入匹配）
    retrieve_tool = make_retrieve_tool_lc(task, retrieval_DB, DB_embeddings, constraint, None)
    db_tool = Tool(name="DrugRetriever", func=retrieve_tool,
                   description="输入json: input_drug和generated_drug，返回符合约束的最相似分子")
    if is_chatmol_gen:  # 调用chatmol（flask）的Tool
        chatmol_tool = Tool.from_function(func=molecule_understanding_tool_func,
                                          name="ChatMolTool",
                                          description="输入json: input和task（describe/generate）")
    # \||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||/

    num_correct_l, num_all_l, num_all_f_l, num_correct_s, num_all_s, num_all_f_s = 0, 0, 0, 0, 0, 0
    # 如log有上一次运行的记录，就从记录的最后一个数据开始记录
    num_all_l, num_correct_l = process_log_file(log_loose)
    num_all_s, num_correct_s = process_log_file(log_strict)
    print(f"尝试读取之前的log文件，\nloose结果num_all={num_all_l}, num_correct={num_correct_l}\nstrict结果num_all={num_all_s}, num_correct={num_correct_s}")
    # 记录开始点(l和s的log理论上是同步的，所以只用其一获得开始下标就行了)，因为num_all在后面的循环里就变了
    start = num_all_l-1 if num_all_l > 1 else 0
    # record = {}  # record字典用于控制不同轮次之间变量的互相访问
    record_loose, record_strict = {}, {}
    logf_loose, logf_strict = open(log_loose, 'a'), open(log_strict, 'a')  # 存放对话记录的文件

    # /||||||||||||||||||||||||||||||||||||||||||||||||  遍历所有数据  ||||||||||||||||||||||||||||||||||||||||||||||||\
    # for index, input_drug in enumerate(input_drug_list):
    for index, input_drug in zip(range(start, len(input_drug_list)), input_drug_list[start:]):
        # # 遍历每条输入
        """204 loose"""
        # yesid = [0,2,4,7,11,16,17,23,27,28,38,45,47,53,61,67,68,69,73,74,79,80,81,88,91,97,99,
        #         106,110,115,119,126,137,142,147,149,150,152,158,163,172,174,175,179,183,187,196]
        """204 strict"""
        # yesid = [0,2,4,11,16,23,27,30,37,38,45,47,49,53,58,69,73,80,88,89,91,99,100,106,107,
        #          110,111,115,119,123,126,129,130,131,137,142,143,147,149,150,152,158,163,172,
        #          174,175,179,183]
        # if index not in yesid:
        #     continue

        ColorText.print(f"{'》' * 10}ind:{index}, {input_drug}{'《' * 10}", ColorText.YELLOW, ColorText.REVERSE)
        print(f"\n>>>>> Sample {index}: {input_drug}", file=logf_loose)
        print(f"\n>>>>> Sample {index}: {input_drug}", file=logf_strict)
        num_all_l, num_all_s = num_all_l + 1, num_all_s + 1

        # 初始化上下文message和药物生成历史record
        def init():
            messages_loose, messages_strict = [], []
            record_loose[input_drug], record_strict[input_drug] = {}, {}  # 初始化record[input_drug]
            # 以C=2为例，record[input_drug]就被初始化成{"skip_conversation_round":-1, "retrieval_conversation":{{"result": 0}, {"result": 1}, {"result": 2}}}
            # 额外加个-1轮表示最开始的查表
            record_loose[input_drug]['retrieval_conversation'] = [{'result': i} for i in range(-1, (args['C'] + 1))]
            record_strict[input_drug]['retrieval_conversation'] = [{'result': i} for i in range(-1, (args['C'] + 1))]
            return messages_loose, messages_strict, record_loose, record_strict

        messages_loose, messages_strict, record_loose, record_strict = init()
        active_constraints = ["loose", "strict"]   # 每个分子有一个此变量表示对应约束优化是否结束
        # ====================================================================== 首轮之前做检索并决定是否跳过对话 =====
        if is_prequery:
            # ColorText.print(f"\n\n{'+' * 15}ind: {index}, 输入药物: {input_drug}{'+' * 15}", ColorText.YELLOW, ColorText.REVERSE)
            ColorText.print(f"{'=' * 10} 【0轮前先尝试，在数据库中检索最优分子】{'=' * 10}", ColorText.YELLOW)

            tool_input_str = json.dumps({
                "input_drug": input_drug, "generated_drug": input_drug  # 让x浪 = input_drug，这样就可以少写一个函数了
            })  # 因为retrieve_and_feedback就是比较input_drug获得条件D，比较generated_drug获得相似度，如果这个g_d换成i_d就刚好满足需求
            retrieved_result_all = json.loads(db_tool.run(tool_input_str))  # 现在这里是loose和strict两个情况结果

            for constr, f in zip(constraint, [logf_loose, logf_strict]):
                res = retrieved_result_all.get(constr, {})
                closest_drug, similarity = res.get("retrieved"), res.get("similarity", 0)

                if closest_drug is None or similarity == 0:
                    ColorText.print(f"[{constr}] 查表没找到满足条件D的分子", ColorText.PURPLE, ColorText.REVERSE)
                else:
                    ColorText.print(f"[{constr}] DB预查询Top-1: {closest_drug} 相似度={similarity}", ColorText.GREEN)
                    if is_same_scaffold(input_drug, closest_drug):
                        ColorText.print(">>> 查表结果和输入药物scaffold相同，任务提前成功结束 <<<", ColorText.GREEN, ColorText.REVERSE)
                        print(f"Pre-query DB: {closest_drug} (similarity={similarity}). Task early stopped.", file=f)

                        if constr == "loose":
                            num_correct_l += 1
                            active_constraints.remove(constr)
                            record_loose[input_drug]['retrieval_conversation'][0]['retrieval_drug'] = closest_drug
                            record_loose[input_drug]['retrieval_conversation'][0]['generated_drug'] = closest_drug
                            record_loose[input_drug]['retrieval_conversation'][0]['answer'] = 'True'
                        elif constr == "strict":
                            num_correct_s += 1
                            active_constraints.remove(constr)
                            record_strict[input_drug]['retrieval_conversation'][0]['retrieval_drug'] = closest_drug
                            record_strict[input_drug]['retrieval_conversation'][0]['generated_drug'] = closest_drug
                            record_strict[input_drug]['retrieval_conversation'][0]['answer'] = 'True'
                        continue  # 跳过后续对话逻辑
                    else:
                        ColorText.print(f"{'>' * 10}  查表结果和输入药物scaffold不相同，继续对话 {'<' * 10}", ColorText.RED, ColorText.REVERSE)
                        print(f"Pre-query DB:  fail. Task continue.", file=logf_loose)
                        print(f"Pre-query DB:  fail. Task continue.", file=logf_strict)
        else:
            ColorText.print(f"{'=' * 10} 【不使用预查询】{'=' * 10}", ColorText.YELLOW)
        # ========================================================================================== 正式对话 =====
        ColorText.print(f"{'=' * 10} 【开始正式对话】{'=' * 10}", ColorText.YELLOW)
        # 构造系统提示
        messages_loose.append(SystemMessage(content='You are an expert in the field of molecular chemistry.'))
        messages_strict.append(SystemMessage(content='You are an expert in the field of molecular chemistry.'))
        # 构造用户输入
        prompt = init_prompt_in_a_conversation(llm_type, task_specific_prompt_dict, input_drug, drug_type, task)
        messages_loose.append(HumanMessage(content=prompt))  # msg中添加用户初始输入
        messages_strict.append(HumanMessage(content=prompt))  # msg中添加用户初始输入

        # ======================================================================================
        # KS   Step 0: 规则库 初始化/加载
        target_prop_lst, target_direc_lst = prop_direc_from_taskid(task)  # 目标属性 优化方向
        rules_dir = args.get('rules_dir', 'rules')  # 规则库路径
        regen_rules = str2bool(args.get('regen_rules', False))  # 是否重新生成规则库

        rb, record_loose[input_drug]['rulebanks'], record_strict[input_drug]['rulebanks'] = {}, {}, {}
        rulebanks_this_mol = {}  # 多属性的时候用，key是prop，value是规则库
        for prop in target_prop_lst:
            rb = {} if regen_rules else load_rulebank(rules_dir, prop)   # 重新生成规则库的话就把rb置空
            if rb and rb.get("rules"):                                   # 读取规则库
                ColorText.print(f"[Step0] 载入规则库: {prop} ({len(rb['rules'])} 条)", ColorText.GREEN)
            else:  # 生成规则库
                rb = build_large_rulebank_batch(prop, llm, llm2, total_rules=300, batch_size=20)
                print("新生成规则库：", json.dumps(rb, indent=2, ensure_ascii=False))
                save_rulebank(rules_dir, prop, rb)
                ColorText.print(f"[Step0] 新生成的规则库({len(rb['rules'])} 条)已保存: {os.path.join(rules_dir, f'{prop}.json')}", ColorText.GREEN)

            record_loose[input_drug]['rulebanks'][prop], record_strict[input_drug]['rulebanks'][prop] = rb, rb  # 存入record，后续可用于提示构造或一致性校验
            rulebanks_this_mol[prop] = rb   # 单属性的时候rb和rulebanks_this_mol[prop]是一回事
        # ======================================================================================

        # ============================================================================对话部分===
        # KS  Step 1 对输入smiles进行分子文本翻译
        # ”分子文本翻译任务“里  把输入分子翻译成文本
        # if use_caption == 'True':
        #     caption = step1_translate(input_drug, llm, 'data/cap2mol_trans_raw/train.txt', 3, None, None)
        #     ColorText.print(f"[Step1] 使用【分子文本翻译任务】获得的inputdrug的结构表述:\n {caption}\n", ColorText.GREEN)


        # KS  Step2 结合规则库的第一次分子生成(不用llm做底层分子编辑操作了)
        for round_index in range(args["C"] + 1):  # =================================================== 正式对话 ===
            # =========================================================================== 获取XR（首轮没有后面才有）===
            if round_index >= 1:  # 这里是因为XR是评估完当前轮结果不合适就需要ReDF，所以把XR记录在了本轮中（下一轮prompt构造使用）
                if "loose" in active_constraints:
                    closest_drug_l = record_loose[input_drug]['retrieval_conversation'][round_index].get('retrieval_drug', None)
                else:
                    closest_drug_l = None

                if "strict" in active_constraints:
                    closest_drug_s = record_strict[input_drug]['retrieval_conversation'][round_index].get('retrieval_drug', None)
                else:
                    closest_drug_s = None
            else:  # 首轮XR = None
                ColorText.print("---首轮对话还没有进行ReDF，没有XR", ColorText.PURPLE, ColorText.UNDERLINE)
                closest_drug_l, closest_drug_s = None, None

            # =============================================================================== 首轮仅基于输入分子做 ===
            ColorText.print(f"{'=' * 7} 第 {round_index} 轮对话 {'=' * 7}", ColorText.YELLOW)
            print(f"----- round_index:{round_index}", file=logf_loose)
            print(f"----- round_index:{round_index}", file=logf_strict)

            # KS  首轮loose和strict的生成逻辑没有区别，且都必须且仅调用一次，故不需要区分需不需要区分active_constraints
            if round_index == 0 and len(active_constraints)!=0:  # 防止两个分子都预查询通过时（长度为0）再走后面的逻辑
                if len(target_prop_lst) == 1:          # KS  单属性优化
                    target_prop, target_direc = target_prop_lst[0], target_direc_lst[0]
                    rb_single = rulebanks_this_mol[target_prop]

                    ColorText.print(f"{'>' * 7} 筛选 {target_prop} 规则库", ColorText.YELLOW)
                    new_rulebank = filter_rules_with_chain_of_thought(rb_single, input_drug, target_prop, llm,
                        5, 300, len(rb_single['rules']), target_direc, 2)

                    ColorText.print("规则库初次筛选完毕", ColorText.GREEN)
                    new_mol_dict = apply_rules_from_cot(input_drug, new_rulebank, llm)
                    print(f"筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict, ensure_ascii=False)}")
                    print(f"【rulbank】{new_mol_dict}", file=logf_loose)
                    print(f"【rulbank】{new_mol_dict}", file=logf_strict)
                else:
                    # ===== 多属性：使用多目标 rulebank 组合 =====
                    ColorText.print(f"{'>' * 7} 多属性首轮筛选 {target_prop_lst}", ColorText.YELLOW)
                    if task == 207:
                        new_rulebank = build_multiobjective_rulebank_for_mol(llm, input_drug, rulebanks_this_mol,
                            target_prop_lst, target_direc_lst, 10,  300,  5, True)
                    else:
                        new_rulebank = build_multiobjective_rulebank_for_mol(llm, input_drug, rulebanks_this_mol,
                            target_prop_lst, target_direc_lst, 10, 300, 5)
                    ColorText.print("多目标规则库首轮筛选完毕", ColorText.GREEN)
                    new_mol_dict = apply_rules_from_cot(input_drug, new_rulebank, llm)
                    print(f"[multi] 筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict, ensure_ascii=False)}")
                    print(f"【rulebank_multi】{new_mol_dict}", file=logf_loose)
                    print(f"【rulebank_multi】{new_mol_dict}", file=logf_strict)

                # 3) 把返回结果中新分子的部分拿出来
                generated_drug = []
                for item in new_mol_dict.get("selected", []):
                    if item.get("success"):
                        generated_drug.append(item.get("new_smiles"))
                # for ind, target_prop in enumerate(target_prop_lst):
                #     ColorText.print(f"{'>' * 7} 筛选 {target_prop} 规则库", ColorText.YELLOW)
                #     # 1) cot筛选规则库                   (llama2上下文短，要把chunksize调小一点)
                #     new_rulebank = filter_rules_with_chain_of_thought(rb, input_drug, target_prop, llm, 15, 300, len(rb['rules']), target_direc_lst[ind], 2)
                #     ColorText.print("规则库初次筛选完毕", ColorText.GREEN)
                #     # 2) 解析规则库 + 执行手动编辑操作
                #     new_mol_dict = apply_rules_from_cot(input_drug, new_rulebank, llm)
                #     print(f"筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict, ensure_ascii=False)}")
                #     print(f"【rulbank】{new_mol_dict}", file=logf_loose)
                #     print(f"【rulbank】{new_mol_dict}", file=logf_strict)   # 这部分没涉及到Δ，所以村一样的

                # KS  这一步缝上chatmol的分子生成
                def is_valid_smiles(smiles: str) -> bool:   # 判断chatmol生成的分子合不合法
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        return mol is not None
                    except:
                        return False

                if str2bool(is_chatmol_gen):
                    query = json.dumps({"input": truncate_prompt(prompt), "task": "generate"})  # 阶段提示防止超长
                    with suppress_everything():
                        response_chatmol = chatmol_tool.run(query)
                    moleculars = json.loads(response_chatmol).get("result")
                    mols = [smi for smi in moleculars if is_valid_smiles(smi)]  # 只保留合法的
                    generated_drug.extend(mols)
                    # ColorText.print("【本轮生成分子】"+'\n'.join(map(str, generated_drug)), ColorText.BLUE, ColorText.ITALIC)
                else:
                    moleculars = None  # 在调用函数里决定用不用chatmol分子生成
                    ColorText.print('\n'.join(map(str, generated_drug)), ColorText.BLUE, ColorText.ITALIC)
                # 首轮loose和strict生成的分子都一样
                generated_drug_loose, generated_drug_strict = generated_drug, generated_drug
            # ==============================================================从第二轮开始结合XR生成==
            else:
                # KS  从第二轮开始strict和loose的生成逻辑就不一样（因为引入了XR，所以生成和记录都得区分）
                generated_drug_loose, generated_drug_strict = [], []

                if "loose" in active_constraints:
                    S0_loose = record_loose[input_drug]['retrieval_conversation'][round_index]['generated_drug']
                    XR_loose = closest_drug_l
                    ColorText.print(f"---(loose)使用上轮ReDF的结果，XR：{XR_loose}", ColorText.PURPLE)

                    if XR_loose is not None:
                        if len(target_prop_lst) == 1:
                            # 单属性：保持原来的 select_rules_for_prompt 流程
                            target_prop, target_direc = target_prop_lst[0], target_direc_lst[0]
                            rb_single = rulebanks_this_mol[target_prop]
                            sel_rules_loose = select_rules_for_prompt(Chem.MolFromSmiles(S0_loose), Chem.MolFromSmiles(XR_loose),
                                                                      rb_single["rules"], target_prop, target_direc, llm, 5)
                            new_mol_dict_loose = apply_rules_from_cot(S0_loose, sel_rules_loose, llm)
                        else:
                            # 多属性：用 build_multiobjective_rulebank_for_round 聚合二轮规则
                            if task == 207:
                                multi_rb_loose = build_multiobjective_rulebank_for_round(llm, S0_loose, XR_loose, rulebanks_this_mol,
                                                                                         target_prop_lst, target_direc_lst, 10, 5, True)
                            else:
                                multi_rb_loose = build_multiobjective_rulebank_for_round(llm, S0_loose, XR_loose, rulebanks_this_mol,
                                                                                         target_prop_lst, target_direc_lst, 10, 5)
                            new_mol_dict_loose = apply_rules_from_cot(S0_loose, multi_rb_loose, llm)

                        print(f"(loose)筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict_loose, ensure_ascii=False)}")
                        print(f"【rulebank】{new_mol_dict_loose}", file=logf_loose)

                        for item in new_mol_dict_loose.get("selected", []):
                            if item.get("success"):
                                generated_drug_loose.append(item.get("new_smiles"))
                    else:
                        ColorText.print("[loose] 本轮无XR，跳过规则筛选", ColorText.PURPLE)

                # if "loose" in active_constraints:
                #     sel_rules_loose = None
                #     S0_loose = record_loose[input_drug]['retrieval_conversation'][round_index]['generated_drug']
                #     XR_loose = closest_drug_l
                #     ColorText.print(f"---(loose)使用上轮ReDF的结果，XR：{XR_loose}", ColorText.PURPLE)
                #     for target_prop, target_direc in zip(target_prop_lst, target_direc_lst):
                #         # 结合生成上轮的生成药物S0和retrieval的目标药物参考XR重新搜索+排序规则库
                #         sel_rules_loose = select_rules_for_prompt(Chem.MolFromSmiles(S0_loose), Chem.MolFromSmiles(XR_loose),
                #                                                   rb["rules"], target_prop, target_direc, llm, 5)
                #     # 2) 解析规则库+执行编辑操作     这里也没支持多属性，实际编辑的是最后一个属性
                #     # 注意 在上一步生成的分子上在做修改，而不是原分子
                #     new_mol_dict_loose = apply_rules_from_cot(S0_loose, sel_rules_loose, llm)
                #     print(f"(loose)筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict_loose, ensure_ascii=False)}")
                #     print(f"【rulebank】{new_mol_dict_loose}", file=logf_loose)
                #     # 3) 把返回结果中新分子的部分拿出来
                #     for item in new_mol_dict_loose.get("selected", []):
                #         if item.get("success"):
                #             generated_drug_loose.append(item.get("new_smiles"))

                if "strict" in active_constraints:
                    S0_strict = record_strict[input_drug]['retrieval_conversation'][round_index]['generated_drug']
                    XR_strict = closest_drug_s
                    ColorText.print(f"---(strict)使用上轮ReDF的结果，XR：{XR_strict}", ColorText.PURPLE)

                    if XR_strict is not None:
                        if len(target_prop_lst) == 1:
                            target_prop, target_direc = target_prop_lst[0], target_direc_lst[0]
                            rb_single = rulebanks_this_mol[target_prop]
                            # 选规则
                            sel_rules_strict = select_rules_for_prompt(Chem.MolFromSmiles(S0_strict), Chem.MolFromSmiles(XR_strict),
                                                                       rb_single["rules"], target_prop, target_direc, llm, 5)
                            new_mol_dict_strict = apply_rules_from_cot(S0_strict, sel_rules_strict, llm)
                        else:
                            if task == 207:
                                multi_rb_strict = build_multiobjective_rulebank_for_round(llm, S0_strict, XR_strict, rulebanks_this_mol,
                                                                                          target_prop_lst, target_direc_lst, 10, 5, True)
                            else:
                                multi_rb_strict = build_multiobjective_rulebank_for_round(llm, S0_strict, XR_strict, rulebanks_this_mol,
                                                                                          target_prop_lst, target_direc_lst, 10, 5)
                            new_mol_dict_strict = apply_rules_from_cot(S0_strict, multi_rb_strict, llm)

                        print(f"(strict)筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict_strict, ensure_ascii=False)}")
                        print(f"【rulebank】{new_mol_dict_strict}", file=logf_strict)

                        for item in new_mol_dict_strict.get("selected", []):
                            if item.get("success"):
                                generated_drug_strict.append(item.get("new_smiles"))
                    else:
                        ColorText.print("[strict] 本轮无XR，跳过规则筛选", ColorText.PURPLE)

                # if "strict" in active_constraints:
                #     sel_rules_strict = None
                #     S0_strict = record_strict[input_drug]['retrieval_conversation'][round_index]['generated_drug']
                #     XR_strict = closest_drug_s
                #     ColorText.print(f"---(strict)使用上轮ReDF的结果，XR：{XR_strict}", ColorText.PURPLE)
                #     # 结合生成上轮的生成药物S0和retrieval的目标药物参考XR重新搜索+排序规则库
                #     for target_prop, target_direc in zip(target_prop_lst, target_direc_lst):
                #         sel_rules_strict = select_rules_for_prompt(Chem.MolFromSmiles(S0_strict), Chem.MolFromSmiles(XR_strict),
                #                                                   rb["rules"], target_prop, target_direc, llm, 5)
                #
                #     # 2) 解析规则库+执行编辑操作     这里也没支持多属性，实际编辑的是最后一个属性
                #     # 注意 在上一步生成的分子上在做修改，而不是原分子
                #     new_mol_dict_strict = apply_rules_from_cot(S0_strict, sel_rules_strict, llm)
                #     # print(f"(strict)筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict_strict, indent=2, ensure_ascii=False)}")
                #     print(f"(strict)筛选出的规则库（仅返回edit_op和rationale字段）和新分子\n{json.dumps(new_mol_dict_strict, ensure_ascii=False)}")
                #     print(f"【rulebank】{new_mol_dict_strict}", file=logf_strict)
                #     # 3) 把返回结果中新分子的部分拿出来
                #     for item in new_mol_dict_strict.get("selected", []):
                #         if item.get("success"):
                #             generated_drug_strict.append(item.get("new_smiles"))

            # ==============================================================筛选和评估函数都是通用的==
            def ReDF(round_index, input_drug, generated_drug_loose, generated_drug_strict, record_loose, record_strict):
                XR_loose, XR_strict = None, None
                # 调用逻辑是一个一个constraint做的，所以得判断一下generated_drug_loose, generated_drug_strict存不存咋
                # ========== loose ==========
                if generated_drug_loose:
                    tool_input_str_loose = json.dumps(
                        {"input_drug": input_drug, "generated_drug": generated_drug_loose}
                    )
                    try:
                        retrieved_result_all_loose = json.loads(db_tool.run(tool_input_str_loose))
                        res_loose = retrieved_result_all_loose.get("loose", {})
                        XR_loose, sim_loose = res_loose.get("retrieved"), res_loose.get("similarity", 0)
                        if XR_loose:
                            ColorText.print(f"[loose] ReDF找到XR: {XR_loose}, 相似度={sim_loose}", ColorText.GREEN)
                            record_loose[input_drug]['retrieval_conversation'][round_index + 1]['retrieval_drug'] = XR_loose
                        else:
                            ColorText.print("[loose] ReDF无有效结果", ColorText.PURPLE)
                            XR_loose = None
                    except Exception as e:
                        ColorText.print(f"[loose] ReDF 检索失败: {e}", ColorText.RED)
                        XR_loose = None
                else:
                    ColorText.print("[loose] ReDF 跳过（无生成分子）", ColorText.PURPLE)

                # ========== strict ==========
                if generated_drug_strict:
                    tool_input_str_strict = json.dumps(
                        {"input_drug": input_drug, "generated_drug": generated_drug_strict}
                    )
                    try:
                        retrieved_result_all_strict = json.loads(db_tool.run(tool_input_str_strict))
                        res_strict = retrieved_result_all_strict.get("strict", {})
                        XR_strict, sim_strict = res_strict.get("retrieved"), res_strict.get("similarity", 0)
                        if XR_strict:
                            ColorText.print(f"[strict] ReDF找到XR: {XR_strict}, 相似度={sim_strict}", ColorText.GREEN)
                            record_strict[input_drug]['retrieval_conversation'][round_index + 1]['retrieval_drug'] = XR_strict
                        else:
                            ColorText.print("[strict] ReDF无有效结果", ColorText.PURPLE)
                            XR_strict = None
                    except Exception as e:
                        ColorText.print(f"[strict] ReDF 检索失败: {e}", ColorText.RED)
                        XR_strict = None
                else:
                    ColorText.print("[strict] ReDF 跳过（无生成分子）", ColorText.PURPLE)

                return XR_loose, XR_strict

            # 因为不同constrain的成功轮次可能不对齐，所以还是用支持单constrain的ReDF
            def ReDF_single(round_index, input_drug, constr, gen_drug, db_tool, record, logf):
                if not isinstance(gen_drug, str) or not gen_drug:
                    ColorText.print(f"[{constr}] ReDF 跳过：gen_drug 非法", ColorText.RED)
                    return None

                tool_input_str = json.dumps({"input_drug": input_drug, "generated_drug": gen_drug})

                try:
                    retrieved_result_all = json.loads(db_tool.run(tool_input_str))  # {"loose": {...}, "strict": {...}}
                except Exception as e:
                    ColorText.print(f"[{constr}] ReDF 检索失败: {e}", ColorText.RED)
                    return None

                res = retrieved_result_all.get(constr, {}) or {}
                XR, sim = res.get("retrieved"), res.get("similarity", 0)

                if XR:
                    ColorText.print(f"[{constr}] ReDF找到XR: {XR}, 相似度={sim}", ColorText.GREEN)
                    record[input_drug]['retrieval_conversation'][round_index + 1]['retrieval_drug'] = XR
                    print(f"[{constr}] ReDF retrieved {XR} (sim={sim})", file=logf)
                    return XR
                else:
                    ColorText.print(f"[{constr}] ReDF无有效结果", ColorText.PURPLE)
                    print(f"[{constr}] ReDF no XR", file=logf)
                    return None

            # 解析+筛选
            for constr, generated_list, logf, record in [
                ("loose", generated_drug_loose, logf_loose, record_loose),
                ("strict", generated_drug_strict, logf_strict, record_strict)
            ]:  # 约束、生成的分子、log文件
                if constr not in active_constraints:  # KS  只处理未完成的约束
                    continue
                # 解析并选最优生成结果  （closetdrug是用来去重的，现在逻辑S0不太可能改出XR，置空也无所谓）
                ColorText.print(f"---解析 [{constr}] 生成分子---", ColorText.CYAN, ColorText.REVERSE)
                parse_status, gen_drug \
                    = parse_molecular_in_response(input_drug, generated_list, None, record, constr, logf, target_direc_lst[0])
                # 评估生成分子是否满足约束
                ColorText.print(f"---评估 [{constr}] 生成分子---", ColorText.BLUE, ColorText.REVERSE)
                eval_status, final_drug \
                    = evaluate_generated_drug(input_drug, gen_drug, task, constr, record, logf)

                # 根据评估结果判断要不要ReDF
                if eval_status == 1:           # KS  优化成功
                    if constr == "loose":
                        num_correct_l += 1     # 优化成功药物总数
                    elif constr == "strict":
                        num_correct_s += 1
                    ColorText.print(f"[{constr}] round {round_index} 评估通过，结束任务", ColorText.GREEN, ColorText.REVERSE)
                    active_constraints.remove(constr)  # KS  移除完成的(成功)约束
                    continue  # break                      # 满足Δ约束，直接结束

                elif eval_status == -1:        # KS  gen_drug（经parse后是一个）不合法，则没法相似性比较，故没法继续
                    if constr == "loose":
                        num_all_f_l += 1
                    elif constr == "strict":
                        num_all_f_s += 1
                    ColorText.print(f"【S:main[{constr}]】 生成分子非法，无法继续retrieval，提前结束 <<<", ColorText.RED)
                    print("fail: Generation error ", file=logf)
                    active_constraints.remove(constr)  # KS  移除完成的(失败)约束
                    continue  # break
                elif eval_status == 0:        # KS  gen_drug正常生成，但评估不通过
                    # 如果X~不正确，那么需要根据C的情况决定是否继续ReDF
                    if round_index < C:     # 再上限轮次以内才能进行ReDF
                        # XR_loose, XR_strict =\
                        #     ReDF(round_index, input_drug, generated_drug_loose, generated_drug_strict, record_loose, record_strict)
                        XR = ReDF_single(round_index, input_drug, constr, gen_drug, db_tool, record, logf)
                        if XR is None:
                            if constr == "loose":
                                num_all_f_l += 1
                            elif constr == "strict":
                                num_all_f_s += 1
                            ColorText.print(f"[{constr}] ReDF 未返回有效 XR，终止该分支", ColorText.RED, ColorText.REVERSE)
                            print("fail:XR", file=logf)
                            active_constraints.remove(constr)  # KS  移除完成的(失败)约束
                            continue   # 不 break 整体循环，因为这里的上级for是遍历constraint的
                        else:
                            ColorText.print(f"[{constr}] round {round_index} ReDF 返回有效 XR，进入下一轮", ColorText.CYAN)
                            # 进入下一轮

                    else:    # 超轮次，停止
                        print("fail:maxround", file=logf_strict)
                        print("fail:maxround", file=logf_loose)
                        ColorText.print(f'本次对话已达到最大轮次C={C}，对话结束', ColorText.RED, ColorText.REVERSE)

            if not active_constraints:  # <<< active_constraints味空的话，就表示两个约束都完成则退出
                ColorText.print("loose和strict均完成，结束当前分子优化", ColorText.GREEN, ColorText.REVERSE)
                break

        print(f'[loose]当前指标 num_correct = {num_correct_l}, acc = {num_correct_l/num_all_l}, num_all = {num_all_l}')
        print(f'[strict]当前指标 num_correct = {num_correct_s}, acc = {num_correct_s / num_all_s}, num_all = {num_all_s}')
        print(f'当前指标 num_correct = {num_correct_l}, acc = {num_correct_l/num_all_l}', file=logf_loose)
        print(f'当前指标 num_correct = {num_correct_s}, acc = {num_correct_s / num_all_s}', file=logf_strict)



    # 最终命中率
    print("---------Final Acc---------", file=logf_loose)
    print(f'Acc = {num_correct_l}/{num_all_l}', file=logf_loose)
    print(f' num_correct = {num_correct_l}, num_all ={num_all_l}, num_all_f ={num_all_f_l}', file=logf_loose)
    ColorText.print(f'Acc = {num_correct_l}/{num_all_l}', ColorText.GREEN)
    ColorText.print(f'Err = {num_all_f_l}', ColorText.GREEN)
    print("----------------------------", file=logf_loose)

    print("---------Final Acc---------", file=logf_strict)
    print(f'Acc = {num_correct_s}/{num_all_s}', file=logf_strict)
    print(f' num_correct = {num_correct_s}, num_all ={num_all_s}, num_all_f ={num_all_f_s}', file=logf_strict)
    ColorText.print(f'Acc = {num_correct_s}/{num_all_s}', ColorText.GREEN)
    ColorText.print(f'Err = {num_all_f_s}', ColorText.GREEN)
    print("----------------------------", file=logf_strict)

    with open(rec_loose, 'w', encoding='utf-8') as rf:
        json.dump(record_loose, rf, ensure_ascii=False)
    with open(rec_strict, 'w', encoding='utf-8') as rf:
        json.dump(record_strict, rf, ensure_ascii=False)
    # return num_all, num_all_f, num_correct

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', action='store', required=True, type=int)
    parser.add_argument('--conversational_LLM', action='store', required=False, type=str, default='chatgpt')
    parser.add_argument('--log_file', action='store', required=False, type=str, default='results/ChatDrug.log', help='每个分子的处理结果出的记录文件')
    parser.add_argument('--record_file', action='store', required=False, type=str, default='results/ChatDrug.json', help='代码里record变量的记录文件')
    parser.add_argument('--constraint', required=False, type=str, default='loose', help='loose是只要分子属性有变化就行，strict则变化量必须满足某阈值')
    parser.add_argument('--seed', required=False, type=int, default=0, help='构造DB的随机种子')
    parser.add_argument('--trial_index', required=False, type=int, default=0, help='取第几个生成的分子作为输出(fix_mol是int时)')
    parser.add_argument('--C', required=False, type=int, default=2, help='一次对话最大轮数')

    parser.add_argument('--fix_mol', required=False, type=str, default=False, help='一次对话最大轮数')
    parser.add_argument('--is_prequery', required=False, type=str, default=False, help='正式对话前要不要查DB')
    parser.add_argument('--is_emb_retri', required=False, type=str, default=False, help='用嵌入比较DB相似度/tanimoto')
    parser.add_argument('--is_chatmol_und', required=False, type=str, default=True, help='加不加chatmol的分子理解部分')
    parser.add_argument('--is_chatmol_gen', required=False, type=str, default=True, help='加不加chatmol的分子生成部分')
    parser.add_argument('--summary_path', required=False, type=str, default='results/summary.csv', help='记录所有实验的文件')

    parser.add_argument('--rules_dir', required=False, type=str, default='rules', help='规则库存放目录，每个属性一份JSON')
    parser.add_argument('--regen_rules', required=False, type=str, default=False, help='是否强制重建规则库文件')

    args = parser.parse_args()
    args = vars(args)
    # main_lc(args, DB=None, DB_emb=None)

    input_drug_list, retrieval_DB, DB_embeddings = load_retrieval_DB2(args['task'], return_embedding=True)  # 只加载一次DB
    main_lc(args, retrieval_DB, DB_embeddings)
