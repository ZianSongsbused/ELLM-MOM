import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5,6,7"
import torch
# print(torch.cuda.device_count())  # 应该输出 7
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, OPTForCausalLM
from .local_models.local_galactica import complete_galactica_molecule
from .local_models.local_llama import complete_llama
from langchain_core.runnables import Runnable
# openai.api_key = "sk-H5WPt8TfJvVFA37pyW7l6DyClM9QaHl7Fj4uiixUdTnw0y7e"


# langchain使用的版本，获取gpt3.5turbo的输出
def complete_chatgpt_lc(temperature=0.2, top_p=1.0):
    # 规则生成阶段给(0.9, 0.9), 补全阶段给(0.2, 1.0)
    llm = ChatOpenAI(
        api_key="your api key",
        base_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo",  # 指定模型名称
        temperature=temperature,  # 降低随机性，使结果更稳定
        top_p=top_p,
    )
    return llm


# 获取deepseekv3的输出
def complete_deepseek_lc():
    # 初始化LangChain的ChatOpenAI客户端
    llm = ChatOpenAI(
        api_key="your api key",  # 替换为你的DeepSeek API密钥
        base_url="https://api.deepseek.com",  # DeepSeek API端点
        model="deepseek-chat",  # 指定模型名称
        temperature=0.2  # 降低随机性，使结果更稳定
    )
    return llm

def get_free_gpu_memory():
    """返回每张GPU剩余显存（MiB）字典，如 {0: 12345, 1: 23456, ...}"""
    free_mem = {}
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        free_mem[i] = stats[0] // (1024 ** 2)  # 转MiB
    return free_mem


def print_model_gpu_memory_usage(model):
    if not hasattr(model, "hf_device_map"):
        print("⚠️ 该模型没有 hf_device_map，可能未使用 device_map='auto' 加载")
        return
    device_usage = {}
    for name, module in model.named_modules():
        device = getattr(module, 'device', None)
        if device is None and hasattr(module, 'weight'):
            device = module.weight.device
        if device is None or not torch.device(device).type == 'cuda':
            continue
        device_id = torch.device(device).index
        device_usage[device_id] = torch.cuda.memory_allocated(device_id)

    print("\n当前模型在各 GPU 上显存分布（单位: MiB）:")
    for device_id in sorted(device_usage):
        used_mem = device_usage[device_id] / 1024 / 1024
        print(f"  - GPU {device_id}: {used_mem:.2f} MiB")


# 从本地路径加载LLM（llama2和galactica）
def load_local_model(conversational_LLM):
    # 通过手动设置每张卡的最大可用显存来实现屏蔽某张卡的效果
    max_memory = {
        0: "15GB",
        1: "24GB",
        # 2: "30GB",
        # 3: "30GB",
        # 4: "0GB",  # 直接禁用4号卡
        # 5: "0GB",
        # 6: "0GB",
    }
    if conversational_LLM == 'llama2':
        localpath = "/home/aita8180/data/mntdata/ziansong/p1/pretrain_models/meta-llama/Llama-2-7b-chat-hf"
        # localpath = "/home/wangshuang/ziansong/p1/pretrain_models/meta-llama/Llama-2-7b-chat-hf"
        print(f"从此处加载llama模型: {localpath}")

        # model = LlamaForCausalLM.from_pretrained(
        #     localpath, return_dict=True, local_files_only=True, load_in_8bit=False,
        #     device_map="auto", low_cpu_mem_usage=True)
        model = LlamaForCausalLM.from_pretrained(
            localpath, return_dict=True, local_files_only=True, load_in_8bit=False,
            device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True)

        tokenizer = AutoTokenizer.from_pretrained(localpath, local_files_only=True)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})   # llama2系列的模型的分词器没有预定义的<PAD>标记
        # 打印 每个卡上大模型占了多少显存
        print_model_gpu_memory_usage(model)
    elif conversational_LLM == 'galactica':
        localpath = "/home/aita8180/data/mntdata/ziansong/chatDrug/sourcecode/models/facebook/galactica-6.7b"
        # localpath = "/home/wangshuang/ziansong/p1/pretrain_models/facebook/galactica-6.7b"
        print(f"从此处加载galactica模型: {localpath}")
        model = OPTForCausalLM.from_pretrained(localpath, return_dict=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(localpath, local_files_only=True)
        # 打印 每个卡上大模型占了多少显存
        print_model_gpu_memory_usage(model)
    else:
        raise ValueError(f"Unsupported model: {conversational_LLM}")

    return model, tokenizer


# 调用本地llm获得response
def complete(messages, model, tokenizer, conversational_LLM, round_index=None):
    if conversational_LLM == 'llama2':
        return complete_llama(messages, model, tokenizer)
    elif conversational_LLM == 'galactica':
        return complete_galactica_molecule(messages, model, tokenizer, round_index)



class LocalLLM(Runnable):
    def __init__(self, model, tokenizer, conversational_LLM):
        self.model = model
        self.tokenizer = tokenizer
        self.conversational_LLM = conversational_LLM

    def _call(self, messages, **kwargs):  # 关键是实现这个方法
        round_index = kwargs.get("round_index", None)
        return self.invoke(messages, round_index=round_index)

    def invoke(self, messages, round_index=None):
        # 转换 langchain messages 为 dict 格式
        formatted_messages = [self._convert_message(m) for m in messages]

        if self.conversational_LLM == 'llama2':
            return complete_llama(formatted_messages, self.model, self.tokenizer)
        elif self.conversational_LLM == 'galactica':
            if round_index is None:
                round_index = (len(formatted_messages) - 1) // 2
            return complete_galactica_molecule(formatted_messages, self.model, self.tokenizer, round_index)
        else:
            raise NotImplementedError(f"Unsupported model: {self.conversational_LLM}")

    def _convert_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

        return {"role": role, "content": message.content}


# 单次运行时返回本地模型（llama和galactica）
def complete_localllm_lc(conversational_LLM):
    model, tokenizer = load_local_model(conversational_LLM)
    llm = LocalLLM(model, tokenizer, conversational_LLM)
    return llm
# 批量运行时返回本地模型（model和tokenizer从外部传入，方便批量运行时只加载一次模型）
def complete_localllm_lc_muti(conversational_LLM, local_llm, local_tokenizer):
    # model, tokenizer = load_local_model(conversational_LLM)
    llm = LocalLLM(local_llm, local_tokenizer, conversational_LLM)
    return llm

# ====== 测试入口函数 ======
def test_local_llm(conversational_LLM):
    model, tokenizer = load_local_model(conversational_LLM)
    llm = LocalLLM(model, tokenizer, conversational_LLM)

    messages = [
        {"role": "system", "content": "You are a chemistry assistant."},
        {"role": "user", "content": "What is the SMILES for Aspirin?"}
    ]
    response = llm.invoke(messages, round_index=0)    # galactica需要llm参数
    print(f"\n💬 [{conversational_LLM}] 回答内容：\n{response}")


def test_local_llm_multi_turn(conversational_LLM):
    model, tokenizer = load_local_model(conversational_LLM)
    llm = LocalLLM(model, tokenizer, conversational_LLM)

    messages = [
        {"role": "system", "content": "You are a chemistry assistant."},
        {"role": "user", "content": "What is the SMILES for Aspirin?"}
    ]

    num_rounds = 3     # 轮数
    user_questions = [
        "What is the SMILES for Aspirin?\n\nAnswer:",
        "What is the SMILES for Ibuprofen?\n\nAnswer:",
        "What is the SMILES for Acetaminophen?\n\nAnswer:'"
    ]

    for round_index in range(num_rounds):
        # 如果不是第0轮，添加新一轮用户输入
        if round_index > 0:
            messages.append(
                {"role": "user", "content": user_questions[round_index]}
            )

        response = llm.invoke(messages, round_index=round_index)

        print(f"\n Round {round_index + 1}: {user_questions[round_index]}")
        print(f"Response: {response}")

        # 加入模型回复用于下一轮
        messages.append({
            "role": "assistant",
            "content": response
        })



# ====== 执行测试 ======
if __name__ == "__main__":
    # test_local_llm("llama2")
    # test_local_llm("galactica")
    # test_local_llm_multi_turn("llama2")
    test_local_llm_multi_turn("galactica")
