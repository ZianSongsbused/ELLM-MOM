import re
import torch
from utils.convenient_utils.wordart import ColorText

def complete_galactica_molecule(messages, model, tokenizer, round_index):
    # print(messages)
    with torch.no_grad():
        if round_index == 0:
            input_text = messages[1]['content']
            input_text = input_text + " [START_I_SMILES]"
        else:
            input_text = ""
            for i in range(len(messages) - 1):
                if i % 2 == 0:
                    input_text += messages[i + 1]['content'] + " [START_I_SMILES]"
                if i % 2 == 1:
                    input_text += messages[i + 1]['content'] + "[END_I_SMILES]" + "\n\n"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')

        vocab_size = tokenizer.vocab_size  # or model.config.vocab_size
        # 检查非法 token
        if torch.any(input_ids >= vocab_size):
            print(f"⚠️ Skipping due to OOV token. Max token: {input_ids.max().item()}, Vocab size: {vocab_size}")
            return "OOV skipped"
        outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.95,
                                 temperature=1.0, use_cache=True, top_k=50, repetition_penalty=1.0, length_penalty=1)

        # 安全检查：是否越界vocab范围
        if input_ids.max().item() >= tokenizer.vocab_size:
            ColorText.print("[Galactica] Token ID 超出 vocab 范围，跳过该输入",ColorText.RED)
            return None
        output_text = tokenizer.decode(outputs[0])
        # print("galactica的原始输出\n", output_text)
        output_text_list = output_text.split("[START_I_SMILES]")
        output_text = output_text_list[2 + round_index * 3].strip()
        output_text_list = output_text.split("[END_I_SMILES]")
        output_text = output_text_list[0].strip()

    return output_text