import torch

B_INST, E_INST = "[INST]", "[/INST]"  # [INST]...[/INST] 用于包裹用户提问
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"  # <<SYS>>...<</SYS>> 是系统角色的 prompt（行为规范等）
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] != "system":
            dialog = [
                         {
                             "role": "system",
                             "content": DEFAULT_SYSTEM_PROMPT,
                         }
                     ] + dialog
        dialog = [
                     {
                         "role": dialog[1]["role"],
                         "content": B_SYS
                                    + dialog[0]["content"]
                                    + E_SYS
                                    + dialog[1]["content"],
                     }
                 ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """

        dialog_tokens = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
                dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def complete_llama(
        dialogs,
        model,
        tokenizer,
        max_new_tokens=1024,  #The maximum numbers of tokens to generate
        seed: int = 42,  #seed value for reproducibility
        do_sample: bool = True,  #Whether or not to use sampling ; use greedy decoding otherwise.
        use_cache: bool = True,
        #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 0.95,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 0.7,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  #The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,  #[optional] Exponential penalty to the length that is used with beam-based generation.
        **kwargs
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)        # 设置随机种子

    # ✅ tokenizer 输出 tokens 并打印长度（调试超长崩溃的根因）
    chats = format_tokens([dialogs], tokenizer)   # dialog是上下文
    chat = chats[0]
    print(f"🔎🔎🔎🔎🔎🔎🔎 Token count for input: {len(chat)}")

    # chats = format_tokens([dialogs], tokenizer)
    # chat = chats[0]

    with torch.no_grad():
        tokens = torch.tensor(chat).long()
        tokens = tokens.unsqueeze(0)
        tokens = tokens.to("cuda:0")
        outputs = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text_list = output_text.split("[/INST]")
        output_text = output_text_list[-1].strip()
    # print(f"llama输出\n{output_text}")
    return output_text
