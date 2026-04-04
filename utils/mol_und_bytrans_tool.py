from exp_textmoltrans.find_context.get_cont import get_examples
from exp_textmoltrans.find_context.prompt_eng import build_prompt_msg
from exp_textmoltrans.query_llm import safe_llm_call
from utils.convenient_utils.wordart import ColorText


def step1_translate(smiles, llm, example_file=None, n_shot=3, model=None, tokenizer=None):
    """
    Step1: 分子 → 文本描述 (caption)
    输入: smiles (str)
    输出: caption (str)
    """
    input = {"molecule": smiles, "caption": None}
    # few-shot上下文
    if n_shot > 0 and example_file is not None:
        _, mol_examples = get_examples(example_file, n_shot, input, "random", "None", model, tokenizer)
        molecule2caption = build_prompt_msg("m2c", mol_examples, smiles)
    else:
        molecule2caption = build_prompt_msg("m2c", None, smiles)
    # print(f"输入给llm的prompt\n{molecule2caption}")
    try:
        caption = safe_llm_call(llm, molecule2caption, key="caption")
    except Exception as e:
        ColorText.print(f"Step1生成caption失败: {e}", ColorText.RED)
        caption = "N/A"

    return caption
