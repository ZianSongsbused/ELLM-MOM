import json
import re


# 清理llm输出中常见的 ```json ``` 包装
def remove_json_markers(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[3:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


# CoT阶段，解析某prop的trendmap
def safe_parse_trend_json(raw):
    """
    解析 LLM 输出的 trend→impact 映射。
    支持 '+1' / '-1' / 无逗号结尾 / 多余文本 的情况。
    返回 dict。
    """
    # 提取最外层大括号内内容
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    content = match.group(0)

    # 去掉 JSON 中的加号，如 "+1" -> 1
    content = re.sub(r":\s*\+1", ": 1", content)
    content = re.sub(r":\s*-1", ": -1", content)

    # 尝试解析
    try:
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[safe_parse_trend_json] JSON解析失败: {e}")
    return {}


# CoT时，llm的打分结果解析（llm评分没生成完整的json时，只解析出完整的部分）
def safe_parse_incomplete_json(raw: str):
    """
    尝试修复和截断模型生成的部分 JSON 数组。
    若中途截断，则删除最后一个不完整对象。
    """
    # 清除常见非JSON标记
    raw = re.sub(r"^[^\[\{]*", "", raw)   # 去掉前面多余字符
    raw = re.sub(r"[^\]\}]*$", "", raw)   # 去掉后面非JSON部分

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 如果是数组开头但没结尾，则逐层检测配对括号
    stack = []
    clean = ""
    for c in raw:
        if c in "[{":
            stack.append(c)
        elif c in "]}":
            if stack:
                stack.pop()
            else:
                continue
        clean += c
        if not stack and c in "]}":
            # 在最后一个完整结构结束处截断
            break

    # 再次尝试解析
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # 如果仍然失败，则尝试删除最后一条不完整对象
        partial = re.sub(r',\s*\{[^\}]*$', '', clean) + "]"
        try:
            return json.loads(partial)
        except Exception:
            return []


# CoT时，llm的打分结果解析
def extract_json_from_llm_output(text):
    # 尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 从后向前扫描寻找有效JSON
    for i in range(len(text)-1, -1, -1):
        try:
            return json.loads(text[i:])
        except json.JSONDecodeError:
            continue

    # 寻找最长的有效JSON数组
    max_valid_length = 0
    result = []
    for start in range(len(text)):
        for end in range(start+1, len(text)+1):
            try:
                candidate = json.loads(text[start:end])
                if isinstance(candidate, list):
                    if (end - start) > max_valid_length:
                        max_valid_length = end - start
                        result = candidate
            except:
                continue
    return result
