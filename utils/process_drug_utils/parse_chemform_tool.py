# =================
# 规则库里的edit_op字段是基团缩写，后面是把这些基团换成对应的smiles的处理逻辑
#  （直接让LM模型操作smiles会多次重复并且和rationale对不上）
# =================
import json
import re

import requests
from rdkit import Chem
from utils.convenient_utils.suppress_useless_print import suppress_everything
from utils.convenient_utils.wordart import ColorText
import json
import os

def smiles_to_smarts(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return Chem.MolToSmarts(mol)



# 修复最后一截不全的json
def _repair_trailing_broken_entry(text: str) -> str:
    """
    处理类似：
    {
      "a": {...},
      "b": {...},
      "bad": {
        "smiles": "*C(=O)CCCC...
    <EOF>
    的情况：删除最后一个 key 行之后的所有内容，
    去掉前一项末尾的逗号，并补上一个收尾的大括号。
    """
    # 找到最后一个 key 行（形如换行 + 两个空格 + 引号）
    last_key_pos = text.rfind('\n  "')
    if last_key_pos == -1:
        # 实在找不到就返回一个最小合法 JSON
        return "{}\n"

    # 保留这个 key 之前的内容
    prefix = text[:last_key_pos].rstrip()

    # 去掉前一个条目后面多余的逗号（如果有的话）
    if prefix.endswith(','):
        prefix = prefix[:-1]

    # 确保最外层对象有收尾 '}'
    # 假设文件一开始就是 '{' 开头，不动它，只是补最后的 '}'
    if not prefix.endswith('\n'):
        prefix += '\n'
    fixed = prefix + '}\n'
    return fixed
# 从外部 JSON 文件加载基团库
# def load_substituents(filepath="substituents.json"):
#     try:
#         with open(filepath, "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {}
# 程序中断的时候有可能会把substituents.json写坏，这版代码里包含修复逻辑
def load_substituents(filepath="./rules/substituents.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 修复错误的json
        fixed_text = _repair_trailing_broken_entry(text)

        try:
            data = json.loads(fixed_text)
        except Exception as e_fix:
            ColorText.print(f"[SubstituentResolver] 修复后仍然不是合法 JSON，放弃修复: {e_fix}", ColorText.RED)
            return {}

        # 修复成功：回写修复后的 JSON
        try:
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(fixed_text)          # 先写到tmp文件里
            os.replace(tmp_path, filepath)   # 写完再替换
            ColorText.print(f"[SubstituentResolver] 已自动修复并回写 {filepath}", ColorText.GREEN)
        except Exception as e_write:
            ColorText.print(f"[SubstituentResolver] 修复结果回写失败: {e_write}", ColorText.RED)

        return data



def save_substituents(subs, path="./rules/substituents.json"):
    with open(path, "w") as f:
        json.dump(subs, f, indent=2)  # 把subs(dict)写进path


# 内置的通用推断规则（定义最常见的基团的映射规则）
def infer_substituent(group: str):
    """内置的基团推断（返回统一格式的 dict）"""
    group = group.strip()

    # 卤素
    if group in ["Cl", "Br", "F", "I"]:
        smiles = f"[{group}]"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": f"halogen ({group})"}
    # 羟基
    if group == "OH":
        smiles = "O"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": "hydroxyl"}
    # 氨基
    if group == "NH2":
        smiles = "N"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": "amine"}
    # 羧基
    if group == "COOH":
        smiles = "C(=O)O"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": "carboxyl"}
    # n氟甲基
    if group.startswith("CF"):
        nF = int(group[2:]) if group[2:].isdigit() else 1
        smiles = "C(" + ")(".join(["F"] * nF) + ")"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": f"fluoroalkyl (CF{nF})"}
    # 硝基
    if group == "NO2":
        smiles = "[N+](=O)[O-]"
        return {"smiles": smiles, "smarts": smiles_to_smarts(smiles), "description": "nitro"}

    return None

# 通用基团解析器
class SubstituentResolver:
    def __init__(self, custom_lib_path='./rules/substituents.json'):
        self.lib_path = custom_lib_path      # 实际的文件路径
        self.custom_lib = load_substituents(self.lib_path)  # 读到内存里的文件内容

    # 验证给出的smiles是不合法
    def validate_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    # 片段不能标准化,完整结构需要标准化,添加判断逻辑
    def _normalize_smiles(self, raw_smiles):
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {raw_smiles}")

        # 1 含有显式片段标记或多片段，【直接 canonical（不做 kekulize）】
        if "*" in raw_smiles or "." in raw_smiles:
            print("【S:SubstituentResolver】进行片段标准化")
            return Chem.MolToSmiles(mol, canonical=True)

        # 2 判断是否含环或芳香原子
        has_ring = mol.GetRingInfo().NumRings() > 0
        has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())

        # 有环或有芳香性，认为是完整分子【进行（kekulize）】
        if has_ring or has_aromatic:
            try:
                print("【S:SubstituentResolver】进行整体标准化")
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:  # Kekulize 失败则回退到不带 kekuleSmiles 的 canonical
                return Chem.MolToSmiles(mol, canonical=True)
            return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
        # 否则就是链状小片段，【直接 canonical（不做 kekulize）】
        else:   # 显式加氢 -> canonical -> 去氢，避免 RDKit 在隐式 H 情况下改变键阶
            mol_h = Chem.AddHs(mol)   # 显式加氢
            smi_h = Chem.MolToSmiles(mol_h, canonical=True, allHsExplicit=True)  # 标准化
            print(smi_h)
            mol2 = Chem.MolFromSmiles(smi_h)
            if mol2 is None:
                return Chem.MolToSmiles(mol, canonical=True)
            mol2 = Chem.RemoveHs(mol2)  # 去氢
            print("【S:SubstituentResolver】进行片段标准化")
            return Chem.MolToSmiles(mol2, canonical=True)

    def fetch_smiles_from_pubchem(self, name):
        """从 PubChem 查询 canonical SMILES（两步 + fallback record JSON）"""
        try:
            # 第一步：查 CID
            cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
            with suppress_everything():
                cid_resp = requests.get(cid_url, timeout=50)
            cid_resp.raise_for_status()
            cid_data = cid_resp.json()

            cids = cid_data.get("IdentifierList", {}).get("CID", [])
            if not cids:
                ColorText.print(f"【PubChem】 查询 {name} 没有返回 CID", ColorText.RED)
                return None
            cid = cids[0]  # 取第一个 CID
            ColorText.print(f"【PubChem】 查询 {name} 获得了 CID {cid}", ColorText.BLUE)

            # 第二步：CID -> CanonicalSMILES
            # 格式 https:/.../pug/<查询主题>/<属性域>/[<输出格式>][?<operation_options>]
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            with suppress_everything():
                smiles_resp = requests.get(smiles_url, timeout=50)
            smiles_resp.raise_for_status()
            smiles_data = smiles_resp.json()

            props = smiles_data.get("PropertyTable", {}).get("Properties", [])
            if props and "CanonicalSMILES" in props[0]:
                raw_smiles = props[0]["CanonicalSMILES"]
                if raw_smiles and self.validate_smiles(raw_smiles):
                    ColorText.print(f"【S: PubChem br1】 查询 {cid} 获得了 SMILES {raw_smiles}", ColorText.BLUE)
                    # 自定义规范化
                    norm_smiles = self._normalize_smiles(raw_smiles)
                    if norm_smiles:
                        ColorText.print(f"【S: RDKIT】 标准化之后 {norm_smiles}", ColorText.CYAN)
                        return norm_smiles
                    return raw_smiles

            # 如果没有CanonicalSMILES这个字段，换种方式拿: CID -> record/JSON
            record_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/JSON"
            with suppress_everything():
                record_resp = requests.get(record_url, timeout=10)
            record_resp.raise_for_status()
            record_data = record_resp.json()

            record_props = (
                record_data.get("PC_Compounds", [{}])[0].get("props", [])
            )

            raw_smiles = None
            for prop in record_props:
                urn = prop.get("urn", {})
                if urn.get("label") == "SMILES":
                    raw_smiles = prop.get("value", {}).get("sval")
                    break

            if raw_smiles and self.validate_smiles(raw_smiles):
                ColorText.print(f"【S: PubChem br2】 查询 {cid} 获得了 SMILES {raw_smiles}", ColorText.BLUE)
                # 用 RDKit 规范化
                norm_smiles = self._normalize_smiles(raw_smiles)
                if norm_smiles:
                    ColorText.print(f"【S: RDKIT】 标准化之后 {norm_smiles}", ColorText.CYAN)
                    return norm_smiles
                return raw_smiles

            print(f"【PubChem】 CID {cid} 没有 SMILES 信息")
            return None

        except Exception as e:
            print(f"【PubChem】 查询 {name} 失败: {e}")
            return None

    def resolve(self, group, llm):
        group = group.strip()
        # 0. 直接尝试当作 SMILES 解析
        try:
            mol = Chem.MolFromSmiles(group)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                smarts = smiles_to_smarts(smiles)
                print("【F: SubstituentResolver br0】直接当作SMILES解析成功", smiles)
                # 存进自定义库，避免重复解析
                self.custom_lib[group] = {
                    "smiles": smiles,
                    "smarts": smarts,
                    "description": f"direct SMILES: {group}"
                }
                save_substituents(self.custom_lib, self.lib_path)
                return smiles
        except Exception:
            pass
        # 1. 先查 substituents.json 库
        if group in self.custom_lib:
            print("【F: SubstituentResolver br1】在标准库里查到了结果", self.custom_lib[group]["smiles"])
            return self.custom_lib[group]["smiles"]
        # 如果 group 不在键中，尝试从 description 中匹配
        for key, value in self.custom_lib.items():      # 遍历 custom_lib 中的每一项
            if group in value["description"]:   # 检查 description 是否包含 group
                print(f"【F: SubstituentResolver br1】在描述中查到了结果，描述为：{value['description']}")
                return value["smiles"]  # 返回匹配到的 smiles

        # 2. 库中没有就内置推断
        inferred = infer_substituent(group)
        if inferred:
            self.custom_lib[group] = inferred
            save_substituents(self.custom_lib, self.lib_path)
            print("【F: SubstituentResolver br2】用内置推断逻辑得到了结果", inferred["smiles"])
            return inferred["smiles"]

        # 4. PubChem 查询
        pubchem_smiles = self.fetch_smiles_from_pubchem(group)
        if pubchem_smiles:
            smarts = smiles_to_smarts(pubchem_smiles)
            print("【F: SubstituentResolver br3】在pubchem里面查到了结果", pubchem_smiles)
            self.custom_lib[group] = {
                "smiles": pubchem_smiles,
                "smarts": smarts,
                "description": f"from PubChem: {group}"
            }
            save_substituents(self.custom_lib, self.lib_path)
            return pubchem_smiles

        # 3. 在没有就调用 LLM
        prompt = f"""
You are a chemistry assistant.
Please convert the following substituent abbreviation into:
1. SMILES (valid)
2. A short 1-2 word description
Return in JSON format with keys: smiles, description.

Substituent: {group}
"""  # 别改字符串拼接，不知道为什么会出错
        with suppress_everything():
            resp = llm.invoke(prompt).content.strip()
        print(resp)

        data = json.loads(resp)
        smiles = data["smiles"].strip()
        print("【F: SubstituentResolver br4】用llm查到了结果", smiles)
        description = data.get("description", "")
        smarts = smiles_to_smarts(smiles)

        # 存储为 dict
        self.custom_lib[group] = {
            "smiles": smiles,
            "smarts": smarts,
            "description": description
        }
        save_substituents(self.custom_lib, self.lib_path)

        return smiles