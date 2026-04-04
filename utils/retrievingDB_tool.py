import json
import pickle

import requests
from e3fp.pipeline import fprints_from_mol
from unimol_tools.predictor import UniMolRepr
from tqdm import tqdm

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from mol2vec.features import mol2alt_sentence
from gensim.models import word2vec
from transformers import AutoTokenizer, AutoModel

from utils.convenient_utils.test_timeout import run_with_timeout
from .convenient_utils.suppress_useless_print import suppress_everything
from .convenient_utils.wordart import ColorText
import time

# USELESS  e3fp没法对应到原来的分子上面
# 生成 E3FP 指纹并提取激活的位索引。
def get_e3fp_bits(smiles, num_conformers=3, bits=1024):
    # 将 SMILES 转换为 RDKit 分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES 字符串。")
    mol = Chem.AddHs(mol)  # 添加氢原子以生成更准确的三维结构

    # E3FP 指纹依赖于分子的三维构象，所以先生成构象
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # 自动分配cpu
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)  # 根据构象参数生成多个conformers
    for conf_id in ids:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)  # 对每个conformer都进行能量最小化，这mol变成了一个包含多个 conformer 的分子对象。

    # 然后再计算 E3FP
    fprints = fprints_from_mol(mol, bits=bits)

    # 对每个构象的E3FP都提取激活的位索引，用set去重
    on_bits = set()
    for fp in fprints:
        on_bits.update(fp.indices)

    return sorted(on_bits)


# KS  ---------- 1 模型加载 ----------
# 1D: ChemBERTa
with suppress_everything():
    molT5_tokenizer = AutoTokenizer.from_pretrained(
        "/home/aita8180/data/mntdata/ziansong/chatDrug/sourcecode/models/ChemBERTa-zinc-base-v1")
        #
        # "/home/wangshuang/ziansong/p1/pretrain_models/seyonec/ChemBERTa-zinc-base-v1"
    molT5_model = AutoModel.from_pretrained(
        "/home/aita8180/data/mntdata/ziansong/chatDrug/sourcecode/models/ChemBERTa-zinc-base-v1")

# 2D: Mol2Vec
with suppress_everything():
    mol2vec_model = word2vec.Word2Vec.load(
        "/home/aita8180/data/mntdata/ziansong/chatDrug/sourcecode/models/mol2vec/model_300dim.pkl")
        # "/home/wangshuang/ziansong/p1/pretrain_models/mol2vec/model_300dim.pkl"
        #

# 3D: Uni-Mol
# ① 初始化 unimol_tools 的表示接口
with suppress_everything():
    unimol_clf = UniMolRepr(
        data_type='molecule',
        remove_hs=False,
        model_name='unimolv1',  # 或者 unimolv2
        # model_size='84m'      # 只有unimolv2时才要加这个参数
    )   # data_type='molecule' 表示小分子，remove_hs=False 保留氢


# ② 还有dimenet++, 调用 Flask 后端的分子嵌入dimenet服务
def molecule_embedding_tool_func(payload: dict) -> str:
    """
    输入格式：{"smiles": "CCO", "target": 7}
    返回格式：{"status": "1", "result": [...]}
    """
    url = "http://localhost:8899/api/mol-embedding"
    try:
        # payload = json.loads(json_text)
        # smiles, target = payload.get("smiles"), payload.get("target")
        # ColorText.print(f"正在使用dimenet计算{smiles}的{target}属性嵌入", ColorText.BLUE)
        # 请求后端
        with suppress_everything():
            response = requests.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()

        if "embedding" in response_data:  # 判断是否是成功响应
            return json.dumps({"status": "1", "result": response_data["embedding"]})
        else:  # 如果没有返回embedding字段，说明失败
            error_msg = response_data.get("error", "No 'embedding' in response")
            return json.dumps({"status": "0", "error": error_msg})

    except Exception as e:
        return json.dumps({"status": "-1", "message": str(e)})


# KS  ---------- 2 嵌入提取函数 ----------
# ① 用【ChemBERTa】计算【1D Embedding】
def embed_fragment_molt5(smiles_fragment):
    inputs = molT5_tokenizer(smiles_fragment, return_tensors="pt")
    with torch.no_grad():
        outputs = molT5_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()


# ② 用【Mol2Vec】计算【2D Embedding】
def embed_fragment_mol2vec(mol, radius=1):
    try:
        Chem.GetSymmSSSR(mol)  # 强制初始化环信息，避免后续 mol2vec 报错
        sentence = mol2alt_sentence(mol, radius=radius)  # 用 mol2vec 分词函数构建“句子”
        vecs = []
        for token in sentence:
            with suppress_everything():
                vec = mol2vec_model.wv[token] if token in mol2vec_model.wv else np.zeros(mol2vec_model.vector_size)
            vecs.append(vec)
        return np.mean(vecs, axis=0) if vecs else np.zeros(mol2vec_model.vector_size)  # 平均该子结构所有 token 的向量
    except Exception as e:
        ColorText.print(f"mol2vec 编码失败：{e}", ColorText.RED)
        return np.zeros(mol2vec_model.vector_size)


# ③ 用【UniMol】计算【3D Embedding】
def embed_fragment_unimol(smiles):
    try:
        with suppress_everything():
            unimol_repr = unimol_clf.get_repr([smiles], return_atomic_reprs=False)
        emb = np.array(unimol_repr['cls_repr'])[0]  # 提取第一个分子的 CLS embedding
        return emb
    except Exception as e:
        print(f"⚠️ UniMolTools 处理出错：{e}")
        return None


# ④ 用【DimeNet】计算【3D Embedding】
def embed_fragment_dimenet(smiles, prop_type):
    # with suppress_everything():
    # 不同的task预测不同的qm9属性嵌入
    if prop_type == 'logP':
        # targets = [0, 1, 2, 3]
        targets = list(range(12))
    elif prop_type == 'QED':
        # targets = [0, 1, 2, 3, 11]
        targets = list(range(12))
    elif prop_type == 'TPSA':
        # targets = [0, 1, 7, 8, 9, 10]
        targets = list(range(12))
    else:
        # targets = [1, 2, 3, 11]
        targets = list(range(12))

    full_embedding = []
    for target in targets:

        input_json = {"smiles": smiles, "target": target}
        response_str = molecule_embedding_tool_func(input_json)  # 这里的返回指不直接是flaskapi定义的了，经由这个函数自己重新定义了
        response = json.loads(response_str)
        # smiles, target = payload.get("smiles"), payload.get("target")
        if response["status"] == '1':
            full_embedding.extend(response["result"])  # 立即拼接
        else:
            ColorText.print(f"DimeNet出错(target={target})：{response.get('error')}，用{128 * len(targets)}的全零向量以替代", ColorText.RED)
            return np.zeros(128 * len(targets))  # 一旦失败就直接全零

    return np.array(full_embedding)


# ④ 不同的task对应不同的优化属性prop，这个函数位二者的转换
def task2prop(task):
    if task == 101 or task == 102:
        return "logP"
    elif task == 103 or task == 104:
        return "QED"
    elif task == 105 or task == 106:
        return "TPSA"
    else:
        return "H"


# KS  3 处理单个smile的嵌入(最终嵌入)
def compute_final_embedding(smiles, prop_type, use_models=None):
    if use_models is None:
        use_models = {"1d": True, "2d": True, "3d_unimol": True, "3d_dimenet": False}
    # print(use_models)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKIT解析不了的smile")

    embedding_1d, embedding_2d, embedding_3d_1, embedding_3d_2 =[],[],[],[]
    ### -------- 1D 信息：基于 BRICS 拆分的子结构序列表示（MolT5） --------
    if use_models.get("1d"):
        # 获得分子的BRICS片段(1D)
        if len(smiles) > 800:
            print("1d：不是我喜欢的smiles，我直接置0")
            embedding_1d = np.zeros(768)
        else:
            frag_smiles_list = list(BRICS.BRICSDecompose(mol))
            # print("ok")
            embeddings_1d = []
            for frag in frag_smiles_list:
                # 使用 ChemBERTa 对每个子结构进行编码，取 CLS 向量（即第一个 token 的向量）
                output_emb = embed_fragment_molt5(frag)
                embeddings_1d.append(output_emb)
            # 聚合所有片段向量（取均值），作为分子的 1D 表征
            embedding_1d = np.mean(embeddings_1d, axis=0) if embeddings_1d else np.zeros(768)  # 为空就置全0
        # print("1d嵌入处理完成...", end='', flush=True)
        # time.sleep(3)
        # print("\r" + " " * 20 + "\r", end='')  # 覆盖并回到行首
    ### -------- 2D 信息：ECFP 位图 → 提取原子环境 → 每个子图 → Mol2Vec --------
    if use_models.get("2d"):
        radius = 2  # ECFP4
        bit_info = {}  # 记录最终哈希列表里活跃的位（即符合当前smile的子环境）
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048, bitInfo=bit_info)  # RDkit中的morgen指纹就是ECFP
        on_bits = list(fp.GetOnBits())  # 提取激活位

        # 获取2D embedding
        embeddings_2d = []
        for bit in on_bits:
            atom_idx, rad = bit_info[bit][0]  # 获取激活位对应的原子 idx 和半径
            env = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx))  # 提取对应的原子环境子图

            # 使用 mol2vec 对每个子图进行编码，返回的是子图每个token的平均向量
            output_emb_2d = embed_fragment_mol2vec(env, radius=1)  # ECFP4在语义上对应mol2vec的r=1
            embeddings_2d.append(output_emb_2d)
        embedding_2d = np.mean(embeddings_2d, axis=0) if embeddings_2d else np.zeros(mol2vec_model.vector_size)
        # print("2d嵌入处理完成...", end='', flush=True)
    ### -------- 3D 信息：E3FP 位图 → 理论上提取子结构 → get_e3fp_bits --------
    if use_models.get("3d_unimol"):
        embedding_3d_1 = embed_fragment_unimol(smiles)
        # embedding_3d_1 = run_with_timeout(embed_fragment_unimol, args=(smiles,), timeout=300)
        if embedding_3d_1 is None:
            print("unimol超时，3D embedding置0")
            embedding_3d_1 = np.zeros(512)
        # print("3d嵌入处理完成...", end='', flush=True)
        # time.sleep(3)
        # print("\r" + " " * 20 + "\r", end='')
    if use_models.get("3d_dimenet"):
        print(f"测试 SMILES: {smiles}, 属性类型: {prop_type}")
        embedding_3d_2 = embed_fragment_dimenet(smiles, prop_type)

    ### -------- 最终拼接：1D | 2D | 3D --------
    # E_final = np.concatenate([embedding_1d, embedding_2d, embedding_3d])
    # print(embedding_1d, embedding_2d, embedding_3d_1, embedding_3d_2)
    E_final = np.concatenate([embedding_1d, embedding_2d, embedding_3d_1, embedding_3d_2])
    return E_final


# KS  4 处理多个smile的嵌入(构造数据库时使用)
# ========= 【新增】批量处理的final embedding（构造数据库的时候用） ========= #
def compute_final_embedding_batch(smiles_list, prop_type, batch_size=64, use_models=None):
    """
    批量处理一堆SMILES，返回对应的最终embedding列表
    """
    if use_models is None:
        use_models = {"1d": True, "2d": True, "3d_unimol": True, "3d_dimenet": False}
    all_embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), colour="#00FF00", desc="生成embedding(epoch)", leave=True, mininterval=2):
        batch = smiles_list[i:i + batch_size]  # 循环每个batch（切片会自动处理越界）
        # 初始化各维度embedding容器
        batch_embeddings_1d, batch_embeddings_2d, batch_embeddings_3d_1, batch_embeddings_3d_2 = [],[],[],[]
        ### -------- 1D embedding (ChemBERTa) --------
        if use_models.get("1d", True):
            for smiles in tqdm(batch, desc="1D embedding - ChemBERTa(batch)", leave=False):
            # for smiles in batch:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    batch_embeddings_1d.append(np.zeros(768))
                    continue
                frag_smiles_list = list(BRICS.BRICSDecompose(mol))
                frag_embeddings = []
                for frag in frag_smiles_list:
                    try:
                        output_emb = embed_fragment_molt5(frag)
                        frag_embeddings.append(output_emb)
                    except Exception:
                        pass
                emb_1d = np.mean(frag_embeddings, axis=0) if frag_embeddings else np.zeros(768)
                batch_embeddings_1d.append(emb_1d)
            batch_embeddings_1d = np.vstack(batch_embeddings_1d)  # shape: (batch_size, 768)

        ### -------- 2D embedding (Mol2Vec) --------
        if use_models.get("2d", True):
            for smiles in tqdm(batch, desc="2D embedding - Mol2Vec(batch)", leave=False):
            # for smiles in batch:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    batch_embeddings_2d.append(np.zeros(mol2vec_model.vector_size))
                    continue
                radius = 2
                bit_info = {}
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048, bitInfo=bit_info)
                on_bits = list(fp.GetOnBits())

                env_embeddings = []
                for bit in on_bits:
                    atom_idx, rad = bit_info[bit][0]
                    env = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx))
                    try:
                        output_emb_2d = embed_fragment_mol2vec(env, radius=1)
                        env_embeddings.append(output_emb_2d)
                    except Exception:
                        pass
                emb_2d = np.mean(env_embeddings, axis=0) if env_embeddings else np.zeros(mol2vec_model.vector_size)
                batch_embeddings_2d.append(emb_2d)
            batch_embeddings_2d = np.vstack(batch_embeddings_2d)  # shape: (batch_size, mol2vec_dim)

        # -------- 3D embedding (UniMol) --------
        if use_models.get("3d_unimol", True):
            try:
                with suppress_everything():
                    unimol_repr = unimol_clf.get_repr(batch, return_atomic_reprs=False)
                batch_embeddings_3d_1 = np.array(unimol_repr['cls_repr'])  # shape: (batch_size, 768)
            except Exception as e:
                print(f"⚠️ UniMol batch出错：{e}")
                batch_embeddings_3d_1 = np.zeros((len(batch), 768))
        # -------- 3D embedding (DimeNet) --------
        if use_models.get("3d_dimenet", True):
            batch_embeddings_3d_2 = []
            for smiles in tqdm(batch, desc="3D embedding - DimeNet(batch)", leave=False):
                embedding = embed_fragment_dimenet(smiles, prop_type)
                batch_embeddings_3d_2.append(embedding)

        # -------- 拼接有效embedding --------
        for j in range(len(batch)):
            components = []

            if use_models.get("1d", True) and batch_embeddings_1d[j] is not None:
                components.append(batch_embeddings_1d[j])
            if use_models.get("2d", True) and batch_embeddings_2d[j] is not None:
                components.append(batch_embeddings_2d[j])
            if use_models.get("3d_unimol", True):
                emb_3d = batch_embeddings_3d_1[j] if j < len(batch_embeddings_3d_1) else None
                if emb_3d is not None:
                    components.append(emb_3d)
            if use_models.get("3d_dimenet", True):
                emb_dm = batch_embeddings_3d_2[j] if j < len(batch_embeddings_3d_2) else None
                if emb_dm is not None:
                    components.append(emb_dm)

            if not components:
                raise ValueError(f"SMILES 编号 {i + j} 无任何有效 embedding")

            E_final = np.concatenate(components, axis=0)
            all_embeddings.append(E_final)

    return np.array(all_embeddings)  # shape: (n_total, feature_dim)


# USELESS  测试dimenetAPI调用的函数，，现在没用了
def test_embed_fragment_dimenet():
    test_smiles = "CCO"  # 乙醇
    test_prop_type = "QED"

    print(f"测试 SMILES: {test_smiles}, 属性类型: {test_prop_type}")
    embedding = embed_fragment_dimenet(test_smiles, test_prop_type)

    if isinstance(embedding, list) and all(isinstance(x, float) for x in embedding):
        print(f"✅ 嵌入获取成功，长度: {len(embedding)}")
    elif isinstance(embedding, np.ndarray):
        print(f"✅ 嵌入获取成功，长度: {embedding.shape}")
    else:
        print("❌ 嵌入获取失败或格式不对")


def cal_emb_testset():
    # === 路径设置 ===
    input_file = "./data/small_molecule/small_molecule_editing.txt"  # 测试集
    output_pkl = "./embed_caches/embedding_cache_optTest.pkl"       # 输出文件
    use_models = {"1d": True, "2d": True, "3d_unimol": True, "3d_dimenet": False}

    with open(input_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"共读取 {len(smiles_list)} 个 SMILES 分子")

    # === 生成嵌入并存储为 dict ===
    embeddings_dict = {}

    for smiles in tqdm(smiles_list, desc="生成 SMILES 嵌入"):
        try:
            emb = compute_final_embedding(smiles, None, use_models=use_models)
            embeddings_dict[smiles] = emb
        except Exception as e:
            print(f"⚠️ 跳过 SMILES: {smiles} {e}")
            embeddings_dict[smiles] = None  # 可选择不存入失败项

    # === 保存为 .pkl 文件 ===
    with open(output_pkl, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"✅ 所有分子嵌入已保存到：{output_pkl}")

if __name__ == "__main__":
    # 运行dimenet嵌入测试
    # test_input = '{"smiles": "CCO", "target": 7}'
    # print(molecule_embedding_tool_func(test_input))
    # test_embed_fragment_dimenet()
    """
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
    print("测试阿司匹林的smiles")
    # 调用 compute_final_embedding 函数计算分子的最终表征

    final_embedding = compute_final_embedding(test_smiles, 'QED')

    # 打印最终的表征向量的形状，验证输出
    print("最终的嵌入向量的形状:", final_embedding.shape)
    print("嵌入向量:", final_embedding)

    test_smiless = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "CCO", "CCO", "CCO"]
    final_embedding2 = compute_final_embedding_batch(test_smiless, 'QED', 2)
    # 打印最终的表征向量的形状，验证输出
    print("最终的嵌入向量的形状:", final_embedding2.shape)
    print("嵌入向量:", final_embedding2)
    """
    print("预存测试集嵌入")
    cal_emb_testset()