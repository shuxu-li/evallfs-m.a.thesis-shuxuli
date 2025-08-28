import os
import sys
import pandas as pd
from pathlib import Path
import json
import numpy as np
import copy
from collections import defaultdict as setdefaultdict
import argparse

#=============== PATH SETUP ============================
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
STRUCTURE = 'data/raw/lfs-structure-v3-simplified.json'
INSTANCE = 'data/processed/lfs-slctd.csv'
TARGETS_FILE = 'data/processed/lfs-structure-5_nodes_filtrated.json'

#=============== tool funcrions ============================
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_csv(file_path):
    return pd.read_csv(file_path, index_col=0)

def find_node_path(tree: dict, target: str ="Able1", path=None) -> list:
    if path is None:
        path = []
    for key, subtree in tree.items():
        current_path = path + [key]
        if key == target:
            return current_path
        elif isinstance(subtree, dict):
            result = find_node_path(subtree, target, current_path)
            if result:
                return result
    return None  # not found

def get_direct_sub(tree: dict, target: str = 'Able_i') -> list:
    path = find_node_path(tree, target)
    if not path:
        return []

    node = tree
    for key in path:
        node = node.get(key, {})
    
    if isinstance(node, dict):
        return list(node.keys())
    return []

def get_contrastive_labels(structure, label_name='Able1'):
    """对于结构里的某个label，返回所有层级的对比反例"""
    # 1. 首先定位到label_name，查看它的层级数量，找出所有的父节点
    super_nodes = find_node_path(structure, label_name)
    if super_nodes is None:
        return {}  # label_name not found, return empty dict
    super_nodes = super_nodes[::-1][1:]  # reverse the path
    depth = len(super_nodes)  # 该label的层级数量

    # 2. 然后从目标节点点开始遍历所有的直接子节点，把结果减去label_name
    contrasts = []
    for i in range(depth):
        layer_contrsts = []
        for sub in get_direct_sub(structure, super_nodes[i]):
            if sub != label_name and sub != super_nodes[i-1]:
                layer_contrsts.append(sub)
        contrasts.append(layer_contrsts)
    return contrasts

def get_leaf_weights(node: dict, name: str, current_weight=1.0) -> dict:
    """
    Since we test all nodes, we should attribute a weight to each leaf node of the target to test, that is distributing from the target node to its 'family_name' nodes.
    parameters:
    - node: the current node in the structure
    - name: the name of the current node
    - current_weight: the weight to be distributed to the leaf nodes
    returns:
    - weights: a dictionary mapping leaf node names to their weights
    """
    weights = {}
    if 'lf_éléments' in node:
        weights[name] = current_weight
    elif 'éléments' in node and node['éléments']:
        num_children = len(node['éléments'])
        for child_name, child_node in node['éléments'].items():
            weights.update(get_leaf_weights(child_node, child_name, current_weight / num_children))
    return weights

def get_elements(target_node: dict, node_name: str = None) -> dict:
    """
    获取所有 lf_id 的叶子节点元素。
    - 如果 target_node 本身是一个叶节点（包含 "lf_éléments"），直接处理；
    - 否则递归处理其子节点；
    返回结构为 {node_name: [lf_id, ...]}
    """
    target_elements = {}

    if isinstance(target_node, dict):
        # ✅ 情况 1：当前节点本身就是叶节点
        if "lf_éléments" in target_node:
            # 若未提供 node_name，则默认取单个键（用于最外层）
            if node_name is None:
                raise ValueError("必须提供 node_name，当 target_node 是叶节点时。")
            target_elements[node_name] = [lf['lf_id'] for lf in target_node['lf_éléments']]
        
        else:
            # ✅ 情况 2：递归查找子节点
            for k, v in target_node.items():
                sub_elements = get_elements(v, node_name=k)
                target_elements.update(sub_elements)

    return target_elements

def get_ex(df: pd.DataFrame, node: dict, node_name: str) -> dict:
    """
    Get positive examples for each target node in jn, from the DataFrame which contains LF examples.
    parameters:
    - df: DataFrame containing LF examples
    - node: JSON structure containing the LF nodes
    - node_name: the name of the node (string)
    returns:
    - targets_pos_instances: a dictionary mapping each node name to its positive example indices
    node_name: [list of idx of positive examples]
    """
    node_elements = get_elements(node, node_name)
    node_instances = {}
    for leaf_name, leaf_ids in node_elements.items():
        leaf_instances_idx = df[df['lf_id'].isin(leaf_ids)].index.tolist()
        node_instances[leaf_name] = leaf_instances_idx

    return node_instances

def select_pos_question_examples(
    pos_instances: dict[str, list[int]],
    weights: dict[str, float],
    rng: np.random.Generator,
    quest_num: int = 30
) -> tuple:
    """
    Select positive examples from pos_instances, according to node-level weights.

    Returns:
    - pos_prompt_instances: pos_instances with selected question indices removed
    - pos_quest_indices: flat list of selected indices for questions
    """

    node_names = list(pos_instances.keys())
    probs = np.array([weights[n] for n in node_names])
    probs = probs / probs.sum()

    pos_quest_indices = []
    pos_prompt_instances = {k: v.copy() for k, v in pos_instances.items()}

    pos_quest_num = quest_num // 2

    while len(pos_quest_indices) < pos_quest_num:
        chosen_node = rng.choice(node_names, p=probs)
        pool = pos_prompt_instances[chosen_node]
        if not pool:
            continue  # 跳过耗尽的类
        chosen_idx = int(rng.choice(pool))
        pos_quest_indices.append(chosen_idx)
        for node in pos_prompt_instances:
                pos_prompt_instances[node] = [idx for idx in pos_prompt_instances[node] if idx not in pos_quest_indices]

    # ✅ 验证 1：pos_prompt_instances 里不能包含 quest 中的任何 idx
    flat_prompt = [idx for lst in pos_prompt_instances.values() for idx in lst]
    overlap = set(flat_prompt) & set(pos_quest_indices)
    assert not overlap, f"[错误] Prompt 与 Quest 存在重合: {overlap}"

    # ✅ 验证 2：quest 里不能有重复 idx
    assert len(pos_quest_indices) == len(set(pos_quest_indices)), "[错误] Quest 中存在重复 idx"

    return  pos_quest_indices, pos_prompt_instances

def select_pos_prompt_for_all_questions(
    pos_prompt_instances: dict[str, list[int]],
    weights: dict[str, float],
    rng: np.random.Generator,
    quest_num: int = 30,
    k_shot_max: int =10
) -> tuple:
    """
    选出30组正例专门用于 prompt， 每组

    Returns:
    - pos_prompt_instances: pos_instances with selected question indices removed
    - pos_quest_indices: flat list of selected indices for questions
    """
    prompt_num = k_shot_max // 2
    node_names = list(pos_prompt_instances.keys())
    probs = np.array([weights[n] for n in node_names])
    probs = probs / probs.sum()

    pos_all_indices = {k: v.copy() for k, v in pos_prompt_instances.items()}
    target_pos_prompt = []
    for i in range(quest_num):
        single_quest_prompt = []
        while len(single_quest_prompt) < prompt_num:
            chosen_node = rng.choice(node_names, p=probs, replace=False)
            pool = pos_all_indices[chosen_node]
            if not pool:
                continue
            chosen_idx = int(rng.choice(pool))
            single_quest_prompt.append(chosen_idx)
        target_pos_prompt.append(single_quest_prompt)
    
        
    return target_pos_prompt


def select_neg_question_examples(neg_instances: dict, 
                                 neg_distr: list,
                                 rng: np.random.Generator,
                                 quest_num: int = 30
                                 ) -> tuple:
    """
    分层累积采样负例，返回每层对应的一组问题索引和可用的 prompt 池。
    """
    target_neg_question = []
    neg_prompt_final = {}
    for i in range(len(neg_distr)):  # 控制采样到第几层
        current_scope_nodes = copy.deepcopy(neg_distr[:i+1])
        current_scope_instances = {k: v for k, v in neg_instances.items()}
        current_scope_weight = 1.0 / len(current_scope_nodes)
        
        # 汇总每个节点的 leaf 权重
        all_scope_weights = {}
        for layer in current_scope_nodes:
            for node in layer:
                current_weights = get_leaf_weights(targets_nodes[node], node, current_scope_weight)
                all_scope_weights.update(current_weights)
        # 准备好选中的leafnodes名称及其权重矩阵
        scope_leaf_names = list(all_scope_weights.keys())
        probs = np.array([all_scope_weights[k] for k in scope_leaf_names])
        probs /= probs.sum()

        # 选择问题索引
        neg_quest_num = quest_num // 2
        neg_quest_indices = []
        neg_prompt_instances = {k: v.copy() for k, v in current_scope_instances.items()}
        
        while len(neg_quest_indices) < neg_quest_num:
            chosen_node = str(rng.choice(scope_leaf_names, p=probs))
            pool = neg_prompt_instances[chosen_node]
            if not pool:
                continue
            chosen_idx = int(rng.choice(pool, replace=False))
            neg_quest_indices.append(chosen_idx)
            for node in neg_prompt_instances:
                neg_prompt_instances[node] = [idx for idx in neg_prompt_instances[node] if idx not in neg_quest_indices]
        # 验证
        flat_prompt = [idx for lst in neg_prompt_instances.values() for idx in lst]
        overlap = set(flat_prompt) & set(neg_quest_indices)
        assert not overlap, f"[错误] Prompt 与 Quest 存在重合: {overlap}"
        assert len(neg_quest_indices) == len(set(neg_quest_indices)), "[错误] Quest 中存在重复 idx"
        # 验证数量 == 15
        assert len(neg_quest_indices) == neg_quest_num, f"[错误] Quest 中的数量不匹配: {len(neg_quest_indices)} != {neg_quest_num}"
        # 记录结果
        target_neg_question.append(neg_quest_indices)
        neg_prompt_final.update(neg_prompt_instances)

    # 返回结果
    return target_neg_question, neg_prompt_final

import copy

def select_neg_prompt_for_all_questions(
    neg_prompt_instances: dict[str, list[int]],
    neg_distr: list,
    rng: np.random.Generator,
    quest_num: int = 30,
    k_shot_max: int =10
) -> tuple:
    """
    对于每个范围，选出30组负例专门用于 prompt， 每组
    """
    prompt_num = k_shot_max // 2
    target_neg_prompt = []
    for i in range(len(neg_distr)): # 先来划分反例范围
        single_scope_prompt = [] # 单层反例
        current_scope_nodes = copy.deepcopy(neg_distr[:i+1])
        current_scope_instances = {k: v for k, v in neg_prompt_instances.items()}
        current_scope_weight = 1.0 / len(current_scope_nodes)
            
            # 汇总每个节点的 leaf 权重
        all_scope_weights = {}
        for layer in current_scope_nodes:
            for node in layer:
                current_weights = get_leaf_weights(targets_nodes[node], node, current_scope_weight)
                all_scope_weights.update(current_weights)
        # 准备好选中的leafnodes名称及其权重矩阵
        scope_leaf_names = list(all_scope_weights.keys())
        probs = np.array([all_scope_weights[k] for k in scope_leaf_names])
        probs /= probs.sum()

        # 选择问题索引
        
        for q in range(quest_num):
            neg_prompt_current_question = []
            neg_prompt_instances = {k: v.copy() for k, v in current_scope_instances.items()}
            while len(neg_prompt_current_question) < prompt_num:
                chosen_node = str(rng.choice(scope_leaf_names, p=probs))
                pool = neg_prompt_instances[chosen_node]
                if not pool:
                    continue
                chosen_idx = int(rng.choice(pool, replace=False))
                neg_prompt_current_question.append(chosen_idx)
            single_scope_prompt.append(neg_prompt_current_question)
        target_neg_prompt.append(single_scope_prompt)
                
    return target_neg_prompt
#=============== main function ============================

def generate_questions(
        simple_structure: dict,
        targets_nodes: dict,
        lfs_instances: pd.DataFrame,
        contrasts_mapping: dict,
        targets_list: list = None,
        random_seed: int = 42,
        quest_num: int = 30,
        k_shot_max: int = 10,
        output_path: str = None
) -> dict:
    """
    生成问题的函数
    parameters:
    - simple_structure: 结构化的 JSON 数据
    - targets_nodes: 目标节点的 JSON 数据
    - lfs_instances: 包含 LF 实例的 DataFrame
    - contrasts_mapping: 对比反例的映射
    - targets_list: 目标节点的列表
    - random_seed: 随机种子
    - quest_num: 每个问题的数量
    - k_shot_max: 每个问题的最大样本数量
    - output_path: 输出路径
    returns:
    - questions: 生成的问题列表
    """
    # 0. 简化实例数据
    lfs_instances = lfs_instances.copy().drop(columns=['target','source']) 
    # 1. 设置随机种子
    rng = np.random.default_rng(random_seed)

    # 2. 初始化问题列表
    questions = {}

    # 3. 遍历目标节点
    targets_list = targets_list if targets_list else list(targets_nodes.keys())
    for target in targets_list:
        target_questions = []
        # 4. 获取正例和反例
        pos_instances = get_ex(lfs_instances, targets_nodes[target], target)
        neg_instances = {}
        for i, contrast in enumerate(contrasts_mapping[target]):
            for j, c in enumerate(contrast):
                neg_instances.update(get_ex(lfs_instances, targets_nodes[c], c))

        # 5. 获取正例权重
        weights = get_leaf_weights(targets_nodes[target], target, 1.0)

        # 6. 选择正例和反例
        pos_quest_indices, pos_prompt_pool = select_pos_question_examples(pos_instances, weights, rng, quest_num)
        neg_quest_indices, neg_prompt_pool = select_neg_question_examples(neg_instances, contrasts_mapping[target], rng, quest_num)
        
        # 7. 选择正例和反例的提示
        pos_prompt = select_pos_prompt_for_all_questions(pos_prompt_pool, weights, rng, quest_num, k_shot_max) # list of list
        neg_prompt = select_neg_prompt_for_all_questions(neg_prompt_pool, contrasts_mapping[target], rng, quest_num, k_shot_max) # list of list of list
        
        question_indx = 1
        for scope, neg_questions in enumerate(neg_quest_indices):
            scope_id = scope + 1
            scope_questions = []
            
            # 处理正例问题
            for i in range(len(pos_quest_indices)):
                if i >= len(pos_prompt):
                    break
                    
                question = setdefaultdict(list)
                # 获取单个问题索引
                quest_idx = pos_quest_indices[i]
                
                # 获取该问题的正例和负例示例
                pos_indices = pos_prompt[i]
                neg_indices = neg_prompt[scope][i]
                
                # 转换为实际的数据
                pos_prompt_ex = [lfs_instances.iloc[idx].to_dict() for idx in pos_indices]
                neg_prompt_ex = [lfs_instances.iloc[idx].to_dict() for idx in neg_indices]
                
                # 获取问题实例
                quest_ex = lfs_instances.iloc[quest_idx].to_dict()
                
                # 填充问题数据
                question["target"] = target
                question["scope"] = scope_id
                question["Q_ID"] = question_indx
                question["expected"] = "Oui"  # 正例
                question["ex_question"] = quest_ex
                question["ex_pos_prompt"] = pos_prompt_ex
                question["ex_neg_prompt"] = neg_prompt_ex
                
                question_indx += 1
                scope_questions.append(dict(question))
            
            # 处理负例问题
            for i in range(len(neg_questions)):
                if i >= len(neg_prompt[scope]):
                    break
                    
                question = setdefaultdict(list)
                # 获取单个问题索引
                quest_idx = neg_questions[i]
                
                # 获取该问题的正例和负例示例
                pos_indices = pos_prompt[i + len(pos_quest_indices)]
                neg_indices = neg_prompt[scope][i + len(pos_quest_indices)]
                
                # 转换为实际的数据
                pos_prompt_ex = [lfs_instances.iloc[idx].to_dict() for idx in pos_indices]
                neg_prompt_ex = [lfs_instances.iloc[idx].to_dict() for idx in neg_indices]
                
                # 获取问题实例
                quest_ex = lfs_instances.iloc[quest_idx].to_dict()
                
                # 填充问题数据
                question["target"] = target
                question["scope"] = scope_id
                question["Q_ID"] = question_indx
                question["expected"] = "Non"  # 负例
                question["ex_question"] = quest_ex
                question["ex_pos_prompt"] = pos_prompt_ex
                question["ex_neg_prompt"] = neg_prompt_ex                
                question_indx += 1
                scope_questions.append(dict(question))            
            target_questions.extend(scope_questions)            
        questions[target] = target_questions  
    # 8. 保存问题
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
                    
if __name__ == "__main__":
    # 1. 加载数据
    STRUCTURE = load_json(STRUCTURE)
    INSTANCE = load_csv(INSTANCE)
    TARGETS = load_json(TARGETS_FILE)

    # 2. 获取对比反例
    contrasts_mapping = {}
    for target in TARGETS:
        contrasts_mapping[target] = get_contrastive_labels(STRUCTURE, target)

    # 3. 生成问题
    instance = INSTANCE.copy()
    targets_nodes = TARGETS.copy()
    generate_questions(
        simple_structure=STRUCTURE,
        targets_nodes=targets_nodes,
        lfs_instances=instance,
        contrasts_mapping=contrasts_mapping,
        random_seed=123,
        quest_num=20,
        k_shot_max=10,
        output_path='experiments/task_binary_3/data/questions-3.json'
    )
    print("问题生成完成！")
