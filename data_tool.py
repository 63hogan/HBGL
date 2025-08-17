from transformers import BertTokenizer
import json
from collections import defaultdict


model_name = 'hfl/chinese-bert-wwm-ext'
print(f"正在从 '{model_name}' 加载 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)

def read_key_value_file(file_path):
    data_map = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Strip leading/trailing whitespace and then split the line by the first colon
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    data_map[key.strip().lower()] = value.strip()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}
    return data_map

def get_ztf_hierarchy_info(ztf_map):
    hiera = defaultdict(set)
    parent_dic = {}
    hiera['root'] = set()
    for label, _ in ztf_map.items():
        lenl = len(label)
        if lenl > 0 and label[:1] in ztf_map:
            hiera['root'].add(label[:1])
            parent_dic[label[:1]] = 'root'
            for i in range(1, lenl):
                p = label[:i]
                l = i+1
                if p in ztf_map:
                    while l <= lenl:
                        if label[:l] in ztf_map:
                            break
                        l+=1
                    if l <= lenl and label[:l] in ztf_map:
                        hiera[p].add(label[:l])
                        parent_dic[label[:l]] = p
    return hiera, parent_dic

def get_train_labels_hiera_info(train_cls_set, ztf_map):
    hiera = defaultdict(set)
    parent_dic = {}
    hiera['root'] = set()
    for label in train_cls_set:
        lenl = len(label)
        if lenl > 0 and label[:1] in ztf_map:
            hiera['root'].add(label[:1])
            parent_dic[label[:1]] = 'root'
            for i in range(1, lenl):
                p = label[:i]
                l = i+1
                if p in ztf_map:
                    while l <= lenl:
                        if label[:l] in ztf_map:
                            break
                        l+=1
                    if l <= lenl and label[:l] in ztf_map:
                        hiera[p].add(label[:l])
                        parent_dic[label[:l]] = p
    return hiera, parent_dic
    
def get_uniq_train_cls_from_json(file_path):
    cls_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'class' in item:
                        cls_set.add(item['class'].lower())
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_number} 行不是有效的JSON格式，已跳过。内容: '{line.strip()}'")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return cls_set

def get_unk_cls_set(train_cls,ztfMp):
    unk_label_num = 0
    unk_cls_set = set()
    # 遍历 train_cls 中的每个类别名称
    for cls in train_cls:
        if cls not in ztfMp:
            print(f"类别 '{cls}' 在 ztfMp 中未找到。")
        else:
            # print(f"类别 '{cls}' 在 ztfMp 中找到，对应值为: {ztfMp[cls]}")
            token_ids = tokenizer.encode(ztfMp[cls])
            # 检查 token_ids 中是否含 unk_token [unk] 的编码
            if tokenizer.unk_token_id in token_ids:
                print(f"类别 '{cls}' 的编码中包含 unk_token [unk] 的编码: {token_ids}")
                print(f"{cls}对应的类别名称是: {ztfMp[cls]}")
                unk_label_num += 1
                unk_cls_set.add(cls)
            else:
                # print(f"类别 '{cls}' 的编码中不包含 unk_token [unk] 的编码: {token_ids}")
                pass
    print(f"总共有 {unk_label_num} 个类别的编码中包含 unk_token [unk] 的编码。")
    return unk_cls_set

def clear_train_cls_from_json(file_path, output_path, unk_cls_set):
    try:
        clear_line_num = 0
        with open(file_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line_number, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'class' in item:
                        if item['class'] not in unk_cls_set:
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        else:
                            # print(f"已移除类别 '{item['class']}' 的数据。")
                            clear_line_num += 1
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_number} 行不是有效的JSON格式，已跳过。内容: '{line.strip()}'")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    print(f"已清除 {clear_line_num} 行包含 unk_cls_set 中类别的数据，并保存到 '{output_path}'。")

def get_label_idx(label_child_hieraset):
    label_idx = {}
    idx = 0
    sorted_roots = sorted(list(label_child_hieraset.get('root', [])))
    for cls in sorted_roots:
        if cls in label_idx:
            continue
        label_idx[cls] = idx
        idx += 1
        # children_to_visit = list(label_child_hieraset.get(cls, []))
        children_to_visit = sorted(list(label_child_hieraset.get(cls, [])))
        while children_to_visit:
            current_child = children_to_visit.pop(0)
            label_idx[current_child] = idx
            idx += 1
            # grandchildren = label_child_hieraset.get(current_child, [])
            grandchildren = sorted(list(label_child_hieraset.get(current_child, [])))
            children_to_visit.extend(grandchildren)
    return label_idx

def check_cnt(label_name_child_hiera_set):
    cnt = 0
    for label_name in label_name_child_hiera_set['root']:
        cnt += 1
        children_to_visit = list(label_name_child_hiera_set.get(label_name, []))
        while children_to_visit:
            current_child = children_to_visit.pop(0)
            cnt += 1
            grandchildren = label_name_child_hiera_set.get(current_child, [])
            children_to_visit.extend(grandchildren)
    return cnt
    

def get_labels_batch(train_label_hiera_set, ROOT_NODE = 'root', MAX_BATCH_SIZE = 500) :
    """
    将层级标签数据划分为多个批次，每个批次大小尽量接近但不超过500。
    优先将来自同一第一层祖先节点的标签放在同一批次中。

    Args:
        train_label_hiera_set: 一个 defaultdict(set)，存储标签的层级关系。
                               train_label_hiera_set['key'] 返回其所有子节点的集合。

    Returns:
        一个二维列表，其中每个一维列表是一个批次的标签集合。
    """
    
    memo = {}
    def get_all_descendants(node):
        if node in memo:
            return memo[node]
        descendants = {node}
        for child in train_label_hiera_set.get(node, set()):
            descendants.update(get_all_descendants(child))
        result = list(descendants)
        memo[node] = result
        return result

    groups_by_ancestor = defaultdict(list)
    if ROOT_NODE not in train_label_hiera_set:
        return []

    first_level_nodes = sorted(list(train_label_hiera_set[ROOT_NODE]))

    for node in first_level_nodes:
        subtree_nodes = get_all_descendants(node)
        
        if len(subtree_nodes) < MAX_BATCH_SIZE:
            groups_by_ancestor[node].append(subtree_nodes)
        else:
            queue = []
            initial_children = sorted(list(train_label_hiera_set.get(node, set())))
            for child in initial_children:
                queue.append((child, [node]))

            if not initial_children:
                groups_by_ancestor[node].append([node])
                continue

            while queue:
                current_node, parents_chain = queue.pop(0)
                
                current_subtree = get_all_descendants(current_node)
                
                if len(parents_chain) + len(current_subtree) < MAX_BATCH_SIZE:
                    groups_by_ancestor[node].append(parents_chain + current_subtree)
                else:
                    children = sorted(list(train_label_hiera_set.get(current_node, set())))
                    if not children:
                        groups_by_ancestor[node].append(parents_chain + current_subtree)
                    else:
                        for child in children:
                            new_parents_chain = parents_chain + [current_node]
                            queue.append((child, new_parents_chain))

    family_batches = []
    for ancestor in sorted(groups_by_ancestor.keys()):
        family_groups = sorted(groups_by_ancestor[ancestor], key=len, reverse=True)
        
        while family_groups:
            current_family_batch = family_groups.pop(0)
            
            i = len(family_groups) - 1
            while i >= 0:
                group = family_groups[i]
                if len(current_family_batch) + len(group) < MAX_BATCH_SIZE:
                    current_family_batch.extend(family_groups.pop(i))
                i -= 1
            family_batches.append(current_family_batch)

    family_batches.sort(key=len, reverse=True)
    
    final_batches = []
    while family_batches:
        current_batch = family_batches.pop(0)
        
        i = len(family_batches) - 1
        while i >= 0:
            group = family_batches[i]
            if len(current_batch) + len(group) < MAX_BATCH_SIZE:
                current_batch.extend(family_batches.pop(i))
            i -= 1
        unique_batch = list(set(current_batch))
        unique_batch = sorted(unique_batch, key=len)
        final_batches.append(unique_batch)
        
    return final_batches
