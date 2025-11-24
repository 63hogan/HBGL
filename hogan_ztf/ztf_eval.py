
import json
import json
import re
import ast
import numpy as np
import pandas as pd
from collections import Counter


        
def eval_output(pred_data_path, true_data_path='/root/autodl-tmp/HBGL/data/ztfData/eval/ori_eval_data_src_tgt.jsonl'):
    print(f"start evaluating file:{pred_data_path}")
    true_data = []
    pred_data = []
    line_count = 0
    parse_errors = 0

    def filter_strings_compact(input_str):
        return [part for part in input_str.split() 
                if part.startswith('__') and part.endswith('__') and len(part) > 4]

    print(f"Loading data from {true_data_path}...")
    try:
        with open(true_data_path, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    item = json.loads(line)
                    true_data.append(filter_strings_compact(item['tgt']))
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    print(f"Warning: Skipping line {line_count} due to JSON parsing error: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file not found at {true_data_path}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        exit()
    print(f"Loaded {len(true_data)} true records (skipped {parse_errors} lines due to parse errors).")

    line_count
    try:
        with open(pred_data_path, 'r') as f:
            for line in f:
                line_count += 1
                pred_data.append(filter_strings_compact(line.strip()))
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {pred_data_path}")
        exit()
    print(f"Loaded {len(pred_data)} pred records.")

    def evaluate_cls_accuracy(pred_cls, true_cls):
        results = {'exact_match':0}
        max_level = 0
        
        if len(true_cls) == 0 :
            return results, max_level
        
        max_level = len(true_cls)
        for i in range(1, max_level+1):
            results[f'level_{i}_match'] = 0
        
        common_length = min(len(pred_cls), len(true_cls))
        for i in range(common_length):
            if pred_cls[i] == true_cls[i]:
                results[f'level_{i+1}_match'] = 1
            else:
                break
        if len(pred_cls) == len(true_cls) and results[f'level_{common_length}_match'] == 1:
            results['exact_match'] = 1
        return results, max_level

    results = []
    all_true_cls_levels = Counter()


    for i in range(len(true_data)):
        j = i
        if i > len(pred_data) - 1:
            j = len(pred_data) - 1
        cls_accuracy, true_max_level = evaluate_cls_accuracy(pred_data[j], true_data[i])
        all_true_cls_levels.update([f'level_{j}_match' for j in range(1, true_max_level + 1)])

        level_1_label = true_data[i][0] if len(true_data[i]) > 0 else None
        
        results.append({**cls_accuracy, 'true_max_level': true_max_level, 'root_category': level_1_label})



    df_results = pd.DataFrame(results)
    clc_accuracy_summary = {'Exact Match Accuracy': df_results['exact_match'].mean()}
    max_level_overall = 0
    for col in df_results.columns:
        if col.startswith('level_') and col.endswith('_match'):
            level_num = int(col.split('_')[1])
            max_level_overall = max(max_level_overall, level_num)
            # Calculate accuracy only for records where the ground truth CLC has this level
            total_at_level = all_true_cls_levels[col] # Get count from our counter
            if total_at_level > 0:
                accuracy_at_level = df_results[col].sum() / total_at_level
            else:
                accuracy_at_level = np.nan # Or 0, based on preference if level never occurs
            clc_accuracy_summary[f'Level {level_num} Accuracy'] = accuracy_at_level
            
    print(f"result of {pred_data_path}:")
    print("\nCLC Classification Accuracy:")
    print(f"  Exact Match Accuracy: {clc_accuracy_summary['Exact Match Accuracy']:.4f}")
    level_accuracies = []
    levels = []
    for i in range(1, max_level_overall + 1):
        key = f'Level {i} Accuracy'
        if key in clc_accuracy_summary and not np.isnan(clc_accuracy_summary[key]):
            acc = clc_accuracy_summary[key]
            count = all_true_cls_levels[f'level_{i}_match']
            print(f"  {key}: {acc:.4f} (based on {count} samples)")
            level_accuracies.append(acc)
            levels.append(f'Level {i}')
            
            
# --- Part 2: 按 Level 1 大类进行细分统计 (新增部分) ---
    print("\n" + "="*40)
    print("BREAKDOWN BY LEVEL 1 CATEGORY:")
    
    # 按 root_category 分组
    # 排除 'Unknown' (如果存在空标签数据)
    valid_groups = df_results[df_results['root_category'] != "Unknown"].groupby('root_category')
    
    # 按类别名称排序输出
    for cat_name, group_df in sorted(valid_groups, key=lambda x: x[0]):
        print(f"\n>>> Category: {cat_name} (Total Samples: {len(group_df)})")
        
        # 1. 该类别的完全匹配率
        exact_acc = group_df['exact_match'].mean()
        print(f"   Exact Match: {exact_acc:.4f}")
        
        # 2. 该类别下每一层的准确率
        # 注意：分母应该是该类别下，Ground Truth 确实拥有该层级的数据量
        for i in range(1, max_level_overall + 1):
            col_match = f'level_{i}_match'
            
            # 如果这一列不存在于当前group (比如该类都很浅)，跳过
            if col_match not in group_df.columns:
                continue
                
            # 分母：该组中，真实深度 >= i 的样本数
            # 因为只有真实深度 >= i，我们才统计了 level_i_match (无论是0还是1)
            # 在 evaluate_cls_accuracy 中，如果层级不够，字典里是不会有那个key的? 
            # 不，原代码里字典初始化只到了 true_max_level。
            # Pandas转换后，缺失值会是 NaN (或者 0，取决于填充)。
            # 这里最稳妥的方法是看 true_max_level >= i
            
            valid_sample_count = len(group_df[group_df['true_max_level'] >= i])
            
            if valid_sample_count > 0:
                # 分子：匹配成功的数量 (fillna(0)防止NaN干扰，虽然上面过滤了)
                match_count = group_df.loc[group_df['true_max_level'] >= i, col_match].sum()
                acc = match_count / valid_sample_count
                print(f"   Level {i}: {acc:.4f} ({match_count}/{valid_sample_count})")
            else:
                # 该类别下没有这么深的数据
                pass

def eval_output_topk(pred_data_path, true_data_path='/root/autodl-tmp/HBGL/data/ztfData/eval/ori_eval_data_src_tgt.jsonl'):
    print(f"start evaluating file: {pred_data_path}")
    true_data = []
    pred_data = []
    line_count = 0
    parse_errors = 0

    def filter_strings_compact(input_str):
        # 提取形如 __label__ 的标签
        return [part for part in input_str.split() 
                if part.startswith('__') and part.endswith('__') and len(part) > 4]

    # 1. 加载真实标签 (True Data)
    print(f"Loading data from {true_data_path}...")
    try:
        with open(true_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    item = json.loads(line)
                    # true_data 中存储的是完整的层级路径列表，例如 ['__A__', '__B__']
                    true_data.append(filter_strings_compact(item['tgt']))
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    print(f"Warning: Skipping line {line_count} due to JSON parsing error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {true_data_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        return
    print(f"Loaded {len(true_data)} true records (skipped {parse_errors} lines due to parse errors).")

    # 2. 加载预测标签 (Pred Data - Top 5)
    # 假设 pred 文件每一行包含了前5个概率最高的 Level 1 标签，用空格分隔
    try:
        with open(pred_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                # pred_data 中存储的是候选列表，例如 ['__Cand1__', '__Cand2__', '__Cand3__', ...]
                pred_data.append(filter_strings_compact(line.strip()))
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {pred_data_path}")
        return
    print(f"Loaded {len(pred_data)} pred records.")

    results = []

    # 3. 评估循环 (计算 Top-5 Hit)
    for i in range(len(true_data)):
        # 对齐索引，防止预测文件行数少于真实文件
        j = i
        if i > len(pred_data) - 1:
            j = len(pred_data) - 1
        
        # 获取当前样本的真实 Level 1 标签
        # 如果真实数据为空（没有标签），跳过或记为 None
        if len(true_data[i]) > 0:
            true_level_1 = true_data[i][0]
        else:
            continue # 没有真实标签无法评估

        # 获取当前样本的预测候选集 (Top 5)
        pred_candidates = pred_data[j]

        # 核心判断：真实标签是否在预测的 Top 5 列表中
        is_hit = 1 if true_level_1 in pred_candidates else 0

        results.append({
            'root_category': true_level_1,
            'is_hit': is_hit
        })

    # 4. 统计结果
    if not results:
        print("No valid data to evaluate.")
        return

    df_results = pd.DataFrame(results)

    # 计算全局准确率 (Overall Accuracy)
    total_accuracy = df_results['is_hit'].mean()

    print(f"result of {pred_data_path}:")
    print("\n" + "="*40)
    print(f"OVERALL LEVEL 1 TOP-5 ACCURACY: {total_accuracy:.4f}")
    print("="*40)

    # 5. 按 Level 1 大类进行细分统计
    print("\nBREAKDOWN BY LEVEL 1 CATEGORY (Top-5 Hit Rate):")
    
    # 按 root_category 分组
    valid_groups = df_results.groupby('root_category')
    
    # 按照类别名称排序输出
    # key=lambda x: x[0] 是按分组的键(类别名)排序
    for cat_name, group_df in sorted(valid_groups, key=lambda x: x[0]):
        total_samples = len(group_df)
        hit_count = group_df['is_hit'].sum()
        acc = group_df['is_hit'].mean()
        
        print(f"Category: {cat_name:<20} | Accuracy: {acc:.4f} ({hit_count}/{total_samples})")