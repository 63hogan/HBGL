
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
        results.append({**cls_accuracy, 'true_max_level': true_max_level})



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
    