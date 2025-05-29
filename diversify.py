import json
import random
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def stratified_sample(data, label_key, percentages):
    # 分类数据
    label_to_items = defaultdict(list)
    for item in data:
        label = item[label_key]
        label_to_items[label].append(item)

    # 采样
    results = {p: [] for p in percentages}
    for label, items in label_to_items.items():
        for p in percentages:
            k = max(1, int(len(items) * p))  # 至少采样1个
            results[p].extend(random.sample(items, k))

    return results

def main():
    input_path = "OpenR1-Math-196k-verified.jsonl"
    output_template = "OpenR1-Math-196k-verified-diverse-{}%.jsonl"
    percentages = [0.02, 0.05, 0.10, 0.20]

    print("加载数据...")
    data = load_jsonl(input_path)
    print(f"共加载 {len(data)} 条数据")

    print("开始分层采样...")
    sampled_data = stratified_sample(data, 'problem_type', percentages)

    for p in percentages:
        out_path = output_template.format(int(p * 100))
        print(f"保存 {int(p*100)}% 的采样数据到 {out_path}，共 {len(sampled_data[p])} 条")
        save_jsonl(sampled_data[p], out_path)

    print("完成！")

if __name__ == "__main__":
    main()
