import json
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_path = "OpenR1-Math-196k-verified.jsonl"
    output_template = "OpenR1-Math-196k-verified-longest-{}%.jsonl"
    # output_template = "OpenR1-Math-196k-verified-shortest-{}%.jsonl"
    percentages = [0.01]

    print("加载数据...")
    data = load_jsonl(input_path)
    print(f"共加载 {len(data)} 条数据")

    print("根据 output 字段长度排序...")
    data_sorted = sorted(data, key=lambda x: len(x.get('output', '')), reverse=True)
    # data_sorted = sorted(data, key=lambda x: len(x.get('output', '')))

    total = len(data)
    for p in percentages:
        k = max(1, int(total * p))  # 至少取一条
        selected = data_sorted[:k]
        out_path = output_template.format(int(p * 100))
        print(f"保存前 {int(p*100)}% 最长的 output 到 {out_path}，共 {len(selected)} 条")
        # print(f"保存前 {int(p*100)}% 最短的 output 到 {out_path}，共 {len(selected)} 条")
        save_jsonl(selected, out_path)

    print("完成！")

if __name__ == "__main__":
    main()

# import json
# from pathlib import Path

# def load_jsonl(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return [json.loads(line) for line in f]

# def save_jsonl(data, path):
#     with open(path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# def main():
#     input_path = "OpenR1-Math-196k-verified.jsonl"
#     # output_template = "OpenR1-Math-196k-verified-longest-{}%.jsonl"
#     # output_template = "OpenR1-Math-196k-verified-shortest-{}%.jsonl"
#     output_template = "OpenR1-Math-196k-verified-middle-{}%.jsonl"
#     percentages = [0.02, 0.05, 0.10]

#     print("加载数据...")
#     data = load_jsonl(input_path)
#     print(f"共加载 {len(data)} 条数据")

#     print("根据 output 字段长度排序...")
#     data_sorted = sorted(data, key=lambda x: len(x.get('output', '')))

#     total = len(data)
#     for p in percentages:
#         k = max(1, int(total * p))  # 至少取一条
#         mid = total // 2
#         half_k = k // 2

#         # 中间截取
#         start = max(0, mid - half_k)
#         end = min(total, mid + half_k)
#         selected = data_sorted[start:end]

#         out_path = output_template.format(int(p * 100))
#         print(f"保存中间 {int(p*100)}% 的 output 到 {out_path}，共 {len(selected)} 条")
#         save_jsonl(selected, out_path)

#     print("完成！")

# if __name__ == "__main__":
#     main()
