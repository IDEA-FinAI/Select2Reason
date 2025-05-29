# 打开当前文件夹下第一个jsonl文件
# 根据字段evol_is_correct_1为True或False，分别加入到两个jsonl文件中，分别是OpenR1-Math-196k-verified-10%-random-evol-1-correct.jsonl和OpenR1-Math-196k-verified-10%-random-evol-1-incorrect.jsonl

import os
import json

# 获取当前文件夹下的所有 jsonl 文件
jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]

if not jsonl_files:
    print("No JSONL files found in the current directory.")
    exit()

# 选择第一个 jsonl 文件
input_file = jsonl_files[0]
correct_file = f"evol-1-correct.jsonl"
incorrect_file = f"evol-1-incorrect.jsonl"

# 读取并分类数据
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(correct_file, 'w', encoding='utf-8') as correct_outfile, \
     open(incorrect_file, 'w', encoding='utf-8') as incorrect_outfile:
    
    for line in infile:
        try:
            data = json.loads(line)
            if data.get("evol_is_correct_1"):
                correct_outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                incorrect_outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print("Skipping invalid JSON line:", line)

print(f"Processing complete. Correct entries saved in {correct_file}, incorrect entries saved in {incorrect_file}.")
