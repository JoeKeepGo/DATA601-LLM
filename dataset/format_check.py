import json
import re

INPUT_FILE = "/home/data601/project/dataset/tmp/synthetic_train_final_v7prompt.jsonl"  # 文件路径

def check_data_health():
    total = 0
    valid_format = 0
    strict_keywords = ["Keywords:", "Context:", "Object Check:", "Impact:"]
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    reasoning = data.get('reasoning', "")
                    total += 1
                    
                    # 检查是否包含所有必须的推理步骤
                    if all(k in reasoning for k in strict_keywords):
                        valid_format += 1
                except:
                    pass
    except FileNotFoundError:
        print("找不到文件！")
        return

    if total == 0:
        print("文件是空的！")
        return

    health_rate = (valid_format / total) * 100
    print(f"总数据量: {total}")
    print(f"符合新格式的数据: {valid_format}")
    print(f"格式符合率: {health_rate:.2f}%")

    if health_rate > 90:
        print("数据健康，格式良好！")
    elif health_rate > 50:
        print("部分中毒")
    else:
        print("严重中毒。")

check_data_health()