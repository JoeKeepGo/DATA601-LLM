import json
from collections import Counter

path = "/home/data601/project/dataset/train/train_quality.jsonl"
levels = []

with open(path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            # 假设你的格式里有 impact_level，如果没有需从 content 解析
            # 这里假设你直接存了 json，如果不是，需要调整解析逻辑
            # 简单粗暴法：直接搜字符串
            if '"impact_level": 1' in line: levels.append(1)
            elif '"impact_level": 2' in line: levels.append(2)
            elif '"impact_level": 3' in line: levels.append(3)
            elif '"impact_level": 4' in line: levels.append(4)
            elif '"impact_level": 5' in line: levels.append(5)
        except: pass

print(Counter(levels))