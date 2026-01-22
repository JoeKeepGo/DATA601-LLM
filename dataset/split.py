import json
import os
import random
import sys

INPUT_FILE = "/home/data601/project/dataset/tmp/obsolete_files/gold_standard.json" 
OUTPUT_DIR = "/home/data601/project/dataset/tmp/dataset_split"
TRAIN_SIZE = 400

def load_tricky_json(file_path):
    print(f"正在读取: {file_path}")
    
    if not os.path.exists(file_path):
        print("文件不存在")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        print("文件是空的")
        sys.exit(1)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print("格式 A: 标准 JSON 列表")
            return data
    except:
        pass
    try:
        print("检测到裸露对象，尝试人工添加 [] 包裹.")
        clean_content = content.rstrip(',')
        wrapped_content = f"[{clean_content}]"
        data = json.loads(wrapped_content)
        print("格式 B: 修复成功")
        return data
    except json.JSONDecodeError as e:
        print(f"   -> 修复失败: {e}")
    try:
        print("尝试按行读取 (JSONL).")
        data = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.endswith(','): 
                line = line[:-1]
            if not line: continue
            data.append(json.loads(line))
        if data:
            print("格式 C: 逐行处理成功")
            return data
    except:
        pass

    print("文件有误。")
    sys.exit(1)

def deduplicate_data(data_list):
    """
    字典转换为排序后的 JSON 字符串进行比对去重。
    """
    unique_data = []
    seen = set()
    
    for item in data_list:
        # sort_keys=True 非常重要，保证 {'a':1, 'b':2} 和 {'b':2, 'a':1} 被视为相同
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        
        if item_str not in seen:
            seen.add(item_str)
            unique_data.append(item)
            
    return unique_data

def main():
    # 加载数据
    data = load_tricky_json(INPUT_FILE)
    original_count = len(data)
    print(f"原始数据条数: {original_count}")

    if original_count == 0:
        print("数据为空，退出。")
        return

    # 去重
    print("正在进行去重.")
    data = deduplicate_data(data)
    deduplicated_count = len(data)
    removed_count = original_count - deduplicated_count
    print(f"去重完成。剩余: {deduplicated_count} 条 (剔除了 {removed_count} 条重复数据)")
    # ====================

    if len(data) < 5:
        print("有效数据太少，请检查源文件是否完整。")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 打乱与切分
    random.seed(42)
    random.shuffle(data)
    
    # 防止越界，逻辑稍微优化了一下
    if len(data) < 500:
        real_train_size = int(len(data) * 0.8) # 如果数据少，强制按 8:2 切分
    else:
        real_train_size = min(TRAIN_SIZE, len(data))

    train_data = data[:real_train_size]
    test_data = data[real_train_size:]
    
    print(f"切分完成: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")

    # 4. 保存为标准 JSONL
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(os.path.join(OUTPUT_DIR, "test.jsonl"), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"文件已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()