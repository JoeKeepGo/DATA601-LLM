import json
import os

# 配置路径
FILE_A_PATH = "/home/data601/project/gold_standard.json"  # 文件 A
FILE_B_PATH = "/home/data601/project/obsolete_files/Combined.json" # 文件 B (处理后的)
FIELD_NAME = "original_comment" # 要对比的字段名

def load_data_robust(file_path):
    """
    兼容标准 JSON (List) 和 JSONL (Line-by-line) 的读取函数
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # 标准 JSON List
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"文件 {os.path.basename(file_path)} 为标准 JSON 列表")
            return data
    except:
        pass

    # JSONL (逐行)
    try:
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            # 简单的清理行尾逗号，防止报错
            if line.endswith(','): line = line[:-1]
            data.append(json.loads(line))
        print(f"文件 {os.path.basename(file_path)} 为 JSONL (逐行)")
        return data
    except Exception as e:
        print(f"无法解析文件 {file_path}: {e}")
        return []

def main():
    # 读取两个文件
    list_a = load_data_robust(FILE_A_PATH)
    list_b = load_data_robust(FILE_B_PATH)

    print(f"文件 A 条数: {len(list_a)}")
    print(f"文件 B 条数: {len(list_b)}")

    # 提取字段到集合 (Set)
    set_a = {item.get(FIELD_NAME, "").strip() for item in list_a if item.get(FIELD_NAME)}
    set_b = {item.get(FIELD_NAME, "").strip() for item in list_b if item.get(FIELD_NAME)}

    # 差集运算
    missing_in_b = set_a - set_b
    
    # missing_in_a
    missing_in_a = set_b - set_a

    print("-" * 30)
    
    # 输出结果
    if not missing_in_b and not missing_in_a:
        print("两个文件的 comment 内容完全一致！")
    else:
        if missing_in_b:
            print(f"文件 B 缺少了 {len(missing_in_b)} 条 (存在于 A 但不在 B):")
            # 打印前3条看看样子
            for i, comment in enumerate(list(missing_in_b)[:3]):
                print(f"   示例 {i+1}: {comment[:50]}...") 
            
            # 保存缺失内容到文件，方便查看
            with open("missing_in_file_b.json", "w", encoding="utf-8") as f:
                json.dump(list(missing_in_b), f, ensure_ascii=False, indent=2)
            print("完整缺失列表已保存至 missing_in_file_b.json")

        print("-" * 10)

        if missing_in_a:
            print(f"文件 B 多出了 {len(missing_in_a)} 条 (存在于 B 但不在 A):")
            for i, comment in enumerate(list(missing_in_a)[:3]):
                print(f"   示例 {i+1}: {comment[:50]}...")
            
            with open("extra_in_file_b.json", "w", encoding="utf-8") as f:
                json.dump(list(missing_in_a), f, ensure_ascii=False, indent=2)
            print("完整新增列表已保存至 extra_in_file_b.json")

if __name__ == "__main__":
    main()