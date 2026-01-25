import json
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 你的训练数据文件路径
DATA_FILE_PATH = "dataset/train/train_quality.jsonl"
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
# ===========================================

def check_raw_file_content(filepath):
    # 模块 1: 检查物理文件内容
    # 确认文件是否已经去除了 original_comment
    print(f"[Module 1] Checking raw file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    leakage_count = 0
    checked_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5: break # 只检查前5行
                
                checked_count += 1
                data = json.loads(line)
                
                # 提取 Assistant 的回复
                messages = data.get("messages", [])
                assistant_content = ""
                for msg in messages:
                    if msg["role"] == "assistant":
                        assistant_content = msg["content"]
                        break
                
                # 检查是否存在泄露字段
                if "original_comment" in assistant_content:
                    print(f"FAIL: Leakage found in line {i+1}")
                    print(f"Content snippet: {assistant_content[:100]}...")
                    leakage_count += 1
                else:
                    print(f"PASS: Line {i+1} is clean.")

    except Exception as e:
        print(f"Error reading file: {e}")

    if leakage_count == 0:
        print("[Module 1] Raw file check PASSED.\n")
    else:
        print(f"[Module 1] Raw file check FAILED. Found {leakage_count} leaks.\n")


def check_dataset_loading(filepath):
    # 模块 2: 检查 HuggingFace 数据集加载与缓存
    # 确认 load_dataset 是否读取了旧的缓存
    print(f"[Module 2] Checking HuggingFace dataset loading")
    
    try:
        # 尝试加载数据集，不使用 cache_dir 参数，使用默认缓存
        dataset = load_dataset('json', data_files=filepath, split='train')
        
        print(f"Loaded {len(dataset)} samples.")
        first_sample = dataset[0]
        
        # 解析 messages
        messages = first_sample.get("messages", [])
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_content = msg["content"]
                break
        
        # 检查加载后的数据
        if "original_comment" in assistant_content:
            print("FAIL: Dataset loaded with 'original_comment'. CACHE ISSUE DETECTED.")
            print("Suggestion: Run 'rm -rf ~/.cache/huggingface/datasets'")
        else:
            print("PASS: Loaded dataset is clean.")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    print("[Module 2] Finished.\n")


def check_tokenizer_template(filepath, model_path):
    # 模块 3: 检查 Tokenizer 解码结果
    # 模拟训练时的输入，查看 labels 是否包含泄露内容
    print(f"[Module 3] Checking Tokenizer Output")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = json.loads(f.readline())
            
        messages = first_line["messages"]
        
        # 使用 chat_template 处理
        try:
            tokenized_output = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            decoded_text = tokenizer.decode(tokenized_output[0], skip_special_tokens=False)
            
            print("Decoded Tokenizer Output:")
            print("-" * 20)
            # 打印部分内容用于人工核对
            print(decoded_text[-300:]) 
            print("-" * 20)
            
            if "original_comment" in decoded_text and '"original_comment"' in decoded_text:
                 # 双重检查，防止原文里本来就有这个词
                print("WARNING: 'original_comment' key found in tokenized output.")
            else:
                print("PASS: No 'original_comment' key found in tokenized text.")
                
        except Exception as e:
            print(f"Tokenizer check skipped or failed (template issue): {e}")
            
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    print("[Module 3] Finished.\n")

if __name__ == "__main__":
    print("Starting Debug Tools...\n")
    
    # 1. 检查文件
    check_raw_file_content(DATA_FILE_PATH)
    
    # 2. 检查缓存加载
    check_dataset_loading(DATA_FILE_PATH)
    
    # 3. 检查 Tokenizer
    if os.path.exists(MODEL_PATH) or "/" in MODEL_PATH:
        check_tokenizer_template(DATA_FILE_PATH, MODEL_PATH)
    else:
        print("Skipping Module 3: Invalid MODEL_PATH configured.")