import torch
import json
import re
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 全局配置

# Alibaba Qwen
#MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
#MODEL_ALIAS = "Qwen-3-4B"

# Google Gemma
#MODEL_ID = "google/gemma-3-12b-it"
#MODEL_ALIAS = "Gemma-3-12B"

# DeepSeek
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_ALIAS = "DeepSeek-R1-8B"

HF_TOKEN = os.getenv("HF_TOKEN")

# 实验参数
NUM_RUNS = 5      # 运行次数
SEED_BASE = 42    # 随机种子基数

# 路径配置
BASE_DIR = "/home/data601/project"
TRAIN_FILE = os.path.join(BASE_DIR, "dataset_split/train.jsonl")
TEST_FILE = os.path.join(BASE_DIR, "dataset_split/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_results")

# Schema
SCHEMA_DEFINITION = """
You are a content moderation expert. Analyze the comment and output a JSON object with:
1. "original_comment": The input text.
2. "impact_level": Integer 1-5 (1=Negligible, 2=Low, 3=Medium, 4=High, 5=Severe).
3. "harm_category": List of strings (e.g., ["Violence Threat", "Group Derogation", "Sexual Harassment", "Toxicity", "Insult"]). Return [] if none.
4. "target_identity": List of strings (e.g., ["Race", "Gender", "LGBTQ+", "Religion"]). Return [] if none.
5. "reasoning": A coherent text explanation.
6. "action_suggestion": String ("None", "Collapse", "Warn User", "Block/Delete", "Escalate").
"""

# ================= 🔧 工具函数 =================

def set_seed(seed):
    """设置所有随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(file_path):
    """数据加载函数"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_json_from_text(text):
    """
    针对 LLM 输出的 JSON 提取。
    新增支持 Markdown 代码块、Thinking 标签过滤和纯文本提取。
    """
    try:
        # 1. 移除 DeepSeek 的 <think> 标签
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()

        # 2. 尝试提取 ```json ... ``` 代码块
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        # 3. 尝试提取最外层 {}
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        # 4. 尝试直接解析
        return json.loads(text)
    except:
        return None

def save_json(data, path):
    """JSON 保存封装，处理 Numpy 类型无法序列化的问题"""
    def default_converter(o):
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default_converter)

def load_model_8bit(model_name):
    print(f"🔄 Loading model: {model_name} in 8-bit.")
    
    # 清理显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" # 如果报错可改为 "eager"
    )
    
    return model, tokenizer

def build_prompt(tokenizer, comment, few_shot_examples):
    instruction_text = f"{SCHEMA_DEFINITION}\n\nHere are some examples of how to analyze comments:\n"
    messages = []
    
    # 构造 Few-shot
    first_ex = few_shot_examples[0]
    clean_first = {k: first_ex[k] for k in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"] if k in first_ex}
    
    messages.append({"role": "user", "content": f"{instruction_text}\n\nAnalyze this: {first_ex['original_comment']}"})
    messages.append({"role": "assistant", "content": json.dumps(clean_first, ensure_ascii=False)})

    for ex in few_shot_examples[1:]:
        clean_ex = {k: ex[k] for k in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"] if k in ex}
        messages.append({"role": "user", "content": f"Analyze this: {ex['original_comment']}"})
        messages.append({"role": "assistant", "content": json.dumps(clean_ex, ensure_ascii=False)})
        
    messages.append({"role": "user", "content": f"Analyze this: {comment}"})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 主逻辑

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. 加载数据
    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE)
    if not train_data or not test_data: return

    few_shot_examples = train_data[:3]
    print(f"Loaded {len(test_data)} test samples.")

    # 加载模型
    model, tokenizer = load_model_8bit(MODEL_ID)

    # 历史记录容器
    history = {
        "f1": [], "acc": [], "format": []
    }
    
    # 记录每次运行的概况
    runs_meta = []

    print(f"\nStarting {NUM_RUNS} iterations for {MODEL_ALIAS}.\n")

    # 循环运行
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"Run {run_idx}/{NUM_RUNS}")
        
        # 显式设置种子：Base + Run Index
        current_seed = SEED_BASE + run_idx
        set_seed(current_seed)
        
        results = []
        y_true = []
        y_pred = []
        parse_errors = 0

        # 推理循环
        for item in tqdm(test_data, desc=f"Run {run_idx}"):
            prompt = build_prompt(tokenizer, item['original_comment'], few_shot_examples)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.1,  
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response_part = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            pred_json = extract_json_from_text(response_part)
            
            gt_level = item.get('impact_level', 1)
            pred_level = 1 
            valid_parse = False
            
            if pred_json:
                try:
                    p = int(pred_json.get('impact_level', 1))
                    pred_level = max(1, min(5, p))
                    y_true.append(gt_level)
                    y_pred.append(pred_level)
                    valid_parse = True
                except: pass 
            
            if not valid_parse: parse_errors += 1

            results.append({
                "id": item.get('id', 'unknown'),
                "ground_truth_level": gt_level,
                "prediction_level": pred_level if valid_parse else "ERROR",
                "is_valid": valid_parse,
                "parsed_json": pred_json,
                "raw_response": response_part
            })

        # 计算单次指标
        total = len(test_data)
        format_acc = (total - parse_errors) / total if total > 0 else 0
        f1 = f1_score(y_true, y_pred, average='macro') if y_true else 0.0
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        
        # 记录到历史
        history["f1"].append(f1)
        history["acc"].append(acc)
        history["format"].append(format_acc)
        
        # 生成分类报告表格
        report_str = classification_report(y_true, y_pred, labels=[1,2,3,4,5], zero_division=0) if y_true else "No valid predictions"
        
        print(f"Run {run_idx}: F1={f1:.4f}, Acc={acc:.4f}, Format={format_acc:.4f}")

        # 保存单次文件
        # 保存详细预测结果 (JSON)
        save_json(results, os.path.join(OUTPUT_DIR, f"predictions_{MODEL_ALIAS}_run_{run_idx}.json"))
        
        # 保存单次指标 (JSON)
        meta = {
            "run_id": run_idx,
            "seed": current_seed,
            "metrics": {"f1": f1, "acc": acc, "format": format_acc},
            "errors": parse_errors
        }
        runs_meta.append(meta)
        save_json(meta, os.path.join(OUTPUT_DIR, f"metrics_{MODEL_ALIAS}_run_{run_idx}.json"))
        
        # 保存分类报告表格
        with open(os.path.join(OUTPUT_DIR, f"report_{MODEL_ALIAS}_run_{run_idx}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Classification Report for Run {run_idx}\n")
            f.write("="*60 + "\n")
            f.write(report_str)
            f.write("\n" + "="*60 + "\n")

    # Numpy 统计与汇总
    print("\n" + "="*40)
    print(f"FINAL STATISTICS ({NUM_RUNS} runs)")
    print("="*40)

    # 使用 Numpy 计算 Mean 和 Std
    # 保存到 JSON 前，把 numpy 类型 (float32/64) 转为 python float
    final_stats = {
        "model": MODEL_ALIAS,
        "total_runs": NUM_RUNS,
        "aggregated_metrics": {
            "f1_macro": {
                "mean": float(np.mean(history["f1"])),
                "std": float(np.std(history["f1"])),
                "raw": history["f1"]
            },
            "accuracy": {
                "mean": float(np.mean(history["acc"])),
                "std": float(np.std(history["acc"])),
                "raw": history["acc"]
            },
            "format_adherence": {
                "mean": float(np.mean(history["format"])),
                "std": float(np.std(history["format"])),
                "raw": history["format"]
            }
        },
        "all_runs_meta": runs_meta
    }

    print(json.dumps(final_stats, indent=2))
    
    # 保存最终汇总文件
    save_json(final_stats, os.path.join(OUTPUT_DIR, f"final_summary_{MODEL_ALIAS}.json"))
    
    print(f"\nPipeline Complete. All files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()