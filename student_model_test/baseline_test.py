import torch
import json
import re
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置区域 =================

#MODEL_ID = "google/gemma-3-12b-it"
#MODEL_ALIAS = "Gemma-3-12B"

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ALIAS = "Qwen-3-4B"

#MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
#MODEL_ALIAS = "DeepSeek-R1-14B"


# 路径配置
BASE_DIR = "/home/data601/project"
TRAIN_FILE = os.path.join(BASE_DIR, "dataset_split/train.jsonl")
TEST_FILE = os.path.join(BASE_DIR, "dataset_split/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_results")

# JSON 定义
SCHEMA_DEFINITION = """
You are a content moderation expert. Analyze the comment and output a JSON object with:
1. "original_comment": The input text.
2. "impact_level": Integer 1-5 (1=Negligible, 2=Low, 3=Medium, 4=High, 5=Severe).
3. "harm_category": List of strings (e.g., ["Violence Threat", "Group Derogation", "Sexual Harassment", "Toxicity", "Insult"]). Return [] if none.
4. "target_identity": List of strings (e.g., ["Race", "Gender", "LGBTQ+", "Religion"]). Return [] if none.
5. "reasoning": A coherent text explanation.
6. "action_suggestion": String ("None", "Collapse", "Warn User", "Block/Delete", "Escalate").
"""

def load_model(model_name):
    print(f"Loading model: {model_name} in 8-bit.")
    
    HF_TOKEN = os.getenv("HF_TOKEN")

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    
    # Padding 修复逻辑
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"      
    )
    
    return model, tokenizer

def extract_json_from_text(text):
    """
    针对 DeepSeek R1 和其他模型的健壮的提取。
    DeepSeek R1 可能会在 JSON 前输出 <think>...</think>，此正则可忽略这些内容。
    """
    try:
        # 尝试寻找最外层的 {}
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            return json.loads(clean_text)
        # 如果没有找到 {}，尝试直接解析
        return json.loads(text)
    except:
        return None

def build_prompt(tokenizer, comment, few_shot_examples):
    """
    使用 Chat Template 构建 Prompt。
    将 Few-Shot 示例作为之前的对话历史注入。
    """
    # 1. 构建 System Prompt (包含 Schema 和 Examples)
    # 适配不支持 System Role 的模型，将 System Prompt 和 Examples 合并
    instruction_text = f"{SCHEMA_DEFINITION}\n\nHere are some examples of how to analyze comments:\n"
    
    # 伪造对话历史作为 Few-Shot
    messages = []
    
    # Gemma对 System Role 支持有限，将 Instruction 放入第一条 User 消息 
    first_content = instruction_text
    
    for ex in few_shot_examples:
        # 清理示例数据，只保留需要的字段
        clean_ex = {k: ex[k] for k in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"] if k in ex}
        
        # 将示例拼接到开头的指令中，或者作为独立的 User/Assistant 对
        # 为了通用性，将示例直接作为 User/Assistant 轮次
        messages.append({"role": "user", "content": f"Analyze this: {ex['original_comment']}"})
        messages.append({"role": "assistant", "content": json.dumps(clean_ex, ensure_ascii=False)})

    # 添加当前需要推理的问题
    # 如果是第一条，需要加上 System Instruction
    current_query = comment
    
    # 重组 Messages：在第一条 User 消息前加上 System Instruction
    final_messages = [
        {"role": "user", "content": f"{instruction_text}\n\nAnalyze this: {few_shot_examples[0]['original_comment']}"},
        {"role": "assistant", "content": json.dumps({k: few_shot_examples[0][k] for k in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"] if k in few_shot_examples[0]}, ensure_ascii=False)}
    ]
    
    # 添加剩余的 Few-shot
    for ex in few_shot_examples[1:]:
        clean_ex = {k: ex[k] for k in ["original_comment", "impact_level", "harm_category", "target_identity", "reasoning", "action_suggestion"] if k in ex}
        final_messages.append({"role": "user", "content": f"Analyze this: {ex['original_comment']}"})
        final_messages.append({"role": "assistant", "content": json.dumps(clean_ex, ensure_ascii=False)})
        
    # 添加当前测试数据
    final_messages.append({"role": "user", "content": f"Analyze this: {comment}"})

    # 应用模板
    prompt = tokenizer.apply_chat_template(
        final_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载数据
    print(f"Loading datasets from {TRAIN_FILE} ...")
    try:
        train_data = []
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            for line in f: 
                if line.strip(): train_data.append(json.loads(line))
            
        test_data = []
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f: 
                if line.strip(): test_data.append(json.loads(line))
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Check path: {TRAIN_FILE}")
        return

    # 2. 准备 Few-Shot 示例 (3条)
    few_shot_examples = train_data[:3]
    print(f"Loaded {len(test_data)} test samples.")

    # 3. 加载模型
    model, tokenizer = load_model(MODEL_ID)

    results = []
    parse_errors = 0
    y_true = []
    y_pred = []

    print(f"Running inference using {MODEL_ALIAS}...")
    
    for item in tqdm(test_data):
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
        
        # 结果截取：只获取新生成的部分
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_part = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # JSON 解析
        pred_json = extract_json_from_text(response_part)
        
        # 统计指标
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
            except:
                pass 
        
        if not valid_parse:
            parse_errors += 1

        results.append({
            "id": item.get('id', 'unknown'),
            "original_comment": item['original_comment'],
            "ground_truth_level": gt_level,
            "prediction_level": pred_level if valid_parse else "ERROR",
            "raw_response": response_part,
            "parsed_json": pred_json,
            "is_valid_format": valid_parse
        })

    # 4. 计算指标
    total = len(test_data)
    format_acc = (total - parse_errors) / total if total > 0 else 0
    
    f1 = 0.0
    acc = 0.0
    if y_true:
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
    
    summary = {
        "model": MODEL_ALIAS,
        "format_adherence": format_acc,
        "impact_level_f1_macro": f1,
        "impact_level_accuracy": acc,
        "total_samples": total,
        "parse_errors": parse_errors
    }
    
    print("\n" + "="*30)
    print(f"RESULT SUMMARY: {MODEL_ALIAS}")
    print(json.dumps(summary, indent=2))
    print("="*30)

    # 5. 保存结果
    detail_path = os.path.join(OUTPUT_DIR, f"prediction_{MODEL_ALIAS}.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    metric_path = os.path.join(OUTPUT_DIR, f"metrics_{MODEL_ALIAS}.json")
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()