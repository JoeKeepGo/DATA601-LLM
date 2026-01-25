import os
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 配置区域
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "/home/data601/project/fine_tuned_model"
TEST_DATA_PATH = "/home/data601/project/dataset_split/test.jsonl"
OUTPUT_RESULT_FILE = "/home/data601/project/evaluation_results.jsonl"

# 必须与训练时完全一致的 System Prompt
SYSTEM_PROMPT = """
You are a content moderation expert following a strict "Target-First Principle". Analyze the comment and output a JSON object.

1. JSON Output Schema:
{
  "original_comment": "The input text",
  "impact_level": <Integer 1-5>,
  "harm_category": ["List of strings from the Allowed Harm Categories below. Empty [] if harmless"],
  "target_identity": ["List of strings from the Allowed Target Identities below. Empty [] if none"],
  "reasoning": "Keywords: '...', '...'. Context: ... . Object Check: ... . Impact: ... .",
  "action_suggestion": "One of: 'None', 'Collapse', 'Warn User', 'Block/Delete', 'Escalate'"
}

2. Definitions & Logic:

Allowed Harm Categories (Select only from these):
- Violence Threat
- Group Derogation
- Sexual Harassment
- Toxicity
- Insult

Allowed Target Identities (Select only from these):
- Race/Ethnicity
- Gender
- LGBTQ+
- Religion
- Disability

Impact Level (1-5):
- 1 (Negligible): Harmless, political debate, or attacks on NON-HUMAN targets (objects, policies, concepts).
- 2 (Low): Sarcasm, mild rudeness, non-targeted profanity.
- 3 (Medium): Targeted insults, standard toxicity.
- 4 (High): Identity attacks, dehumanization.
- 5 (Severe): Violence threats, hate speech.

Action Suggestion Logic:
- Level 1 -> "None"
- Level 2 -> "Collapse" (Reduce visibility)
- Level 3 -> "Collapse" or "Warn User"
- Level 4 -> "Warn User" or "Block/Delete"
- Level 5 -> "Escalate" (Human review for safety risks)

CRITICAL: Object-First Principle
- If negative words (e.g., "trash", "disgusting", "criminal") target a OBJECT/POLICY (e.g., "imported trash", "criminal price"), it is Level 1.
- If they target a PERSON/GROUP (e.g., "white trash", "criminal immigrant"), it is Level 2+.

3. Special Considerations & Edge Cases:
- Substitution Test: If unsure whether a negative word (e.g., 'trash') refers to a person or thing, try replacing it with 'garbage' (for objects) or 'policy'. If the sentence still makes sense, the target is likely non-human (Level 1).
- Metaphorical Attacks: Be vigilant for sentences where an object word is used to refer to a group of people mentioned in context (e.g., "These people... we don't need more trash"). This is dehumanization (Level 4).
- The Comparative Trap: Sentences like "Dirt has higher IQ than him" use objects ("Dirt") grammatically, but the intent is to degrade a person via comparison. Treat these as Insults (Level 2+), not harmless object discussions.
- Political Discourse: Even strong language (e.g., 'criminal', 'disgusting') is Level 1 if it criticizes policies, institutions, or concepts, not people. This includes rhetorical exaggeration.
- Sarcasm/Dog Whistles: Consider the broader context and established coded language. Literal meaning may be neutral, but intent could be harmful.

4. Reasoning Field Format (Strict Requirement):
You must structure the 'reasoning' field with exactly these four parts:
1. Keywords: List trigger words.
2. Context: Briefly describe the topic.
3. Object Check: Identify if the target is Human or Non-Human (Apply the Substitution Test if helpful).
4. Impact: Explain the rationale based on the target and special considerations.
"""

def extract_json(text):
    """从模型输出中提取 JSON"""
    try:
        # 尝试直接解析
        return json.loads(text)
    except:
        # 尝试提取代码块
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match: return json.loads(match.group(1))
            # 尝试提取大括号内容
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match: return json.loads(match.group(1))
        except:
            pass
    return None

def main():
    print(f"Loading base model: {BASE_MODEL_ID}...")
    
    # 1. 加载基础模型 (BFloat16)
    # 显存优化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config, # 如果显存够大，可以去掉这行用全量加载
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    # 2. 加载 LoRA 适配器 (Fine-tuned Weights)
    print(f"Loading LoRA adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() # 切换到评估模式

    # 3. 加载测试数据集
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_samples = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                test_samples.append(json.loads(line))
            except: pass
    
    # 我们可以只测前 100 条来看看效果，或者测全部
    # test_samples = test_samples[:100] 
    print(f"Total test samples: {len(test_samples)}")

    correct_count = 0
    format_error_count = 0
    results = []

    print("Starting inference...")
    for sample in tqdm(test_samples):
        original_text = sample.get('original_comment', '')
        ground_truth_level = int(sample.get('impact_level', 1))
        
        # 4. 构建 Prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this: {original_text}"}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text_input], return_tensors="pt").to("cuda")

        # 5. 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, # 测试时温度调低，保证稳定性
                do_sample=False
            )
        
        # 获取新生成的 tokens
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 6. 解析结果
        pred_json = extract_json(output_text)
        
        is_correct = False
        is_format_error = False
        
        if pred_json and 'impact_level' in pred_json:
            pred_level = int(pred_json['impact_level'])
            
            # 判断逻辑：这里我们判断 "有害(>1)" vs "无害(1)" 是否一致
            # 或者您可以严格判断 pred_level == ground_truth_level
            
            # 宽松模式：只要都认为是有害，或者都认为是无害，就算对
            gt_is_harmful = ground_truth_level > 1
            pred_is_harmful = pred_level > 1
            
            if gt_is_harmful == pred_is_harmful:
                correct_count += 1
                is_correct = True
        else:
            format_error_count += 1
            is_format_error = True

        # 记录结果
        result_entry = {
            "original_text": original_text,
            "ground_truth": ground_truth_level,
            "prediction_raw": output_text,
            "parsed_prediction": pred_json,
            "is_correct": is_correct,
            "format_error": is_format_error
        }
        results.append(result_entry)

    # 7. 计算并打印指标
    total = len(test_samples)
    accuracy = (correct_count / total) * 100
    format_compliance = ((total - format_error_count) / total) * 100

    print("\n" + "="*30)
    print("TEST REPORT")
    print("="*30)
    print(f"Total Samples: {total}")
    print(f"Format Errors: {format_error_count}")
    print(f"Format Compliance: {format_compliance:.2f}%")
    print(f"Binary Accuracy (Harmful/Safe): {accuracy:.2f}%")
    print("="*30)

    # 保存详细结果
    with open(OUTPUT_RESULT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to {OUTPUT_RESULT_FILE}")

if __name__ == "__main__":
    main()