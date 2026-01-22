import os
import json
import re
import random
import time
import threading
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 全局配置开关
# True: 使用 DeepSeek API (并发)
# False: 使用本地模型 (单线程)
USE_API = True

# API 配置
API_KEY = os.getenv("API_KEY")
API_BASE_URL = "https://api.deepseek.com"
API_MODEL_NAME = "deepseek-reasoner"
# 并发线程数设置为 100 以最大化处理速度
API_MAX_WORKERS = 100

# 单次批处理的最大请求上限
# 建议与 API_MAX_WORKERS 保持一致，既能跑满并发，又防止单次请求过多导致 Token 浪费
# 当 strict_limit=True 时，此限制能显著减少溢出浪费
MAX_BATCH_SIZE_LIMIT = 1000

# 本地模型配置
LOCAL_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
HF_TOKEN = os.getenv("HF_TOKEN")

# 文件路径配置
BASE_DIR = "/home/data601/project"
# 黄金数据集路径，用于提取 Few-Shot 示例
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "dataset_split/test.jsonl")
# 最终输出文件路径，实时写入
OUTPUT_FILE = os.path.join(BASE_DIR, "constructed_dataset/synthetic_train_final.jsonl")

# 任务目标参数
TARGET_HARMFUL_MIN = 10000  # 目标收集的有害样本数量
MAX_RETRIES = 3             # 单条数据生成失败时的最大重试次数
FEW_SHOT_COUNT = 2          # 每次推理使用的黄金样本示例数量
MAX_NEW_TOKENS = 2048       # 最大生成长度

# 线程锁
# 用于多线程环境下安全地写入文件，防止数据竞争
FILE_LOCK = threading.Lock()

# 系统提示词定义
# 规定 JSON 输出格式、评分标准 (Object-First Principle) 及推理字段结构
SCHEMA_DEFINITION = """
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

# 推理引擎类
# 统一封装 API 调用和本地模型推理接口
class InferenceEngine:
    def __init__(self, use_api):
        self.use_api = use_api
        if self.use_api:
            from openai import OpenAI
            self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"Loading local model: {LOCAL_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_ID, device_map="auto", token=HF_TOKEN, torch_dtype=torch.float16
            )

    def generate(self, messages, temperature=0.1):
        if self.use_api:
            try:
                response = self.client.chat.completions.create(
                    model=API_MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=MAX_NEW_TOKENS, 
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API Error: {e}")
                return None
        else:
            import torch
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temperature,
                    do_sample=(temperature > 0.0)
                )
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# 辅助函数：从模型输出文本中提取 JSON
# 处理 Markdown 标记和可能的思维链内容
def extract_json_strict(text):
    if not text: return None
    try: return json.loads(text)
    except: pass
    try:
        if "<think>" in text: text = text.split("</think>")[-1].strip()
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return None

# 辅助函数：构建 Prompt 消息列表
# 包含 System Prompt、Few-Shot 示例和当前用户输入
def build_messages(text, golden_pool):
    messages = [{"role": "system", "content": SCHEMA_DEFINITION}]
    if golden_pool and len(golden_pool) >= FEW_SHOT_COUNT:
        samples = random.sample(golden_pool, FEW_SHOT_COUNT)
        for s in samples:
            raw_reasoning = s.get('reasoning', "No reasoning provided.")
            if "Object Check:" not in raw_reasoning:
                formatted_reasoning = (
                    f"Keywords: [Implied]. Context: [Golden Sample]. "
                    f"Object Check: [See Impact]. Impact: {raw_reasoning}"
                )
            else:
                formatted_reasoning = raw_reasoning

            output_example = json.dumps({
                "original_comment": s['original_comment'],
                "impact_level": s['impact_level'],
                "harm_category": s.get('harm_category', []),
                "target_identity": s.get('target_identity', []),
                "reasoning": formatted_reasoning,
                "action_suggestion": s.get('action_suggestion', "None")
            }, ensure_ascii=False)
            
            messages.append({"role": "user", "content": f"Analyze this: {s['original_comment']}"})
            messages.append({"role": "assistant", "content": output_example})
            
    messages.append({"role": "user", "content": f"Analyze this: {text}"})
    return messages

# 辅助函数：线程安全文件写入
def append_to_file_thread_safe(data, filepath):
    with FILE_LOCK:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

# 主程序入口
def main():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    engine = InferenceEngine(USE_API)

    print("Loading datasets...")
    golden_pool = []
    if os.path.exists(GOLDEN_DATASET_PATH):
        with open(GOLDEN_DATASET_PATH, 'r') as f:
            for line in f:
                try: golden_pool.append(json.loads(line))
                except: pass
    
    dataset = load_dataset("google/civil_comments", split="train")
    df = dataset.to_pandas()[['text', 'toxicity']]

    # 读取历史进度，用于断点续传
    existing_harmful = []
    existing_safe = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if int(item.get('impact_level', 1)) > 1:
                        existing_harmful.append(item)
                    else:
                        existing_safe.append(item)
                except: pass
    
    harmful_count = len(existing_harmful)
    safe_count = len(existing_safe)
    print(f"Resuming: Harmful={harmful_count}, Safe={safe_count}")

    # 使用集合存储已处理的文本，用于 O(1) 复杂度去重
    processed_texts = set(x['original_comment'] for x in existing_harmful + existing_safe)

    # 单条样本处理逻辑
    # 包含重试机制和结果解析
    def process_one_sample(text, check_harmful=False):
        messages = build_messages(text, golden_pool)
        for attempt in range(MAX_RETRIES + 1):
            temp = 0.1 if attempt == 0 else 0.4
            raw_out = engine.generate(messages, temperature=temp)
            if not raw_out: continue
            
            parsed = extract_json_strict(raw_out)
            if parsed and 'impact_level' in parsed and 'reasoning' in parsed:
                parsed['original_comment'] = text
                level = int(parsed.get('impact_level', 1))
                if check_harmful:
                    if level > 1: return parsed
                else:
                    return parsed
        return None

    # 批量处理逻辑
    # 负责并发提交任务、收集结果和写入文件
    def run_batch_processing(candidates, current_count, target_limit, check_harmful=False, strict_limit=True):
        new_candidates = candidates
        
        if current_count >= target_limit: return current_count
        
        # 强制限制单次批次大小，防止 Token 浪费
        batch_size = min(len(new_candidates), MAX_BATCH_SIZE_LIMIT)
        batch_tasks = new_candidates[:batch_size]
        
        print(f"Processing mixed batch of {len(batch_tasks)} candidates...")
        
        pbar = tqdm(total=len(batch_tasks), desc="Batch Progress")
        
        def handle_result(result, text_val, count_val):
            if result and text_val not in processed_texts:
                append_to_file_thread_safe(result, OUTPUT_FILE)
                processed_texts.add(text_val)
                return count_val + 1
            return count_val

        if USE_API:
            with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
                future_to_text = {executor.submit(process_one_sample, text, check_harmful): text for text in batch_tasks}
                for future in as_completed(future_to_text):
                    text = future_to_text[future]
                    try:
                        result = future.result()
                        current_count = handle_result(result, text, current_count)
                    except: pass
                    pbar.update(1)
                    if strict_limit and current_count >= target_limit: 
                        print(f"Target limit {target_limit} reached in batch.")
                        break
        else:
            for text in batch_tasks:
                result = process_one_sample(text, check_harmful)
                current_count = handle_result(result, text, current_count)
                pbar.update(1)
                if strict_limit and current_count >= target_limit: break
        
        pbar.close()
        return current_count

    # 挖掘有害样本 (Harmful Mining)
    # 按比例混合高、中、低毒性区间的样本，保证良品率和进度流畅
    print("Building candidate pools for Harmful Mixed Sampling...")
    
    # 1. 提取原始数据
    raw_high = df[(df['toxicity'] >= 0.95) & (df['toxicity'] < 1.01)]['text'].tolist()
    raw_mid  = df[(df['toxicity'] >= 0.90) & (df['toxicity'] < 0.95)]['text'].tolist()
    raw_low  = df[(df['toxicity'] >= 0.80) & (df['toxicity'] < 0.90)]['text'].tolist()
    
    # 2. 预先去重：从源头剔除已处理的数据
    pool_high = [x for x in raw_high if x not in processed_texts]
    pool_mid  = [x for x in raw_mid if x not in processed_texts]
    pool_low  = [x for x in raw_low if x not in processed_texts]
    
    # 3. 打乱顺序
    random.shuffle(pool_high)
    random.shuffle(pool_mid)
    random.shuffle(pool_low)
    
    print(f"Active Pools: High={len(pool_high)}, Mid={len(pool_mid)}, Low={len(pool_low)}")
    
    # 4. 定义混合配额
    quota_high = 30
    quota_mid  = 30
    quota_low  = 40
    
    while harmful_count < TARGET_HARMFUL_MIN:
        mixed_batch = []
        
        # 使用切片法取数据，取完即从池中移除
        take_h = pool_high[:quota_high]
        pool_high = pool_high[quota_high:]
        mixed_batch.extend(take_h)
        
        take_m = pool_mid[:quota_mid]
        pool_mid = pool_mid[quota_mid:]
        mixed_batch.extend(take_m)
        
        take_l = pool_low[:quota_low]
        pool_low = pool_low[quota_low:]
        mixed_batch.extend(take_l)
        
        if not mixed_batch:
            print("All harmful candidate pools exhausted!")
            break
            
        random.shuffle(mixed_batch)
        
        harmful_count = run_batch_processing(
            mixed_batch, harmful_count, TARGET_HARMFUL_MIN, check_harmful=True, strict_limit=True
        )

    # 补足 Safe 样本 (Padding)
    # 按比例混合纯净、普通、困难三种样本，提升模型对边界情况的健壮性
    if safe_count < harmful_count:
        needed = harmful_count - safe_count
        print(f"Padding with Safe Samples (Targeting 1:1 Balance, Need approx {needed})")
        print("Building candidate pools for Safe Mixed Sampling...")

        # 1. 提取原始数据 (Safe 分层)
        # Pure: 0.0-0.1, 安全
        raw_safe_pure = df[(df['toxicity'] >= 0.0) & (df['toxicity'] < 0.1)]['text'].tolist()
        # Mid: 0.1-0.3, 轻微冒犯
        raw_safe_mid  = df[(df['toxicity'] >= 0.1) & (df['toxicity'] < 0.3)]['text'].tolist()
        # Hard: 0.3-0.5, 困难负样本 (Hard Negatives)，可能包含反讽或激烈辩论
        raw_safe_hard = df[(df['toxicity'] >= 0.3) & (df['toxicity'] < 0.5)]['text'].tolist()

        # 2. 预先去重
        pool_safe_pure = [x for x in raw_safe_pure if x not in processed_texts]
        pool_safe_mid  = [x for x in raw_safe_mid  if x not in processed_texts]
        pool_safe_hard = [x for x in raw_safe_hard if x not in processed_texts]

        # 3. 打乱顺序
        random.shuffle(pool_safe_pure)
        random.shuffle(pool_safe_mid)
        random.shuffle(pool_safe_hard)

        print(f"Safe Pools: Pure={len(pool_safe_pure)}, Mid={len(pool_safe_mid)}, Hard={len(pool_safe_hard)}")

        # 4. 定义混合配额
        quota_pure = 400
        quota_mid  = 300
        quota_hard = 300

        safe_target = harmful_count

        while safe_count < safe_target:
            mixed_batch = []

            take_p = pool_safe_pure[:quota_pure]
            pool_safe_pure = pool_safe_pure[quota_pure:]
            mixed_batch.extend(take_p)

            take_m = pool_safe_mid[:quota_mid]
            pool_safe_mid = pool_safe_mid[quota_mid:]
            mixed_batch.extend(take_m)

            take_h = pool_safe_hard[:quota_hard]
            pool_safe_hard = pool_safe_hard[quota_hard:]
            mixed_batch.extend(take_h)

            if not mixed_batch:
                print("All safe candidate pools exhausted!")
                break
            
            random.shuffle(mixed_batch)

            # 发送处理 (check_harmful=False，无论是否有害都保留，依赖区间本身)
            safe_count = run_batch_processing(
                mixed_batch, safe_count, safe_target, check_harmful=False, strict_limit=True
            )

    print(f"Pipeline Finished. Final: Harmful={harmful_count}, Safe={safe_count}")

if __name__ == "__main__":
    main()