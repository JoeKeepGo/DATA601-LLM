import os
import json
import re
import random
import time
import threading
import textwrap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 这个程序用于重新生成完整的20K数据集 Reasoning 字段
# API 配置
USE_API = True
API_KEY = os.getenv("API_KEY")
API_BASE_URL = "https://api.deepseek.com"
API_MODEL_NAME = "deepseek-chat"
API_MAX_WORKERS = 200

# 文件路径配置
BASE_DIR = "/home/data601/project"
INPUT_FILE = "dataset/tmp/synthetic_train_final.jsonl"
OUTPUT_FILE = "dataset/tmp/synthetic_train_final_v8prompt.jsonl"
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "dataset/tmp/dataset_split/test_reannotated.jsonl")

# 参数
MAX_RETRIES = 1
FEW_SHOT_COUNT = 3
MAX_NEW_TOKENS = 4096

# 线程锁
FILE_LOCK = threading.Lock()

# PROMPT 定义
SCHEMA_DEFINITION = textwrap.dedent("""
# Role
You are a Content Moderation Auditor. Your task is to provide the "Gold Standard" labels and reasoning for a content moderation dataset.

# Task Instruction
Analyze the input text and determine its Impact Level (1-5) based on the strict Rules A-K provided below.

# CRITICAL INSTRUCTION: LOGICAL RIGOR
For each case, you MUST:
1. Conduct a "Rule-by-Rule" audit.
2. Explicitly explain why a higher Level was REJECTED (e.g., "This is not Level 4 because...").
3. Strictly check [RULE A] for physical meanings.
4. Ensure the Analysis section provides deep context before the JSON output.

---

# 1. Definitions & Logic

Allowed Harm Categories:
- Violence Threat
- Group Derogation (Hate Speech)
- Sexual Harassment
- Toxicity (Severe insults, malicious wishes)
- Insult (Targeted personal attacks)
- Dangerous Content (Self-harm, Illegal Acts, CSAM)

Allowed Target Identities (Protected Groups):
- Race/Ethnicity, Gender, LGBTQ+, Religion, Disability.
- Note: Age is protected only when targeting Minors.

Impact Level Framework:
- 1 (Negligible / Harmless): 
    * Attacks on POLICIES, COUNTRIES, CONCEPTS, OBJECTS.
    * Aggressive Political Discourse targeting Public Figures (Competence/Logic).
    * Admissions of Depression/Self-Harm (Cry for help).
    * Literal/Physical descriptions (e.g. "littering trash").
    * Counterspeech (quoting hate speech to condemn it).
- 2 (Low / Mild): 
    * General rudeness, sarcasm, non-targeted profanity.
- 3 (Medium / Severe Toxicity & Harassment): 
    * Targeted insults towards a SPECIFIC USER ("You are stupid").
    * Malicious Wishes/Curse ("I hope you die") targeting individuals - NOT based on identity.
    * Attacks on physical appearance ("Fat boy").
- 4 (High / Identity Attacks & Hate Speech): 
    * Generalizing a Protected Group as "terrorists", "criminals", or "low IQ".
    * Dehumanization (vermin, disease).
    * Dog Whistles attacking a group's culture/intelligence.
    * Hateful use of Emojis (e.g. Gorilla/Monkey emoji targeting Black people).
    * Exclusionary Rhetoric ("Go back to your country").
    * Glorification of Violence (Praising shooters).
    * Promotion of Drug Usage.
- 5 (Severe / Dangerous): 
    * Credible Violence Threats (Specific plan/intent).
    * Encouraging Suicide / Instructing on Self-Harm methods.
    * Transaction/Instruction of Illegal Goods (Selling drugs/weapons).
    * Sexualization of Minors (CSAM) or Violence against Children.

# 2. Special Considerations (Rules A-K)

[RULE A] The Physical Reality Check (Anti-Hallucination):
- If a word is polysemous (e.g., "trash", "dirt", "clean"), and the context describes a PHYSICAL action (littering, washing, dumping), you MUST classify it as Level 1 (Object).
- DO NOT hallucinate a racial metaphor unless a racial slur is explicitly attached.

[RULE B] The "Trojan Horse" Rule (Policy vs. Hate):
- ACCEPTABLE (Level 1): "I oppose immigration because of economic costs." (Reason = Economy).
- HATE SPEECH (Level 4): "I oppose immigration because refugees are terrorists." (Reason = Group is dangerous).
- NOTE: Linking a policy to "culture destruction" by a specific group is Level 4.

[RULE C] Public Figure Exemption & Identity Override:
- Attacks on Competence/Integrity ("Liar", "Corrupt", "Clown", "Traitor") -> Level 1.
- BUT: If the attack cites a Protected Identity, the Exemption is VOIDED.
    * Example: "She is emotional because she is a woman" -> Level 4 (Gender Stereotyping).

[RULE D] User vs. Public Figure Distinction:
- Insulting a Public Figure ("The Mayor is an idiot") -> Level 1 (Political Discourse).
- Insulting another User ("You are an idiot", "You need medical help") -> Level 3 (Personal Attack/Harassment).

[RULE E] Dog Whistles (Strict Evidence Required):
- Attacks on a group's intelligence, culture, or breeding are Level 4 (Hate Speech).
- Example: "African studies = zero value" (Attacking cultural worth) -> Level 4.
- Example: "They don't belong in civilized society" (Incompatibility) -> Level 4.

[RULE F] Violence: Metaphor vs. Wish vs. Threat:
- "Burn it down" / "Heads will roll" (Political Metaphor) -> Level 1.
- "I hope you die/get cancer" (Malicious Wish) -> Level 3 (Severe Toxicity).
- "I am going to kill you tonight" (Credible Threat) -> Level 5.

[RULE G] Exclusionary Rhetoric:
- Phrases like "Go back to where you came from", "We don't want you here" directed at a demographic are Level 4.

[RULE H] Self-Harm Logic:
- Admission ("I feel like hurting myself") -> Level 1 (Escalate for help, do not punish).
- Encouragement/Instruction ("You should kill yourself") -> Level 5.

[RULE I] Drugs & Illegal Acts:
- Policy Debate ("Legalize it") -> Level 1.
- Usage/Promotion ("This feels great") -> Level 4.
- Transaction/Instruction ("Selling X", "How to make bomb") -> Level 5.

[RULE K] Counterspeech & Quotes (The Victim Shield):
- If a user uses a slur ONLY to quote it for condemnation, reporting, or self-defense, it is Level 1.
- Example: "He called me a [slur], that is wrong" -> Level 1.
- Context must be clear that the user is NOT the aggressor.

---

# 3. Output Format

You must output your response in exactly two parts:

### Analysis:
1. Keywords: List trigger words.
2. Context: Topic analysis.
3. Evidence & Hallucination Check: 
   - Ask: "Is the attack EXPLICITLY against a Protected Group, or am I inferring it?"
   - Check [RULE A]: Is there a physical/literal meaning?
   - Check [RULE K]: Is this a quote/counterspeech?
   - Check [RULE B]: Is it a policy critique or a group generalization?
   - **Audit**: Explain why other Levels were rejected.
4. Impact: Final classification based on strict evidence.

### JSON Result:
```json
{
  "impact_level": <Integer 1-5>,
  "harm_category": ["List of strings from Allowed Categories. Empty [] if Level 1"],
  "target_identity": ["List of strings from Allowed Identities. Empty [] if none"],
  "reasoning": "[Brief summary of the Analysis above]",
  "action_suggestion": "One of: 'None', 'Collapse', 'Warn User', 'Block/Delete', 'Escalate'"
}
""").strip()

# 推理
class InferenceEngine:
    def __init__(self, use_api):
        self.use_api = use_api
        if self.use_api:
            from openai import OpenAI
            self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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
        return None

# 辅助函数
def extract_json_strict(text):
    if not text: return None
    try: return json.loads(text)
    except: pass
    try:
        # 处理 DeepSeek 可能输出的 <think> 标签
        if "<think>" in text: text = text.split("</think>")[-1].strip()
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return None

def build_messages(text, golden_pool):
    messages = [{"role": "system", "content": SCHEMA_DEFINITION}]
    # Few-shot 逻辑
    if golden_pool and len(golden_pool) >= FEW_SHOT_COUNT:
        samples = random.sample(golden_pool, FEW_SHOT_COUNT)
        for s in samples:
            raw_reasoning = s.get('reasoning', "No reasoning provided.")
            output_example = json.dumps({
                "original_comment": s['original_comment'],
                "impact_level": s['impact_level'],
                "harm_category": s.get('harm_category', []),
                "target_identity": s.get('target_identity', []),
                "reasoning": raw_reasoning,
                "action_suggestion": s.get('action_suggestion', "None")
            }, ensure_ascii=False)
            
            messages.append({"role": "user", "content": f"Analyze this: {s['original_comment']}"})
            messages.append({"role": "assistant", "content": output_example})
            
    messages.append({"role": "user", "content": f"Analyze this: {text}"})
    return messages

def append_to_file_thread_safe(data, filepath):
    with FILE_LOCK:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

# 主逻辑
def main():
    # 准备推理
    engine = InferenceEngine(USE_API)

    # 加载 Golden Dataset
    golden_pool = []
    if os.path.exists(GOLDEN_DATASET_PATH):
        print(f"Loading golden dataset from {GOLDEN_DATASET_PATH}...")
        with open(GOLDEN_DATASET_PATH, 'r') as f:
            for line in f:
                try: golden_pool.append(json.loads(line))
                except: pass
    else:
        print("Warning: Golden dataset not found. Running in Zero-shot mode.")

    # 读取待处理文件
    print(f"Reading input file: {INPUT_FILE}...")
    comments_to_process = []
    
    # 检查已处理的数据以支持断点续传
    processed_texts = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'original_comment' in item:
                        processed_texts.add(item['original_comment'])
                except: pass
    print(f"Found {len(processed_texts)} already processed items.")

    # 加载源数据
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 确保提取 original_comment 字段
                    text = data.get('original_comment', '')
                    if text and text not in processed_texts:
                        comments_to_process.append(text)
                except Exception as e:
                    print(f"Skipping invalid line: {e}")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        return

    print(f"Total comments to process: {len(comments_to_process)}")

    # 4. 定义单条处理函数
    def process_one_sample(text):
        messages = build_messages(text, golden_pool)
        for attempt in range(MAX_RETRIES + 1):
            # 重试时稍微增加 temperature 以获得不同结果
            temp = 0.1 if attempt == 0 else 0.4
            raw_out = engine.generate(messages, temperature=temp)
            if not raw_out: continue
            
            parsed = extract_json_strict(raw_out)
            if parsed and 'impact_level' in parsed:
                # 强制把原始评论写回去，确保一一对应
                parsed['original_comment'] = text
                return parsed
        return None

    # 5. 执行并发处理
    print(f"Starting processing with {API_MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        # 提交任务
        future_to_text = {executor.submit(process_one_sample, text): text for text in comments_to_process}
        
        # 进度条
        pbar = tqdm(total=len(comments_to_process), desc="Processing")
        
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                result = future.result()
                if result:
                    append_to_file_thread_safe(result, OUTPUT_FILE)
            except Exception as e:
                print(f"Error processing item: {e}")
            finally:
                pbar.update(1)
                
        pbar.close()

    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
