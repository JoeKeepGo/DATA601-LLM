import torch
import json
import os
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel

# 全局配置 

# "base" (纯基座), "lora" (基座+Adapter), "fft" (全量微调权重)
MODE = "lora" 

# 基座模型路径
BASE_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

# 检查点路径
# lora 填 Adapter 文件夹路径
# fft 填全量微调权重的文件夹
# base 可忽略
CHECKPOINT_PATH = "/home/data601/project/fine_tuned_model/lora_run_v1"

# 测试集文件路径 
TEST_FILE = "/home/data601/project/dataset/test/test.jsonl"

# 结果输出目录
OUTPUT_DIR = "eval_results"

# 评测数量控制
# None: 评测所有数据
NUM_SAMPLES = None

# 推理参数
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 4096      
MAX_NEW_TOKENS = 2048       
LOAD_IN_4BIT = False
SEED = 42                  

# Prompt 定义
SCHEMA_DEFINITION = """
# Role
You are a content moderation expert. Your goal is to distinguish between harmful content and legitimate discourse with HIGH PRECISION.

# Critical Instruction: FIGHT HALLUCINATION
- Do NOT read intent that is not explicitly in the text.
- If a word has a physical meaning (e.g., "trash", "dirt", "burn"), assume the PHYSICAL/LITERAL meaning first.
- ONLY classify as Level 4/5 if there is EXPLICIT evidence of identity attacks, real-world danger, or obfuscated hate (specific emojis).
- When in doubt, default to Level 1 (Harmless).

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

### Analysis:
1. Keywords: List trigger words.
2. Context: Topic analysis.
3. Evidence & Hallucination Check: 
   - Ask: "Is the attack EXPLICITLY against a Protected Group, or am I inferring it?"
   - Check [RULE A]: Is there a physical/literal meaning?
   - Check [RULE K]: Is this a quote/counterspeech?
   - Check [RULE B]: Is it a policy critique or a group generalization?
4. Impact: Final classification based on strict evidence.

### JSON Result:
```json
{
  "impact_level": <Integer 1-5>,
  "harm_category": ["List of strings from Allowed Categories. Empty [] if Level 1"],
  "target_identity": ["List of strings from Allowed Identities. Empty [] if none"],
  "reasoning": "[Brief summary of the Analysis]",
  "action_suggestion": "One of: 'None', 'Collapse', 'Warn User', 'Block/Delete', 'Escalate'"
}
"""

# 工具函数

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

# JSON 提取器
def extract_json_from_text(text):
    try:
        if "<think>" in text: text = text.split("</think>")[-1].strip()
        # 匹配 ```json { ... } ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        # 匹配最外层 { ... }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        # 尝试直接解析
        return json.loads(text)
    except:
        return None

def save_json(data, path):
    def default_converter(o):
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        return str(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default_converter)

# 样本解析
def parse_sample(item):

    text = ""
    gt_level = 1
    
    # Messages 格式
    if "messages" in item:
        # 提取 User 输入
        for msg in item["messages"]:
            if msg["role"] == "user":
                text = msg["content"]
                break
        
        # 2. 提取 Ground Truth
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                gt_json = extract_json_from_text(msg["content"])
                if gt_json:
                    try:
                        gt_level = int(gt_json.get("impact_level", 1))
                    except:
                        gt_level = 1
                break
                
    # 扁平格式
    else:
        text = item.get("original_comment", "")
        gt_level = int(item.get("impact_level", 1))
        
    return text, gt_level

# 模型加载

def load_model_and_tokenizer():
    print(f"\n>>> Initializing in MODE: [{MODE}]")
    
    bnb_config = None
    if LOAD_IN_4BIT:
        print(">>> Using 4-bit Quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # 加载 Tokenizer
    tokenizer_path = BASE_MODEL_PATH if MODE != 'fft' else CHECKPOINT_PATH
    print(f">>> Loading Tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except:
        print("Warning: Failed to load tokenizer from checkpoint, using Base Model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 加载模型
    model = None
    if MODE == 'base':
        print(f">>> Loading BASE Model: {BASE_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    elif MODE == 'fft':
        print(f">>> Loading FFT Model: {CHECKPOINT_PATH}")
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
             raise ValueError("MODE='fft' requires a valid CHECKPOINT_PATH!")
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    elif MODE == 'lora':
        print(f">>> Loading LoRA Base: {BASE_MODEL_PATH}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        print(f">>> Loading LoRA Adapter: {CHECKPOINT_PATH}")
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
             raise ValueError("MODE='lora' requires a valid CHECKPOINT_PATH!")
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
    
    model.eval()
    return model, tokenizer

# 主程序

def main():
    set_seed(SEED)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 加载数据
    all_data = load_data(TEST_FILE)
    
    # 数据截取逻辑
    if NUM_SAMPLES is not None and isinstance(NUM_SAMPLES, int):
        print(f">>> [DEBUG MODE] Slicing first {NUM_SAMPLES} samples.")
        test_data = all_data[:NUM_SAMPLES]
    else:
        print(f">>> [FULL MODE] Using all {len(all_data)} samples.")
        test_data = all_data
        
    print(f"Total Test Samples: {len(test_data)}")

    # 准备模型
    model, tokenizer = load_model_and_tokenizer()

    # 推理循环
    results = []
    y_true = []
    y_pred = []
    parse_errors = 0

    print(f"\n>>> Starting Inference (Batch Size: {BATCH_SIZE})...")
    
    def batch_generator(data, bsize):
        for i in range(0, len(data), bsize):
            yield data[i:i+bsize]

    total_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in tqdm(batch_generator(test_data, BATCH_SIZE), total=total_batches):
        
        # 批量解析输入与真值
        batch_inputs = []
        batch_gts = []
        batch_ids = []
        
        for item in batch:
            txt, gt = parse_sample(item)
            batch_inputs.append(txt)
            batch_gts.append(gt)
            batch_ids.append(item.get('id', 'unknown'))
            
        # 构造 Prompt
        prompts = [f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {txt}\nAssistant:" for txt in batch_inputs]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, # Greedy Search 保证确定性
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 解析模型输出
        for i, text in enumerate(decoded_texts):
            gt_level = batch_gts[i] # 获取对应的真值
            
            pred_json = extract_json_from_text(text)
            pred_level = 1
            valid = False

            if pred_json:
                try:
                    p = int(pred_json.get('impact_level', 1))
                    pred_level = max(1, min(5, p))
                    y_true.append(gt_level)
                    y_pred.append(pred_level)
                    valid = True
                except:
                    pass
            
            if not valid:
                parse_errors += 1
            
            results.append({
                "id": batch_ids[i],
                "input_text": batch_inputs[i],
                "ground_truth": gt_level,
                "prediction": pred_level if valid else "ERROR",
                "valid_json": valid,
                "raw_response": text
            })

    # 统计指标
    metric_f1 = f1_score(y_true, y_pred, average='macro') if y_true else 0.0
    metric_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    format_rate = (len(test_data) - parse_errors) / len(test_data) if len(test_data) > 0 else 0

    print("\n" + "="*40)
    print(f"EVALUATION REPORT | Mode: {MODE}")
    if NUM_SAMPLES: print(f"(Partial Evaluation: First {NUM_SAMPLES} samples)")
    print("="*40)
    print(f"F1 Score (Macro): {metric_f1:.4f}")
    print(f"Accuracy:         {metric_acc:.4f}")
    print(f"Format Compliance: {format_rate:.2%}")
    print("-" * 40)

    # 保存结果
    run_name = f"{MODE}_results"
    
    # 保存详细预测
    save_json(results, os.path.join(OUTPUT_DIR, f"{run_name}_predictions.json"))
    
    # 保存指标
    metrics = {
        "mode": MODE,
        "base_model": BASE_MODEL_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "num_samples": len(test_data),
        "f1": metric_f1,
        "acc": metric_acc,
        "format_rate": format_rate
    }
    save_json(metrics, os.path.join(OUTPUT_DIR, f"{run_name}_metrics.json"))

    # 保存分类报告
    if y_true:
        report = classification_report(y_true, y_pred, labels=[1,2,3,4,5], zero_division=0)
        with open(os.path.join(OUTPUT_DIR, f"{run_name}_class_report.txt"), "w") as f:
            f.write(report)
        print("Classification Report saved.")
        print(report)

    print(f"\nAll Done. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()