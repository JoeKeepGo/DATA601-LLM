import argparse
import json
import os
import re
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

# 默认配置 (可被命令行参数覆盖)

# "base" (纯基座), "lora" (基座+Adapter), "fft" (全量微调权重)
DEFAULT_MODE = "lora" 

# 基座模型路径
DEFAULT_BASE_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

# 检查点路径
# lora 填 Adapter 文件夹路径
# fft 填全量微调权重的文件夹
# base 可忽略
DEFAULT_CHECKPOINT_PATH = "/home/data601/project/fine_tuned_model/lora_run_v1"

# 测试集文件路径 
DEFAULT_TEST_FILE = "/home/data601/project/dataset/test/test.jsonl"

# 结果输出目录
DEFAULT_OUTPUT_DIR = "eval_results"

# 评测数量控制
# None: 评测所有数据
DEFAULT_NUM_SAMPLES = None

# 推理参数
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_SEED = 42
DEFAULT_FIX_MISTRAL_REGEX = True
DEFAULT_SAMPLE_STRATEGY = "first"
DEFAULT_STOP_ON_JSON = True
DEFAULT_STOP_STRINGS = ["\n}\n```", "}\n```"]

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

# CLI 参数

def _default_run_name(mode, checkpoint_path):
    if checkpoint_path:
        return os.path.basename(os.path.abspath(checkpoint_path.rstrip(os.sep)))
    return mode

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation worker")
    parser.add_argument("--mode", type=str, choices=["base", "lora", "fft"], default=DEFAULT_MODE)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--load_in_4bit", action="store_true", default=DEFAULT_LOAD_IN_4BIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fix_mistral_regex", action="store_true", default=DEFAULT_FIX_MISTRAL_REGEX)
    parser.add_argument("--no_fix_mistral_regex", action="store_false", dest="fix_mistral_regex")
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["first", "random"],
        default=DEFAULT_SAMPLE_STRATEGY,
        help="How to pick samples when num_samples is set.",
    )
    parser.add_argument("--stop_on_json", action="store_true", default=DEFAULT_STOP_ON_JSON)
    parser.add_argument("--no_stop_on_json", action="store_false", dest="stop_on_json")
    return parser.parse_args()

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

def load_model_and_tokenizer(mode, base_model_path, checkpoint_path, load_in_4bit, fix_mistral_regex):
    print(f"\n>>> Initializing in MODE: [{mode}]")

    bnb_config = None
    if load_in_4bit:
        print(">>> Using 4-bit Quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # 加载 Tokenizer
    tokenizer_path = base_model_path if mode != "fft" else checkpoint_path
    print(f">>> Loading Tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            fix_mistral_regex=fix_mistral_regex,
        )
    except:
        print("Warning: Failed to load tokenizer from checkpoint, using Base Model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            fix_mistral_regex=fix_mistral_regex,
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 加载模型
    model = None
    if mode == "base":
        print(f">>> Loading BASE Model: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    elif mode == "fft":
        print(f">>> Loading FFT Model: {checkpoint_path}")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError("MODE='fft' requires a valid CHECKPOINT_PATH!")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    elif mode == "lora":
        print(f">>> Loading LoRA Base: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        print(f">>> Loading LoRA Adapter: {checkpoint_path}")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError("MODE='lora' requires a valid CHECKPOINT_PATH!")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        raise ValueError(f"Unknown MODE: {mode}")

    model.eval()
    return model, tokenizer

# 主程序

def run_evaluation(args):
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    run_name = args.run_name or _default_run_name(args.mode, args.checkpoint_path)
    num_samples = args.num_samples
    if num_samples is not None and num_samples <= 0:
        num_samples = None
    sample_strategy = args.sample_strategy

    # 加载数据
    all_data = load_data(args.test_file)

    # 数据截取逻辑
    if num_samples is not None and isinstance(num_samples, int):
        if num_samples >= len(all_data):
            print(f">>> [FULL MODE] Using all {len(all_data)} samples.")
            test_data = all_data
        elif sample_strategy == "random":
            rng = np.random.RandomState(args.seed)
            indices = rng.permutation(len(all_data))[:num_samples]
            test_data = [all_data[i] for i in indices]
            print(f">>> [DEBUG MODE] Randomly sampling {num_samples} samples (seed={args.seed}).")
        else:
            print(f">>> [DEBUG MODE] Slicing first {num_samples} samples.")
            test_data = all_data[:num_samples]
    else:
        print(f">>> [FULL MODE] Using all {len(all_data)} samples.")
        test_data = all_data

    print(f"Run Name: {run_name}")
    print(f"Total Test Samples: {len(test_data)}")

    # 准备模型
    model, tokenizer = load_model_and_tokenizer(
        args.mode,
        args.base_model_path,
        args.checkpoint_path,
        args.load_in_4bit,
        args.fix_mistral_regex,
    )

    # 推理循环
    results = []
    y_true = []
    y_pred = []
    parse_errors = 0

    print(f"\n>>> Starting Inference (Batch Size: {args.batch_size})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def batch_generator(data, bsize):
        for i in range(0, len(data), bsize):
            yield data[i:i + bsize]

    total_batches = (len(test_data) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    generation_config = model.generation_config
    if generation_config is None:
        generation_config = GenerationConfig.from_model_config(model.config)
    else:
        generation_config = GenerationConfig.from_dict(generation_config.to_dict())
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.stop_strings = DEFAULT_STOP_STRINGS if args.stop_on_json else None

    for batch in tqdm(batch_generator(test_data, args.batch_size), total=total_batches):

        # 批量解析输入与真值
        batch_inputs = []
        batch_gts = []
        batch_ids = []

        for item in batch:
            txt, gt = parse_sample(item)
            batch_inputs.append(txt)
            batch_gts.append(gt)
            batch_ids.append(item.get("id", "unknown"))

        # 构造 Prompt
        prompts = [f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {txt}\nAssistant:" for txt in batch_inputs]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                tokenizer=tokenizer,
            )

        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 解析模型输出
        for i, text in enumerate(decoded_texts):
            gt_level = batch_gts[i]  # 获取对应的真值

            pred_json = extract_json_from_text(text)
            pred_level = 1
            valid = False

            if pred_json:
                try:
                    p = int(pred_json.get("impact_level", 1))
                    pred_level = max(1, min(5, p))
                    y_true.append(gt_level)
                    y_pred.append(pred_level)
                    valid = True
                except:
                    pass

            if not valid:
                parse_errors += 1

            results.append(
                {
                    "id": batch_ids[i],
                    "input_text": batch_inputs[i],
                    "ground_truth": gt_level,
                    "prediction": pred_level if valid else "ERROR",
                    "valid_json": valid,
                    "raw_response": text,
                }
            )

    elapsed = time.time() - start_time

    # 统计指标
    metric_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    metric_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    format_rate = (len(test_data) - parse_errors) / len(test_data) if len(test_data) > 0 else 0

    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT | Run: {run_name} | Mode: {args.mode}")
    if args.mode != "base":
        print(f"Checkpoint: {args.checkpoint_path}")
    if num_samples:
        if sample_strategy == "random":
            print(f"(Partial Evaluation: Random {num_samples} samples, seed={args.seed})")
        else:
            print(f"(Partial Evaluation: First {num_samples} samples)")
    print("=" * 40)
    print(f"F1 Score (Macro): {metric_f1:.4f}")
    print(f"Accuracy:         {metric_acc:.4f}")
    print(f"Format Compliance: {format_rate:.2%}")
    print(f"Parse Errors:     {parse_errors}")
    print(f"Elapsed:          {elapsed:.2f}s")
    print("-" * 40)

    # 保存结果
    predictions_path = os.path.join(args.output_dir, f"{run_name}_predictions.json")
    metrics_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    class_report_path = os.path.join(args.output_dir, f"{run_name}_class_report.txt")

    # 保存详细预测
    save_json(results, predictions_path)

    # 保存指标
    metrics = {
        "run_name": run_name,
        "mode": args.mode,
        "base_model": args.base_model_path,
        "checkpoint": args.checkpoint_path,
        "test_file": args.test_file,
        "num_samples": len(test_data),
        "requested_num_samples": args.num_samples,
        "sample_strategy": sample_strategy,
        "f1": metric_f1,
        "acc": metric_acc,
        "format_rate": format_rate,
        "parse_errors": parse_errors,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "stop_on_json": args.stop_on_json,
        "stop_strings": DEFAULT_STOP_STRINGS if args.stop_on_json else None,
        "elapsed_seconds": elapsed,
    }
    save_json(metrics, metrics_path)

    # 保存分类报告
    if y_true:
        report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0)
        with open(class_report_path, "w") as f:
            f.write(report)
        print("Classification Report saved.")
        print(report)

    print(f"\nAll Done. Results saved to: {args.output_dir}")

def main():
    args = parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
