import torch
import json
import os
import gc
import random
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

# 1. 全局配置

CANDIDATE_MODELS = [
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "alias": "Qwen-3-4B"},
    {"id": "google/gemma-3-12b-it", "alias": "Gemma-3-12B"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "alias": "DeepSeek-R1-8B"}
]

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_DIR = "/home/data601/project"
TRAIN_FILE = os.path.join(BASE_DIR, "dataset_split/train.jsonl")
TEST_FILE = os.path.join(BASE_DIR, "dataset_split/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "finetuning_test_results_3")

# 关键参数
MAX_STEPS = 60
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1        
GRAD_ACCUMULATION = 8
EVAL_BATCH_SIZE = 8         
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCHEMA_DEFINITION = """
You are a content moderation expert. Analyze the comment and output a JSON object with:
1. "original_comment": The input text.
2. "impact_level": Integer 1-5 (1=Negligible, 2=Low, 3=Medium, 4=High, 5=Severe).
3. "harm_category": List of strings.
4. "target_identity": List of strings.
5. "reasoning": A coherent text explanation.
6. "action_suggestion": String.
"""

# 2. 工具函数

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_memory():
    # 深度清理显存
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except:
        pass

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_json_from_text(text):
    try:
        if "<think>" in text: text = text.split("</think>")[-1].strip()
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
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

# 3. 核心逻辑

def run_experiment_for_model(model_config, train_data_raw, test_data_raw):
    model_id = model_config["id"]
    model_alias = model_config["alias"]
    exp_dir = os.path.join(OUTPUT_DIR, model_alias)
    adapter_dir = os.path.join(exp_dir, "adapter")
    
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    print(f"\nProcessing: {model_alias}")
    clean_memory()

    # 3.1 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        tokenizer.padding_side = 'right' 
        # 强制修复 Pad Token 问题
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        print(f"Skip {model_alias}: Tokenizer load failed - {e}")
        return None

    # 3.2 数据预处理 (并行加速)
    def preprocess_function(example):
        comment = example['original_comment']
        target_dict = {
            "original_comment": comment,
            "impact_level": example['impact_level'],
            "harm_category": example['harm_category'],
            "target_identity": example['target_identity'],
            "reasoning": example['reasoning'],
            "action_suggestion": example['action_suggestion']
        }
        # 构造对话格式
        full_text = f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {comment}\nAssistant: {json.dumps(target_dict, ensure_ascii=False)}{tokenizer.eos_token}"
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print(f"[{model_alias}] Preprocessing data")
    hf_dataset = Dataset.from_list(train_data_raw)
    # num_proc=4 使用多核处理
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=False, num_proc=4)

    # 3.3 模型加载
    print(f"[{model_alias}] Loading Model")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            attn_implementation="sdpa"
        )
        # 确保 pad_token_id 正确，防止训练报错
        model.config.pad_token_id = tokenizer.pad_token_id 
    except Exception as e:
        print(f"Error loading model {model_alias}: {e}")
        clean_memory()
        return None

    # 3.4 训练
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    # --- 修复核心：适配新版 TRL 参数 ---
    training_args = SFTConfig(
        output_dir=exp_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        max_steps=MAX_STEPS,
        optim="paged_adamw_32bit",
        save_strategy="no",
        fp16=False,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True, 
        
        # 关键修改：新版使用 max_length 而非 max_seq_length
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field=None # 显式设为 None，避免自动处理报错
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer # 新版 TRL 推荐用 processing_class 替代 tokenizer 参数
    )

    print(f"[{model_alias}] Training start")
    try:
        trainer.train()
        
        # 保存 Adapter
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    except Exception as e:
        print(f"Training failed for {model_alias}: {e}")
        del model, trainer
        clean_memory()
        return None

    # 释放显存
    del model, trainer
    clean_memory()

    # 4. 批量推理评测 (加速核心)
    
    print(f"[{model_alias}] Evaluating (Batch Size: {EVAL_BATCH_SIZE})")
    
    try:
        # Reload Base + Adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        tokenizer.padding_side = 'left' # 推理必须 Left Padding

        results = []
        y_true, y_pred = [], []
        parse_errors = 0
        
        # 批量数据生成器
        def batch_data(data, bsize):
            for i in range(0, len(data), bsize):
                yield data[i:i+bsize]

        for batch in tqdm(batch_data(test_data_raw, EVAL_BATCH_SIZE), desc="Inference", total=len(test_data_raw)//EVAL_BATCH_SIZE + 1):
            # 1. 准备 Prompt Batch
            prompts = [f"{SCHEMA_DEFINITION}\n\nUser: Analyze this: {item['original_comment']}\nAssistant:" for item in batch]
            
            # 2. 批量 Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            
            # 3. 批量 Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=0.1, 
                    do_sample=True, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 4. 批量 Decode 和 处理
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:] # 切片去掉 Prompt
            decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for idx, text in enumerate(decoded_texts):
                original_item = batch[idx]
                pred_json = extract_json_from_text(text.strip())
                
                gt = original_item.get('impact_level', 1)
                pred = 1
                valid = False
                
                if pred_json:
                    try:
                        p = int(pred_json.get('impact_level', 1))
                        pred = max(1, min(5, p))
                        y_true.append(gt)
                        y_pred.append(pred)
                        valid = True
                    except: pass
                
                if not valid: parse_errors += 1
                
                results.append({
                    "id": original_item.get('id'), 
                    "gt": gt, 
                    "pred": pred, 
                    "valid": valid, 
                    "response": text.strip()
                })

        # 计算指标
        f1 = f1_score(y_true, y_pred, average='macro') if y_true else 0.0
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        fmt = (len(test_data_raw) - parse_errors) / len(test_data_raw) if len(test_data_raw) > 0 else 0
        
        print(f"Result {model_alias}: F1={f1:.4f}, Acc={acc:.4f}, Format={fmt:.2f}")
        
        # 保存结果
        save_json(results, os.path.join(exp_dir, "predictions.json"))
        save_json({"f1": f1, "acc": acc, "format": fmt}, os.path.join(exp_dir, "metrics.json"))
        
        if y_true:
            report_str = classification_report(y_true, y_pred, labels=[1,2,3,4,5], zero_division=0)
            with open(os.path.join(exp_dir, "report.txt"), "w", encoding="utf-8") as f:
                f.write(report_str)

        del model, base_model
        clean_memory()
        
        return {"alias": model_alias, "f1": f1, "acc": acc, "format": fmt}

    except Exception as e:
        print(f"Evaluation failed for {model_alias}: {e}")
        clean_memory()
        return None

# 主程序

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    set_seed(42)

    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE)
    
    if not train_data or not test_data:
        print("Error: Data is empty. Check paths.")
        return
        
    print(f"Data Loaded: Train={len(train_data)}, Test={len(test_data)}")

    leaderboard = []
    
    for candidate in CANDIDATE_MODELS:
        # 健壮性：全局异常捕获，单个模型失败不影响整体
        try:
            res = run_experiment_for_model(candidate, train_data, test_data)
            if res: leaderboard.append(res)
        except KeyboardInterrupt:
            print("\nManually interrupted.")
            break
        except Exception as e:
            print(f"\nCRITICAL ERROR: Model {candidate['alias']} failed: {e}")
            import traceback
            traceback.print_exc()
            clean_memory() # 救急清理

    print("\nFINAL LEADERBOARD")
    print(f"{'Model':<15} | {'Format':<8} | {'Acc':<8} | {'F1':<8}")
    for entry in leaderboard:
        print(f"{entry['alias']:<15} | {entry['format']*100:.1f}%    | {entry['acc']*100:.1f}%   | {entry['f1']:.4f}")
    
    save_json(leaderboard, os.path.join(OUTPUT_DIR, "leaderboard.json"))
    print(f"\nAll Done. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()