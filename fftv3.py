import os
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig

# 配置区域
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = "Full-FT-ManualMasking"

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "/home/data601/project/dataset/train/train_quality.jsonl"
OUTPUT_DIR = "/home/data601/project/fine_tuned_model/full/Full-FT-ManualMasking"

LEARNING_RATE = 2e-5     
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 3
BATCH_SIZE = 8           
GRADIENT_ACCUMULATION = 8 

def train():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # WandB 设置
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_WATCH"] = "false"
    
    if "WANDB_API_KEY" in os.environ:
        wandb.login()
    else:
        print("Warning: WANDB_API_KEY not found in env.")

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16, 
        attn_implementation="sdpa" 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Qwen 默认 <|im_end|>
    tokenizer.padding_side = "right"
    
    # =================================================================
    # 【核心修复】手动预处理函数
    # 不依赖 TRL 的模板魔法，手动将 User/System 部分的 Label 设为 -100
    # =================================================================
    def process_func(example):
        messages = example["messages"]
        input_ids = []
        labels = []
        
        # Qwen 特殊 Token ID (根据 Qwen3/2.5 词表)
        # 通常: <|im_start|> = 151644, <|im_end|> = 151645 (具体以 tokenizer 为准)
        im_start_tokens = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False) 

        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # 构建：<|im_start|>role\ncontent<|im_end|>\n
            role_tokens = tokenizer.encode(role, add_special_tokens=False)
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            
            # 拼接当前段落的 tokens
            # 结构: [im_start] + [role] + [nl] + [content] + [im_end] + [nl]
            current_tokens = (
                im_start_tokens + 
                role_tokens + 
                nl_tokens + 
                content_tokens + 
                im_end_tokens + 
                nl_tokens
            )
            
            input_ids.extend(current_tokens)
            
            if role == "assistant":
                # 如果是助手回复，我们需要计算 Loss
                # 但是：<|im_start|>assistant\n 这部分依然不应该算 Loss
                # 只有 content + <|im_end|> 需要算
                
                # 计算前缀长度: <|im_start|>assistant\n
                prefix_len = len(im_start_tokens) + len(role_tokens) + len(nl_tokens)
                
                # 助手的 Label: 前缀部分是 -100，内容部分是 token id
                current_labels = [-100] * prefix_len + current_tokens[prefix_len:]
                labels.extend(current_labels)
            else:
                # 如果是 User 或 System，全部设为 -100
                labels.extend([-100] * len(current_tokens))

        # 截断处理
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            labels = labels[:MAX_SEQ_LENGTH]
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }
    
    # =================================================================
    
    print(f"Loading and processing dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 手动应用预处理
    print("Pre-processing dataset manually (Masking User/System tokens)...")
    dataset = dataset.map(process_func, remove_columns=["messages"])

    print("Initializing SFTConfig...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=False,
        bf16=True, 
        tf32=True,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8, 
        lr_scheduler_type="cosine", 
        warmup_ratio=0.05,
        optim="adamw_torch_fused",
        logging_steps=1,
        save_strategy="steps",  
        save_steps=100, 
        save_total_limit=2,
        report_to="wandb",      
        run_name=WANDB_RUN_NAME,
        
        # 【重要】因为我们已经手动处理了 input_ids 和 labels
        # 所以这里不需要再指定 dataset_text_field，也不需要 assistant_only_loss
        dataset_text_field=None, 
        packing=False,
        remove_unused_columns=False, # 防止 Dataset 中的 input_ids 被过滤
    )
    
    # SFTTrainer 在检测到 input_ids/labels 已存在时，会直接使用它们
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer, 
    )

    print("Starting Full Fine-Tuning...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete.")

if __name__ == "__main__":
    train()