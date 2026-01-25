import os
import wandb
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig

# WandB 配置
WANDB_API_KEY = "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"
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = "Full-SFT (20k-5e5)"

# 2. 路径与模型配置
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "/home/data601/project/dataset/train/train_quality.jsonl"
OUTPUT_DIR = "/home/data601/project/fine_tuned_model/full/20k_5e5"

# 全量微调超参数
LEARNING_RATE = 5e-5     
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 3
# 显存与批次配置
BATCH_SIZE = 16          
GRADIENT_ACCUMULATION = 32

def train():
    # 显存碎片优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Initializing WandB for Project: {WANDB_PROJECT} (Entity: {WANDB_ENTITY})")
    
    # WandB 自动登录与配置
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_WATCH"] = "false"
    
    try:
        wandb.login(key=WANDB_API_KEY)
        print(">>> WandB Login Successful!")
    except Exception as e:
        print(f"!!! WandB Login Failed: {e}")

    print(f"Loading model: {MODEL_ID}")

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16, 
        attn_implementation="sdpa" 
    )
    
    # 加载并配置分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH 

    # 加载数据集
    print(f"Loading dataset from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 配置训练参数
    print("Initializing SFTConfig...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        
        # 硬件加速
        fp16=False,
        bf16=True, 
        tf32=True,
        
        # 显存优化
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # 数据加载
        # 有22 CPU核心
        dataloader_num_workers=8, 
        
        # 学习率策略
        lr_scheduler_type="cosine", 
        warmup_ratio=0.03,
        
        # 优化器
        optim="adamw_torch_fused",
        
        # 日志与保存
        logging_steps=1,        
        save_strategy="steps",  
        save_steps=200,         
        save_total_limit=3,
        
        # WandB 报告
        report_to="wandb",      
        run_name=WANDB_RUN_NAME,
        
        # 基础配置
        dataset_text_field="messages", 
        packing=False,
    )
    
    training_args.max_seq_length = MAX_SEQ_LENGTH

    # 初始化 Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting Full Fine-Tuning...")
    trainer.train()

    # 保存模型
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete.")

if __name__ == "__main__":
    train()