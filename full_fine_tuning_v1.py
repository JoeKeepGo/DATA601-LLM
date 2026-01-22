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

# 配置区域
# WandB 配置
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = "Full-FT-2e5-20K" 

# 路径与模型配置
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "/home/data601/project/dataset/train/train_quality.jsonl"
OUTPUT_DIR = "/home/data601/project/fine_tuned_model/full/Full-FT-2e5-20K"

# 超参数调整
LEARNING_RATE = 2e-5     
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 3

# Batch Size 策略
BATCH_SIZE = 8           
GRADIENT_ACCUMULATION = 8 

def train():
    # 显存碎片优化 (PyTorch 推荐新写法 PYTORCH_ALLOC_CONF，但也兼容旧写法)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # WandB 设置
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_WATCH"] = "false"
    
    # 登录 WandB
    if "WANDB_API_KEY" in os.environ:
        wandb.login()
    else:
        # 上传时请删除 Key
        wandb.login(key="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")
        print("Warning: WANDB_API_KEY not found in env.")

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

    # 注入支持 TRL 掩码的训练专用模板
    # 必须显式定义模板并使用 {% generation %} 标签，否则 assistant_only_loss 会报错
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}"
            "{% elif message['role'] == 'user' %}"
                "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{'<|im_start|>assistant\n'}}"
                # trl 计算 Loss 核心点
                "{% generation %}"
                    "{{message['content'] + '<|im_end|>\n'}}"
                "{% endgeneration %}"
            "{% endif %}"
        "{% endfor %}"
    )

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
        dataloader_num_workers=8, 
        
        # 学习率策略
        lr_scheduler_type="cosine", 
        warmup_ratio=0.05,
        
        # 优化器
        optim="adamw_torch_fused",
        
        # 日志与保存
        logging_steps=1,
        save_strategy="steps",  
        save_steps=100, 
        save_total_limit=2,
        
        # WandB 报告
        report_to="wandb",      
        run_name=WANDB_RUN_NAME,
        
        # 基础配置
        dataset_text_field="messages", 
        dataset_kwargs={
            "add_special_tokens": False, # 使用自定义模板时通常设为 False
        },
        completion_only_loss=False,
        assistant_only_loss=True,   # 仅计算助手回复 Loss
        packing=False,
    )
    
    training_args.max_seq_length = MAX_SEQ_LENGTH

    # 初始化 Trainer
    print("Initializing SFTTrainer...")
    # 传入 tokenizer，SFTTrainer 自动应用刚才设置的 chat_template
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