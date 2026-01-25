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
    # 显存碎片优化
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

    # 加载基础模型
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16, 
        attn_implementation="sdpa" 
    )
    
    # 加载并配置分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right" # 训练通常建议右侧 Padding
    tokenizer.model_max_length = MAX_SEQ_LENGTH 
    
    # 设置独立的 Pad Token，避免与 EOS 冲突
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        # 如果词表中没有 <|endoftext|>，则退回使用 eos_token
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token = tokenizer.eos_token

    # 获取原始模板
    chat_template = tokenizer.chat_template
    if chat_template is None:
        # 如果模型没有默认模板，使用标准的 ChatML 模板
        print("Warning: No default chat template found. Using standard ChatML template.")
        chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}"
            "{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% endif %}"
            "{% if message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% generation %}"
            "{{ message['content'] + '<|im_end|>\n' }}"
            "{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
        )
    else:
        # 如果有模板，尝试注入 {% generation %} 标签
        # 使用通用的 ChatML 替换逻辑
        print("Applying TRL generation mask to existing template...")
        tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}"
                "{% elif message['role'] == 'user' %}"
                    "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{'<|im_start|>assistant\n'}}"
                    "{% generation %}"
                        "{{message['content'] + '<|im_end|>\n'}}"
                    "{% endgeneration %}"
                "{% endif %}"
            "{% endfor %}"
        )

    # 验证模板是否生效
    print("Verifying chat template...")
    test_msg = [{"role": "assistant", "content": "test"}]
    # 尝试应用模板看是否报错，确保 logic 正确
    try:
        tokenizer.apply_chat_template(test_msg, tokenize=False)
        print("Chat template verified.")
    except Exception as e:
        print(f"Error in chat template: {e}")

    # 加载数据集
    print(f"Loading dataset from {DATA_PATH}")
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
        # 硬件配置
        fp16=False,
        bf16=True, 
        tf32=True,
        # 显存优化
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 数据加载宽度
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
        # WandB 报告配置
        report_to="wandb",      
        run_name=WANDB_RUN_NAME,
        
        # 基础配置
        dataset_text_field="messages", 
        assistant_only_loss=True,
        packing=False,
    )
    
    training_args.max_seq_length = MAX_SEQ_LENGTH

    # 初始化 Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer, # 传入修改好模板的 tokenizer
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