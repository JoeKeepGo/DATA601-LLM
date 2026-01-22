import os
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

# ==========================================
# 核心配置区域
# ==========================================
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "/home/data601/project/constructed_dataset/sft_train_test.jsonl"
OUTPUT_DIR = "/home/data601/project/fine_tuned_model"

# 训练超参数
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

def train():
    print(f"Loading model: {MODEL_ID}")
    print("Mode: BFloat16 (Native support enabled)")

    # 1. 显存优化配置 (4-bit QLoRA)
    # 关键：计算精度设为 bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载基础模型
    # 关键：模型权重直接加载为 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    # 3. 加载并配置分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # 修复 trl 版本兼容性：直接注入长度属性
    tokenizer.model_max_length = MAX_SEQ_LENGTH 

    # 4. 手动准备 PEFT 模型
    # 不依赖 SFTTrainer 自动创建，防止它重置精度
    print("Manually preparing PEFT model...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)

    # 5. 【关键修复】确保 LoRA 参数是 BFloat16
    # 遍历所有可训练参数，强制转为 bf16，防止混用导致报错
    print("Enforcing BFloat16 for trainable parameters...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)

    model.print_trainable_parameters()

    # 6. 加载数据集
    print(f"Loading dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 7. 配置训练参数
    print("Initializing SFTConfig...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        
        # 核心切换：关闭 fp16，开启 bf16
        fp16=False, 
        bf16=True,  
        
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_32bit",
        dataset_text_field="messages", 
        packing=False,
    )
    
    # 暴力注入 max_seq_length，避开构造函数检查
    training_args.max_seq_length = MAX_SEQ_LENGTH

    # 8. 初始化 Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=None,  # 手动处理了 PEFT，这里填 None
        args=training_args,
        processing_class=tokenizer, # 兼容新版 trl 参数名
    )

    print("Starting training...")
    trainer.train()

    # 9. 保存模型
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()