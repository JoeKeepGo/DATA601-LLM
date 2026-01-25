import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def train():
    # 1. 配置路径和参数
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    data_path = "/home/data601/project/dataset/train/train_quality.jsonl"
    output_dir = "/home/data601/project/output"

    # 2. 加载 Tokenizer
    # 注意：trust_remote_code=True 对 Qwen 系列通常是必须的
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # ---------------------------------------------------------------------------
    # 【核心修复】注入支持 trl 掩码的训练专用模板
    # 原因：Qwen3 原生模板用于推理，缺少 {% generation %} 标签，导致 assistant_only_loss 无法识别助手回复
    # ---------------------------------------------------------------------------
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}"
            "{% elif message['role'] == 'user' %}"
                "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{'<|im_start|>assistant\n'}}"
                # 关键点：将助手回复包裹在 generation 块中，trl 才能正确计算 loss
                "{% generation %}"
                    "{{message['content'] + '<|im_end|>\n'}}"
                "{% endgeneration %}"
            "{% endif %}"
        "{% endfor %}"
    )

    # 3. 加载数据集
    # 确保数据集列名为 'messages' 且格式为 List
    dataset = load_dataset("json", data_files=data_path, split="train")

    # 4. 配置 SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=2048,             # 根据显存调整
        per_device_train_batch_size=4,   # 根据显存调整
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        
        # 你的核心需求：仅计算助手回复的 Loss
        assistant_only_loss=True,        
        
        # Qwen3 必须指定 pad_token，通常使用 <|endoftext|> 或 <|im_end|>
        # 这里建议如果不使用 packing，显式设置 pad_token_id
        packing=False, 
    )

    # 5. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    # 6. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # 7. 开始训练
    print("验证模板掩码逻辑...")
    # 可选：简单打印一个样本看是否还会报错
    # trainer.processing_class.apply_chat_template(dataset['messages'], return_assistant_tokens_mask=True)
    
    trainer.train()

if __name__ == "__main__":
    train()