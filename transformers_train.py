import os

# 如果你仍想尝试 Unsloth，取消下面的注释
# os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
import logging
import sys
import gc
import transformers
import wandb
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==================== 全局配置 ====================

# WandB 配置
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = "FFT-5k-5e4-1ep-32x2-22Jan-1"
WANDB_KEY = "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"

# 路径与模型
BASE_DIR = "/home/data601/project"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
DATA_PATH = os.path.join(BASE_DIR, "dataset/train/train.jsonl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset/test/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model", WANDB_RUN_NAME)

# 模型加载参数
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOAD_IN_4BIT = False 

# 数据集控制
DATA_SAMPLE_COUNT = 5000 
DATASET_BATCHED = True
DATASET_TEXT_FIELD = "text"
ADD_GENERATION_PROMPT = False
TEMPLATE_TOKENIZE = False 

# 微调模式
USE_LORA = False

# 随机种子
RANDOM_SEED = 3407

# LoRA 配置
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# 梯度检查点
USE_GRADIENT_CHECKPOINTING = True

# 训练超参数
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
NUM_EPOCHS = 1
WARMUP_RATIO = 0.01

# 日志与保存配置
LOGGING_STEPS = 1
SAVE_STRATEGY = "steps"
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2

# 验证与评估配置
EVAL_STRATEGY = "steps"
EVAL_STEPS = 5
EVAL_BATCH_SIZE = 16
TEST_SIZE = 0.01

# 优化器配置
OPTIMIZER = "adamw_torch_fused" 
WEIGHT_DECAY = 0.1
LR_SCHEDULER_TYPE = "cosine"

# 梯度裁剪与 NEFTune
MAX_GRAD_NORM = 0.5
NEFTUNE_NOISE_ALPHA = 10

# Packing
PACKING = False

# Loss 计算方式配置
LOSS_METHOD = "tail_weighted"  # "default", "token_micro", "sentence_macro", "tail_weighted"

# tail_weighted 参数
TAIL_WEIGHT = 1.5
TAIL_PORTION = 0.3

# Test 结果可视化
ENABLE_WANDB_TEST_TABLE = True
WANDB_TEST_TABLE_SAMPLES = 50
WANDB_TEST_MAX_NEW_TOKENS = 2048
WANDB_TABLE_TEXT_TRUNCATE = 4096


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.flatten()
    labels = labels.flatten()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if LOSS_METHOD == "default":
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch, **kwargs,
            )

        # 获取 labels
        labels = inputs.get("labels", None)
        if labels is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch, **kwargs,
            )

        # 前向传播 - 这里使用原生 transformers，logits 会正常返回
        outputs = model(**inputs)
        logits = outputs.logits

        # CausalLM shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab = shift_logits.size(-1)

        # per-token CE
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        mask = (shift_labels != -100).to(loss_per_token.dtype)

        if LOSS_METHOD == "token_micro":
            denom = mask.sum().clamp_min(1.0)
            loss = (loss_per_token * mask).sum() / denom

        elif LOSS_METHOD == "sentence_macro":
            per_seq_sum = (loss_per_token * mask).sum(dim=1)
            per_seq_cnt = mask.sum(dim=1).clamp_min(1.0)
            loss = (per_seq_sum / per_seq_cnt).mean()

        elif LOSS_METHOD == "tail_weighted":
            weights = torch.ones_like(loss_per_token)
            seq_len = weights.size(1)
            tail_start = int(seq_len * (1.0 - TAIL_PORTION))
            tail_start = max(0, min(seq_len, tail_start))
            weights[:, tail_start:] = TAIL_WEIGHT

            weighted = loss_per_token * mask * weights
            denom = (mask * weights).sum().clamp_min(1.0)
            loss = weighted.sum() / denom

        else:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch, **kwargs,
            )

        return (loss, outputs) if return_outputs else loss


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "train.log")

    transformers.logging.disable_default_handler()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    metric_logger = logging.getLogger("metrics")
    metric_logger.setLevel(logging.INFO)
    metric_logger.handlers = []
    metric_logger.propagate = False
    
    metric_file_handler = logging.FileHandler(log_file, mode='a')
    metric_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    metric_logger.addHandler(metric_file_handler)

    logger.info(f"Training started. Logging to {log_file}")

    # WandB 登录
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_WATCH"] = "false"
    
    try:
        wandb.login(key=WANDB_KEY)
    except Exception as e:
        logger.warning(f"WandB login warning: {e}")

    # ========== 使用原生 Transformers 加载模型 ==========
    logger.info(f">>> Loading model: {MODEL_ID}")
    logger.info(f"    DType: {DTYPE}")
    logger.info(f"    Load in 4bit: {LOAD_IN_4BIT}")

    # 配置量化（如果需要）
    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 配置 LoRA 或全量微调
    if USE_LORA:
        logger.info(f">>> Applying LoRA adapters (Rank={LORA_RANK})...")
        if LOAD_IN_4BIT:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        logger.warning(">>> Running Full Parameter Fine-Tuning.")
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info(">>> All parameters unfrozen for Full Fine-Tuning.")

    # 启用梯度检查点
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info(">>> Gradient checkpointing enabled.")

    # 加载数据集
    logger.info(f">>> Loading dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    if DATA_SAMPLE_COUNT is not None:
        total_len = len(dataset)
        limit = min(DATA_SAMPLE_COUNT, total_len)
        logger.info(f">>> Limiting dataset to {limit} samples...")
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(limit))
    else:
        logger.info(f">>> Using FULL dataset ({len(dataset)} samples).")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                c, 
                tokenize=TEMPLATE_TOKENIZE,
                add_generation_prompt=ADD_GENERATION_PROMPT
            ) for c in convos
        ]
        return {DATASET_TEXT_FIELD: texts}
    
    logger.info(">>> Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=DATASET_BATCHED)

    logger.info(">>> Splitting dataset into Train and Eval sets...")
    dataset_split = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    logger.info(f"    Train Samples: {len(train_dataset)}")
    logger.info(f"    Eval Samples:  {len(eval_dataset)}")

    # 加载 Test 集
    test_dataset = None
    if TEST_DATA_PATH:
        logger.info(f">>> Loading TEST dataset from {TEST_DATA_PATH}")
        test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

        if DATASET_TEXT_FIELD not in test_dataset.column_names:
            if "messages" not in test_dataset.column_names:
                logger.error(f"!!! ERROR: Test dataset missing 'messages' or '{DATASET_TEXT_FIELD}' field.")
                sys.exit(1)
            logger.info(">>> Formatting TEST dataset...")
            test_dataset = test_dataset.map(formatting_prompts_func, batched=DATASET_BATCHED)

        logger.info(f"    Test Samples: {len(test_dataset)}")

    # 配置训练器
    logger.info(">>> Initializing Trainer...")
    
    final_lr = LEARNING_RATE
    if not USE_LORA and final_lr > 5e-5:
        logger.warning(f"WARNING: Learning rate {final_lr} might be too high for Full FT.")

    trainer = CustomSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        dataset_text_field=DATASET_TEXT_FIELD,
        max_seq_length=MAX_SEQ_LENGTH,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=final_lr,
            neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
            max_grad_norm=MAX_GRAD_NORM,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=LOGGING_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=RANDOM_SEED,
            report_to="wandb",
            run_name=WANDB_RUN_NAME,
            dataset_kwargs={"add_special_tokens": False},
            packing=PACKING,
        ),
    )

    # 移除控制台 Log
    callbacks_to_remove = [
        c for c in trainer.callback_handler.callbacks 
        if isinstance(c, PrinterCallback)
    ]
    for c in callbacks_to_remove:
        trainer.remove_callback(c)
    
    class FileLogCallback(PrinterCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                metric_logger.info(f"Step {state.global_step}: {logs}")
                
    trainer.add_callback(FileLogCallback)

    # 开始训练
    logger.info(">>> Starting Training...")
    logger.info(">>> Verifying trainable parameters...")
 
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")
    
    if trainable_params == 0:
        logger.error("!!! ERROR: No trainable parameters found. Gradients will be 0. !!!")
        sys.exit(1)

    trainer_stats = trainer.train()

    # Test 集评估
    if test_dataset is not None:
        logger.info(">>> Running final evaluation on Test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        logger.info(f">>> Test metrics: {test_metrics}")

        if wandb.run is not None:
            try:
                wandb.run.summary.update(test_metrics)
            except Exception as e:
                logger.warning(f"WandB summary update warning: {e}")

        # WandB Table
        if ENABLE_WANDB_TEST_TABLE and wandb.run is not None:
            try:
                logger.info(">>> Generating predictions for WandB table (sampled)...")
                n = min(WANDB_TEST_TABLE_SAMPLES, len(test_dataset))
                sampled = test_dataset.shuffle(seed=RANDOM_SEED).select(range(n))
                table = wandb.Table(columns=["Prompt", "Ground Truth", "Model Output"])

                model.eval()
                for i in range(n):
                    row = sampled[i]
                    prompt = None
                    gt = None

                    if "messages" in row and row["messages"] is not None:
                        msgs = row["messages"]
                        try:
                            last_a = None
                            for j in range(len(msgs) - 1, -1, -1):
                                if msgs[j].get("role") == "assistant":
                                    last_a = j
                                    break
                            if last_a is not None:
                                gt = msgs[last_a].get("content", None)
                                prompt_msgs = msgs[:last_a]
                            else:
                                prompt_msgs = msgs

                            prompt = tokenizer.apply_chat_template(
                                prompt_msgs,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        except Exception:
                            prompt = row.get(DATASET_TEXT_FIELD, None)
                    else:
                        prompt = row.get(DATASET_TEXT_FIELD, None)

                    if prompt is None:
                        continue

                    def _trunc(s):
                        if s is None:
                            return ""
                        s = str(s)
                        return (s[:WANDB_TABLE_TEXT_TRUNCATE] + "...") if len(s) > WANDB_TABLE_TEXT_TRUNCATE else s

                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH,
                    )
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=WANDB_TEST_MAX_NEW_TOKENS,
                            do_sample=False,
                            use_cache=True,
                        )

                    prompt_len = inputs["input_ids"].shape[1]
                    gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)

                    table.add_data(_trunc(prompt), _trunc(gt), _trunc(gen_text))

                wandb.log({"test_predictions_sample": table})
                logger.info(">>> WandB table logged: test_predictions_sample")
            except Exception as e:
                logger.warning(f"WandB table generation warning: {e}")

        if wandb.run is not None and TEST_DATA_PATH:
            try:
                artifact = wandb.Artifact(name="test_dataset", type="dataset")
                artifact.add_file(TEST_DATA_PATH)
                wandb.log_artifact(artifact)
                logger.info(">>> WandB artifact logged: test_dataset")
            except Exception as e:
                logger.warning(f"WandB artifact log warning: {e}")

    # 保存模型
    logger.info(f">>> Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR) 
    tokenizer.save_pretrained(OUTPUT_DIR)
                
    logger.info(">>> Training Complete.")

    # 显存清理
    logger.info(">>> Cleaning up GPU memory...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
