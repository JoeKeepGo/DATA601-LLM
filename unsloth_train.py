import os
import hashlib
import shutil

cache_paths = [
    "/root/autodl-tmp/home/data601/project/unsloth_compiled_cache",
    os.path.expanduser("~/.cache/unsloth"),
]
for p in cache_paths:
    if os.path.exists(p):
        shutil.rmtree(p)

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import unsloth
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
import logging
import sys
import gc
import transformers
import wandb
import weave
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers.trainer_callback import PrinterCallback

# ==================== 全局配置 ====================

# WandB 配置
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = "FFT-5k-5e4-1ep-32x2-23Jan-1"

def _default_wandb_run_id(run_name: str) -> str:
    return hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:8]

WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID") or _default_wandb_run_id(WANDB_RUN_NAME)
WANDB_KEY = "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"

# 路径与模型
BASE_DIR = "/home/data601/project"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
DATA_PATH = os.path.join(BASE_DIR, "dataset/train/train.jsonl")
# 独立 Test 集路径若为 None 则跳过 Test 评估
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset/test/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model", WANDB_RUN_NAME)

# 模型加载参数
# None = 自动检测 (通常为 bfloat16); torch.float16 = fp16; torch.bfloat16 = bf16
DTYPE = None 

# True = 使用 4bit 量化加载 (省显存, 推荐); False = 使用 16bit 加载 (高精度)
LOAD_IN_4BIT = False 

# 数据集控制
# 设置为整数 (e.g., 1000) 使用部分数据; None 使用完整数据集
DATA_SAMPLE_COUNT = 5000 

# 数据集映射时的批处理开关 (通常 True 更快)
DATASET_BATCHED = True

# 训练时用于读取文本的列名
DATASET_TEXT_FIELD = "text"

# 格式化模板参数
# 是否在 prompt 末尾添加生成提示 (例如 "\n<|im_start|>assistant\n")
ADD_GENERATION_PROMPT = False

# apply_chat_template 是否直接分词
# 注意: SFTTrainer 通常需要文本输入 (tokenize=False)，除非在外部做完 tokenization
TEMPLATE_TOKENIZE = False 

# 微调模式
USE_LORA = False # True=LoRA, False=全量微调

# 随机种子
RANDOM_SEED = 3407

# LoRA 配置 (仅 USE_LORA=True 生效)
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# 梯度检查点: Unsloth 推荐使用 "unsloth" 字符串，全量微调时可用 True
USE_GRADIENT_CHECKPOINTING = "unsloth" 

# 训练超参数 (SFTConfig)
MAX_SEQ_LENGTH = 4096     # 上下文长度
LEARNING_RATE = 5e-4      # 学习率 (LoRA: 2e-4, Full: 2e-5)
BATCH_SIZE = 16            # Per Device Batch Size
GRAD_ACCUMULATION = 2     # 梯度累积步数，默认 2
NUM_EPOCHS = 1            # 训练轮数
WARMUP_RATIO = 0.01       # 预热比例

# 日志与保存配置
LOGGING_STEPS = 1         # 每隔多少步打印一次日志
SAVE_STRATEGY = "steps"   # 保存策略: "steps" (按步数) 或 "epoch" (按轮数)
SAVE_STEPS = 50          # 每隔多少步保存一次 Checkpoint (仅当 SAVE_STRATEGY="steps" 时生效)
SAVE_TOTAL_LIMIT = 2      # 最多保留多少个 Checkpoint

# 验证与评估配置
EVAL_STRATEGY = "steps"   # 开启评估：按步数进行
EVAL_STEPS = 10            # 每多少步评估一次
EVAL_BATCH_SIZE = 8      # 验证集的 Batch Size
TEST_SIZE = 0.01          # 验证集比例 (% 数据用于验证)

# 优化器配置
# 选项: "adamw_8bit", "adamw_torch", "adamw_torch_fused"
OPTIMIZER = "adamw_torch_fused" 
WEIGHT_DECAY = 0.1 # 加大权重衰减，防止过拟合，默认 0.01
LR_SCHEDULER_TYPE = "cosine" # 选项: "linear", "cosine", "constant"

# 梯度裁剪与 NEFTune
MAX_GRAD_NORM = 0.5
NEFTUNE_NOISE_ALPHA = 10

# Packing
PACKING = False
REMOVE_UNUSED_COLUMNS = False

# Loss 计算方式配置
# 选项:
# "default" 完全使用 TRL/SFTTrainer 默认 loss
# "token_micro" 按有效 token 数做 micro-average（sum / #valid_tokens）
# "sentence_macro" 先每条样本平均，再对 batch 平均（对短样本权重大）
# "tail_weighted" 对序列尾部 token 加权（后半段更重要）
LOSS_METHOD = "tail_weighted"

# 仅当 LOSS_METHOD="tail_weighted" 时生效
TAIL_WEIGHT = 1.5          # 尾部权重倍数
TAIL_PORTION = 0.3         # 尾部比例：0.5 表示最后 50% token

# Test 结果可视化（WandB）
ENABLE_WANDB_TEST_TABLE = True # True: 训练结束后在 Test 集抽样生成预测并上传 WandB Table
WANDB_TEST_TABLE_SAMPLES = 50 # 抽样条数
WANDB_TEST_MAX_NEW_TOKENS = 2048 # 每条样本生成的最大 token 数
WANDB_TABLE_TEXT_TRUNCATE = 4096 # 记录到 Table 的文本截断长度（字符数）

# 指标计算函数
# 在 GPU 上直接计算 argmax，避免传输巨大的 Logits 到 CPU，避免计算 Accuracy 产生 OOM
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

# 计算验证集的 Accuracy
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    # 展平数据
    preds = preds.flatten()
    labels = labels.flatten()
    
    # 过滤掉 Padding 部分 (-100)
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]
    
    # 计算准确率
    accuracy = (preds == labels).mean()
    
    return {"accuracy": accuracy}


# 自定义 Loss 计算

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # 默认
        if LOSS_METHOD == "default":
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        outputs = model(**inputs)
        try:
            logits = outputs.logits
        except NotImplementedError:
            # Unsloth may disable logits; fall back to model loss if available.
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
        if not torch.is_tensor(logits):
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
        if logits.numel() == 0:
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        labels = inputs.get("labels", None)
        if labels is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        # CausalLM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab = shift_logits.size(-1)

        # per-token CE（none reduction）
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        # 有效 token mask（忽略 -100）
        mask = (shift_labels != -100).to(loss_per_token.dtype)

        if LOSS_METHOD == "token_micro":
            denom = mask.sum().clamp_min(1.0)
            loss = (loss_per_token * mask).sum() / denom

        elif LOSS_METHOD == "sentence_macro":
            # 每条样本：sum / #valid，再对 batch mean
            per_seq_sum = (loss_per_token * mask).sum(dim=1)
            per_seq_cnt = mask.sum(dim=1).clamp_min(1.0)
            loss = (per_seq_sum / per_seq_cnt).mean()

        elif LOSS_METHOD == "tail_weighted":
            # 对尾部 token 加权
            weights = torch.ones_like(loss_per_token)
            seq_len = weights.size(1)
            tail_start = int(seq_len * (1.0 - TAIL_PORTION))
            tail_start = max(0, min(seq_len, tail_start))
            weights[:, tail_start:] = TAIL_WEIGHT

            weighted = loss_per_token * mask * weights
            denom = (mask * weights).sum().clamp_min(1.0)
            loss = weighted.sum() / denom

        else:
            # 未知配置退回默认
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if logits is None:
            return loss, logits, labels
        if isinstance(logits, (tuple, list)):
            if len(logits) == 0 or not torch.is_tensor(logits[0]):
                return loss, None, labels
        elif not torch.is_tensor(logits):
            return loss, None, labels
        return loss, logits, labels

# 训练代码
def train():

    # 设置日志记录
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
    metric_logger.propagate = False # 禁止传播
    
    metric_file_handler = logging.FileHandler(log_file, mode='a')
    metric_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    metric_logger.addHandler(metric_file_handler)

    logger.info(f"Training started. Logging to {log_file}")

    # 登录 WandB
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_WATCH"] = "false"
    os.environ.setdefault("WANDB_NAME", WANDB_RUN_NAME)
    os.environ.setdefault("WANDB_RUN_ID", WANDB_RUN_ID)
    os.environ.setdefault("WANDB_RESUME", "allow")
    
    try:
        wandb.login(key=WANDB_KEY)
    except Exception as e:
        logger.warning(f"WandB login warning: {e}")
    try:
        if wandb.run is None:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=WANDB_RUN_NAME,
                id=WANDB_RUN_ID,
                resume="allow",
            )
    except Exception as e:
        logger.warning(f"WandB init warning: {e}")

    # 加载模型
    logger.info(f">>> Loading model: {MODEL_ID}")
    logger.info(f"    DType: {DTYPE if DTYPE else 'Auto'}")
    logger.info(f"    Load in 4bit: {LOAD_IN_4BIT}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 配置 LoRA
    if USE_LORA:
        logger.info(f">>> Applying LoRA adapters (Rank={LORA_RANK})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = LORA_RANK,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = LORA_ALPHA,
            lora_dropout = LORA_DROPOUT, 
            bias = "none",    
            use_gradient_checkpointing = USE_GRADIENT_CHECKPOINTING,
            random_state = RANDOM_SEED,
        )
    else:
        logger.warning(">>> Running Full Parameter Fine-Tuning. Warning: High VRAM usage! (60GB+)")
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info(">>> All parameters unfrozen for Full Fine-Tuning.")

    # 加载并处理数据集
    logger.info(f">>> Loading dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 数据集截取逻辑
    if DATA_SAMPLE_COUNT is not None:
        total_len = len(dataset)
        limit = min(DATA_SAMPLE_COUNT, total_len)
        logger.info(f">>> Limiting dataset to {limit} samples...")
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(limit))
    else:
        logger.info(f">>> Using FULL dataset ({len(dataset)} samples).")

    # 定义格式化函数
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                c, 
                tokenize = TEMPLATE_TOKENIZE,
                add_generation_prompt = ADD_GENERATION_PROMPT
            ) for c in convos
        ]
        return { DATASET_TEXT_FIELD : texts } 
    
    logger.info(">>> Formatting dataset...")
    # 使用 batched 配置
    dataset = dataset.map(formatting_prompts_func, batched = DATASET_BATCHED)

    # 划分训练集和验证集
    logger.info(">>> Splitting dataset into Train and Eval sets...")
    dataset_split = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    logger.info(f"    Train Samples: {len(train_dataset)}")
    logger.info(f"    Eval Samples:  {len(eval_dataset)}")


    # 加载独立 Test 集
    test_dataset = None
    if TEST_DATA_PATH:
        logger.info(f">>> Loading TEST dataset from {TEST_DATA_PATH}")
        test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

        # 如果 test 数据还没格式化为 text，则复用同样的 formatting 函数
        if DATASET_TEXT_FIELD not in test_dataset.column_names:
            if "messages" not in test_dataset.column_names:
                logger.error(f"!!! ERROR: Test dataset missing 'messages' or '{DATASET_TEXT_FIELD}' field.")
                sys.exit(1)
            logger.info(">>> Formatting TEST dataset...")
            test_dataset = test_dataset.map(formatting_prompts_func, batched = DATASET_BATCHED)

        logger.info(f"    Test Samples: {len(test_dataset)}")

    # 配置训练器
    logger.info(">>> Initializing Trainer...")
    
    # 自动检查全量微调的学习率警告
    final_lr = LEARNING_RATE
    if not USE_LORA and final_lr > 5e-5:
        logger.warning(f"WARNING: Learning rate {final_lr} might be too high for Full FT.")

    trainer = CustomSFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset, 
        dataset_text_field = DATASET_TEXT_FIELD,
        max_seq_length = MAX_SEQ_LENGTH,

        # 传入指标计算函数
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        
        args = SFTConfig(
            output_dir = OUTPUT_DIR,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUMULATION,
            warmup_ratio = WARMUP_RATIO,
            num_train_epochs = NUM_EPOCHS,
            learning_rate = final_lr,

            # 梯度裁剪与 NEFTune
            neftune_noise_alpha = NEFTUNE_NOISE_ALPHA,
            max_grad_norm = MAX_GRAD_NORM,

            # 硬件精度配置
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            
            # 日志与保存配置
            logging_steps = LOGGING_STEPS,
            save_strategy = SAVE_STRATEGY,
            save_steps = SAVE_STEPS,
            save_total_limit = SAVE_TOTAL_LIMIT,

            # 评估配置
            eval_strategy = EVAL_STRATEGY,
            eval_steps = EVAL_STEPS,
            per_device_eval_batch_size = EVAL_BATCH_SIZE,
            
            # 优化器与调度器配置
            optim = OPTIMIZER,
            weight_decay = WEIGHT_DECAY,
            lr_scheduler_type = LR_SCHEDULER_TYPE,
            
            seed = RANDOM_SEED,
            report_to = "wandb",
            run_name = WANDB_RUN_NAME,
            
            # 数据集参数
            dataset_kwargs = {"add_special_tokens": False},
            packing = PACKING,
            remove_unused_columns = REMOVE_UNUSED_COLUMNS,
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
 
    # 打印可训练参数详情
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")
    
    # 如果 trainable_params 为 0，程序应该在这里报错或警告
    if trainable_params == 0:
        logger.error("!!! ERROR: No trainable parameters found. Gradients will be 0. !!!")
        sys.exit(1)

    trainer_stats = trainer.train()

    # 训练结束后对独立 Test 集做最终评估
    if test_dataset is not None:
        logger.info(">>> Running final evaluation on Test set...")

        # Ensure test dataset is tokenized to match the collator's expectations.
        eval_packing = trainer.args.packing if trainer.args.eval_packing is None else trainer.args.eval_packing
        processing_class = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        test_dataset = trainer._prepare_dataset(
            test_dataset,
            processing_class,
            trainer.args,
            eval_packing,
            formatting_func=None,
            dataset_name="test",
        )

        # Ensure WandB run is initialized for eval logging.
        report_to = trainer.args.report_to
        wandb_enabled = (
            (isinstance(report_to, str) and report_to == "wandb")
            or (isinstance(report_to, (list, tuple, set)) and "wandb" in report_to)
        )
        if wandb_enabled and wandb.run is None:
            try:
                wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    name=WANDB_RUN_NAME,
                    id=WANDB_RUN_ID,
                    resume="allow",
                )
            except Exception as e:
                logger.warning(f"WandB init warning: {e}")
                try:
                    from transformers.integrations import WandbCallback

                    trainer.remove_callback(WandbCallback)
                except Exception as e2:
                    logger.warning(f"WandB callback remove warning: {e2}")

        # 获取评估结果
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        # 打印并保存到本地日志
        logger.info(f">>> Test metrics: {test_metrics}")

        # 更新 WandB 的 Summary，确保 Runs 总览面板能看到 test_* 指标
        if wandb.run is not None:
            try:
                wandb.run.summary.update(test_metrics)
            except Exception as e:
                logger.warning(f"WandB summary update warning: {e}")

        # 抽样生成预测并上传 WandB Table
        # 用 generate（而非 trainer.predict 的全量 logits），避免输出巨大 logits 导致 OOM
        if ENABLE_WANDB_TEST_TABLE and wandb.run is not None:
            try:
                logger.info(">>> Generating predictions for WandB table (sampled)...")

                # 固定抽样
                n = min(WANDB_TEST_TABLE_SAMPLES, len(test_dataset))
                sampled = test_dataset.shuffle(seed=RANDOM_SEED).select(range(n))

                table = wandb.Table(columns=["Prompt", "Ground Truth", "Model Output"])

                model.eval()
                for i in range(n):
                    row = sampled[i]

                    # 优先从原始 messages 构建 prompt，否则退回 text 字段
                    prompt = None
                    gt = None

                    if "messages" in row and row["messages"] is not None:
                        msgs = row["messages"]
                        try:
                            # ground truth：取最后一个 assistant 的内容
                            last_a = None
                            for j in range(len(msgs) - 1, -1, -1):
                                if msgs[j].get("role") == "assistant":
                                    last_a = j
                                    break
                            if last_a is not None:
                                gt = msgs[last_a].get("content", None)
                                prompt_msgs = msgs[:last_a]  # 不包含最终 assistant，避免答案泄漏
                            else:
                                prompt_msgs = msgs

                            # prompt：添加 generation prompt，输出将从 assistant 开始生成
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

                    # 文本截断
                    def _trunc(s):
                        if s is None:
                            return ""
                        s = str(s)
                        return (s[:WANDB_TABLE_TEXT_TRUNCATE] + "...") if len(s) > WANDB_TABLE_TEXT_TRUNCATE else s

                    # tokenize + generate
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

                    # 只解码新生成部分
                    prompt_len = inputs["input_ids"].shape[1]
                    gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)

                    table.add_data(_trunc(prompt), _trunc(gt), _trunc(gen_text))

                wandb.log({"test_predictions_sample": table})
                logger.info(">>> WandB table logged: test_predictions_sample")
            except Exception as e:
                logger.warning(f"WandB table generation warning: {e}")

        # 上传测试集文件本身作为 Artifact 备份
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
