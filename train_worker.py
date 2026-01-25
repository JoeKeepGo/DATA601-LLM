import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
import shutil
import sys
import unsloth
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
import transformers
import wandb
import weave
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback

# ==================== 1. 命令行参数解析 (ArgParse) ====================
parser = argparse.ArgumentParser(description="Unsloth Fine-Tuning Worker Script")

# [基础配置]
parser.add_argument("--run_name", type=str, required=True, help="Unique name for this run (used for WandB and Output Dir)")
parser.add_argument("--output_dir_base", type=str, default="/home/data601/project/fine_tuned_model", help="Base directory for saving models")
parser.add_argument("--dataset_size", type=int, default=None, help="Number of samples to use (None for all)")
parser.add_argument("--seed", type=int, default=3407, help="Random seed")
parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile")
parser.add_argument("--tf32", action="store_true", dest="tf32", help="Enable TF32")
parser.add_argument("--no_tf32", action="store_false", dest="tf32", help="Disable TF32")
parser.set_defaults(tf32=None)

# [训练超参]
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Per device batch size")
parser.add_argument("--grad_accumulation", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
parser.add_argument("--max_steps", type=int, default=-1, help="Max steps override (-1 disables)")
parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")

# [模式选择: LoRA vs FFT]
parser.add_argument("--use_lora", action="store_true", help="Enable LoRA (Default is False/FFT if not specified)")
parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA Dropout")

# [优化器与Loss配置]
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant"], help="LR scheduler type")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max grad norm (0 disables clipping)")
parser.add_argument("--neftune_noise_alpha", type=float, default=10, help="NEFTune noise alpha (0 disables)")
parser.add_argument("--loss_method", type=str, default="tail_weighted", choices=["default", "token_micro", "sentence_macro", "tail_weighted"], help="Loss method")
parser.add_argument("--tail_weight", type=float, default=1.5, help="Tail weight for tail_weighted loss")
parser.add_argument("--tail_portion", type=float, default=0.3, help="Tail portion in (0, 1] for tail_weighted loss")

# [评估与日志]
parser.add_argument("--logging_steps", type=int, default=1, help="Logging interval in steps")
parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Evaluation strategy")
parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation interval in steps")
parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Save strategy")
parser.add_argument("--save_steps", type=int, default=50, help="Save interval in steps")
parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
parser.add_argument("--metric_for_best_model", type=str, default=None, help="Metric for best model")
parser.add_argument("--greater_is_better", type=str, default=None, help="Whether greater metric is better (true/false)")
parser.add_argument("--dataloader_num_workers", type=int, default=0, help="DataLoader workers")

# [其他高级配置]
parser.add_argument("--gradient_checkpointing", action="store_true", dest="gradient_checkpointing", help="Enable gradient checkpointing")
parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
parser.set_defaults(gradient_checkpointing=None)
parser.add_argument("--bf16", action="store_true", default=None, help="Force bf16")
parser.add_argument("--fp16", action="store_true", default=None, help="Force fp16")
parser.add_argument("--load_in_4bit", action="store_true", help="Use 4bit quantization loading")

args = parser.parse_args()

if args.tail_portion <= 0 or args.tail_portion > 1:
    parser.error("--tail_portion must be in (0, 1].")
if args.tail_weight <= 0:
    parser.error("--tail_weight must be > 0.")
if args.max_grad_norm < 0:
    parser.error("--max_grad_norm must be >= 0.")
if args.neftune_noise_alpha < 0:
    parser.error("--neftune_noise_alpha must be >= 0.")
if args.bf16 and args.fp16:
    parser.error("--bf16 and --fp16 cannot both be enabled.")

def _parse_optional_bool(value, name):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    parser.error(f"--{name} must be a boolean value (true/false).")

# ==================== 2. 全局配置映射 (Global Config) ====================
# 将命令行参数映射回全局变量，以保持后续逻辑不变

# 清理缓存路径
cache_paths = [
    "/root/autodl-tmp/home/data601/project/unsloth_compiled_cache",
    os.path.expanduser("~/.cache/unsloth"),
]
for p in cache_paths:
    if os.path.exists(p):
        shutil.rmtree(p)

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "0" if args.torch_compile else "1"

if args.tf32 is not None:
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

# WandB 配置
WANDB_PROJECT = "DATA601"
WANDB_ENTITY = "joeyang97"
WANDB_RUN_NAME = args.run_name  # 动态获取

def _default_wandb_run_id(run_name: str) -> str:
    return hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:8]

WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID") or _default_wandb_run_id(WANDB_RUN_NAME)
WANDB_KEY = "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"

# 路径与模型
BASE_DIR = "/home/data601/project"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
DATA_PATH = os.path.join(BASE_DIR, "dataset/train/train.jsonl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset/test/test.jsonl")
OUTPUT_DIR = os.path.join(args.output_dir_base, args.run_name) # 动态路径

# 动态参数映射
DTYPE = None 
LOAD_IN_4BIT = args.load_in_4bit
GREATER_IS_BETTER = _parse_optional_bool(args.greater_is_better, "greater_is_better")

if args.bf16:
    USE_BF16 = True
    USE_FP16 = False
elif args.fp16:
    USE_BF16 = False
    USE_FP16 = True
else:
    USE_BF16 = torch.cuda.is_bf16_supported()
    USE_FP16 = not USE_BF16
DATA_SAMPLE_COUNT = args.dataset_size 
DATASET_BATCHED = True
DATASET_TEXT_FIELD = "text"
ADD_GENERATION_PROMPT = False
TEMPLATE_TOKENIZE = False 

# 微调模式
USE_LORA = args.use_lora
RANDOM_SEED = args.seed

# LoRA 配置
LORA_RANK = args.lora_rank
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
GRADIENT_CHECKPOINTING = args.gradient_checkpointing
if GRADIENT_CHECKPOINTING is None:
    LORA_GRADIENT_CHECKPOINTING = "unsloth"
    FFT_GRADIENT_CHECKPOINTING = False
else:
    LORA_GRADIENT_CHECKPOINTING = "unsloth" if GRADIENT_CHECKPOINTING else False
    FFT_GRADIENT_CHECKPOINTING = GRADIENT_CHECKPOINTING

# 训练超参数
MAX_SEQ_LENGTH = args.max_seq_length
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
GRAD_ACCUMULATION = args.grad_accumulation
NUM_EPOCHS = args.num_epochs
WARMUP_RATIO = args.warmup_ratio
MAX_STEPS = args.max_steps

# 日志与保存配置
LOGGING_STEPS = args.logging_steps
SAVE_STRATEGY = args.save_strategy
SAVE_STEPS = args.save_steps
SAVE_TOTAL_LIMIT = args.save_total_limit

# 验证与评估配置
EVAL_STRATEGY = args.eval_strategy
EVAL_STEPS = args.eval_steps
EVAL_BATCH_SIZE = 8      
TEST_SIZE = 0.01          
TEST_EVAL_DURING_TRAIN = True 

# 其他训练控制
LOAD_BEST_MODEL_AT_END = args.load_best_model_at_end
METRIC_FOR_BEST_MODEL = args.metric_for_best_model
DATALOADER_NUM_WORKERS = args.dataloader_num_workers

# 优化器与Loss配置
OPTIMIZER = "adamw_torch_fused" 
WEIGHT_DECAY = 0.1 
LR_SCHEDULER_TYPE = args.lr_scheduler_type
MAX_GRAD_NORM = args.max_grad_norm
NEFTUNE_NOISE_ALPHA = args.neftune_noise_alpha
PACKING = False
REMOVE_UNUSED_COLUMNS = False
LOSS_METHOD = args.loss_method
TAIL_WEIGHT = args.tail_weight
TAIL_PORTION = args.tail_portion

# WandB Table 配置
ENABLE_WANDB_TEST_TABLE = True 
WANDB_TEST_TABLE_SAMPLES = 50 
WANDB_TEST_MAX_NEW_TOKENS = 2048 
WANDB_TABLE_TEXT_TRUNCATE = 4096 
WANDB_TABLE_BATCH_SIZE = 4  
WANDB_TABLE_INCLUDE_MESSAGES = True  
WANDB_TABLE_INCLUDE_RAW_TEXT = True  
WANDB_TABLE_LOG_TOKEN_COUNTS = True  
WANDB_TABLE_LOG_CHAR_COUNTS = True   

# ==================== 3. 核心逻辑 (保持原样) ====================

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
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

        outputs = model(**inputs)
        try:
            logits = outputs.logits
        except NotImplementedError:
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)
        
        if not torch.is_tensor(logits):
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)
        if logits.numel() == 0:
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

        labels = inputs.get("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.size(-1)

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
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if logits is None: return loss, logits, labels
        if isinstance(logits, (tuple, list)) and (len(logits) == 0 or not torch.is_tensor(logits[0])):
            return loss, None, labels
        elif not torch.is_tensor(logits):
            return loss, None, labels
        return loss, logits, labels

def _ensure_special_tokens(tokenizer, logger):
    def _token_id(token):
        if token is None:
            return None
        return tokenizer.convert_tokens_to_ids(token)

    eos_token = tokenizer.eos_token
    eos_id = _token_id(eos_token)
    if eos_id is None:
        for cand in ("<|im_end|>", "<|endoftext|>", "</s>"):
            cand_id = _token_id(cand)
            if cand_id is not None:
                tokenizer.eos_token = cand
                eos_token = cand
                eos_id = cand_id
                logger.warning(f"Adjusted eos_token to '{cand}' to match tokenizer vocab.")
                break
    if eos_id is None:
        logger.warning("No valid eos_token found in tokenizer vocab; leaving eos_token unset.")
        eos_token = None

    pad_token = tokenizer.pad_token
    pad_id = _token_id(pad_token)
    if pad_id is None:
        for cand in (pad_token, "<|endoftext|>", "</s>", eos_token):
            if cand is None:
                continue
            cand_id = _token_id(cand)
            if cand_id is not None:
                tokenizer.pad_token = cand
                pad_token = cand
                pad_id = cand_id
                logger.warning(f"Adjusted pad_token to '{cand}' to match tokenizer vocab.")
                break
    if pad_id is None and eos_token is not None:
        tokenizer.pad_token = eos_token
        pad_token = eos_token
        logger.warning(f"Using eos_token '{eos_token}' as pad_token.")

    return eos_token, pad_token

def train():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # WandB Login
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
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, id=WANDB_RUN_ID, resume="allow")
    except Exception as e:
        logger.warning(f"WandB init warning: {e}")

    logger.info(f">>> Loading model: {MODEL_ID}")
    logger.info(f"    DType: {DTYPE if DTYPE else 'Auto'}")
    logger.info(f"    Load in 4bit: {LOAD_IN_4BIT}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    eos_token, pad_token = _ensure_special_tokens(tokenizer, logger)

    if USE_LORA:
        logger.info(f">>> Applying LoRA adapters (Rank={LORA_RANK})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = LORA_RANK,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = LORA_ALPHA,
            lora_dropout = LORA_DROPOUT, 
            bias = "none",    
            use_gradient_checkpointing = LORA_GRADIENT_CHECKPOINTING,
            random_state = RANDOM_SEED,
        )
    else:
        logger.warning(">>> Running Full Parameter Fine-Tuning. Warning: High VRAM usage! (60GB+)")
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info(">>> All parameters unfrozen for Full Fine-Tuning.")
        if FFT_GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()
            logger.info(">>> Gradient checkpointing enabled for Full Fine-Tuning.")

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
        texts = [tokenizer.apply_chat_template(c, tokenize=TEMPLATE_TOKENIZE, add_generation_prompt=ADD_GENERATION_PROMPT) for c in convos]
        return { DATASET_TEXT_FIELD : texts } 

    logger.info(">>> Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched = DATASET_BATCHED)
    logger.info(">>> Splitting dataset into Train and Eval sets...")
    dataset_split = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    logger.info(f"    Train Samples: {len(train_dataset)}")
    logger.info(f"    Eval Samples:  {len(eval_dataset)}")

    test_dataset_raw = None
    if TEST_DATA_PATH:
        logger.info(f">>> Loading TEST dataset from {TEST_DATA_PATH}")
        test_dataset_raw = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
        if DATASET_TEXT_FIELD not in test_dataset_raw.column_names:
            if "messages" not in test_dataset_raw.column_names:
                logger.error(f"!!! ERROR: Test dataset missing 'messages' or '{DATASET_TEXT_FIELD}' field.")
                sys.exit(1)
            logger.info(">>> Formatting TEST dataset...")
            test_dataset_raw = test_dataset_raw.map(formatting_prompts_func, batched = DATASET_BATCHED)
        logger.info(f"    Test Samples: {len(test_dataset_raw)}")

    logger.info(">>> Initializing Trainer...")
    final_lr = LEARNING_RATE
    if not USE_LORA and final_lr > 5e-5:
        logger.warning(f"WARNING: Learning rate {final_lr} might be too high for Full FT.")

    sft_config_kwargs = dict(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        warmup_ratio = WARMUP_RATIO,
        num_train_epochs = NUM_EPOCHS,
        learning_rate = final_lr,
        neftune_noise_alpha = NEFTUNE_NOISE_ALPHA,
        max_grad_norm = MAX_GRAD_NORM,
        fp16 = USE_FP16,
        bf16 = USE_BF16,
        logging_steps = LOGGING_STEPS,
        save_strategy = SAVE_STRATEGY,
        save_steps = SAVE_STEPS,
        save_total_limit = SAVE_TOTAL_LIMIT,
        eval_steps = EVAL_STEPS,
        per_device_eval_batch_size = EVAL_BATCH_SIZE,
        optim = OPTIMIZER,
        weight_decay = WEIGHT_DECAY,
        lr_scheduler_type = LR_SCHEDULER_TYPE,
        seed = RANDOM_SEED,
        report_to = "wandb",
        run_name = WANDB_RUN_NAME,
        dataset_kwargs = {"add_special_tokens": False},
        packing = PACKING,
        remove_unused_columns = REMOVE_UNUSED_COLUMNS,
    )
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    if MAX_STEPS is not None and MAX_STEPS > 0 and "max_steps" in sft_config_params:
        sft_config_kwargs["max_steps"] = MAX_STEPS
    if "evaluation_strategy" in sft_config_params:
        sft_config_kwargs["evaluation_strategy"] = EVAL_STRATEGY
    elif "eval_strategy" in sft_config_params:
        sft_config_kwargs["eval_strategy"] = EVAL_STRATEGY
    if "dataloader_num_workers" in sft_config_params:
        sft_config_kwargs["dataloader_num_workers"] = DATALOADER_NUM_WORKERS
    if "load_best_model_at_end" in sft_config_params:
        sft_config_kwargs["load_best_model_at_end"] = LOAD_BEST_MODEL_AT_END
    if METRIC_FOR_BEST_MODEL is not None and "metric_for_best_model" in sft_config_params:
        sft_config_kwargs["metric_for_best_model"] = METRIC_FOR_BEST_MODEL
    if GREATER_IS_BETTER is not None and "greater_is_better" in sft_config_params:
        sft_config_kwargs["greater_is_better"] = GREATER_IS_BETTER
    if args.torch_compile and "torch_compile" in sft_config_params:
        sft_config_kwargs["torch_compile"] = True
    if GRADIENT_CHECKPOINTING is not None and "gradient_checkpointing" in sft_config_params:
        sft_config_kwargs["gradient_checkpointing"] = GRADIENT_CHECKPOINTING
    if "dataset_text_field" in sft_config_params:
        sft_config_kwargs["dataset_text_field"] = DATASET_TEXT_FIELD
    if "max_seq_length" in sft_config_params:
        sft_config_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "max_length" in sft_config_params:
        sft_config_kwargs["max_length"] = MAX_SEQ_LENGTH
    if eos_token is not None and "eos_token" in sft_config_params:
        sft_config_kwargs["eos_token"] = eos_token
    if pad_token is not None and "pad_token" in sft_config_params:
        sft_config_kwargs["pad_token"] = pad_token

    trainer_kwargs = dict(
        model = model,
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset, 
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        args = SFTConfig(**sft_config_kwargs),
    )
    trainer_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_init_params:
        trainer_kwargs["dataset_text_field"] = DATASET_TEXT_FIELD
    if "max_seq_length" in trainer_init_params:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    trainer = CustomSFTTrainer(**trainer_kwargs)

    test_dataset_eval = None
    if test_dataset_raw is not None:
        logger.info(">>> Preparing TEST dataset for evaluation...")
        eval_packing = trainer.args.packing if trainer.args.eval_packing is None else trainer.args.eval_packing
        processing_class = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        test_dataset_eval = trainer._prepare_dataset(
            test_dataset_raw,
            processing_class,
            trainer.args,
            eval_packing,
            formatting_func=None,
            dataset_name="test",
        )

        if TEST_EVAL_DURING_TRAIN:
            class TestEvalCallback(TrainerCallback):
                def __init__(self, trainer, test_dataset):
                    self.trainer = trainer
                    self.test_dataset = test_dataset
                    self._in_test_eval = False

                def on_evaluate(self, args, state, control, **kwargs):
                    if self.test_dataset is None:
                        return control
                    if not getattr(self.trainer, "is_in_train", False):
                        return control
                    if self._in_test_eval:
                        return control
                    self._in_test_eval = True
                    try:
                        self.trainer.evaluate(
                            eval_dataset=self.test_dataset,
                            metric_key_prefix="test",
                        )
                    except Exception as e:
                        logger.warning(f"Test eval warning: {e}")
                    finally:
                        self._in_test_eval = False
                    return control

            trainer.add_callback(TestEvalCallback(trainer, test_dataset_eval))

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

    if test_dataset_raw is not None:
        logger.info(">>> Running final evaluation on Test set...")

        if test_dataset_eval is None:
            eval_packing = trainer.args.packing if trainer.args.eval_packing is None else trainer.args.eval_packing
            processing_class = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
            test_dataset_eval = trainer._prepare_dataset(
                test_dataset_raw,
                processing_class,
                trainer.args,
                eval_packing,
                formatting_func=None,
                dataset_name="test",
            )

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

        test_metrics = trainer.evaluate(eval_dataset=test_dataset_eval, metric_key_prefix="test")

        logger.info(f">>> Test metrics: {test_metrics}")

        if wandb.run is not None:
            try:
                wandb.run.summary.update(test_metrics)
            except Exception as e:
                logger.warning(f"WandB summary update warning: {e}")

        if ENABLE_WANDB_TEST_TABLE and wandb.run is not None:
            try:
                logger.info(">>> Generating predictions for WandB table (sampled)...")

                n = min(WANDB_TEST_TABLE_SAMPLES, len(test_dataset_raw))
                sampled = test_dataset_raw.shuffle(seed=RANDOM_SEED).select(range(n))

                table_columns = ["Index", "Prompt", "Ground Truth", "Model Output"]
                if WANDB_TABLE_INCLUDE_MESSAGES:
                    table_columns.append("Messages JSON")
                if WANDB_TABLE_INCLUDE_RAW_TEXT:
                    table_columns.append("Raw Text")
                if WANDB_TABLE_LOG_TOKEN_COUNTS:
                    table_columns.extend(["Prompt Tokens", "GT Tokens", "Gen Tokens"])
                if WANDB_TABLE_LOG_CHAR_COUNTS:
                    table_columns.extend(["Prompt Chars", "GT Chars", "Gen Chars"])
                table_columns.append("Exact Match")
                table = wandb.Table(columns=table_columns)

                def _trunc(s):
                    if s is None:
                        return ""
                    s = str(s)
                    return (s[:WANDB_TABLE_TEXT_TRUNCATE] + "...") if len(s) > WANDB_TABLE_TEXT_TRUNCATE else s

                rows = []
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

                    messages_json = None
                    if WANDB_TABLE_INCLUDE_MESSAGES and "messages" in row and row["messages"] is not None:
                        try:
                            messages_json = json.dumps(row["messages"], ensure_ascii=True, default=str)
                        except Exception:
                            messages_json = None

                    raw_text = row.get(DATASET_TEXT_FIELD, None) if WANDB_TABLE_INCLUDE_RAW_TEXT else None

                    rows.append(
                        {
                            "index": i,
                            "prompt": prompt,
                            "gt": gt,
                            "messages_json": messages_json,
                            "raw_text": raw_text,
                        }
                    )

                if len(rows) == 0:
                    logger.warning("WandB table generation warning: no valid prompts found.")
                else:
                    model.eval()
                    orig_padding_side = tokenizer.padding_side
                    tokenizer.padding_side = "left"
                    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                    batch_size = max(1, WANDB_TABLE_BATCH_SIZE)
                    try:
                        for start in range(0, len(rows), batch_size):
                            batch = rows[start : start + batch_size]
                            prompts = [r["prompt"] for r in batch]

                            inputs = tokenizer(
                                prompts,
                                return_tensors="pt",
                                padding=True,
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
                                    pad_token_id=pad_token_id,
                                )

                            input_len = inputs["input_ids"].shape[1]
                            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

                            for bi, r in enumerate(batch):
                                seq = gen_ids[bi]
                                seq_len = int(seq.shape[0])
                                if pad_token_id is not None:
                                    while seq_len > 0 and int(seq[seq_len - 1]) == pad_token_id:
                                        seq_len -= 1
                                gen_text = tokenizer.decode(seq[input_len:seq_len], skip_special_tokens=True)

                                row_data = [r["index"], _trunc(r["prompt"]), _trunc(r["gt"]), _trunc(gen_text)]
                                if WANDB_TABLE_INCLUDE_MESSAGES:
                                    row_data.append(_trunc(r["messages_json"]))
                                if WANDB_TABLE_INCLUDE_RAW_TEXT:
                                    row_data.append(_trunc(r["raw_text"]))
                                if WANDB_TABLE_LOG_TOKEN_COUNTS:
                                    gt_tokens = None
                                    if r["gt"] is not None:
                                        gt_tokens = len(
                                            tokenizer(
                                                r["gt"],
                                                add_special_tokens=False,
                                                truncation=True,
                                                max_length=MAX_SEQ_LENGTH,
                                            )["input_ids"]
                                        )
                                    gen_tokens = max(0, seq_len - input_len)
                                    row_data.extend([int(prompt_lens[bi]), gt_tokens, int(gen_tokens)])
                                if WANDB_TABLE_LOG_CHAR_COUNTS:
                                    row_data.extend(
                                        [
                                            len(r["prompt"]) if r["prompt"] is not None else None,
                                            len(r["gt"]) if r["gt"] is not None else None,
                                            len(gen_text) if gen_text is not None else None,
                                        ]
                                    )
                                exact_match = None
                                if r["gt"] is not None and gen_text is not None:
                                    exact_match = r["gt"].strip() == gen_text.strip()
                                row_data.append(exact_match)
                                table.add_data(*row_data)
                    finally:
                        tokenizer.padding_side = orig_padding_side

                if len(rows) > 0:
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

    logger.info(f">>> Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR) 
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(">>> Training Complete.")
    logger.info(">>> Cleaning up GPU memory...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
