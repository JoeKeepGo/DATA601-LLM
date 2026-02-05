import subprocess
import os
import sys
import time
from collections import deque

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    print("Missing dependency: rich. Install with: pip install rich")
    raise SystemExit(1) from exc

# 常量定义
GLOBAL_BATCH = 64  # 【核心控制变量】所有实验严格对齐全局 Batch Size，确保 LR 可比性
BASE_SEED = 42     # 固定随机种子，确保实验可复现

# 自动计算梯度累积步数
def auto_grad_accum(batch_size: int) -> int:
    # Global Batch Size = 64。
    # Accum_Steps = 64 / Batch_Size
    return max(1, GLOBAL_BATCH // batch_size)

# 基础配置
COMMON = {
    # 复现性与稳定性
    "seed": BASE_SEED,
    "torch_compile": False,        # 毕设建议关闭，避免动态图编译带来的额外排错成本
    "tf32": True,                  # 开启 TensorFloat-32 (Ampere架构加速)
    "max_grad_norm": 1.0,          # 梯度裁剪，防止梯度爆炸

    # 学习率调度
    "lr_scheduler_type": "cosine", # 余弦退火：前期学习快，后期微调精细
    "warmup_ratio": 0.03,          # 默认预热比例 (LoRA 实验中会覆盖此值)

    # Loss 策略
    "loss_method": "tail_weighted", # 针对 CoT (思维链) 优化的加权 Loss
    "tail_weight": 1.5,
    "tail_portion": 0.30,

    # 评估与日志
    "logging_steps": 1,           # 高频日志，方便画 WandB 曲线
    "eval_strategy": "steps",
    "eval_steps": 10,              # 每 50 步测一次 Validation Loss
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,         # 最多存3个档，节省硬盘
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,    # Loss 越小越好

    # 运行效率
    "gradient_checkpointing": True,# 显存优化技术
    "dataloader_num_workers": 4,   # 数据加载进程数
    
    # 精度
    # "bf16": True,                # A100/A800/4090 建议开启
    # "fp16": False,
}

# 实验组菜单
experiments = [
    # ==========================================================
    # Exp7: Final Production (15k 全量生产 - 修正版)
    # 依据: 
    #   1. Exp1/2 证明 FFT 高 LR 必死 (F1 0.00)，必须放弃 FFT 或用极低 LR。
    #   2. Exp3 证明 LoRA 有效 (F1 0.65)，但 Exp4 证明 3e-4 太大。
    #   3. 只有 "低 LR (5e-5) + 增加轮数" 才能在保护逻辑的同时优化格式。
    # ==========================================================
    dict(COMMON, **{
        "run_name": "Exp7_Final_LoRA_15k_LowLR",
        "use_lora": True,              # FFT 风险太大
        "dataset_size": None,          # 全量
        "learning_rate": 5e-5,         # 比 Exp3(1e-4) 还要低，为了稳住格式
        "lora_rank": 64,               # Exp6 证明 256 没用，回到 64
        "lora_alpha": 128,
        "batch_size": 32,
        "grad_accumulation": auto_grad_accum(32),
        # 低 LR 需要更多步数来收敛。
        "num_epochs": 3,               
        "warmup_ratio": 0.05,          # Exp5 证明 Warmup 必须要有
        "save_steps": 100,             
        "eval_steps": 50,
    }),
]

# 控制器逻辑
LOG_BUFFER_LINES = 200

def _format_duration(seconds):
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"

def _format_value(value, default="n/a"):
    if value is None:
        return default
    return str(value)

class HeaderView:
    def __init__(self):
        self.exp = None
        self.idx = 0
        self.total = 0
        self.total_start = time.time()
        self.exp_start = None
        self.status = "idle"
        self.log_file = None

    def set_experiment(self, exp, idx, total, log_file):
        self.exp = exp
        self.idx = idx
        self.total = total
        self.log_file = log_file
        self.exp_start = time.time()
        self.status = "running"

    def set_status(self, status):
        self.status = status

    def __rich__(self):
        if not self.exp:
            return Panel("Waiting for experiments...", title="Experiment Status", border_style="blue")

        exp = self.exp
        run_name = exp.get("run_name", "unknown")
        use_lora = exp.get("use_lora", False)
        dataset_size = _format_value(exp.get("dataset_size"), "all")
        batch_size = exp.get("batch_size")
        grad_accum = exp.get("grad_accumulation")
        global_batch = "n/a"
        if isinstance(batch_size, int) and isinstance(grad_accum, int):
            global_batch = str(batch_size * grad_accum)

        if use_lora:
            mode = f"LoRA rank={_format_value(exp.get('lora_rank'))}, alpha={_format_value(exp.get('lora_alpha'))}"
        else:
            mode = "FFT (full fine-tune)"

        remaining = max(0, self.total - self.idx)
        now = time.time()
        total_elapsed = _format_duration(now - self.total_start)
        exp_elapsed = _format_duration(now - (self.exp_start or now))

        status_color = "white"
        if self.status.startswith("running"):
            status_color = "yellow"
        elif self.status.startswith("success"):
            status_color = "green"
        elif self.status.startswith("failed"):
            status_color = "red"

        status_text = f"[{status_color}]{self.status}[/{status_color}]"

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Experiment:", f"[{self.idx}/{self.total}] {run_name}")
        table.add_row("Mode:", mode)
        table.add_row("Batch:", f"bs={_format_value(batch_size)} | accum={_format_value(grad_accum)} | global={global_batch}")
        table.add_row(
            "Data:",
            f"dataset={dataset_size} | epochs={_format_value(exp.get('num_epochs'))} | max_steps={_format_value(exp.get('max_steps'))}",
        )
        table.add_row(
            "LR:",
            f"lr={_format_value(exp.get('learning_rate'))} | warmup={_format_value(exp.get('warmup_ratio'))} | sched={_format_value(exp.get('lr_scheduler_type'))}",
        )
        table.add_row(
            "Eval/Save:",
            f"eval={_format_value(exp.get('eval_strategy'))}@{_format_value(exp.get('eval_steps'))} | save={_format_value(exp.get('save_strategy'))}@{_format_value(exp.get('save_steps'))} | keep={_format_value(exp.get('save_total_limit'))}",
        )
        table.add_row(
            "Progress:",
            f"remaining={remaining} | exp_elapsed={exp_elapsed} | total_elapsed={total_elapsed} | status={status_text}",
        )
        if self.log_file:
            table.add_row("Log File:", self.log_file)

        return Panel(table, title="Experiment Status", border_style="blue")

class LogView:
    def __init__(self, max_lines=200):
        self.lines = deque(maxlen=max_lines)

    def append(self, line):
        if line is None:
            return
        self.lines.append(line)

    def clear(self):
        self.lines.clear()

    def __rich_console__(self, console, options):
        if not self.lines:
            yield Panel(Text("Waiting for output..."), title="Live Output", border_style="cyan")
            return

        lines = list(self.lines)
        max_height = options.max_height
        if max_height is not None:
            interior = max(1, max_height - 2)
            if len(lines) > interior:
                lines = lines[-interior:]
        text = Text("\n".join(lines))
        yield Panel(text, title="Live Output", border_style="cyan")

def _add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])

def _add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)

def _add_bool_with_neg(cmd, flag, neg_flag, value):
    if value is True:
        cmd.append(flag)
    elif value is False:
        cmd.append(neg_flag)

def run_experiments():
    console = Console()
    python_executable = sys.executable 
    # 确保这里的文件名 worker 脚本名字一致
    script_path = "train_worker.py" 

    if not os.path.exists(script_path):
        console.print(f"[red]Error: Could not find {script_path}[/red]")
        return

    header = HeaderView()
    log_view = LogView(LOG_BUFFER_LINES)
    layout = Layout()
    layout.split(Layout(name="header", size=11), Layout(name="body"))
    layout["header"].update(header)
    layout["body"].update(log_view)

    total = len(experiments)
    header.total_start = time.time()
    log_view.append(f"Starting {total} experiments...")

    with Live(layout, console=console, refresh_per_second=4, screen=True):
        for i, exp in enumerate(experiments, start=1):
            log_view.clear()

            # 构建命令行参数
            cmd = [python_executable, script_path, "--run_name", exp["run_name"]]
            _add_arg(cmd, "--seed", exp.get("seed", COMMON["seed"]))
            _add_arg(cmd, "--learning_rate", exp.get("learning_rate", 2e-4))
            _add_arg(cmd, "--dataset_size", exp.get("dataset_size"))
            _add_arg(cmd, "--batch_size", exp.get("batch_size", 16))
            _add_arg(cmd, "--grad_accumulation", exp.get("grad_accumulation", 2))
            _add_arg(cmd, "--num_epochs", exp.get("num_epochs", 1))
            _add_arg(cmd, "--max_steps", exp.get("max_steps"))
            _add_arg(cmd, "--warmup_ratio", exp.get("warmup_ratio", COMMON["warmup_ratio"]))
            _add_arg(cmd, "--lr_scheduler_type", exp.get("lr_scheduler_type", COMMON["lr_scheduler_type"]))
            _add_arg(cmd, "--max_grad_norm", exp.get("max_grad_norm", COMMON["max_grad_norm"]))
            _add_arg(cmd, "--neftune_noise_alpha", exp.get("neftune_noise_alpha"))
            _add_arg(cmd, "--loss_method", exp.get("loss_method", COMMON["loss_method"]))
            _add_arg(cmd, "--tail_weight", exp.get("tail_weight", COMMON["tail_weight"]))
            _add_arg(cmd, "--tail_portion", exp.get("tail_portion", COMMON["tail_portion"]))
            _add_arg(cmd, "--logging_steps", exp.get("logging_steps", COMMON["logging_steps"]))
            _add_arg(cmd, "--eval_strategy", exp.get("eval_strategy", COMMON["eval_strategy"]))
            _add_arg(cmd, "--eval_steps", exp.get("eval_steps", COMMON["eval_steps"]))
            _add_arg(cmd, "--save_strategy", exp.get("save_strategy", COMMON["save_strategy"]))
            _add_arg(cmd, "--save_steps", exp.get("save_steps", COMMON["save_steps"]))
            _add_arg(cmd, "--save_total_limit", exp.get("save_total_limit", COMMON["save_total_limit"]))
            _add_arg(cmd, "--metric_for_best_model", exp.get("metric_for_best_model", COMMON["metric_for_best_model"]))
            _add_arg(cmd, "--greater_is_better", exp.get("greater_is_better", COMMON["greater_is_better"]))
            _add_arg(cmd, "--dataloader_num_workers", exp.get("dataloader_num_workers", COMMON["dataloader_num_workers"]))
            _add_flag(cmd, "--load_best_model_at_end", exp.get("load_best_model_at_end", False))
            _add_flag(cmd, "--torch_compile", exp.get("torch_compile", COMMON.get("torch_compile", False)))
            _add_bool_with_neg(cmd, "--tf32", "--no_tf32", exp.get("tf32", COMMON.get("tf32")))
            _add_bool_with_neg(
                cmd,
                "--gradient_checkpointing",
                "--no_gradient_checkpointing",
                exp.get("gradient_checkpointing", COMMON.get("gradient_checkpointing")),
            )
            if exp.get("bf16"):
                cmd.append("--bf16")
            if exp.get("fp16"):
                cmd.append("--fp16")

            # 处理布尔开关和LoRA参数
            if exp.get("use_lora", False):
                cmd.append("--use_lora")
                cmd.extend(["--lora_rank", str(exp.get("lora_rank", 16))])
                cmd.extend(["--lora_alpha", str(exp.get("lora_alpha", 16))])
                cmd.extend(["--lora_dropout", str(exp.get("lora_dropout", 0))])
            
            # 处理可选参数
            if exp.get("output_dir_base"):
                cmd.extend(["--output_dir_base", exp["output_dir_base"]])

            # 日志文件名
            log_file = f"log_{exp['run_name']}.txt"
            header.set_experiment(exp, i, total, log_file)
            log_view.append(f"Experiment [{i}/{total}]: {exp['run_name']}")
            log_view.append(f"Config: {exp}")
            log_view.append(f"Logs: {log_file}")
            log_view.append(f"Command: {' '.join(cmd)}")

            start_time = time.time()
            return_code = 1
            try:
                with open(log_file, "w") as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    if process.stdout is None:
                        raise RuntimeError("Failed to capture process output.")
                    for line in process.stdout:
                        f.write(line)
                        f.flush()
                        log_view.append(line.rstrip("\n"))
                    return_code = process.wait()
            except Exception as exc:
                log_view.append(f"Controller error: {exc}")

            duration = time.time() - start_time
            if return_code == 0:
                header.set_status(f"success ({duration/60:.2f}m)")
                log_view.append(f"Success! Duration: {duration/60:.2f} mins")
            else:
                header.set_status(f"failed (code={return_code})")
                log_view.append(f"Failed! Check {log_file} for details.")

            # 回收显存
            time.sleep(10)

if __name__ == "__main__":
    run_experiments()
