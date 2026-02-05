import glob
import os
import re
import subprocess
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

# ==================== Config ====================

COMMON = {
    "base_model_path": "Qwen/Qwen3-4B-Instruct-2507",
    "test_file": "/home/data601/project/dataset/test/test.jsonl",
    "output_dir": "/home/data601/project/eval_results",
    "num_samples": 500,
    "sample_strategy": "random",
    "batch_size": 64,
    "max_seq_length": 2048 ,
    "max_new_tokens": 1400,
    "load_in_4bit": False,
    "seed": 42,
    "stop_on_json": True,
}

MODEL_ROOT = "/home/data601/project/fine_tuned_model"

AUTO_DISCOVER = False
MODEL_ROOTS = [
    MODEL_ROOT,
]
INCLUDE_BASE_MODEL = False
INCLUDE_REGEX = None
EXCLUDE_REGEX = None
SKIP_EXISTING = True

MANUAL_EXPERIMENTS = [
    dict(
        COMMON,
        **{
            "run_name": "Exp7_Final_LoRA_15k_LowLR",
            "mode": "lora",
            "checkpoint_path": os.path.join(MODEL_ROOT, "Exp7_Final_LoRA_15k_LowLR"),
        },
    ),
]

LOG_BUFFER_LINES = 200

# ==================== Helpers ====================

def _format_duration(seconds):
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"

def _short_path(path, max_len=60):
    if not path:
        return "n/a"
    path = os.path.abspath(path)
    if len(path) <= max_len:
        return path
    return f"...{path[-max_len:]}"

def _detect_mode(path):
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return "lora"
    if os.path.isfile(os.path.join(path, "config.json")):
        return "fft"
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return "fft"
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return "fft"
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return "fft"
    if glob.glob(os.path.join(path, "model-*.safetensors")):
        return "fft"
    return None

def _collect_models(root):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []

    kind = _detect_mode(root)
    if kind:
        return [(root, kind)]

    models = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        kind = _detect_mode(path)
        if kind:
            models.append((path, kind))
    return models

def _include_name(name):
    if INCLUDE_REGEX and not re.search(INCLUDE_REGEX, name):
        return False
    if EXCLUDE_REGEX and re.search(EXCLUDE_REGEX, name):
        return False
    return True

def build_experiments():
    experiments = []

    if AUTO_DISCOVER:
        for root in MODEL_ROOTS:
            for path, mode in _collect_models(root):
                run_name = os.path.basename(path.rstrip(os.sep))
                if not _include_name(run_name):
                    continue
                exp = dict(COMMON)
                exp.update(
                    {
                        "run_name": run_name,
                        "mode": mode,
                        "checkpoint_path": path,
                    }
                )
                experiments.append(exp)

    if INCLUDE_BASE_MODEL:
        exp = dict(COMMON)
        exp.update(
            {
                "run_name": "base_model",
                "mode": "base",
                "checkpoint_path": None,
            }
        )
        experiments.append(exp)

    experiments.extend(MANUAL_EXPERIMENTS)
    experiments.sort(key=lambda x: x.get("run_name", ""))
    return experiments

def _add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])

def _add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)

# ==================== UI ====================

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
            return Panel("Waiting for evaluations...", title="Evaluation Status", border_style="blue")

        exp = self.exp
        run_name = exp.get("run_name", "unknown")
        mode = exp.get("mode", "unknown")
        checkpoint = _short_path(exp.get("checkpoint_path"))
        test_file = _short_path(exp.get("test_file"))

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
        elif self.status.startswith("skipped"):
            status_color = "blue"

        status_text = f"[{status_color}]{self.status}[/{status_color}]"

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Evaluation:", f"[{self.idx}/{self.total}] {run_name}")
        table.add_row("Mode:", mode)
        table.add_row("Checkpoint:", checkpoint)
        table.add_row("Dataset:", test_file)
        table.add_row(
            "Batch/Seq:",
            f"bs={exp.get('batch_size')} | max_seq={exp.get('max_seq_length')} | max_new={exp.get('max_new_tokens')}",
        )
        table.add_row(
            "Samples:",
            f"num_samples={exp.get('num_samples', 'all')}",
        )
        table.add_row(
            "Progress:",
            f"remaining={remaining} | exp_elapsed={exp_elapsed} | total_elapsed={total_elapsed} | status={status_text}",
        )
        if self.log_file:
            table.add_row("Log File:", self.log_file)

        return Panel(table, title="Evaluation Status", border_style="blue")

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

# ==================== Runner ====================

def run_evaluations():
    console = Console()
    python_executable = sys.executable
    script_path = "eval_worker.py"

    if not os.path.exists(script_path):
        console.print(f"[red]Error: Could not find {script_path}[/red]")
        return

    experiments = build_experiments()
    if not experiments:
        console.print("[red]No evaluations found. Check MODEL_ROOTS or MANUAL_EXPERIMENTS.[/red]")
        return

    header = HeaderView()
    log_view = LogView(LOG_BUFFER_LINES)
    layout = Layout()
    layout.split(Layout(name="header", size=11), Layout(name="body"))
    layout["header"].update(header)
    layout["body"].update(log_view)

    total = len(experiments)
    header.total_start = time.time()
    log_view.append(f"Starting {total} evaluations...")

    with Live(layout, console=console, refresh_per_second=4, screen=True):
        for i, exp in enumerate(experiments, start=1):
            log_view.clear()

            log_file = f"log_eval_{exp.get('run_name', 'unknown')}.txt"
            header.set_experiment(exp, i, total, log_file)

            metrics_path = os.path.join(exp["output_dir"], f"{exp['run_name']}_metrics.json")
            if SKIP_EXISTING and os.path.exists(metrics_path):
                header.set_status("skipped (metrics exists)")
                log_view.append(f"Skipped: {metrics_path} already exists.")
                continue

            cmd = [python_executable, script_path, "--run_name", exp["run_name"], "--mode", exp["mode"]]
            _add_arg(cmd, "--base_model_path", exp.get("base_model_path"))
            _add_arg(cmd, "--checkpoint_path", exp.get("checkpoint_path"))
            _add_arg(cmd, "--test_file", exp.get("test_file"))
            _add_arg(cmd, "--output_dir", exp.get("output_dir"))
            _add_arg(cmd, "--num_samples", exp.get("num_samples"))
            _add_arg(cmd, "--sample_strategy", exp.get("sample_strategy"))
            _add_arg(cmd, "--batch_size", exp.get("batch_size"))
            _add_arg(cmd, "--max_seq_length", exp.get("max_seq_length"))
            _add_arg(cmd, "--max_new_tokens", exp.get("max_new_tokens"))
            _add_arg(cmd, "--seed", exp.get("seed"))
            _add_flag(cmd, "--load_in_4bit", exp.get("load_in_4bit", False))
            if exp.get("stop_on_json", True):
                cmd.append("--stop_on_json")
            else:
                cmd.append("--no_stop_on_json")

            log_view.append(f"Evaluation [{i}/{total}]: {exp['run_name']}")
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

            time.sleep(5)

if __name__ == "__main__":
    run_evaluations()
