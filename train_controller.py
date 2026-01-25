import subprocess
import os
import sys
import time

# Default advanced options (keep in sync with train_worker.py defaults)
COMMON_ADVANCED_ARGS = {
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "neftune_noise_alpha": 10,
    "loss_method": "tail_weighted",
    "tail_weight": 1.5,
    "tail_portion": 0.3,
}

# 实验配置菜单
experiments = [
    # 实验 1: LoRA Baseline (标准对照组)
    dict(COMMON_ADVANCED_ARGS, **{
        "run_name": "exp1_lora_baseline",
        "use_lora": True,
        "learning_rate": 2e-4,
        "dataset_size": 500, # 使用部分数据
        "lora_rank": 16,
        "lora_alpha": 16,
        "batch_size": 16,
    }),
    
    # 实验 2: LoRA High Rank (测试高Rank表现)
    dict(COMMON_ADVANCED_ARGS, **{
        "run_name": "exp2_lora_rank64",
        "use_lora": True,
        "learning_rate": 2e-4,
        "dataset_size": 500,
        "lora_rank": 64,  # Rank 变大
        "lora_alpha": 128, # Alpha 通常设为 2*Rank
        "batch_size": 16,
    }),

    # 实验 3: FFT Full Fine-Tuning (全参数微调)
    # 注意：FFT 显存消耗极大，必须减小 Batch Size 和 LR
    dict(COMMON_ADVANCED_ARGS, **{
        "run_name": "exp3_fft_full_param",
        "use_lora": False,    # 关闭 LoRA = FFT
        "learning_rate": 1e-5, # FFT 学习率要小 (1e-5 或 5e-6)
        "dataset_size": 200,  # 数据少一点，防止跑太久
        "batch_size": 4,       # 减小 Batch Size 防止 OOM
        "grad_accumulation": 8 # 增加累积步数以弥补 Batch Size
    }),
    
    # 实验 4: Debug Run (快速测试代码是否跑通)
    dict(COMMON_ADVANCED_ARGS, **{
        "run_name": "exp0_debug_run",
        "use_lora": True,
        "learning_rate": 2e-4,
        "dataset_size": 50, # 只跑50条
        "num_epochs": 1,
    }),
]

# 控制器逻辑

def run_experiments():
    python_executable = sys.executable 
    # 确保这里的文件名 worker 脚本名字一致
    script_path = "train_worker.py" 

    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        return

    print(f"Starting {len(experiments)} experiments...\n")

    for i, exp in enumerate(experiments):
        print(f"{'='*60}")
        print(f"Experiment [{i+1}/{len(experiments)}]: {exp['run_name']}")
        print(f"Config: {exp}")
        print(f"{'='*60}")

        # 构建命令行参数
        cmd = [
            python_executable, script_path,
            "--run_name", exp["run_name"],
            "--learning_rate", str(exp.get("learning_rate", 2e-4)),
            "--dataset_size", str(exp.get("dataset_size", 5000)),
            "--batch_size", str(exp.get("batch_size", 16)),
            "--grad_accumulation", str(exp.get("grad_accumulation", 2)),
            "--num_epochs", str(exp.get("num_epochs", 1)),
            "--lr_scheduler_type", exp.get("lr_scheduler_type", COMMON_ADVANCED_ARGS["lr_scheduler_type"]),
            "--max_grad_norm", str(exp.get("max_grad_norm", COMMON_ADVANCED_ARGS["max_grad_norm"])),
            "--neftune_noise_alpha", str(exp.get("neftune_noise_alpha", COMMON_ADVANCED_ARGS["neftune_noise_alpha"])),
            "--loss_method", exp.get("loss_method", COMMON_ADVANCED_ARGS["loss_method"]),
            "--tail_weight", str(exp.get("tail_weight", COMMON_ADVANCED_ARGS["tail_weight"])),
            "--tail_portion", str(exp.get("tail_portion", COMMON_ADVANCED_ARGS["tail_portion"])),
        ]

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
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Logs will be saved to: {log_file}")
        
        start_time = time.time()
        
        with open(log_file, "w") as f:
            try:
                # 启动子进程，将输出重定向到文件
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                duration = time.time() - start_time
                print(f"Success! Duration: {duration/60:.2f} mins\n")
            
            except subprocess.CalledProcessError:
                print(f"Failed! Check {log_file} for details.\n")
                # 实验失败是否终止后续实验？
                # sys.exit(1) 

        # 回收显存
        time.sleep(10)

if __name__ == "__main__":
    run_experiments()
