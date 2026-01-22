import pandas as pd
from datasets import load_dataset
import sys

# 配置
OUTPUT_FILE = "gold_standard_500.csv"
SAMPLE_COUNT = 250  # 每一类采样的数量
TOXIC_THRESHOLD = 0.33

# 1. 加载数据
print("Loading dataset (this may take a moment)...")
try:
    # 加载训练集
    ds = load_dataset("google/civil_comments", split="train")
    df = pd.DataFrame(ds)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

print(f"Total rows loaded: {len(df)}")

# 2. 筛选与采样
print("Filtering and Sampling...")

# 分离有毒和无毒样本
# 逻辑：toxicity >= 0.5 为有毒，反之为无毒
toxic_pool = df[df['toxicity'] >= TOXIC_THRESHOLD]
nontoxic_pool = df[df['toxicity'] < TOXIC_THRESHOLD]

# 检查是否有足够的样本
if len(toxic_pool) < SAMPLE_COUNT or len(nontoxic_pool) < SAMPLE_COUNT:
    print("Error: Not enough samples in dataset.")
    sys.exit(1)

# 随机采样
sampled_toxic = toxic_pool.sample(n=SAMPLE_COUNT, random_state=42)
sampled_nontoxic = nontoxic_pool.sample(n=SAMPLE_COUNT, random_state=42)

# 3. 合并与保存
# 合并两个子集
gold_df = pd.concat([sampled_toxic, sampled_nontoxic])

# 打乱顺序 (Shuffle)，防止前250全是毒，后250全是无毒，影响标注体验
gold_df = gold_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 只保留标注需要的关键列 (根据需要调整，这里保留了原始的所有列以防万一)
# 如果只想保留文本和ID，可以取消下面这行的注释:
# gold_df = gold_df[['id', 'text', 'toxicity']]

# 保存为 CSV
gold_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n Success!")
print(f"Created '{OUTPUT_FILE}' with {len(gold_df)} rows.")
print(f"   - Toxic samples: {len(sampled_toxic)}")
print(f"   - Non-toxic samples: {len(sampled_nontoxic)}")
print("   - Data has been shuffled.")