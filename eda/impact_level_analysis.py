import json
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
try:
    with open('Combined.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 统计数据 (确保 1-5 都有值)
    counts = df['impact_level'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index, counts.values, color='#4c72b0', edgecolor='black', alpha=0.7)
    
    # 格式化
    plt.title('Distribution of Impact Levels', fontsize=15)
    plt.xlabel('Impact Level', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5], ['1\nNegligible', '2\nLow', '3\nMedium', '4\nHigh', '5\nSevere'])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 在柱子上显示数字
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.show()

except Exception as e:
    print(f"无法绘图: {e}")