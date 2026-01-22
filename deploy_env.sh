#!/bin/bash
set -e

ENV_NAME="data601"
YAML_FILE="environment.yml"

# 1. 初始化 Conda
eval "$(conda shell.bash hook)"

# 2. 针对新显卡的优化环境变量
export TORCH_CUDA_ARCH_LIST="9.0" # Blackwell 架构对应 9.0+
export CUDA_CACHE_DISABLE=0       # 启用编译缓存

# 3. 创建/更新环境
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Updating environment..."
    conda env update --name $ENV_NAME --file $YAML_FILE --prune
else
    echo "Creating new environment..."
    conda env create --name $ENV_NAME --file $YAML_FILE
fi

echo "Deployment successful!"