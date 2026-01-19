#!/bin/bash
#SBATCH --job-name=verl-6                # 作业名称
#SBATCH --output=MARL-agent-math-only.log # 标准输出日志文件
#SBATCH --error=MARL-agent-math-only.log   # 错误日志文件
#SBATCH --ntasks=1                     # 任务数
#SBATCH --cpus-per-task=4              # 每个任务的 CPU 核数
#SBATCH --mem=512G                     # 内存分配
#SBATCH --time=4-00:00:00              # 最大运行时间 3 天
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8                   # 请求 1 个 GPU

if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
  # Optional: provide your conda.sh path via CONDA_SH
  source "${CONDA_SH}"
  if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "${CONDA_ENV}"
  fi
fi

# python upload.py

DATA_DIR="${DATA_DIR:-output/interact/gpt-5_HLE_none_RARE_free_recruited}"
WORKERS="${WORKERS:-8}"
python -u global_score_only_math.py --data_dir "${DATA_DIR}" --workers "${WORKERS}" --verbose