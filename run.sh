#!/bin/bash
# 激活 conda 环境并运行 TradingAgents CLI

source ~/anaconda3/etc/profile.d/conda.sh
conda activate langchain
python cli/main.py "$@"
