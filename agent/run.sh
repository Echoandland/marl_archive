CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B --port 8001
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-8B --port 8002 --enable-reasoning --reasoning-parser deepseek_r1
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8003 

python main.py --dataset rarebench/RAMEDIS --model Qwen/Qwen3-8B --few_shot none --difficulty advanced --num_samples 200 --port 8002

python main.py --dataset rarebench/RAMEDIS --model meta-llama/Llama-3.1-8B-Instruct --few_shot none --difficulty advanced --num_samples 200 --port 8000
python main.py --dataset rarebench/ALL --model openai/gpt-oss-20b --few_shot none --difficulty advanced --port 8000   2>&1 | tee logs/rarebench_gpt-oss-20b_advanced.log
python main.py --dataset rarebench/ALL --model Qwen/Qwen3-8B --few_shot none --difficulty advanced --port 8002   2>&1 | tee logs/rarebench_qwen3_8B_advanced.log
python main.py --dataset rarebench/ALL --model o3-2025-04-16 --few_shot none --difficulty advanced 2>&1 | tee logs/rarebench_o3_advanced.log

python main.py --dataset rarebench/ALL --model o3-2025-04-16 --few_shot none --method MARL --difficulty advanced 2>&1 | tee -a  logs_ouragent/rarebench_o3_advanced.log
python main.py --dataset rarebench/ALL --model Qwen/Qwen3-8B --few_shot none --method MARL --port 8002 --difficulty advanced 2>&1 | tee -a  logs_ouragent/rarebench_qwen3_8B_advanced.log
python main.py --dataset rarebench/ALL --model openai/gpt-oss-20b --few_shot none --method MARL --port 8000 --difficulty advanced 2>&1 | tee -a  logs_ouragent/rarebench_gpt_oss_20B_advanced.log
python main_async.py --dataset rarebench/ALL --model gpt-5 --few_shot none --method MARL --port 8000 --difficulty advanced 2>&1 | tee -a  logs_ouragent/rarebench_gpt_5_advanced.log
python main_async.py --dataset rarebench/ALL --model gpt-5 --few_shot none --port 8000 --difficulty advanced 2>&1 | tee -a  logs/rarebench_gpt_5_advanced.log
python main_async.py --dataset rarebench/ALL --model Qwen/Qwen3-8B --few_shot none --method RARE --port 8002 --difficulty advanced 2>&1 | tee -a  logs_rareagent/rarebench_qwen3_advanced.log
python main_async.py --dataset rarebench/ALL --model openai/gpt-oss-20b --few_shot none --method RARE --port 8000 --difficulty advanced 2>&1 | tee -a  logs_rareagent/rarebench_oss_advanced.log
python main_async.py --dataset rarebench/ALL --model gpt-5 --few_shot none --method RARE --port 8000 --difficulty advanced 2>&1 | tee -a  logs_rareagent/rarebench_gpt_5_advanced.log
python main_async.py --dataset rarebench/ALL --few_shot none --method RARE --multi_expert 
python main_async.py --dataset rarebench/ALL --model gemini-2.5-pro --few_shot none --method RARE --difficulty new_version

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-8B --port 8002 

vllm serve openai/gpt-oss-20b

python main.py --dataset rarebench/RAMEDIS --model Qwen/Qwen3-8B --few_shot none --few_shot none --method RARE --difficulty advanced --port 8002
python main.py --dataset rarebench/ALL --model Qwen/Qwen3-30B-A3B-Instruct-2507 --few_shot none --difficulty advanced --num_samples 200 --port 8003
python main.py --dataset rarebench/RAMEDIS --model o3-2025-04-16 --few_shot none --difficulty advanced --num_samples 200 --port 8000

python main.py --dataset rarebench/RAMEDIS --model meta-llama/Llama-3.1-8B-Instruct --few_shot none --difficulty advanced --num_samples 200 --port 8000
