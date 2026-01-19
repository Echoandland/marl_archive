import os
import json
import random
import argparse
import asyncio
import concurrent.futures
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable

from utils_with_experience_math import (
    Agent, Group, load_data, create_question, process_rareagent
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HLE')
parser.add_argument('--split', type=str, default='test', choices=['test', 'train'])
parser.add_argument('--few_shot', type=str, default='none', choices=['none', 'random', 'dynamic'])
parser.add_argument('--model', type=str, default='gpt-5')
parser.add_argument('--method', type=str, default='RARE', choices=['RARE'])
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--max_turns', type=int, default=4)
parser.add_argument('--max_specialists', type=int, default=4)
parser.add_argument('--o3_summary', action='store_true')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--concurrency', type=int, default=12, help="Max concurrent problems")
parser.add_argument('--multi_expert', action='store_true', help="Use multiple expert models for RARE method")
parser.add_argument('--use_experience', action='store_true', help="Use experience KB for specialists (if available)")
parser.add_argument('--free_recruitment', action='store_true', help="Recruiter defines specialties freely without preset pool")
parser.add_argument('--unified_experience', action='store_true', help="Use one shared experience KB for all agents (ignore per-role KB)")

args = parser.parse_args()

# 文件名后缀：根据开关组合（顺序固定，便于匹配）
SUFFIX_PARTS = []
if getattr(args, "use_experience", False):
    SUFFIX_PARTS.append("useexperience")
if getattr(args, "free_recruitment", False):
    SUFFIX_PARTS.append("free_recruited")
SUFFIX = "_".join(SUFFIX_PARTS)

def _with_suffix(name: str, sep: str = "_") -> str:
    return f"{name}{sep}{SUFFIX}" if SUFFIX else name

# Load data
test_qa, examplers = load_data(args.dataset)
samples = test_qa if args.split == 'test' else examplers

# 基础输出目录（带后缀）
split_tag = '' if args.split == 'test' else f'_{args.split}'
base_out = (
    f"output/multiagents_{args.dataset}{split_tag}_{args.few_shot}_{args.method}"
    if args.multi_expert
    else f"output/{args.model}_{args.dataset}{split_tag}_{args.few_shot}_{args.method}"
)
base_out = _with_suffix(base_out)

# 交互日志目录（带后缀）
inter_path = (
    f'output/interact/multiagents_{args.dataset}{split_tag}_{args.few_shot}_{args.method}'
    if args.multi_expert
    else f'output/interact/{args.model}_{args.dataset}{split_tag}_{args.few_shot}_{args.method}'
)
inter_path = _with_suffix(inter_path)

os.makedirs(base_out, exist_ok=True)
os.makedirs(inter_path, exist_ok=True)

import asyncio

async def process_sample(no, sample, sem):
    async with sem:
        # 每个样本的文件名（带后缀）
        result_file = os.path.join(
            base_out,
            f"problem_{no}{('_' + SUFFIX) if SUFFIX else ''}.json"
        )
        interaction_history_file = os.path.join(
            inter_path,
            f"problem_{no}_interaction{('_' + SUFFIX) if SUFFIX else ''}.json"
        )

        # 若已存在且有预测答案则跳过
        if os.path.exists(result_file):
            try:
                rs = json.load(open(result_file, 'r', encoding='utf-8-sig'))
                if rs.get('predict_answer', '') != '':
                    return None
                else:
                    cprint(f"[Sample {no}] needs re-produce", 'yellow')
            except Exception:
                pass

        question, img_path = await asyncio.to_thread(create_question, sample, args.dataset, args.few_shot)

        final_answer, interaction_history = await asyncio.to_thread(
            process_rareagent, sample, question, args.model, args,
            args.max_turns, args.max_specialists, result_file, interaction_history_file
        )
        cprint(f"[Sample {no}] final answer: {final_answer}", 'green')

        result = {
            'problem': question,
            'problem_idx': sample.get('problem_idx', no),
            'golden_answer': sample.get('golden_answer', None),
            'predict_answer': final_answer
        }
        cprint(f"[Sample {no}] pipeline wrote result to {result_file}", 'yellow')
        return result

async def main():
    loop = asyncio.get_event_loop()
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    # loop.set_default_executor(executor)

    # Create semaphore inside async function to avoid "different loop" error
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []
    for no, sample in enumerate(samples):
        if args.num_samples is not None and args.num_samples >= 0 and no == args.num_samples:
            break
        tasks.append(asyncio.create_task(process_sample(no, sample, sem)))

    cprint(f"[DEBUG] Created {len(tasks)} tasks for samples 0-{len(samples)-1}", 'cyan')
    cprint(f"[DEBUG] Concurrency limit: {args.concurrency}", 'cyan')
    
    # gather all with exception capture to avoid dropping remaining tasks on first error
    results_all = await asyncio.gather(*tasks, return_exceptions=True)
    cprint(f"[DEBUG] Gathered {len(results_all)} results", 'cyan')
    
    # Check for exceptions
    exceptions = [r for r in results_all if isinstance(r, Exception)]
    if exceptions:
        cprint(f"[DEBUG] Found {len(exceptions)} exceptions:", 'red')
        for i, exc in enumerate(exceptions[:5]):  # Show first 5
            cprint(f"  Exception {i}: {type(exc).__name__}: {str(exc)[:200]}", 'red')
    
    results = [r for r in results_all if r and not isinstance(r, Exception)]
    cprint(f"[DEBUG] Valid results: {len(results)}", 'cyan')

    # 汇总文件名（带后缀）
    out_file = _with_suffix(f"output/{args.model}_{args.dataset}{split_tag}_summary")
    out_file = f"{out_file}.json"

    with open(out_file, 'w', encoding='utf-8-sig') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    cprint(f"All results saved to {out_file}", 'magenta')

if __name__ == '__main__':
    asyncio.run(main())
