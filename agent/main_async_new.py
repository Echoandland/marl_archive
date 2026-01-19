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
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)
from utils_with_experience import process_rareagent, process_cot, determine_multiagents


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rarebench/ALL')
parser.add_argument('--few_shot', type=str, default='none', choices=['none', 'random', 'dynamic'])
parser.add_argument('--model', type=str, default='gpt-5')
parser.add_argument('--method', type=str, default='RARE')
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--max_turns', type=int, default=3)
parser.add_argument('--max_specialists', type=int, default=3)
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--o3_summary', action='store_true')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--concurrency', type=int, default=2000, help="Max concurrent API calls")
parser.add_argument('--multi_expert', action='store_true', help="Use multiple expert models for RARE method")
parser.add_argument('--use_experience', action='store_true', help="Use multiple expert experiences for RARE method")

args = parser.parse_args()

# Load data and prepare output directory
test_qa, examplers = load_data(args.dataset)
base_out = f"output/multiagents__{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}" if args.multi_expert else f"output/{args.model}_{args.dataset}_{args.few_shot}_{args.difficulty}_expert{args.max_specialists}_turn{args.max_turns}_{args.method}"
inter_path = f'output/interact/multiagents_{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}' if args.multi_expert else f'output/interact/{args.model}_{args.dataset}_{args.few_shot}_{args.difficulty}_expert{args.max_specialists}_turn{args.max_turns}_{args.method}'

os.makedirs(base_out, exist_ok=True)
os.makedirs(inter_path, exist_ok=True)


# Semaphore to rate-limit concurrent tasks
sem = asyncio.Semaphore(args.concurrency)  # e.g., max 30 concurrent API calls

async def process_sample(no, sample):
    async with sem:
        # prepare file paths
        result_file = os.path.join(base_out, f"patient_{no}.json")
        interaction_history_file = os.path.join(inter_path, f"patient_{no}_interaction.json")
        if os.path.exists(result_file):
            rs = json.load(open(result_file, 'r', encoding='utf-8-sig'))
            if rs.get('predict_diagnosis') != '':
                return None
            else:
                cprint(f"[Sample {no}] needs re-produce", 'yellow') 

        # synchronous parts can run in thread
        question, img_path = await asyncio.to_thread(create_question, sample, args.dataset, args.few_shot)
        if args.method == 'MDAgents':
            final_decision, interaction_history = await asyncio.to_thread(process_advanced_query, question, args.model, args)
        elif args.method == 'RARE':
            # if no in [1, 3, 29, 46, 52, 91, 107, 117, 166, 181, 185, 195, 233, 238, 275, 291, 294, 313, 328, 331, 363, 384, 395, 402, 410, 415, 437, 442, # 444, 456, 482, 488, 544, 555, 578, 611, 656, 707, 712, 713, 714, 723, 734, 740, 752, 784, 876, 902, 914, 920, 925, 936, 942, 996, 1018, 1039, # 1042, 1054, 1108]:
            #     difficulty = 'cot'
            # else:
            #     difficulty = await asyncio.to_thread(determine_multiagents, question, args.difficulty)
            # cprint(f"[Sample {no}] difficulty: {difficulty}", 'cyan')
            # if difficulty == 'cot':
            #     final_decision = await asyncio.to_thread(process_cot, sample, question, args.model, result_file, args)
            # else:
            final_decision, interaction_history = await asyncio.to_thread(process_rareagent, sample, question, args.model, args, args.max_turns, args.max_specialists, result_file, interaction_history_file, enable_experience=args.use_experience)
            cprint(f"[Sample {no}] final decision: {final_decision}", 'green')

  
        # build result dict
        if args.dataset == 'medqa':
            result = {
                'question': question,
                'label': sample['answer_idx'],
                'answer': sample['answer'],
                'options': sample['options'],
                'response': final_decision,
                'difficulty': args.difficulty
            }
        else:
            result = {
                'question': question,
                'patient_info': sample['patient_idx'],
                'golden_diagnosis': sample['golden_diagnosis'],
                'predict_diagnosis': final_decision,
                'predict_rank': None
            }

        cprint(f"[Sample {no}] saved to {result_file}", 'yellow')
        return result

async def main():
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    loop.set_default_executor(executor)

    tasks = []
    for no, sample in enumerate(test_qa):
        if args.num_samples is not None and no == args.num_samples:
            break
        # 把协程包装成 Task
        # if no in [516, 518, 519, 520, 1031, 10, 522, 12, 525, 1032, 1041, 20, 532, 533, 23, 535, 536, 537, 27, 28, 29, 538, 31, 542, 544, 546, 35, 548, 1056, 38, 39, 40, 551, 46, 561, 562, 1074, 564, 53, 1076, 56, 57, 570, 1083, 60, 62, 574, 64, 578, 579]:
        tasks.append(asyncio.create_task(process_sample(no, sample)))

    # 显示进度（等待每个已完成的 Task）
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await fut

    # 从同一批 Task 收集结果（Task 可重复 await）
    results = [r for r in await asyncio.gather(*tasks) if r]

    out_file = f"output/{args.model}_{args.dataset}_{args.difficulty}.json"
    with open(out_file, 'w', encoding='utf-8-sig') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    cprint(f"All results saved to {out_file}", 'magenta')

if __name__ == '__main__':
    asyncio.run(main())
