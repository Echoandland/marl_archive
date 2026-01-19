import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
    # process_rareagent
)
from utils_with_experience_light import (
    process_rareagent
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rarebench/ALL')
parser.add_argument('--few_shot', type=str, default='none', choices=['none', 'random', 'dynamic'])
parser.add_argument('--model', type=str, default='gpt-5')
parser.add_argument('--method', type=str, default='RARE')
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--max_turns', type=int, default=2)
parser.add_argument('--max_specialists', type=int, default=2)
parser.add_argument('--difficulty', type=str, default='light')
parser.add_argument('--o3_summary', action='store_true')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--multi_expert', action='store_true', help="Use multiple expert models for RARE method")
parser.add_argument('--use_experience', action='store_true', help="Use multiple expert experiences for RARE method")

args = parser.parse_args()

# model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

results = []
os.makedirs(f'output/{args.model}_{args.dataset}_{args.few_shot}', exist_ok=True)
os.makedirs(f'output/interact/{args.model}_{args.dataset}_{args.few_shot}', exist_ok=True)

base_out = f"output/multiagents__{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}" if args.multi_expert else f"output/{args.model}_{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}"
inter_path = f'output/interact/multiagents_{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}' if args.multi_expert else f'output/interact/{args.model}_{args.dataset}_{args.few_shot}_{args.difficulty}_{args.method}'

for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break
    #if no not in [45, 65, 95, 101, 127, 145, 146, 158, 165, 275, 287, 313, 321, 415, 423, 462, 482, 487, 531, 625, 652, 662, 664, 676, 684, 688, 698, 707, 718, 728, 737, 750, 760, 768, 771, 773, 787, 788, 802, 809, 819, 835, 836, 839, 841, 869, 890, 896, 907, 908, 910, 913, 918, 930, 936, 942, 943, 944, 949, 953, 987, 996, 1007, 1009, 1055, 1058, 1091]:
        #continue
    print(f"\n[INFO] no: {no}")
    total_api_calls = 0

    if 'rarebench' in args.dataset:
        result_file = os.path.join(base_out, f"patient_{no}.json")
        if os.path.exists(result_file):
            continue
        interaction_history_file = os.path.join(inter_path, f"patient_{no}_interaction.json")

    question, img_path = create_question(sample, args.dataset, args.few_shot)
    if args.method == 'MDAgents':
        final_decision, interaction_history = process_advanced_query(question, args.model, args)
    elif args.method == 'RARE':
        if args.use_experience:
            from utils_with_experience_light import process_rareagent
        else:
            from utils_without_experience_light import process_rareagent
        final_decision, interaction_history = process_rareagent(sample, question, args.model, args, args.max_turns, args.max_specialists)
        print(f"final decision: {final_decision}")
        
    if args.dataset == 'medqa':
        results.append({
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision,
            'difficulty': args.difficulty
        })
    elif 'rarebench' in args.dataset:
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(inter_path, exist_ok=True)
        results.append({
            'question': question,
            'patient_info': sample['patient_idx'],
            'golden_diagnosis': sample['golden_diagnosis'],
            "predict_diagnosis": final_decision,
            "predict_rank": None
        })
        res = {
            'question': question,
            'patient_info': sample['patient_idx'],
            'golden_diagnosis': sample['golden_diagnosis'],
            "predict_diagnosis": final_decision,
            "predict_rank": None
        }
    json.dump(res, open(result_file, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    json.dump(interaction_history, open(interaction_history_file, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print(f"patient {no} finished")

# Save results
path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)

with open(f'output/{args.model}_{args.dataset}_{args.difficulty}.json', 'w') as file:
    json.dump(results, file, indent=4)