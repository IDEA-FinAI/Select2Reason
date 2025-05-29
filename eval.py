import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
import torch
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import set_seed
from utils.parser import *
from utils.math_normalization import *
from utils.grader import *

ENGLISH_BENCH = ["olympiadbench", "gaokao2023en", "math500"]
CHINESE_BENCH = ["gaokao2024_mix", "kaoyan", "gaokao_math_qa"]
STEM_BENCH = []
EXPERT_BENCH = ["aime24", "aime25", "amc23"]
NORMAL_BENCH = ENGLISH_BENCH + CHINESE_BENCH + STEM_BENCH

def infer(args):
    model_name = "_".join(args.model_path.split("/")[-3:])
    model = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.93,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.test_set == "all":
        test_sets = ENGLISH_BENCH + CHINESE_BENCH + STEM_BENCH + EXPERT_BENCH
    elif args.test_set == "normal":
        test_sets = NORMAL_BENCH
    elif args.test_set == "expert":
        test_sets = EXPERT_BENCH
    else:
        test_sets = [args.test_set]
    
    normal_test_sets = [ts for ts in test_sets if ts in NORMAL_BENCH]
    expert_test_sets = [ts for ts in test_sets if ts in EXPERT_BENCH]
    
    def batch_infer(test_sets, temperature, top_p, n_sampling):
        if not test_sets:
            return
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens, 
            n=n_sampling
        )
        
        prompt_batch = []
        batch_indices = [0]
        for test_set in test_sets:
            with open(f"eval_data/{test_set}/test.jsonl") as f:
                test_samples = [json.loads(line) for line in f]
                for test_sample in test_samples:
                    question = parse_question(test_sample, test_set)
                    messages = [
                        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                        {"role": "user", "content": question}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt_batch.append(text)
            batch_indices.append(batch_indices[-1] + len(test_samples))
                    
        # start_time = time.time()
        outputs = model.generate(prompt_batch, sampling_params)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken: {elapsed_time:.2f}s")
        
        for index, test_set in enumerate(test_sets):
            with open(f"eval_data/{test_set}/test.jsonl") as f:
                test_samples = [json.loads(line) for line in f]
                
            total_pass = 0
            total_maj = 0
            total_output_tokens = 0
            
            os.makedirs(f"eval_logs/{test_set}/{model_name}", exist_ok=True)
            log_file = f"eval_logs/{test_set}/{model_name}/{datetime.now().strftime('%m%d%H%M')}.txt"
            with open(log_file, "w") as log:
                log.write(f"Model loaded from {args.model_path}\n")
                log.write(f"CUDA device count: {torch.cuda.device_count()}\n")
                log.write(f"Temperature: {temperature}\n")
                log.write(f"Top-p: {top_p}\n")
                log.write(f"Repetition penalty: {args.repetition_penalty}\n")
                log.write(f"N sampling: {n_sampling}\n")
                log.write(f"Max tokens: {args.max_tokens}\n")
                log.write(f"Random seed: {args.seed}\n")
                log.write("*" * 50 + "\n")
                
                for idx, (test_sample, prompt, output) in enumerate(zip(test_samples, prompt_batch[batch_indices[index]:batch_indices[index + 1]], outputs[batch_indices[index]:batch_indices[index + 1]])):
                    log.write(f"SAMPLE {idx + 1}\n")
                    log.write(f"Prompt: {prompt}\n")

                    gold_ans, gold_choice = parse_ground_truth(test_sample, test_set)
                    log.write(f"Gold answer: {gold_ans}\n")
                    log.write(f"Gold choice: {gold_choice}\n")    
                    log.write("*" * 30 + "\n")
                    
                    responses = [output.outputs[i].text for i in range(n_sampling)]
                    pred_answers = [extract_answer(response) for response in responses]
                    output_tokens = [len(output.outputs[i].token_ids) for i in range(n_sampling)]
                    
                    pas = 0
                    for i in range(n_sampling):
                        log.write(f"LLM Response {i + 1}: {responses[i]}\n")
                        log.write(f"Predicted answer {i + 1}: {pred_answers[i]}\n")
                        if pred_answers[i] is None:
                            log.write("Answer not found\n")
                        is_correct = check_is_correct(pred_answers[i], gold_ans, gold_choice)
                        if is_correct:
                            pas += 1
                            log.write("Check Current Answer: Correct\n")
                        else:
                            log.write("Check Current Answer: Incorrect\n")
                        log.write(f"Length of output tokens {i + 1}: {output_tokens[i]}\n")
                        log.write("*" * 30 + "\n")

                    log.write(f"SAMPLE {idx + 1}\n")
                    avg_output_tokens = sum(output_tokens) / n_sampling
                    total_output_tokens += avg_output_tokens                    

                    # pass@1
                    log.write(f"pass@1: {pas / n_sampling:.3f}\n")
                    total_pass += pas / n_sampling
                    
                    # maj@16
                    counter = Counter([ans for ans in pred_answers if ans != ''])
                    log.write(f"Counter: {counter}\n")
                    if counter:
                        maj_answer = counter.most_common(1)[0][0]
                    else:
                        maj_answer = None
                    log.write(f"Majority Answer: {maj_answer}\n")
                    maj_is_correct = check_is_correct(maj_answer, gold_ans, gold_choice) if maj_answer else False
                    if maj_is_correct:
                        log.write("Check Majority Answer: Correct\n")
                        total_maj += 1
                    else:
                        log.write("Check Majority Answer: Incorrect\n")
                        
                    log.write("*" * 50 + "\n")
                
                log.write(f"Total samples count: {len(test_samples)}\n")
                log.write(f"Total pass@1: {total_pass / len(test_samples):.3f}\n")
                log.write(f"Total maj@{n_sampling}: {total_maj / len(test_samples):.3f}\n")
                log.write(f"Average length of output tokens: {total_output_tokens / len(test_samples):.2f}\n")
    
    batch_infer(normal_test_sets, temperature=0, top_p=1, n_sampling=1)
    batch_infer(expert_test_sets, temperature=0.6, top_p=0.95, n_sampling=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct") # open-r1/OpenR1-Qwen-7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B open-thoughts/OpenThinker-7B
    parser.add_argument("--test_set", type=str, default="all", choices=ENGLISH_BENCH + CHINESE_BENCH + STEM_BENCH + EXPERT_BENCH + ["all", "normal", "expert"])
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    infer(args)