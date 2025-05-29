import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
import torch
import math
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import set_seed

def select(args):
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
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        n=1,
        logprobs=2
    )
    
    prompt_batch = []
    with open(args.dataset) as f:
        select_samples = [json.loads(line) for line in f]
        for select_sample in select_samples:
            question = select_sample["instruction"]
            messages = [
                {"role": "system", "content": "Please judge the difficulty of this instruction and return 1 if difficult or 0 if not."},
                {"role": "user", "content": question}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_batch.append(text)
            
    outputs = model.generate(prompt_batch, sampling_params)
    
    scores = []
        
    for select_sample, output in zip(select_samples, outputs):
        lp: dict = output.outputs[0].logprobs[0]  # the dict mapping token IDs→Logprob objects
        # Find the two candidates “1” vs “0”
        probs = {entry.decoded_token: math.exp(entry.logprob) for entry in lp.values()}
        total = sum(probs.values())
        difficulty_score = probs.get("1", 0.0) / total
        
        select_sample["difficulty_score"] = difficulty_score
        scores.append(select_sample)
    
    os.makedirs(f"difficulty_scores/{model_name}", exist_ok=True)

    base_path = f"difficulty_scores/{model_name}/OpenR1-Math-196k-verified-difficulty"
    # Write full scored file
    with open(f"{base_path}.jsonl", "w", encoding="utf-8") as out_f:
        for sample in scores:
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    # Sort descending by difficulty_score
    sorted_scores = sorted(scores, key=lambda x: x["difficulty_score"], reverse=True)
    total = len(sorted_scores)

    for pct in (2, 5, 10):
        k = max(1, int(total * pct / 100))
        subset = sorted_scores[:k]
        out_file = f"{base_path}-{pct}%.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for sample in subset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Wrote top {pct}% ({k} samples) → {out_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default=f"OpenR1-Math-196k-verified.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    select(args)