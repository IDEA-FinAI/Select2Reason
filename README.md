# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 🕶️

## 📄 Paper & Resources
[![arXiv](https://img.shields.io/badge/Arxiv-2505.17266-AD1C18.svg?logo=arXiv)](https://www.arxiv.org/abs/2505.17266)
[![hf_model_data](https://img.shields.io/badge/%F0%9F%A4%97-Models&Datasets-48A9DC)](https://huggingface.co/collections/cehao/select2reason-6837fbb9231cf484dd49a066)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚀 Model Running

The official implementation of [Select2Reason-Qwen-7B](https://huggingface.co/cehao/Select2Reason-Qwen-7B) is trained on 10% selected high-quality instructions from [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cehao/Select2Reason-Qwen-7B"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

# CoT
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

```

## 📊 Evaluation

```bash
python eval.py --model_path cehao/Select2Reason-Qwen-7B --test_set aime24
```

## 🎯 Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning and preference optimization, which provides an efficient training pipiline. The hyperparameters are given in our paper.

## 🙏 Acknowledgments
Special thanks to:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)

## 📓 Cite our Work
```python
@article{yang2025select2reason,
  title={Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning},
  author={Yang, Cehao and Lin, Xueyuan and Xu, Chengjin and Jiang, Xuhui and Wu, Xiaojun and Liu, Honghao and Xiong, Hui and Guo, Jian},
  journal={arXiv preprint arXiv:2505.17266},
  year={2025}
}
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
