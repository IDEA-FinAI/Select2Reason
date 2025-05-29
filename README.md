# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 🕶️

## 📄 Paper & Resources
[![arXiv](https://img.shields.io/badge/Arxiv-2505.17266-AD1C18.svg?logo=arXiv)](https://www.arxiv.org/abs/2505.17266)
[![hf_model_data](https://img.shields.io/badge/%F0%9F%A4%97-Models&Datasets-48A9DC)](https://huggingface.co/collections/cehao/select2reason-6837fbb9231cf484dd49a066)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

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
