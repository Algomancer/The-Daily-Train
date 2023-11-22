# The Daily Train
![Logo of The Daily Train](https://raw.githubusercontent.com/Algomancer/The-Daily-Train/main/assets/logo.webp)

**Objective:** To create the best text model possible, trained from scratch on a single h100 (or potentially h200) within a 24-hour period.

## The Plan

### Daily Training
Every day, we will train the latest code from the main branch on a single h100 GPU. Got an idea? Contribute by submitting a pull request. The training of the baseline model begins today. [Training progress link to be added soon]

### Focus Areas
- Exploring architectural improvements in text models.
- Experiments with small models on a consistent dataset to find general computational efficiencies in architecture.
- Encouraging novel and unconventional approaches.

### What We Aren't Doing
- Innovating in areas like quantization, efficient fine-tuning, or compute optimizations (e.g., flash attention). Note: We will utilize these technologies.
- Altering datasets frequently. The initial dataset will remain static, with potential expansions as the project evolves.
- Training a big model.
- Following popular trends without critical analysis.

### What We Are Doing
- Testing numerous small models for computational efficiency.
- Embracing experimental and unconventional ideas.
- Planning for a monthly training cycle if successful.

**Action Point:** Contribute with code! Submit your pull requests. [Join our Discord for discussions and bounties](https://discord.gg/T4TtwVXn).

### Efficiency Metrics
Compute efficiency, defined as the compute required to achieve a set test loss with given design and hyper-parameter choices, is our key metric. More efficient models need fewer GPUs, while less efficient ones need more. By setting a fixed training budget and timeframe, we aim to rapidly identify and explore promising approaches.

## How to Contribute

- **Submit a Pull Request:** Code > Ideas

## Funding

- **Self-Funding:** I'll personally fund a dedicated h100 for continuous operation.
- **Community Support:** We're open to scaling our best ideas with community support. Contact us if you have available compute resources and are interested in our work.

## Acknowledgements
Special thanks to:
- [@Lightning-AI](https://github.com/Lightning-AI/) for [Lit-GPT](https://github.com/Lightning-AI/lit-gpt)
- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

## License
The Daily Train is released under the [Apache 2.0 License](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE).
