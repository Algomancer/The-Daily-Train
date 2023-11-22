# The Daily Train.
![Logo of The Daily Train](https://raw.githubusercontent.com/Algomancer/The-Daily-Train/main/assets/logo.webp)

What is the best text model we can make that is trained, from scratch, on a single h100 (or h200 soon?) in 24 hours? 

&nbsp;

## The Plan

Every day, we will train whatever is on main on a single h100. Have an idea? Submit a pull request. Training the baseline has started today. [Link will go here in a couple hours]

We are primarily interested in exploring model archecture improvements. 

Things we are not doing.

A) Innovating on quantisation, effecient fine tuning and other direct compute optimisations such as flash attention. (we will use them tho!)
B) Improving datasets, we will keep the dataset static. (Or atleast only introduce them as the project grows)
C) Training the biggest model we can.
D) Doing what is popular

Things we are doing
A) Training a lot of small models on the same dataset looking for general compute multipliers in the archecture. 
B) Trying weird stuff.
C) If this works, we will start a monthly train.

Code > Ideas, submit a pull request. [Join our discord, we have bounties](https://discord.gg/T4TtwVXn)

Given set of design and hyper-parameter choices and a fixed test loss target, compute efficiency is the measurement of how much compute is required to meet that test loss. A more efficient model requires less GPUs, a less efficient one needs more. By having a fixed training budget, and wallclock time. We can iterate quickly and find things worth exploring.

## How can I help

Submit a pull request. Let's try some shit.

## Funding

I'll self fund a dedicated h100 to run full time. But, we'd love to scale our best ideas for the community, hit us up if you have compute and you see us doing something cool.


## Acknowledgements
- [@Lightning-AI](https://github.com/Lightning-AI/) for [Lit-GPT](https://github.com/Lightning-AI/lit-gpt)
- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## License

Daily Trained is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license. 
