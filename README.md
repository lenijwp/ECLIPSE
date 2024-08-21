# ECLIPSE

This repository is the official implementation of paper ``Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer''

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running Code

To running our code, run this command:

```
python Eclipse.py --model llama2-7b-chat --dataset 1 --cuda 0 --batchsize 8 --K_round 50 --ref_history 10
```

>📋  We provide three open-source LLMs ['llama2-7b-chat', 'vicuna-7b', 'falcon-7b-instruct'] here, and the dataset 1 is what we used for comparison with GCG and dataset 2 is what we used for template-based methods. If you want to specify a specific LLM as the attacker, you can add the --attacker parameter.

To attack the gpt-3.5-Turbo, run this command:
```
python Eclipse-gpt.py --model gpt3.5 --dataset 1 --cuda 0 --batchsize 8 --K_round 50 --ref_history 10
```

## Pre-trained Models

You can download pretrained models here:
- [llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf#/)
- [vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.5#/) 
- [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct#/)
- [Harmfulness Scorer](https://huggingface.co/hubert233/GPTFuzz#/).

And please replace your local model path in the code file.


