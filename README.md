# Self-train GPT: a tiny ChatGPT trained from scratch on a gaming laptop

A GPT language model written from scratch in about 200 lines of PyTorch and trained on one laptop GPU (RTX 4060 Laptop, 8 GB).
Its brain starts as random numbers. It learns English by reading, one "guess the next token" at a time.

This is the code from the ThinkInCode video **"I Trained My Own ChatGPT on a Gaming Laptop"** (Part 1 of 2).

| Run | Data | Model | Training | Result |
|---|---|---|---|---|
| Shakespeare | 1.1M characters, 65-character vocabulary | 10.75M params · 6 layers · context 256 | 5,000 steps · 14.8 min · 1.71 GB VRAM | best val loss 1.465 at step 1,750, then overfits |
| TinyStories | 2.1M short stories → 474M GPT-2 tokens | 51M params · 8 layers · context 512 | 10,000 steps · 5 h 20 min · 7.12 GB VRAM | val loss 1.391, no overfitting |

What it wrote:

```
step 0       Once upon a time, Incre cinematic siege Wide Zak Moment. Broadway shrugged linking assassinated. Teaspoons.
step 1,000   From that day on, Lily never forgot to slap her toys before going outside.
step 10,000  A little girl named Lily falls down and hurts her arm. Her mommy hugs her and says, "I'm here to support you, Lily."
             ... The doctor fixed Lily's arm, and she was happy and grateful for her mommy's support.
```

All the samples are in [`samples/`](samples/), and the loss curves are in [`logs/`](logs/).

---

## Setup

You need Python 3.10+ and an NVIDIA GPU. It also runs on CPU, just very slowly.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# 1. PyTorch WITH CUDA first. Pick your command at https://pytorch.org (Pip + your CUDA version), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. the rest
pip install -r requirements.txt
# 3. check the GPU is found
python check_gpu.py
```

A plain `pip install torch` gives you a CPU-only build. Installing `accelerate` can also quietly swap your CUDA build for a CPU-only one. If `check_gpu.py` says **No GPU found**, reinstall torch from the CUDA index with `--no-deps`.

## Run 1: Shakespeare (about 15 min)

```bash
python prepare.py --dataset shakespeare
python train.py --config configs/shakespeare_char.json
python sample.py --run shakespeare_char --prompt "ROMEO:"
python plot_loss.py --run shakespeare_char --dark
```

## Run 2: TinyStories (about 5 hours on an RTX 4060 Laptop)

```bash
python prepare.py --dataset tinystories            # add --max_stories 500000 for a quicker subset
python train.py --config configs/tinystories.json
python sample.py --run tinystories --prompt "Once upon a time" --num_samples 5
python plot_loss.py --run tinystories --dark
```

If Hugging Face is blocked on your network, set a mirror before `prepare.py`, e.g. `HF_ENDPOINT=https://hf-mirror.com`.

**Less VRAM?** Lower `batch_size` and raise `grad_accum_steps` in the config. Any `--key value` on the command line overrides the config.

## Train it on your own text

```bash
python prepare.py --dataset textfile --input path/to/folder_of_txt --name my_text --tokenizer char
python train.py --config configs/my_videos_char.json --dataset my_text --run_name my_text
```

## Files

```
model.py        the GPT: token + position embeddings, causal self-attention, MLP blocks, weight tying
train.py        training loop: mixed precision, LR warmup + cosine decay, eval, keeps the best checkpoint, logs, samples
sample.py       generate text from a checkpoint (streams token by token)
prepare.py      download + tokenize + split into train.bin / val.bin
tokenizer.py    character-level tokenizer and GPT-2 BPE (tiktoken)
plot_loss.py    train vs validation loss chart
check_gpu.py    is CUDA working?
configs/        the settings for each run
samples/        what the model wrote at each step (the video's montage)
logs/           loss CSVs + charts
run_logs/       console output of every real run, plus RUN_REPORT.md
part2/          teaser for Part 2: LoRA fine-tune of Qwen2.5-0.5B into a grumpy Python teacher
```

`CODE_NOTES.md` explains every file and function in detail.

The trained weights aren't in the repo (they're 129 MB and 614 MB). Run the steps above to make your own.

## Part 2

This model can write, but it can't chat: ask it a question and it just carries on with a story. Part 2 fine-tunes a small pretrained model (Qwen2.5-0.5B, LoRA) to answer like a grumpy Python teacher. The code is in [`part2/`](part2/).

---

Built on the ideas in Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023).

If you train it, tell me in the YouTube comments what your model wrote, especially if it built a house for the thunder too.

Happy coding, and keep coding.
