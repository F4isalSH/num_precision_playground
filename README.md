# Numerical Precision Playground

A small neural network trained on MNIST to compare inference across FP32, FP16, BF16, and INT8 — measuring accuracy, memory, and latency to demonstrate real-world quantization tradeoffs.

## How it works

A 3-layer fully connected network (784 → 128 → 64 → 10) is trained on handwritten digit images. The same trained weights are then stored in four precision formats and evaluated on 10,000 test images.

## Why it matters

Models like GPT-4 and Claude have billions of weights. Quantizing from FP32 to INT8 cuts memory by 75%, saving millions in infrastructure costs while maintaining nearly the same quality. This project demonstrates that principle at a small scale.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```

Open `notebook.ipynb` and run all cells top to bottom. MNIST downloads automatically on first run.

## What each format is

- **FP32** — 32 bits, 7 digits of precision, baseline
- **FP16** — 16 bits, 3 digits of precision, half the memory
- **BF16** — 16 bits, same range as FP32 but less precision, designed for ML
- **INT8** — 8 bits, 256 discrete values, smallest and fastest
