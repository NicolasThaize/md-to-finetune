# AutoTrain (Hugging Face) — Quick Install and Usage

## Installation

```bash
pip install autotrain-advanced
```
Copy `.env.example` file to `.env` and fill variables.

### Optional (PyTorch with CUDA)
```bash
# CUDA 12.1 wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## One usage example

Launch autotrain based on config.yaml file:

```bash
autotrain --config ./config.yaml
```
