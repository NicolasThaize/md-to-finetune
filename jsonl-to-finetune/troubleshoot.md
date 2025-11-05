I'll create comprehensive documentation covering all the issues we encountered and their solutions. Let me first gather information about the current state of the project to provide complete context.


[3 tools called]

Now I'll create comprehensive documentation covering all the issues and solutions we encountered:

# GPT-OSS-20B LoRA Fine-tuning Pipeline: Complete Troubleshooting Guide

## Project Overview

This project implements a complete pipeline for fine-tuning the GPT-OSS-20B model using LoRA (Low-Rank Adaptation) technique with JSONL training data. The pipeline supports multiple data formats and includes comprehensive error handling for common issues encountered during fine-tuning.

## Architecture

The project follows clean architecture principles with clear separation of concerns:

```
jsonl-to-finetune/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── data/                  # Data layer
│   ├── __init__.py
│   └── jsonl_loader.py    # JSONL loading and validation
└── core/                  # Business logic
    ├── __init__.py
    ├── config.py         # LoRA and training configuration
    ├── tokenizer.py      # Tokenizer management
    ├── lora_trainer.py   # LoRA training implementation
    └── pipeline.py       # Main orchestrator
```

## Issues Encountered and Solutions

### Issue 1: JSONL Format Validation Failure

**Problem**: The validation step was failing with the error:
```
Validation failed: Line 1: Missing required fields. Expected 'question/answer' or 'user/assistant'
❌ Error: JSONL format validation failed
```

**Root Cause**: The training data used the modern chat format with `"messages"` containing an array of `{"role": "user", "content": "..."}` and `{"role": "assistant", "content": "..."}` objects, but the validation code only supported the older `"question"/"answer"` or `"user"/"assistant"` formats.

**Solution**: Enhanced the JSONL loader to support three formats:

1. **Question/Answer format**: `{"question": "...", "answer": "..."}`
2. **User/Assistant format**: `{"user": "...", "assistant": "..."}`
3. **Messages format**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

**Implementation**: Updated `data/jsonl_loader.py` to include Harmony format validation with support for:
- Roles: `system`, `developer`, `user`, `assistant` (other roles treated as tool names)
- Channels: `final`, `analysis`, `commentary`
- Tool messages with proper recipient targeting

### Issue 2: Model Offloading Error

**Problem**: Training failed with the error:
```
❌ Error: You can't move a model that has some modules offloaded to cpu or disk.
```

**Root Cause**: The model was loaded with `device_map="auto"` which offloaded some layers to CPU/disk to fit in GPU memory, but the code was trying to move the entire model to a specific dtype with `.to(dtype=model_dtype)`.

**Solution**: Added device mapping detection and conditional dtype conversion:

```python
# Check if model uses device mapping
if hasattr(model, 'hf_device_map') and model.hf_device_map:
    # Model is using device mapping - don't move it
    print("Model loaded with device mapping - skipping dtype conversion")
else:
    # Model is fully loaded - can safely convert dtype
    model = model.to(dtype=model_dtype)
```

### Issue 3: Data Collator Configuration Access Error

**Problem**: Training failed with the error:
```
❌ Error: 'CustomDataCollator' object has no attribute 'config'
```

**Root Cause**: The `CustomDataCollator` class was trying to access `self.config.fp16` but didn't have access to the training configuration.

**Solution**: Modified the data collator to accept configuration parameters directly:

```python
class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, fp16=True):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.fp16 = fp16  # Store fp16 as instance variable
    
    def __call__(self, features):
        batch = super().__call__(features)
        model_dtype = torch.float16 if self.fp16 else torch.float32
        # ... dtype conversion logic
```

### Issue 4: MXFP4 Quantization and Dtype Mismatch

**Problem**: Training failed with the error:
```
Using MXFP4 quantized models requires a GPU, we will default to dequantizing the model to bf16
❌ Error: expected m1 and m2 to have the same dtype, but got: struct c10::Half != struct c10::BFloat16
```

**Root Cause**: The model was automatically dequantized to BFloat16 due to MXFP4 quantization, but the training configuration expected Float16, creating a dtype mismatch.

**Solution**: Implemented comprehensive dtype consistency management:

1. **Disabled problematic quantization**: Removed `quantization_config=None` that was causing issues
2. **Explicit dtype setting**: Used only `dtype=model_dtype` (removed deprecated `torch_dtype`)
3. **Enhanced dtype conversion**: Added explicit dtype conversion for offloaded models
4. **LoRA dtype consistency**: Ensured LoRA parameters match the base model dtype

### Issue 5: Model Loading NoneType Error

**Problem**: Model loading failed with the error:
```
❌ Error: 'NoneType' object has no attribute 'to_dict'
```

**Root Cause**: The `quantization_config=None` parameter was causing the model loading to fail silently and return None.

**Solution**: Cleaned up model loading parameters and added comprehensive error handling:

```python
# Simplified model loading
model = AutoModelForCausalLM.from_pretrained(
    self.config.base_model,
    dtype=model_dtype,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    # Removed problematic quantization_config=None
)

# Added model validation
if model is None:
    raise RuntimeError("Failed to load model - model is None")
```

### Supported Data Formats

1. **Question/Answer**:
```jsonl
{"question": "What is AI?", "answer": "Artificial Intelligence is..."}
```

2. **User/Assistant**:
```jsonl
{"user": "How does a neural network work?", "assistant": "A neural network..."}
```

3. **Messages (Harmony format)**:
```jsonl
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language..."}]}
```
