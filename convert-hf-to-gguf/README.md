# Conversion of HF fine tuned model to .gguf

## Installation
1. Install `llama.cpp`:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .
   source ./bin/activate
   pip install -r ./llama.cpp requirements.txt
   ```
3. Alternatively install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

## Usage
Run the `llama.cpp` converter on a merged Hugging Face model directory:
```bash
python ./llama.cpp/convert_hf_to_gguf.py path/to/model --outfile mistral7b-local-ft.gguf
```
Create a `Makefile`:
```makefile
FROM path/to/gguf
```
Then build the model for Ollama:
```bash
ollama create llmname -f ./Makefile
```
## LoRA conversion
Before running the `llama.cpp` converter, merge the adapter into the base model:
```bash
python ./lora-pipe.py --basemodel base/model/path --tunedmodel /path/to/lora-adapter --output ./merged-model
```
### Parameters
- `--basemodel`: Hugging Face model identifier or local path for the base model.
- `--tunedmodel`: Local path to the fine-tuned LoRA adapter.
- `--output`: Destination directory for the merged model and tokenizer.

