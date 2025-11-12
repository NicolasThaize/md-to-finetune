# Conversion of HF fine tuned model to .gguf

## Installation
1. llama.cpp installation :
```bash
git clone https://github.com/ggerganov/llama.cpp
```

2. Create and activate a python venv : 
```bash
python -m venv .
source ./bin/activate
pip install -r ./llama.cpp requirements.txt
```

3. Alternatively, install cuda pytorch 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Usage
```bash
python .\llama.cpp\convert_hf_to_gguf.py path/to/model --outfile mistral7b-local-ft.gguf
```

Create a Makefile 
```makefile
FROM C:\Users\Travail\Documents\huggingface\md-to-finetune\convert-hf-to-gguf\pending-to-ollama\mistral7b-local-ft\mistral7b-local-ft.gguf
```
```bash
ollama create llmname -f ./Makefile
```


### To convert a lora fined tuned model :
Before using llama.cpp converter, use 
```bash
python ./lora-pipe.py
```
