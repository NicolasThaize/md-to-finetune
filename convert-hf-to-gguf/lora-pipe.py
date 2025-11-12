from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load and merge LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, r"C:\\Users\\Travail\\Documents\\huggingface\\md-to-finetune\\finetuning-autotrain\\mistral7b-local-ft")
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(r"C:\\Users\\Travail\\Documents\\huggingface\\md-to-finetune\\convert-hf-to-gguf")
tokenizer.save_pretrained(r"C:\\Users\\Travail\\Documents\\huggingface\\md-to-finetune\\convert-hf-to-gguf")