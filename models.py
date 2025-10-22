from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("How are you?")

print(encoded_input["input_ids"])

print(tokenizer.decode(encoded_input["input_ids"]))
