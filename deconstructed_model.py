from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# The following code imports essential tools and libraries:
# - AutoTokenizer: A Hugging Face tool that turns text into tokens (numbers) that the model understands.
# - AutoModel & AutoModelForSequenceClassification: Pre-built model loaders for generic or classification tasks.
# - torch: The PyTorch library, which handles tensors and deep learning computations.

# The checkpoint variable holds the name of a specific pretrained model. This model was already trained on a sentiment analysis dataset.

# The tokenizer is loaded using the model checkpoint. The tokenizer changes raw text into input tokens that the model can process.

# raw_inputs is a list of example sentences. These sentences will be classified by the model.

# inputs uses the tokenizer to:
#   - Convert text to tokens (numbers)
#   - Add padding to ensure all sequences are the same length
#   - Truncate sentences if they're too long
#   - Output PyTorch tensors for model input

# model loads a pretrained classification model corresponding to the checkpoint. This model can predict sentiment (positive/negative) for the input.

# outputs contains the model's raw predictions (logits). Passing **inputs unpacks the dictionary into keyword arguments for the model.

# predictions applies softmax to the logits to convert them into probabilities for each class (label).

# print(predictions) shows the prediction probabilities for each input sentence.
# print(model.config.id2label) displays the mapping from class indices (like 0 or 1) to human-readable labels (like "NEGATIVE" or "POSITIVE").


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Load the tokenizer, which is a class that tokenizes text into a format that can be used by the model.


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # Tokenize the input text. 


model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # Load the model, which is a class that makes classification predictions based on the input text.
outputs = model(**inputs) # Pass the input text to the model. This returns a tensor of shape (batch_size, sequence_length, hidden_size).

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # Apply a softmax function to the logits to get the normalized predictions.

print(predictions) # Print the predictions.
print(model.config.id2label) # Print the label for each prediction. This is a dictionary that maps the index of the prediction to the label.
