import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Loading the T5 model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate summary
def generate_summary(input_text, max_length=150):
    input_text = "summarize: " + input_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

# Defining path to the .txt file
file_path = r"text.txt"  # Update this path to point to your input file

# Reading content of the .txt file
with open(file_path, "r", encoding='utf-8') as file:
    text_to_summarize = file.read()

# Output
summary = generate_summary(text_to_summarize)
print("Summarizer:", summary)
print("done")