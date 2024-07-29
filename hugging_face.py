from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def summarize_body(body):
    # Tokenize the text
    inputs = tokenizer.encode("summarize: " + body, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary using T5 - can adjust the reponse parameters here
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary