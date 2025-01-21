from bert_score import score as bert_score

# Function to validate logic
def validate_logic(dataloader, model, tokenizer, device, max_eval_samples=10):
    references = []
    predictions = []

    for i, batch in enumerate(dataloader):
        for text in batch:
            print(text)
            sentences = text.split(".")
            input_context = sentences[0].strip() + "."
            reference_continuation = ".".join(sentences[1:]).strip()
            break
    # for text in validation_texts[:max_eval_samples]:
    #     # Split into context and reference
    #     sentences = text.split(".")
    #     input_context = sentences[0].strip() + "."
    #     reference_continuation = ".".join(sentences[1:]).strip()

    #     if not reference_continuation:
    #         continue

    #     # Tokenize input context
    #     input_tokens = tokenizer(input_context, return_tensors="pt").to(device)

    #     # Tokenize reference to determine continuation length
    #     reference_tokens = tokenizer(reference_continuation, return_tensors="pt").input_ids
    #     reference_length = reference_tokens.size(1)

    #     # Generate continuation
    #     max_length = input_tokens["input_ids"].size(1) + reference_length
    #     outputs = model.generate(input_tokens["input_ids"], max_length=max_length)
    #     generated_continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #     # Extract only the generated continuation (exclude input context)
    #     generated_continuation = generated_continuation[len(input_context):].strip()

    #     # Append for BERTScore evaluation
    #     references.append(reference_continuation)
    #     predictions.append(generated_continuation)
        
    #     print(f"Input Context: {input_context}")
    #     print(f"Reference Continuation: {reference_continuation}")
    #     print(f"Generated Continuation: {generated_continuation}")
    #     print("-" * 50)

    # # Compute BERTScore
    # precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=True)
    # print(f"BERTScore - Precision: {precision.mean().item():.4f}, Recall: {recall.mean().item():.4f}, F1: {f1.mean().item():.4f}")

# Run the validation logic
#validate_logic(validation_texts, model, tokenizer, device)
