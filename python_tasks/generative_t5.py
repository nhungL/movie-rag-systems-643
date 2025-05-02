import os
os.environ["HF_HOME"] = "/mnt/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/hf_cache"
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
    
def generate_answer(query, top_titles, top_docs, model, tokenizer, max_input_len=512, max_output_len=300):
    context = "\n\n".join([f"Movie: {title}\n{plot[:300]}" for title, plot in zip(top_titles[:1], top_docs[:1])])
    prompt = (
        f"You are an expert in movies.\n"
        f"Task: Explain in **one detailed sentence** why this movie is a good recommendation for the question: {query}\n\n"
        f"Context: \n{context}\n\n"
    )
    print(prompt)
    encoded = tokenizer(prompt)
    print(f"Prompt token length: {len(encoded['input_ids'])}")
    
    print(f"\nUser query: {query}")
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=max_input_len, truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=max_output_len, num_beams=4, early_stopping=False, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_path', type=str, default="retrieval_results.json")
    args = parser.parse_args()

    # Load top retrieved results
    import json
    with open(args.retrieval_path, "r") as f:
        results = json.load(f)
    top_titles = results["titles"]
    top_docs = results["docs"]
    query = results["query"]

    # Load model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Generate and print
    answer = generate_answer(query, top_titles, top_docs, model, tokenizer)
    print(f"Movie bot: {answer}")