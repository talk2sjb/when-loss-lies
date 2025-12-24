import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def predict(model_dir, prompt):
    print(f"Loading model from {model_dir}...")
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate output
    # We use max_new_tokens to limit the answer length
    # do_sample=False ensures deterministic output (greedy decoding), which is usually better for math
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=10, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False 
    )

    # Decode and print the result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned math model")
    parser.add_argument("--model_dir", type=str, default="./math_model_output", help="Directory where the model is saved")
    parser.add_argument("prompt", type=str, nargs="?", default="SUM 305 1000", help="Input prompt, e.g., 'SUM 100 200'")
    
    args = parser.parse_args()
    
    result = predict(args.model_dir, args.prompt)
    print(f"Generated Output: {result}")