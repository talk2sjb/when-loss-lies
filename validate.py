import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def validate(model_dir, test_file):
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        return
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' does not exist.")
        return

    output_filepath = os.path.join(model_dir, "eval_result.txt")
    print(f"Validation output will be saved to: {output_filepath}")

    with open(output_filepath, 'w') as output_file:
        print(f"Loading model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"Reading test data from {test_file}...")
        with open(test_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        total = 0
        correct = 0

        start_log = f"Starting validation on {len(lines)} examples...\n"
        print(start_log, end='')
        output_file.write(start_log)
        
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 4:
                continue

            op, num1, num2, expected = parts
            prompt = f"{op} {num1} {num2}"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_parts = generated_text.split()

            # The result should be the 4th token (index 3)
            predicted = generated_parts[3] if len(generated_parts) >= 4 else None

            is_correct = False
            diff_val = "N/A"
            if predicted is not None:
                # For fractions, compare up to 3 decimal places
                if '.' in expected:
                    try:
                        predicted_float = round(float(predicted), 3)
                        expected_float = round(float(expected), 3)
                        diff_val = predicted_float - expected_float
                        if predicted_float == expected_float:
                            is_correct = True
                    except (ValueError, IndexError):
                        is_correct = False  # Prediction was not a valid float
                # For integers, do an exact match
                elif predicted == expected:
                    is_correct = True
                    diff_val = 0
                else:
                    try:
                        diff_val = int(expected) - int(predicted)
                    except (ValueError, IndexError):
                        is_correct = False  # Prediction was not a valid int

            if is_correct:
                correct += 1
                mark = "✅"
            else:
                mark = "❌"

            total += 1
            log_line = f"Test {i+1}: {prompt} : Predicted: '{predicted}', Expected: '{expected}', Diff: {diff_val} {mark}\n"
            print(log_line, end='')
            output_file.write(log_line)

            if (i + 1) % 10 == 0:
                # This progress indicator is for the console only and not written to the file
                print(f"Processed {i + 1}/{len(lines)} - Accuracy: {correct/total:.2%}", end="\r")

        summary_lines = [
            f"\nValidation Complete.\n",
            f"Total: {total}\n",
            f"Correct: {correct}\n"
        ]
        if total > 0:
            summary_lines.append(f"Accuracy: {correct/total:.2%}\n")

        for summary_line in summary_lines:
            print(summary_line, end='')
            output_file.write(summary_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate math model accuracy.")
    parser.add_argument("--model_dir", type=str, default="./math_model_output_750_epoch", help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", type=str, default="math_ops.txt", help="Path to the test file")
    
    args = parser.parse_args()
    validate(args.model_dir, args.test_file)