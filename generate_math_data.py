import argparse
import random

def generate_file(filename, num_lines):
    operators = ["SUM", "SUBTRACT", "MULTIPLY", "DIVIDE"]
    
    with open(filename, 'w') as f:
        for _ in range(num_lines):
            op = random.choice(operators)
            num1 = random.randint(1, 1000)
            num2 = random.randint(1, 1000)
            
            if op == "SUM":
                result = num1 + num2
            elif op == "SUBTRACT":
                result = num1 - num2
            elif op == "MULTIPLY":
                result = num1 * num2
            elif op == "DIVIDE":
                # Using standard division
                result = num1 / num2
            
            # Write the formatted line to the file
            f.write(f"{op} {num1} {num2} {result}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a text file with random arithmetic operations.")

    parser.add_argument("-n", "--lines", type=int, default=10, help="Number of lines to generate (default: 100)")
    parser.add_argument("-o", "--output", type=str, default="training.txt", help="Output filename (default: training.txt)")
    
    args = parser.parse_args()
    
    generate_file(args.output, args.lines)
    print(f"Successfully generated {args.lines} lines in '{args.output}'")