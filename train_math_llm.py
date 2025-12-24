import argparse
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train_model(train_file, model_name, output_dir, epochs, batch_size, checkpoint):
    # 1. Load the dataset
    print(f"Loading dataset from {train_file}...")
    raw_dataset = load_dataset("text", data_files={"train": train_file})
    dataset = raw_dataset["train"].train_test_split(test_size=0.1)

    # 2. Load Tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 and similar models often don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize the dataset
    # We use a short max_length because the math operations are short lines
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Data Collator
    # mlm=False ensures we are doing Causal Language Modeling (predicting the next token)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Load Model
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=500,
        lr_scheduler_type="constant",
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        push_to_hub=False,
        # Use CPU if CUDA is not available, though it will be slow
        no_cuda=False,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # 8. Train
    print("Starting training...")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)

    # 9. Save the model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

    # 10. Plot Training Metrics
    print("Generating training plots...")
    log_history = trainer.state.log_history

    train_steps = []
    train_loss = []
    train_lr = []
    eval_steps = []
    eval_loss = []

    for log in log_history:
        if "loss" in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])
            if "learning_rate" in log:
                train_lr.append(log["learning_rate"])
        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_loss.append(log["eval_loss"])

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label="Training Loss")
    plt.plot(eval_steps, eval_loss, label="Validation Loss", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot LR
    plt.subplot(1, 2, 2)
    plt.plot(train_steps, train_lr, label="Learning Rate", color='orange')
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()

    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Causal LLM on a text file.")
    parser.add_argument("--train_file", type=str, default="training.txt", help="Path to the training text file.")
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Pretrained model to fine-tune (e.g., gpt2, distilgpt2).")
    parser.add_argument("--output_dir", type=str, default="./math_model_output",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from (e.g., ./math_model_output/checkpoint-500).")

    args = parser.parse_args()

    train_model(args.train_file, args.model_name, args.output_dir, args.epochs, args.batch_size, args.checkpoint)
