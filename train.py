import torch
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a T5 model (T5-efficient-mini) on time series data")
    parser.add_argument("--model_name", type=str, default="T5-efficient-mini", help="Model name or path")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--output_dir", type=str, default="./models/t5_time_series", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every X steps")
    return parser.parse_args()

def preprocess_function(examples, tokenizer):
    inputs = [ex["input"] for ex in examples]
    targets = [ex["target"] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

def main():
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    train_dataset = load_dataset("json", data_files=args.train_data)["train"].map(lambda x: preprocess_function(x, tokenizer))
    val_dataset = load_dataset("json", data_files=args.val_data)["train"].map(lambda x: preprocess_function(x, tokenizer))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
if __name__ == "__main__":
    main()