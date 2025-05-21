import argparse
import json
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import exploring_data_layout as loader
from sklearn.metrics import accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")
    
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./bert_output", help="Output directory")
    
    return parser.parse_args()

def load_jsonl(file_path):
    data = loader.get_data(file_path)
    data_x = []
    data_y = []

    for i in range(len(data)):
        data_x.append(data[i]['claim'])
        if data[i]['claim_label'] == 'SUPPORTS':
            data_y.append(0)
        elif data[i]['claim_label'] == 'REFUTES':
            data_y.append(1)
        elif data[i]['claim_label'] == 'DISPUTED':
            data_y.append(2)
        elif data[i]['claim_label'] == 'NOT_ENOUGH_INFO':
            data_y.append(3)
        else:
            print("error with dataset!")
            return None

    return data_x, data_y

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    tracker = EmissionsTracker(project_name="BERT")
    tracker.start()
    args = parse_args()

    train_x, train_y = load_jsonl('data/train_data.jsonl')
    dev_x, dev_y = load_jsonl('data/dev_data.jsonl')

    train_dataset = Dataset.from_list([{'sentence': x, 'label': int(y)} for x, y in zip(train_x, train_y)])
    dev_dataset = Dataset.from_list([{'sentence': x, 'label': int(y)} for x, y in zip(dev_x, dev_y)])

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True)
    dev_dataset = dev_dataset.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=4)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        #evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    tracker.stop()

if __name__ == "__main__":
    main()

