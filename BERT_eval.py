from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import argparse
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from exploring_data_layout import get_data  # If you're using your custom loader
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="eval trained BERT for text classification")

    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--output_dir", type=str, default="./", help="the path to the saved model")
    parser.add_argument("--temp_logs_path", type=str, default="./", help="the path where temp logs can be saved")
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Supported", "Not Supported"])
    disp.plot()
    plt.show()
    return {"accuracy": acc, "precision": precision, "recal": recall, "f1": f1}

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

def load_jsonl(file_path):
    data = get_data(file_path)
    data_x = []
    data_y = []

    for item in data:
        data_x.append(item['claim'])
        if item['claim_label'] == 'SUPPORTS':
            data_y.append(0)
        elif item['claim_label'] in ('REFUTES', 'DISPUTED') or item['claim_label'] == 'NOT_ENOUGH_INFO':
            data_y.append(1)
        else:
            raise ValueError("Unknown label in dataset")

    return data_x, data_y


def main():
    args = parse_args()

    print(args)

    eval_x, eval_y = load_jsonl('data/test_data.jsonl')

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    eval_dataset = Dataset.from_list([{'sentence': x, 'label': int(y)} for x, y in zip(eval_x, eval_y)])
    eval_dataset = eval_dataset.map(lambda x: tokenizer(x['sentence'], padding='max_length', truncation=True, max_length=128), batched=True)
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


    #output_dir = "./bert_output/checkpoint-154"
    model = BertForSequenceClassification.from_pretrained(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)


    training_args = TrainingArguments(
        output_dir=args.temp_logs_path,
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(eval_dataset)
    print("Evaluation Results:", results)

if __name__ == '__main__':
    main()
