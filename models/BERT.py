import argparse
import json
import wandb
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import exploring_data_layout as loader
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

# This parses command line arguments that specify the model architechture, various hyperparameters,
# and other important information like the output directory to save the best performing model.
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")
    
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./bert_output", help="Output directory")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of update to accumulate before preforming MBSGD")
    
    return parser.parse_args()

# This function loads data from jsonl files and extracts the string of the claim
# and, depending on the given label, class 0 or 1. Since we combined three classes in order
# to do binary classification, if the label is SUPPORTS it is assigned class 0, and if it
# is anything else it is assigned class 1.
def load_jsonl(file_path):
    data = loader.get_data(file_path)
    data_x = []
    data_y = []

    for i in range(len(data)):
        data_x.append(data[i]['claim'])
        if data[i]['claim_label'] == 'SUPPORTS':
            data_y.append(0)
        elif data[i]['claim_label'] == 'REFUTES' or data[i]['claim_label'] == 'DISPUTED' or data[i]['claim_label'] == 'NOT_ENOUGH_INFO':
            data_y.append(1)
        #elif data[i]['claim_label'] == 'NOT_ENOUGH_INFO':
        #    data_y.append(2)
        else:
            print("error with dataset!")
            return None

    return data_x, data_y

# This is a function that takes in our given sentences and tokenizes them using a pre-trained tokenizer
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

# This function computes accuracy and f1 score for the given predictions and labels
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    # This initializes and starts the emissions tracker. This tracker will track things such as energy consuption and CO^2 emissions. 
    # This information will be used in the analysis of the models along with other evaluation metrics.
    tracker = EmissionsTracker(project_name="BERT")
    tracker.start()
    args = parse_args()

    # Loading the train and dev sets.
    # For evaluation of our models capablity to generlize to new points we will use the dev set. This will be used for hyperprameter sweeps.
    # Training data will be used to train the model and calculate loss to use MBSGD for weight update.
    train_x, train_y = load_jsonl('../data/train_data.jsonl')
    dev_x, dev_y = load_jsonl('../data/dev_data.jsonl')

    train_dataset = Dataset.from_list([{'sentence': x, 'label': int(y)} for x, y in zip(train_x, train_y)])
    dev_dataset = Dataset.from_list([{'sentence': x, 'label': int(y)} for x, y in zip(dev_x, dev_y)])
    
    # We are defining a pretrained tokenizer to get the static embeddings.
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Here we are using our function to tokenize all our training and evaluation datapoints.
    # No training is being done here since these tokens have already been learned.
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True)
    dev_dataset = dev_dataset.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Here we are loading a pre-trained BERT model. This means will be loading in weights that have alreay been 
    # trained on a larger dataset. We are loading in the encoder portion, and the final classification is randonly initialized.
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Here we are defining all the training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir, #defines where the model checkpoints and preditions will be saved
        learning_rate=args.lr, #init learing rate for MBSGD for AdamW
        per_device_train_batch_size=args.batch_size,  #defines the batch size per device when training
        per_device_eval_batch_size=args.batch_size, #defines the batch size per device on the dev set
        num_train_epochs=args.epochs, #number of epochs or passes through the training dataset
        weight_decay=0.01, #applies weight decay to everything but bias and layerNorm weights.
        logging_dir=f"{args.output_dir}/logs", #directory for where to log data
        logging_steps=10, #how often to log data during training
        eval_strategy='epoch', #evaluation on dev_set will be done every epoch and logged/reported
        save_strategy="best",  # how often to save weights (set to best means only save the weights if the dev acc is better).
        load_best_model_at_end=True, #makes sure last evaluation is done on the best model and makes sure the best model is saved
        metric_for_best_model='accuracy', #best model is choosen based on dev accuracy
        gradient_accumulation_steps=args.grad_accum, # number of updates steps to accumulate before doing MBSGD
    )

    # Gives all the defined pieces to the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # This is the step where the model gets fine-tuned.
    # We are using a pretrained model but this implemtation does not freeze any weights for fine-tuning so we are updating:
        # embedings
        # positional encodings
        # all attention weights (W_q, W_k, W_v, and W_o)
        # encoder block FFNN
        # layer Norm beta and gamma
        # final linear output layer (learned from random weight init)
    trainer.train()
    trainer.save_model(args.output_dir)
    wandb.finish() # fixes brocken pipe error

    tracker.stop() # stops tracking emissions

if __name__ == "__main__":
    main()

