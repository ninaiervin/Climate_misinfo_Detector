import argparse
from sklearn.calibration import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import exploring_data_layout as loader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from joblib import dump, load
from codecarbon import EmissionsTracker

# This parses command line arguments that specify various hyperparameters.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-hi", type=int, default=100, help="This is the hidden layer size")
    parser.add_argument("-act", type=str, default='relu', help="This is the hidden activation")
    parser.add_argument("-a", type=float, default=0.0001, help="This is alpha for strength of reg")
    parser.add_argument("-b", type=int, default=100, help="This is the batch size")
    parser.add_argument("-lrt", type=str, default='adaptive', help="This is the learning rate type")
    parser.add_argument("-lr", type=float, default=0.001, help="This is the learning rate")
    parser.add_argument("-m", type=int, default=10000, help="This is the max iterations")
    parser.add_argument("-w", type=bool, default=False, help="if true we will do warm up")
    parser.add_argument("-e", type=bool, default=False, help="If ture we will do early stopping")
    
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
        else:
            print("error with dataset!")
            return None

    return data_x, data_y

def main():
    # This initializes and starts the emissions tracker. This tracker will track things such as energy consuption and CO^2 emissions. 
    # This information will be used in the analysis of the models along with other evaluation metrics.
    tracker = EmissionsTracker(project_name="NN")
    tracker.start()

    args = parse_args()

    # This initializes the sentence transformer which is used to embed the climate claims
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # This loads the train and dev data, and then encodes the climate claim string using the sentence transformer
    x_train, y_train = load_jsonl('../data/train_data.jsonl')
    x_dev, y_dev = load_jsonl('../data/dev_data.jsonl')

    encoded_x_train = sentence_embedding_model.encode(x_train)
    encoded_x_dev = sentence_embedding_model.encode(x_dev)

    # This initializes the neural network model, trains it, and gets the predictions on the dev and train data
    model = MLPClassifier(hidden_layer_sizes=(args.hi,3), activation=args.act, alpha=args.a,batch_size=args.b, learning_rate=args.lrt, learning_rate_init=args.lr, max_iter=args.m, warm_start=args.w, early_stopping=args.e)
    model.fit(encoded_x_train, y_train)
    dev_y_pred = model.predict(encoded_x_dev)
    train_y_pred = model.predict(encoded_x_train)

    # This computes the accuracy, precision, recall, and f1 metrics based on the dev predictions and train predictions
    dev_acc = accuracy_score(y_dev, dev_y_pred)
    dev_pr = precision_score(y_dev, dev_y_pred, average='macro')
    dev_recall = recall_score(y_dev, dev_y_pred, average='macro')
    dev_f1 = f1_score(y_dev, dev_y_pred, average='macro')

    train_acc = accuracy_score(y_train, train_y_pred)
    train_pr = precision_score(y_train, train_y_pred, average='macro')
    train_recall = recall_score(y_train, train_y_pred, average='macro')
    train_f1 = f1_score(y_train, train_y_pred, average='macro')

    # This checks in the accuracy of the current model is the best one so far, and if it is, 
    # it saves the model parameters and updates the best accuracy variable
    global best_acc
    if dev_acc > best_acc:
        best_acc = dev_acc
        model_params = f'log_reg_params_{dev_acc:.4f}.joblib'
        dump(model, model_params)

    # This stops the emissions tracker and saves the grams of CO2 produced
    emissions_kg = tracker.stop()
    if emissions_kg is None:
        emissions_kg = 0
    emissions_g = emissions_kg * 1000
    
    # This prints the computed metrics, as well as the emissions, and creates and displays a confusion matrix.  
    print(f"Accuracy: train={train_acc}, dev={dev_acc}")
    print(f"Precision: train={train_pr}, dev={dev_pr}")
    print(f"Recall: train={train_recall}, dev={dev_recall}")
    print(f"F1-score: train={dev_f1}, dev={dev_f1}")
    matrix = confusion_matrix(y_dev, dev_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()