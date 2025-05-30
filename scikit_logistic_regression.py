from codecarbon import EmissionsTracker
from joblib import dump
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import exploring_data_layout as loader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Saves the best accuracy to a global variable
best_acc = 0

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
    tracker = EmissionsTracker(project_name="LR")
    tracker.start()

    # This initializes the sentence transformer which is used to embed the slimate claims
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # This loads the train and dev data, and then encodes the climate claim string using the sentence transformer
    x_train, y_train = load_jsonl('data/train_data.jsonl')
    x_dev, y_dev = load_jsonl('data/dev_data.jsonl')

    encoded_x_train = sentence_embedding_model.encode(x_train)
    encoded_x_dev = sentence_embedding_model.encode(x_dev)

    # This initializes the logistic regression model, trains it, and gets the predictions on the dev data
    model = LogisticRegression(class_weight='balanced', multi_class='multinomial')
    model.fit(encoded_x_train, y_train)
    y_pred = model.predict(encoded_x_dev)

    # This computes the accuracy, precision, recall, and f1 metrics based on the dev predictions
    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred, average='macro')
    recall = recall_score(y_dev, y_pred, average='macro')
    f1 = f1_score(y_dev, y_pred, average='macro')

    # This checks in the accuracy of the current model is the best one so far, and if it is, 
    # it saves the model parameters and updates the best accuracy variable
    global best_acc
    if accuracy > best_acc:
        best_acc = accuracy
        model_params = f'log_reg_params_{accuracy:.4f}.joblib'
        dump(model, model_params)

    # This stops the emissions tracker and saves the grams of CO2 produced
    emissions_kg = tracker.stop()
    if emissions_kg is None:
        emissions_kg = 0
    emissions_g = emissions_kg * 1000

    # This prints the computed metrics, as well as the emissions, and creates and displays a confusion matrix.  
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Emmisions (g): {emissions_g}")
    matrix = confusion_matrix(y_dev, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()