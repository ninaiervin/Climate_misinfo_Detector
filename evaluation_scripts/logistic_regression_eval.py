from joblib import load
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import exploring_data_layout as loader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

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
    x_test, y_test = load_jsonl('data/test_data.jsonl')

    # This loads the saved logistic regression model, which has a dev accuracy of 0.6818
    model = load('../saved_models/log_reg_params_0.6818.joblib')

    # This initializes the sentence transformer which is used to embed the climate claims
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    encoded_x_test = sentence_embedding_model.encode(x_test)
    
    # The loaded model is used to make predictions on the test data
    test_y_pred = model.predict(encoded_x_test)

    # This computes the accuracy, precision, recall, and f1 metrics based on the test predictions
    test_acc = accuracy_score(y_test, test_y_pred)
    test_pr = precision_score(y_test, test_y_pred, average='macro')
    test_recall = recall_score(y_test, test_y_pred, average='macro')
    test_f1 = f1_score(y_test, test_y_pred, average='macro')

    # This prints the computed metrics and creates and displays a confusion matrix for the test set.  
    print(f"Accuracy: {test_acc}")
    print(f"Precision: {test_pr}")
    print(f"Recall: {test_recall}")
    print(f"F1-score: {test_f1}")
    matrix = confusion_matrix(y_test, test_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Supported", "Not Supported"])
    disp.plot()
    plt.show()

if __name__ == '__main__':
    main()