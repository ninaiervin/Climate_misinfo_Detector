from codecarbon import EmissionsTracker
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import exploring_data_layout as loader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

tracker = EmissionsTracker(project_name="LR")
tracker.start()

sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#loads data from jsonl files and extracts climate sentance and given label
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

x_train, y_train = load_jsonl('data/train_data.jsonl')
x_dev, y_dev = load_jsonl('data/dev_data.jsonl')

encoded_x_train = sentence_embedding_model.encode(x_train)
encoded_x_dev = sentence_embedding_model.encode(x_dev)

le = LabelEncoder()
'''
0: DISPUTED
1: NOT_ENOUGH_INFO
2: REFUTES
3: SUPPORTS
'''
encoded_y_train = le.fit_transform(y_train)
encoded_y_dev = le.fit_transform(y_dev)


model = LogisticRegression(class_weight='balanced', multi_class='multinomial')
model.fit(encoded_x_train, encoded_y_train)
y_pred = model.predict(encoded_x_dev)

accuracy = accuracy_score(encoded_y_dev, y_pred)
precision = precision_score(encoded_y_dev, y_pred, average='macro')
recall = recall_score(encoded_y_dev, y_pred, average='macro')
f1 = f1_score(encoded_y_dev, y_pred, average='macro')

matrix = confusion_matrix(encoded_y_dev, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
disp.plot()
tracker.stop()
plt.show()
