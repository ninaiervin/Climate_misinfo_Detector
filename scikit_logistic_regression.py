from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import exploring_data_layout as loader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

train_data = loader.get_data('data/train_data.jsonl')
dev_data = loader.get_data('data/dev_data.jsonl')

x_train = []
y_train = []

for i in range(len(train_data)):
    x_train.append(train_data[i]['claim'])
    y_train.append(train_data[i]['claim_label'])

x_dev = []
y_dev = []

for i in range(len(dev_data)):
    x_dev.append(dev_data[i]['claim'])
    y_dev.append(dev_data[i]['claim_label'])

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
plt.show()