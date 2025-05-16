import argparse
from sklearn.calibration import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import exploring_data_layout as loader
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("-hi", type=int, default=100, help="This is the hidden layer side")
parser.add_argument("-act", type=str, default='relu', help="This is the hidden activation")
parser.add_argument("-a", type=float, default=0.0001, help="This is alpha for strength of reg")
parser.add_argument("-b", type=int, default=100, help="This is the batch size")
parser.add_argument("-lrt", type=str, default='adaptive', help="This is the learning rate type")
parser.add_argument("-lr", type=float, default=0.001, help="This is the learning rate")
parser.add_argument("-m", type=int, default=10000, help="This is the max iterations")
parser.add_argument("-w", type=bool, default=False, help="if true we will do warm up")
parser.add_argument("-e", type=bool, default=False, help="If ture we will do early stopping")
args = parser.parse_args()

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


model = MLPClassifier(hidden_layer_sizes=(args.hi,3), activation=args.act, alpha=args.a,batch_size=args.b, learning_rate=args.lrt, learning_rate_init=args.lr, max_iter=args.m, warm_start=args.w, early_stopping=args.e)
model.fit(encoded_x_train, encoded_y_train)
dev_y_pred = model.predict(encoded_x_dev)
train_y_pred = model.predict(encoded_x_train)

dev_acc = accuracy_score(encoded_y_dev, dev_y_pred)
dev_pr = precision_score(encoded_y_dev, dev_y_pred, average='macro')
dev_recall = recall_score(encoded_y_dev, dev_y_pred, average='macro')
dev_f1 = f1_score(encoded_y_dev, dev_y_pred, average='macro')

train_acc = accuracy_score(encoded_y_train, train_y_pred)
train_pr = precision_score(encoded_y_train, train_y_pred, average='macro')
train_recall = recall_score(encoded_y_train, train_y_pred, average='macro')
train_f1 = f1_score(encoded_y_train, train_y_pred, average='macro')

matrix = confusion_matrix(encoded_y_dev, dev_y_pred)

print(f"Accuracy: train={train_acc}, dev={dev_acc}")
print(f"Precision: train={train_pr}, dev={dev_pr}")
print(f"Recall: train={train_recall}, dev={dev_recall}")
print(f"F1-score: train={dev_f1}, dev={dev_f1}")
print(matrix)
