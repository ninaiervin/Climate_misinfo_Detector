import json
import random
        
def get_data(path):
    with open(path, 'r') as file:
        data =  [json.loads(line) for line in file]
    return data

def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            json_string = json.dumps(item) + '\n'
            file.write(json_string)

def split_and_save(data, train_name, dev_name, test_name):
    random.shuffle(data)
    split1 = int(len(data)*0.8)
    train_list = data[:split1]
    temp_list = data[split1:]
    split2 = int(len(temp_list)//2)
    test_list = temp_list[:split2]
    dev_list = temp_list[split2:]

    save_jsonl(train_list, train_name)
    save_jsonl(dev_list, dev_name)
    save_jsonl(test_list, test_name)

def main():
    data = get_data('data/climate-fever-dataset-r1.jsonl')
    #split_and_save(data, 'data/train_data1.jsonl', 'data/dev_data1.jsonl', 'data/test_data1.jsonl')   do not uncomment--with change our train, dev, and test data!!
    train_data = get_data('data/train_data.jsonl')
    dev_data = get_data('data/dev_data.jsonl')

    print(len(train_data), len(dev_data))

    print(data[-1].keys())

if __name__ == '__main__':
    main()
