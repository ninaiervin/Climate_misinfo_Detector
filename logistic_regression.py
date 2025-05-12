import json


def get_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def main():
    data = 'nothing here'
    try:
        data = get_data('data/climate-fever-dataset-r1.jsonl')
    except json.JSONDecodeError:
        print('problem with json')

    print(data)

if __name__ == 'main':
    main()
