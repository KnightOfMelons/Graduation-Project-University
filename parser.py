import requests
import json

def get_data_by_id(id):
    url = f"http://127.0.0.1:5000/api/data"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Предполагаем, что data - это список словарей
        filtered_data = [item for item in data if item.get('ID') == int(id)]
        return filtered_data
    else:
        return None

def save_data_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    id = input("Введите ID: ")
    data = get_data_by_id(id)
    if data:
        file_path = input("Введите путь для сохранения файла: ")
        if not file_path.endswith('.json'):
            file_path += '.json'
        save_data_to_file(data, file_path)
        print(f"Данные успешно сохранены в файл: {file_path}")
    else:
        print("Не удалось получить данные по указанному ID.")

if __name__ == "__main__":
    main()