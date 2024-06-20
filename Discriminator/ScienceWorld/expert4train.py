import os
import json
import re
# 指定要搜索的文件夹路径


def extracted_content(item):
    if "input" in item:
        input_text = item["input"]
        
        index_dot = input_text.find('. </s>')
        # index_comma = input_text.find(',')
        
        # if index_dot == -1:
        #     index = index_comma
        # elif index_comma == -1:
        #     index = index_dot
        # else:
        #     index = min(index_dot, index_comma)
        
        if index_dot != -1:
            extracted_content = input_text[:index_dot]
            
        extracted_content += ". "
        pattern = r'<extra_id_(\d+)>(.*?)\('
        matches = re.findall(pattern, input_text)
        for match in matches:
            extra_id = int(match[0])
            content = match[1].strip()  # 去除内容前后的空格
            extracted_content = extracted_content + str(extra_id) + ". " + content + ". "
    return extracted_content


def rawdataexact():
    folder_path = '/home/qha2sgh/SwiftSage/data_utils/data_v4/data_dir/'

# 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        new_data = []
        for file in files:
            if file.endswith('train.json') or file.endswith('val.json'):
                with open(os.path.join(folder_path, file), 'r') as f:
                    original_data = [json.loads(line.strip()) for line in f]

                # 修改字段信息并存入新 JSON 文件
                for item in original_data:
                    new_item = {
                        'input': item['input'],
                        'action': item['target'],
                        'score': 1
                    }
                    new_data.append(new_item)
                print("file has complete: " + file)
                # 将新数据写入新 JSON 文件，并添加换行符
        with open('outputexpert4train_423.json', 'a') as f:
            for item in new_data:
                json.dump(item, f)
                f.write('\n')

        print("New JSON file has been created successfully!")
                            
            # 将匹配结果转换为 JSON 格式
            # 将结果写入 JSON 文件
            
def compression_version_train():
    folder_path = '/home/qha2sgh/SwiftSage/data_utils/data_v4/data_dir/'
    for root, dirs, files in os.walk(folder_path):
        new_data = []
        for file in files:
            if file.endswith('train.json') or file.endswith('val.json'):
                with open(os.path.join(folder_path, file), 'r') as f:
                    original_data = [json.loads(line.strip()) for line in f]

                # 修改字段信息并存入新 JSON 文件
                for item in original_data:
                    new_item = {
                        'input': extracted_content(item),
                        'action': item['target'],
                        'score': 1
                    }
                    new_data.append(new_item)
                print("file has complete: " + file)
                # 将新数据写入新 JSON 文件，并添加换行符
        with open('outputexpert4train_423.json.json', 'a') as f:
            for item in new_data:
                json.dump(item, f)
                f.write('\n')

        print("New JSON file has been created successfully!")

def compression_version_val():
    folder_path = '/home/qha2sgh/SwiftSage/data_utils/data_v4/data_dir/'
    for root, dirs, files in os.walk(folder_path):
        new_data = []
        for file in files:
            if file.endswith('val.json'):
                with open(os.path.join(folder_path, file), 'r') as f:
                    original_data = [json.loads(line.strip()) for line in f]

                # 修改字段信息并存入新 JSON 文件
                for item in original_data:
                    new_item = {
                        'input': extracted_content(item),
                        'action': item['target'],
                        'score': 1
                    }
                    new_data.append(new_item)
                print("file has complete: " + file)
                # 将新数据写入新 JSON 文件，并添加换行符
        with open('outputexpert4val_423.json.json', 'a') as f:
            for item in new_data:
                json.dump(item, f)
                f.write('\n')

        print("New JSON file has been created successfully!")

if __name__ == "__main__":
    compression_version_val()