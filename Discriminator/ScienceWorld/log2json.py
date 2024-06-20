import os
import json
import re
# 指定要搜索的文件夹路径

folder_prefix = "/home/qha2sgh/SwiftSage/fast_slow_logs/random_data"
cnt = 0

def extracted_content(item):

    input_text = item.strip()
    
    index_dot = input_text.find('.')
    index_comma = input_text.find(',')
    
    if index_dot == -1:
        index = index_comma
    elif index_comma == -1:
        index = index_dot
    else:
        index = min(index_dot, index_comma)
    
    if index != -1:
        extracted_content = input_text[:index]
        
    extracted_content += ". "
    pattern = r'<extra_id_(\d+)>(.*?)\('
    matches = re.findall(pattern, input_text)
    for match in matches:
        extra_id = int(match[0])
        content = match[1].strip()  # 去除内容前后的空格
        extracted_content = extracted_content + str(extra_id) + ". " + content + ". "
    return extracted_content



# 遍历文件夹
for root, dirs, files in os.walk(folder_prefix):
    for dir in dirs:
        # 检查文件夹名是否包含"gen_mis_task"
        if "random" in dir:
            print(f"Searching records in folder: {dir}")
            cnt += 1
            # 遍历文件夹中的文件
            for filename in os.listdir(os.path.join(folder_prefix, dir)):
                if filename.endswith(".log"):
                    file_path = os.path.join(folder_prefix, dir, filename)
                    print(f"Processing file: {file_path}")
                    # 读取 log 文件
                    with open(file_path, 'r') as f:
                        log_data = f.read()

                    # 定义正则表达式模式
                    input_pattern = r'InputStr:\s(.*?)(?=\[)'
                    action_pattern = r'Action:\s(.*?)(?=\[)'
                    score_pattern = r'Average score:\s(.*?)(?=\[)'

                    # 匹配所有记录中的 input 和 action
                    input_matches = re.findall(input_pattern, log_data, re.DOTALL)
                    action_matches = re.findall(action_pattern, log_data, re.DOTALL)
                    score_matches = re.findall(score_pattern, log_data, re.DOTALL)
                    score = float(score_matches[-1])/100.0
                    
                    
                    # 将匹配结果转换为 JSON 格式
                    # 将结果写入 JSON 文件
                    
                    with open('outputsub_all.json', 'a') as f:
                        for input_text, action_text in zip(input_matches, action_matches):
                            data = {"input": extracted_content(input_text).strip(), "action": action_text.strip(), "score": score}
                            json.dump(data, f)
                            f.write('\n')
        else:
            print("no such dictionary: " + dir)
print(cnt)