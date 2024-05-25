import os
import re
import csv
# 定义存储结果的集合
result_set = set()

def find_best_episode(scores_by_episode):
    best_episode = None
    best_final_score = float('-inf')
    shortest_path_length = float('inf')

    for episode, scores in scores_by_episode.items():
        final_score = scores[-1]
        path_length = len(scores)

        if (final_score > best_final_score) or (final_score == best_final_score and path_length < shortest_path_length):
            best_episode = episode
            best_final_score = final_score
            shortest_path_length = path_length

    return best_episode, scores_by_episode[best_episode] if best_episode is not None else []

def extract_task_number(filename):
    match = re.search(r'task(\d+)', filename)
    return match.group(1) if match else None

# 遍历文件夹中的文件
folder_path = "/home/qhf/ubuntu/git_clone_collection/SwiftSage/logs/0512/llama3"
with open('strallama3', 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Task Number', 'Episode', 'Score'])
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 "task" 开头且以 "-score.txt" 结尾
        if filename.startswith("task") and filename.endswith(".json"):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)
            task_number = extract_task_number(filename)
            scores_by_episode = {}
            current_episode = None
            score_pattern = re.compile(r'"score":\s*"(\d+\.\d+)"')
            episode_pattern = re.compile(r'"episodeIdx":\s*(\d+)')
            # 读取文件内容
            with open(file_path, 'r') as file:
                for line in file:
                    episode_match = episode_pattern.search(line)
                    score_match = score_pattern.search(line)

                    if episode_match:
                        current_episode = int(episode_match.group(1))
                        scores_by_episode[current_episode] = []
                    elif score_match and current_episode is not None:
                        score = float(score_match.group(1))
                        scores_by_episode[current_episode].append(score)
            best_episode, best_scores = find_best_episode(scores_by_episode)
            row = [task_number, best_episode] + best_scores
            csvwriter.writerow(row)
            print(f"Task {task_number} is done")




# 输出结果集合
# for item in result_set:
#     print(item)
