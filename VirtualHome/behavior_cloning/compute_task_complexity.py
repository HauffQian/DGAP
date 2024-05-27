import os


def compute_complexity():
    faithful_case = [2, 4, 13, 19, 26, 38, 39, 41, 46, 53, 59, 65, 73, 74, 81, 83, 93, 98]
    bad_case = faithful_case + [43, 61, 62, 85, 86, 88, 1, 3, 10, 14, 15, 16, 18, 20, 21, 23, 25, 28, 30, 31, 32, 37,
                                40, 44, 45, 49, 50, 52, 56, 57, 64, 68, 72, 77, 79, 80, 95]

    current_dir = '/home/tannl/LID/behavior_cloning'
    file_name = 'NovelTasks_complexity.txt'
    file_path = os.path.join(current_dir, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(1, 101):
        if i in faithful_case:
            lines[i - 1] = lines[i - 1].rstrip('\n') + '  faithfulness: False\n'
        else:
            lines[i - 1] = lines[i - 1].rstrip('\n') + '  faithfulness: True\n'

        if i in bad_case:
            lines[i - 1] = lines[i - 1].rstrip('\n') + '  success: False\n'
        else:
            lines[i - 1] = lines[i - 1].rstrip('\n') + '  success: True\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)


def compute_relevance():
    current_dir = '/home/tannl/LID/behavior_cloning'
    file_name = 'NovelTasks_complexity.txt'
    file_path = os.path.join(current_dir, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    success_group = [0, 0, 0, 0, 0, 0, 0, 0]
    group_total_num = [0, 0, 0, 0, 0, 0, 0, 0]
    faithful_group = [0, 0, 0, 0, 0, 0, 0, 0]
    for line in lines:
        parts = line.split()
        info = {}
        key, value = parts[0], parts[1]
        info[key.strip()] = value.strip()
        key, value = parts[2], parts[3]
        info[key.strip()] = value.strip()
        key, value = parts[4], parts[5]
        info[key.strip()] = value.strip()
        key, value = parts[6], parts[7]
        info[key.strip()] = value.strip()
        task_id = int(info['task_id:'])
        min_steps = int(info['min_steps:'])
        faithfulness = info['faithfulness:']
        # print(faithfulness)
        success = info['success:']
        if min_steps <= 10:
            group_total_num[0] += 1
            if faithfulness == 'True':
                faithful_group[0] += 1
            if success == 'True':
                success_group[0] += 1
        elif min_steps <= 15:
            group_total_num[1] += 1
            if faithfulness == 'True':
                faithful_group[1] += 1
            if success == 'True':
                success_group[1] += 1
        elif min_steps <= 20:
            group_total_num[2] += 1
            if faithfulness == 'True':
                faithful_group[2] += 1
            if success == 'True':
                success_group[2] += 1
        elif min_steps <= 25:
            group_total_num[3] += 1
            if faithfulness == 'True':
                faithful_group[3] += 1
            if success == 'True':
                success_group[3] += 1
        elif min_steps <= 30:
            group_total_num[4] += 1
            if faithfulness == 'True':
                faithful_group[4] += 1
            if success == 'True':
                success_group[4] += 1
        elif min_steps <= 35:
            group_total_num[5] += 1
            if faithfulness == 'True':
                faithful_group[5] += 1
            if success == 'True':
                success_group[5] += 1
        elif min_steps <= 40:
            group_total_num[6] += 1
            if faithfulness == 'True':
                faithful_group[6] += 1
            if success == 'True':
                success_group[6] += 1
        else:
            group_total_num[7] += 1
            if faithfulness == 'True':
                faithful_group[7] += 1
            if success == 'True':
                success_group[7] += 1

    print('success_group: ', success_group)
    print('faithful_group: ', faithful_group)
    print('total_group', group_total_num)


def compute_result():
    total_group = [2, 123, 31, 33, 37, 38, 35, 1]
    success_group1 = [1, 78, 3, 0, 0, 0, 0, 0]
    faithful_group1 =  [1, 87, 5, 0, 0, 0, 0, 0]
    success_group2 =  [0, 7, 6, 12, 12, 9, 6, 1]
    faithful_group2 =  [0, 10, 11, 14, 19, 12, 15, 1]
    success_group3 = [1, 10, 6, 11, 6, 8, 3, 0]
    faithful_group3 = [1, 13, 10, 15, 15, 15, 13, 0]
    faith_res = []
    faith_ = []
    success_res = []
    success_ = []
    for i in range(0, 7):
        success = success_group1[i] + success_group2[i] + success_group3[i]
        success_.append(round(success / total_group[i], 4))
        success_res.append(success)
        faithful = faithful_group1[i] + faithful_group2[i] + faithful_group3[i]
        faith_res.append(faithful)
        faith_.append(round(faithful / total_group[i], 4))

    print("success_group: " + str(success_res))
    print("faithful_group: " + str(faith_res))
    print(faith_)
    print(success_)

# compute_complexity()
# compute_relevance()
compute_result()
