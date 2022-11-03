import codecs
import os
import json


def state_task_score(result_path):
    """给eval的输出目录，统计各个任务的结果"""
    result_file_list = os.listdir(result_path)
    result_file_list = [file_name for file_name in result_file_list if file_name.endswith('json')]

    stat_dict = {}  # key:task_name, value: {template_name, score}
    for file_name in result_file_list:
        all_result = json.load(open(os.path.join(result_path, file_name), 'r'))

        for result_dict in all_result:
            if result_dict['dataset_config_name']:
                task_name = f'{result_dict["dataset_name"]}/{result_dict["dataset_config_name"]}'
            else:
                task_name = result_dict['dataset_name']

            if task_name not in stat_dict:
                stat_dict[task_name] = {}

            template_id = result_dict['template_id'] if 'template_id' in result_dict else result_dict['template_name']
            stat_dict[task_name][template_id] = result_dict['evaluation']['accuracy']

    # 输出每个任务的mean，med， 结果写入文件
    output_file = open(os.path.join(result_path, 'result_summary.txt'), 'w')
    output_file.write(f'task_name\tmean\tmed\n')
    for task_name, result in stat_dict.items():
        # print(result)
        scores = list(result.values())
        mean = sum(scores) / len(scores)
        scores.sort()  # 排序
        if len(scores) % 2 == 0:
            med = scores[len(scores)//2 - 1] + scores[len(scores)//2]
            med = med / 2
        else:
            med = scores[len(scores)//2]
        mean = mean * 100
        med = med * 100
        print(f'{task_name} score:{scores}, mean: {mean}, med: {med}')
        output_file.write(f'{task_name}\t{mean}\t{med}\n')


if __name__ == '__main__':
    result_path = '/mfs/shaonan/moonshot/t-zero/evaluation/T0_3B_result'
    state_task_score(result_path)
