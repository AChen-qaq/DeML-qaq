from datasets import load_dataset
import csv
import os
import codecs
import json


def download_and_save_single_dataset(dataset_name, subset_name, output_dir):
    uniq_name = dataset_name if subset_name is None else f'{dataset_name}_{subset_name}'
    dataset_save_dir = os.path.join(output_dir, uniq_name)

    if os.path.exists(os.path.join(dataset_save_dir, 'train.json')):
        print(f'跳过已有的数据集：{uniq_name}')
        return
    os.makedirs(os.path.join(output_dir, uniq_name), exist_ok=True)

    print(f'Processing dataset {dataset_name}, subset: {subset_name}')

    if subset_name is None or subset_name in ['r1', 'r2', 'r3']:
        raw_dataset = load_dataset(dataset_name)
    else:
        raw_dataset = load_dataset(dataset_name, subset_name)

    # 训练集
    if subset_name in ['r1', 'r2', 'r3']:
        train_dataset = raw_dataset[f'train_{subset_name}']
    else:
        train_dataset = raw_dataset['train']
    with codecs.open(os.path.join(dataset_save_dir, 'train.json'), 'w', encoding='utf-8') as f:
        for example in train_dataset:
            f.write(json.dumps(example) + '\n')

    # 开发集
    if subset_name in ['r1', 'r2', 'r3'] or 'validation' in raw_dataset:
        if subset_name in ['r1', 'r2', 'r3']:
            dev_dataset = raw_dataset[f'dev_{subset_name}']
        else:
            dev_dataset = raw_dataset['validation']
        with codecs.open(os.path.join(dataset_save_dir, 'validation.json'), 'w', encoding='utf-8') as f:
            for example in dev_dataset:
                f.write(json.dumps(example) + '\n')
    else:
        print(f'dataset {dataset_name} do not have validation!')

    if subset_name in ['r1', 'r2', 'r3'] or 'test' in raw_dataset:
        if subset_name in ['r1', 'r2', 'r3']:
            test_dataset = raw_dataset[f'test_{subset_name}']
        else:
            test_dataset = raw_dataset['test']
        with codecs.open(os.path.join(dataset_save_dir, 'test.json'), 'w', encoding='utf-8') as f:
            for example in test_dataset:
                f.write(json.dumps(example) + '\n')
    else:
        print(f'dataset {dataset_name} do not have test!')


def download_and_save():
    """把T0训练/测试所需要的所有数据保存下来（不然每次load 线上的不现实）"""

    # dataset_csv = csv.DictReader(open('datasets.csv'))

    # dataset_list = []

    # for row in dataset_csv:
    #     if row['subset'] == '':
    #         row["subset"] = None
    #     if row["do_eval"] == 'BIAS_FAIRNESS' or row['do_train'] == 'BIAS_FAIRNESS':
    #         continue
    #     # anli 需要特别处理
    #     if row["HF_name"] == 'anli':
    #         dataset_list.append(('anli', 'r1'))
    #         dataset_list.append(('anli', 'r2'))
    #         dataset_list.append(('anli', 'r3'))
    #     else:
    #         dataset_list.append((row["HF_name"], row["subset"]))

    # 使用我们自己的task_list
    list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/all.list'
    dataset_list = []
    with codecs.open(list_filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                dataset_list.append(task_tuple)
            else:
                dataset_list.append((task_tuple[0], None))

    output_dir = '/home/yanan/shaonan/data/T0_dataset'
    for dataset_name, subset_name in dataset_list:
        download_and_save_single_dataset(dataset_name, subset_name, output_dir)

    # 测试是否保存了
    # for dataset_name, subset_name in dataset_list:
    #     uniq_name = dataset_name if subset_name is None else f'{dataset_name}_{subset_name}'
    #     uniq_dir = os.path.join(output_dir, uniq_name)
    #     temp_raw = load_dataset('json', data_files={'train': os.path.join(uniq_dir, 'train.json')})


if __name__ == '__main__':
    download_and_save()