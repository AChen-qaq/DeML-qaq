import os
import csv

# import datasets.utils.tqdm_utils
import datasets
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)

from templates import DatasetTemplates
import codecs
import json
from multiprocessing import Pool
from functools import partial


def special_for_triviaqa(uniq_task_dir):
    all_raw_dataset = {}
    if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
        train_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'train.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                train_data_dict['question'].append(example['question'])
                train_data_dict['answer'].append(example['answer'])

        train_dataset = Dataset.from_dict(train_data_dict)
        all_raw_dataset['train'] = train_dataset

    if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
        dev_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'validation.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                dev_data_dict['question'].append(example['question'])
                dev_data_dict['answer'].append(example['answer'])

        dev_dataset = Dataset.from_dict(dev_data_dict)
        all_raw_dataset['validation'] = dev_dataset
    if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
        test_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'test.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                test_data_dict['question'].append(example['question'])
                test_data_dict['answer'].append(example['answer'])

        test_dataset = Dataset.from_dict(test_data_dict)
        all_raw_dataset['test'] = test_dataset

    return all_raw_dataset


def special_for_wikihop(uniq_task_dir):
    all_raw_dataset = {}
    if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
        train_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'train.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                train_data_dict['id'].append(example['id'])
                train_data_dict['question'].append(example['question'])
                train_data_dict['answer'].append(example['answer'])
                train_data_dict['candidates'].append(example['candidates'])
                train_data_dict['supports'].append(example['supports'])

        train_dataset = Dataset.from_dict(train_data_dict)
        all_raw_dataset['train'] = train_dataset

    if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
        dev_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'validation.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                dev_data_dict['id'].append(example['id'])
                dev_data_dict['question'].append(example['question'])
                dev_data_dict['answer'].append(example['answer'])
                dev_data_dict['candidates'].append(example['candidates'])
                dev_data_dict['supports'].append(example['supports'])

        dev_dataset = Dataset.from_dict(dev_data_dict)
        all_raw_dataset['validation'] = dev_dataset
    if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
        test_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'test.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                test_data_dict['id'].append(example['id'])
                test_data_dict['question'].append(example['question'])
                test_data_dict['answer'].append(example['answer'])
                test_data_dict['candidates'].append(example['candidates'])
                test_data_dict['supports'].append(example['supports'])

        test_dataset = Dataset.from_dict(test_data_dict)
        all_raw_dataset['test'] = test_dataset

    return all_raw_dataset


def filter_invalid_data(uniq_task_name, template_name, raw_dataset):
    filtered_dataset = raw_dataset
    if uniq_task_name == 'super_glue_copa':
        # print(f'start filter')
        if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
            filtered_dataset = raw_dataset.filter(lambda example: example['question'] == 'effect')
        if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
            filtered_dataset = raw_dataset.filter(lambda example: example['question'] == 'cause')

    return filtered_dataset


def get_uniq_task_name(dataset_name, subset_name=None, template_name=None):
    uniq_task_name = dataset_name
    if subset_name is not None:
        uniq_task_name += f'_{subset_name}'

    if template_name is not None:
        template_name = template_name.replace('\\', '_')
        template_name = template_name.replace('-', '_')
        template_name = template_name.replace('?', '_')
        template_name = '_'.join(template_name.split())

        uniq_task_name += f'_{template_name}'

    return uniq_task_name


def preprocess_for_baseline():
    """预处理所有数据集，预处理后的所有数据集包括字段：input_ids, attention_mask, labels"""
    # dataset_csv = csv.DictReader(open('/home/yanan/shaonan/t-zero/datasets_temp.csv'))
    list_filename = '/localdata/DeML/t-zero-share/config/pretraining/single.list'
    # list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/test.list'

    origin_data_dir = '/localdata/codebook/data/T0_dataset'  # 保存原始数据集的目录
    tokenizer = AutoTokenizer.from_pretrained('/share/huggingface_models/t5-large-lm-adapt')
    template_dir = '/localdata/DeML/t-zero-share/templates_feature'
    output_dir = '/localdata/DeML/data'

    datasets.disable_progress_bar()

    os.makedirs(output_dir, exist_ok=True)

    max_length = 384
    target_max_length = 32
    padding = False

    # 读取需要使用的数据集，这里是csv，后面也可能用list文件
    # dataset_list = []
    # for row in dataset_csv:
    #     if row['subset'] == '':
    #         row["subset"] = None
    #     if row["do_eval"] == 'BIAS_FAIRNESS' or row['do_train'] == 'BIAS_FAIRNESS':
    #         continue
    #     if row["HF_name"] == 'anli':
    #         dataset_list.append(('anli', 'r1'))
    #         dataset_list.append(('anli', 'r2'))
    #         dataset_list.append(('anli', 'r3'))
    #     else:
    #         dataset_list.append((row["HF_name"], row["subset"]))

    # 不使用dataset.csv，发现有数据不全，使用我们自己的任务列表
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

    print(dataset_list)

    # 分别处理各个数据集
    for dataset_name, subset_name in dataset_list:
        uniq_task_name = dataset_name if subset_name is None else f'{dataset_name}_{subset_name}'
        uniq_task_dir = os.path.join(origin_data_dir, uniq_task_name)
        # 读数据集
        data_files = {}
        if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
            data_files['train'] = os.path.join(uniq_task_dir, 'train.json')
        # 只处理训练集, validation特殊情况很多，很麻烦
        if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
            data_files['validation'] = os.path.join(uniq_task_dir, 'validation.json')
        if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
            if 'validation' not in data_files:
                # 仅在没有validation的时候才用test
                data_files['test'] = os.path.join(uniq_task_dir, 'test.json')

        print(f'loading dataset: {uniq_task_name}')
        if dataset_name.startswith('trivia'):
            raw_dataset = special_for_triviaqa(uniq_task_dir)
        elif dataset_name.startswith('wiki_hop'):
            raw_dataset = special_for_wikihop(uniq_task_dir)
        else:
            raw_dataset = load_dataset('json', data_files=data_files)

        # trec的列名有错误
        if dataset_name == 'trec':
            for split in raw_dataset:
                raw_dataset[split] = raw_dataset[split].rename_column('label-fine', 'label_fine')
                raw_dataset[split] = raw_dataset[split].rename_column('label-coarse', 'label_coarse')

        column_names = raw_dataset['train'].column_names

        def preprocess_train(examples):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            # answer_choices = []  # 如果需要有answer_choice把这个
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                inputs_and_targets = template.apply(ex)
                # print(inputs_and_targets)  # debug
                if len(inputs_and_targets) == 2:
                    input, target = inputs_and_targets
                else:
                    input, target = '', ''

                # ex_answer_choices = template.get_answer_choices_list(ex)
                # if ex_answer_choices:
                #     answer_choices.append(ex_answer_choices)

                # assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)

            model_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
            )

            with tokenizer.as_target_tokenizer():
                tokenized_targets = tokenizer(
                    target_texts,
                    padding=padding,
                    max_length=target_max_length,
                    truncation=True,
                    add_special_tokens=True,  # 加上eos效果好
                )
                model_inputs['labels'] = [
                    [(t if t != tokenizer.pad_token_id else -100) for t in targets]
                    for targets in tokenized_targets["input_ids"]
                ]
            # if len(answer_choices) > 0:
            #     model_inputs['answer_choices'] = answer_choices
            return model_inputs

        # 读prompt
        if dataset_name == 'anli':
            prompts = DatasetTemplates(dataset_name, template_dir=template_dir)
        else:
            prompts = DatasetTemplates(dataset_name if subset_name is None else f"{dataset_name}/{subset_name}",
                                       template_dir=template_dir)
        template_list = prompts.templates.keys()
        print(f'{dataset_name}/{subset_name}的模板列表：{template_list}')

        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name
            origin_template_name = template.name
            # 去掉空格,去掉/
            template_name = template_name.replace('\\', '_')
            template_name = template_name.replace('-', '_')
            template_name = template_name.replace('?', '_')
            template_name = '_'.join(template_name.split())

            prompted_task_name = f'{uniq_task_name}_{template_name}'

            prompted_output_dir = os.path.join(output_dir, prompted_task_name)
            os.makedirs(prompted_output_dir, exist_ok=True)

            # train/validation/test
            for split in raw_dataset:

                # 如果已经存在，那么跳过
                # if os.path.exists(os.path.join(prompted_output_dir, f'{split}.json')) \
                #         and os.path.getsize(os.path.join(prompted_output_dir, f'{split}.json')) > 0:
                #     print(f'跳过已有的数据集：{prompted_output_dir}/{split}.json', flush=True)
                #     continue

                print(f'processing prompted_task_name: {prompted_task_name}, split: {split}')
                split_dataset = raw_dataset[split]
                # print(f'before filter: {split_dataset}')
                split_dataset = filter_invalid_data(uniq_task_name, origin_template_name, split_dataset)
                # print(f'after filter: {split_dataset}')

                # temp_output = template.apply(split_dataset[0])
                # if len(temp_output) == 2:
                #     inputs, target = temp_output
                #     if target is None or len(target) == 0:
                #         print(f'忽略没有标签的split: {prompted_task_name}, split: {split}')
                #         continue

                if dataset_name == 'trec' or dataset_name.startswith('wiki_qa'):
                    # 多线程会报错
                    preprocessed_split_dataset = split_dataset.map(preprocess_train, batched=True,
                                                                   remove_columns=column_names,
                                                                   load_from_cache_file=False)
                else:
                    preprocessed_split_dataset = split_dataset.map(preprocess_train, batched=True,
                                                                   remove_columns=column_names, num_proc=64,
                                                                   load_from_cache_file=False)
                print(f'处理后的数据集字段：{preprocessed_split_dataset.column_names}')
                with codecs.open(os.path.join(prompted_output_dir, f'{split}.json'), 'w',
                                 encoding='utf-8') as f:
                    for example in preprocessed_split_dataset:
                        if len(example['input_ids']) <= 1 or len(example['labels']) == 0:
                            # print(f'忽略有问题的样本：{example}')
                            continue
                        f.write(json.dumps(example) + '\n')


def stat_dataset_len(raw_dataset):
    src_len_sum = 0
    src_len_max = 0
    tgt_len_sum = 0
    tgt_len_max = 0

    for example in raw_dataset:
        ex_src_len = len(example['input_ids'])
        ex_tgt_len = len(example['labels'])

        src_len_sum += ex_src_len
        tgt_len_sum += ex_tgt_len

        src_len_max = max(src_len_max, ex_src_len)
        tgt_len_max = max(tgt_len_max, ex_tgt_len)

    src_len_avg = src_len_sum / len(raw_dataset)
    tgt_len_avg = tgt_len_sum / len(raw_dataset)

    return src_len_avg, src_len_max, tgt_len_avg, tgt_len_max


def truncate_and_filter():
    list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/train_cla.list'
    # list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/all.list'  # 用于统计长度
    # list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/temp.list'  # 用于调参的小集合
    # list_filename = '/home/yanan/shaonan/t-zero/config/pretraining/train_all.list'
    template_dir = '/home/yanan/shaonan/t-zero/templates_feature'
    data_dir = '/home/yanan/shaonan/data/preprocessed_baseline'
    output_dir = '/home/yanan/shaonan/data'

    max_length = 384
    target_max_length = 32
    dataset_max_size = 250000
    eval_dataset_max_size = 10000

    # stat_len_file = codecs.open('stat_len.txt', 'w', encoding='utf-8')
    # datasets.logging.set_verbosity_debug()
    datasets.disable_progress_bar()

    task_list = []
    with codecs.open(list_filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                task_list.append(task_tuple)
            else:
                task_list.append((task_tuple[0], None))

    print(f'任务列表: {task_list}', flush=True)

    raw_train_dataset_list = []
    raw_eval_dataset_list = []
    print(f'开始读取训练数据集')
    for dataset_name, subset_name in task_list:

        # WARNING: 注意是否有读进来是空的template
        if dataset_name == 'anli':
            prompts = DatasetTemplates(dataset_name, template_dir=template_dir)
        else:
            prompts = DatasetTemplates(f"{dataset_name}" if subset_name is None else f"{dataset_name}/{subset_name}",
                                       template_dir=template_dir)
        template_list = prompts.templates.keys()
        print(f'任务{dataset_name}/{subset_name}的template列表：{prompts.templates.keys()}', flush=True)
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            unique_taks_name = get_uniq_task_name(dataset_name, subset_name, template_name)

            # 通过是否有选项和metric过滤生成任务
            if not template.answer_choices:
                print(f'过滤没有选项的数据集{unique_taks_name}')
                continue
            if 'Accuracy' not in template.metadata.metrics:
                print(f'过滤没不使用accuracy的数据集{unique_taks_name}')
                continue

            print(f'读取训练集数据: {unique_taks_name}', flush=True)
            unique_taks_data_dir = os.path.join(data_dir, unique_taks_name)
            data_files = {}
            if os.path.exists(os.path.join(unique_taks_data_dir, 'train.json')):
                data_files['train'] = os.path.join(unique_taks_data_dir, 'train.json')
            # 只处理训练集, validation特殊情况很多，很麻烦
            if os.path.exists(os.path.join(unique_taks_data_dir, 'validation.json')):
                data_files['validation'] = os.path.join(unique_taks_data_dir, 'validation.json')
            if os.path.exists(os.path.join(unique_taks_data_dir, 'test.json')):
                if 'validation' not in data_files:
                    # 仅在没有validation的时候才用test
                    data_files['test'] = os.path.join(unique_taks_data_dir, 'test.json')
            raw_dataset = load_dataset('json', data_files=data_files)

            # 这里只读训练集, 以后可能还需要开发集
            train_raw_dataset = raw_dataset['train']
            if len(train_raw_dataset) > dataset_max_size:
                template_num = len(template_list)
                train_raw_dataset = train_raw_dataset.shuffle()
                print(f'before sample too large train dataset {unique_taks_name}: {len(train_raw_dataset)}', flush=True)
                train_raw_dataset = train_raw_dataset.select(range(0, dataset_max_size // template_num))
                print(f'after sample too large train dataset {unique_taks_name}: {len(train_raw_dataset)}', flush=True)

            raw_train_dataset_list.append(train_raw_dataset)

            # 统计数据集长度，不需要就注释掉
            # src_len_avg, src_len_max, tgt_len_avg, tgt_len_max = stat_dataset_len(raw_dataset)
            # print(f'src_len_avg: {src_len_avg}, src_len_max: {src_len_max}, '
            #       f'tgt_len_avg: {tgt_len_avg}, tgt_len_max: {tgt_len_max}', flush=True)
            # stat_len_file.write(f'{unique_taks_name}\t{src_len_avg}\t{src_len_max}\t{tgt_len_avg}\t{tgt_len_max}\n')

            if 'validation' in raw_dataset:
                eval_raw_dataset = raw_dataset['validation']
            elif 'test' in raw_dataset:
                eval_raw_dataset = raw_dataset['test']
            else:
                print(f'数据集{unique_taks_name}没有训练或测试集！！')
                continue
            if len(eval_raw_dataset) > dataset_max_size:
                template_num = len(template_list)
                eval_raw_dataset = eval_raw_dataset.shuffle()
                print(f'before sample too large eval dataset {unique_taks_name}: {len(eval_raw_dataset)}', flush=True)
                eval_raw_dataset = eval_raw_dataset.select(range(0, dataset_max_size // template_num))
                print(f'after sample too large eval dataset {unique_taks_name}: {len(eval_raw_dataset)}', flush=True)

            raw_eval_dataset_list.append(eval_raw_dataset)

    combined_train_dataset = concatenate_datasets(raw_train_dataset_list)
    combined_eval_dataset = concatenate_datasets(raw_eval_dataset_list)

    if max_length != 1024 or target_max_length != 256:
        # 这里可以选择截断或者concat
        # combined_dataset = combined_dataset.map(truncate_function, num_proc=64)
        # print(f'debug line 63: {combined_dataset[0]}')
        print(f'before filter too long train example: {len(combined_train_dataset)}', flush=True)
        combined_train_dataset = combined_train_dataset.filter(lambda ex: len(ex['input_ids']) < max_length and
                                                                          len(ex['labels']) < target_max_length,
                                                               num_proc=64, load_from_cache_file=False)
        print(f'after filter too long train example: {len(combined_train_dataset)}', flush=True)

        print(f'before filter too long eval example: {len(combined_eval_dataset)}', flush=True)
        combined_eval_dataset = combined_eval_dataset.filter(lambda ex: len(ex['input_ids']) < max_length and
                                                                        len(ex['labels']) < target_max_length,
                                                             num_proc=64, load_from_cache_file=False)
        print(f'after filter too long eval example: {len(combined_eval_dataset)}', flush=True)

    if len(combined_eval_dataset) > eval_dataset_max_size:
        # eval set最多1w个
        combined_eval_dataset = combined_eval_dataset.shuffle()
        combined_eval_dataset = combined_eval_dataset.select(range(0, eval_dataset_max_size))

    # 写入磁盘
    with codecs.open(os.path.join(output_dir,
                                  f'combined_train_v2_src_{max_length}_tgt_{target_max_length}_maxsize_{dataset_max_size}.json'),
                     'w') as f:
        for example in combined_train_dataset:
            f.write(json.dumps(example) + '\n')

    with codecs.open(os.path.join(output_dir,
                                  f'combined_eval_v2_src_{max_length}_tgt_{target_max_length}_maxsize_{dataset_max_size}.json'),
                     'w') as f:
        for example in combined_eval_dataset:
            f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    preprocess_for_baseline()
    # truncate_and_filter()
