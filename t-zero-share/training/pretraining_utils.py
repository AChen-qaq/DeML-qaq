import codecs
import os
from templates import DatasetTemplates
from datasets import load_dataset, concatenate_datasets


def load_training_task_list(list_filename):
    """读取要训练的任务列表"""
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

    return task_list


def load_all_dataset(args, logger, task_list, data_dir, template_dir):
    logger.info(f'开始读取训练数据集')
    if args.train_file is not None and args.eval_file is not None:
        combined_dataset = load_dataset('json', data_files={'train': args.train_file, 'validation': args.eval_file})

        return combined_dataset['train'], combined_dataset['validation']

    raw_dataset_list = []
    for dataset_name, subset_name in task_list:
        prompts = DatasetTemplates(f"{dataset_name}" if subset_name is None else f"{dataset_name}/{subset_name}",
                                   template_dir=template_dir)
        template_list = prompts.templates.keys()
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            # 通过是否有选项和metric过滤生成任务
            if not template.answer_choices:
                continue
            if 'Accuracy' not in template.metadata.metrics:
                continue

            unique_taks_name = get_uniq_task_name(dataset_name, subset_name, template_name)
            logger.info(f'读取训练集数据: {unique_taks_name}')
            unique_taks_data_dir = os.path.join(data_dir, unique_taks_name)
            raw_dataset = load_dataset('json', data_files={'train': os.path.join(unique_taks_data_dir, 'train.json')})

            # 这里只读训练集, 以后可能还需要开发集
            raw_dataset = raw_dataset['train']
            if len(raw_dataset) > 500000:
                template_num = len(template_list)
                raw_dataset = raw_dataset.shuffle(seed=42)
                print(f'debug line 44: {raw_dataset[0]}')
                raw_dataset = raw_dataset.select(range(0, 500000//template_num))

            raw_dataset_list.append(raw_dataset)

    def truncate_function(examples):
        for i in range(len(examples)):
            if len(examples['input_ids'][i]) > args.max_length:
                examples['input_ids'][i] = examples['input_ids'][i][:args.max_length]
                examples['input_ids'][i][-1] = 1   # eos
                examples['attention_mask'][i] = examples['attention_mask'][i][:args.max_length]
            if len(examples['labels'][i]) > args.target_max_length:
                examples['labels'][i] = examples['labels'][i][:args.target_max_length]
        return examples

    combined_dataset = concatenate_datasets(raw_dataset_list)

    if args.max_length != 1024 or args.target_max_length != 256:
        # 这里可以选择截断或者concat
        # combined_dataset = combined_dataset.map(truncate_function, num_proc=64)
        # print(f'debug line 63: {combined_dataset[0]}')
        combined_dataset = combined_dataset.filter(lambda ex: len(ex['input_ids']) < args.max_length and
                                                              len(ex['labels']) < args.target_max_length)

    # TODO: 这里随机切分了训练和开发集
    combined_dataset = combined_dataset.shuffle(seed=42)
    print(f'debug 70: {combined_dataset[0]}')
    combined_eval_dataset = combined_dataset.select(range(0, 5000))
    combined_train_dataset = combined_dataset.select(range(5000, len(combined_dataset)))

    return combined_train_dataset, combined_eval_dataset


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



