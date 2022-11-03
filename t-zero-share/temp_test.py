# from templates_test import DatasetTemplates
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import nn

from multiprocessing import Pool
import json
import os
import torch


def test_templete():
    prompts = DatasetTemplates('super_glue/rte')

    prompt_template = prompts['GPT-3 style']
    print(prompt_template)
    print(f'prompt: {prompt_template.jinja}, type: {type(prompt_template.jinja)}')


def test_concat():
    raw_datasets = load_dataset('anli', split=f'test_r1')
    label_list = raw_datasets['label']
    label_type_set = set(label_list)
    print(f'label_type_set: {len(label_type_set)}')
    dataset_per_type = []
    for label_type in label_type_set:
        dataset_per_type.append(raw_datasets.filter(lambda x: x['label'] == label_type))

    for dataset in dataset_per_type:
        dataset.select


def test_prompt_tuning():
    tokenizer = T5Tokenizer.from_pretrained('')


def test_trivia():
    dir = '/home/yanan/shaonan/data/T0_dataset/trivia_qa_unfiltered'

    train_file = open(os.path.join(dir, 'validation.json'))
    data_dict = {'question': [], 'answer': []}

    count = 0
    for line in train_file:
        example = json.loads(line)
        data_dict['question'].append(example['question'])
        data_dict['answer'].append(example['answer'])
        count += 1
        if count == 10:
            break

    raw = Dataset.from_dict(data_dict)
    print(raw)
    print(raw[0])



if __name__ == '__main__':
    pass