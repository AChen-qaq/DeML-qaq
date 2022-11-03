from transformers import T5ForConditionalGeneration
from torch import nn
import torch
from evaluation.templates import DatasetTemplates
from datasets import load_dataset


class T5Wrap(nn.Module):
    def __init__(self, path_to_T5):
        super().__init__()

        self.prompt_embedding = nn.Embedding(10, 4096)
        self.T5_model = T5ForConditionalGeneration.from_pretrained(path_to_T5)

    def forward(
            self,
            input_ids=None,
            prompt_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        pt_emb = self.prompt_embedding(prompt_ids)
        token_emb = self.T5_model.shared(input_ids)

        input_emb = torch.cat((pt_emb, token_emb), dim=1)

        output = self.T5_model(inputs_embeds=input_emb, attention_mask=attention_mask,
                               labels=labels, return_dict=return_dict)

        print(output)

    def to_cuda_and_parallel(self):
        self.prompt_embedding.cuda()
        self.T5_model.parallelize()


def generate_example():
    dataset_name = 'commonsense_qa'
    dataset_subname = None
    raw_datasets = load_dataset(dataset_name, dataset_subname, split='train')

    prompts = DatasetTemplates(dataset_name, dataset_subname, '/mfs/shaonan/moonshot/t-zero/templates')
    template_list = prompts.templates.keys()
    print(f'模板列表：{template_list}')
    for template_id in template_list:
        template = prompts.templates[template_id]
        column_names = raw_datasets.column_names
        example = raw_datasets[0]

        input, target = template.apply(example)
        print(template_id)
        print(input)
        print(target)


def print_template_id_and_name():
    dataset_name = 'anli'
    dataset_subname = None

    prompts = DatasetTemplates(dataset_name, dataset_subname, '/mfs/shaonan/moonshot/t-zero/templates')
    template_list = prompts.templates.keys()
    print(f'模板列表：{template_list}')

    for template_id in template_list:
        template = prompts.templates[template_id]
        print(f'{template.id}\t{template.name}')


if __name__ == '__main__':
    # model = T5ForConditionalGeneration.from_pretrained('/home/yanan/shaonan/pretrained_model/t5-xxl-lm-adapt')

    # wrap_model = T5Wrap('/home/yanan/shaonan/pretrained_model/t5-xxl-lm-adapt')
    # wrap_model.to_cuda_and_parallel()

    print_template_id_and_name()