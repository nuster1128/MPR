import json
from openai import OpenAI
import os, requests
import torch

LocalModelList = ['Qwen2.5-3B','Qwen2.5-7B']

def RAG_map(model_name):
    mixture_model_dict = {
        'SASFTOracle': 'Oracle',
        'MASFTOracle': 'Oracle',
        'BlockOracle': 'Oracle',
        'SASFTSparseRAG': 'SparseRAG',
        'MASFTSparseRAG': 'SparseRAG',
        'BlockSparseRAG': 'SparseRAG',
        'SASFTDenseRAG': 'DenseRAG',
        'MASFTDenseRAG': 'DenseRAG',
        'BlockDenseRAG': 'DenseRAG',
        'SASFTTreeRAG': 'TreeRAG',
        'MASFTTreeRAG': 'TreeRAG',
        'BlockTreeRAG': 'TreeRAG',
        'SASFTGraphRAG': 'GraphRAG',
        'MASFTGraphRAG': 'GraphRAG',
        'BlockGraphRAG': 'GraphRAG'
    }

    return model_name if model_name not in mixture_model_dict else mixture_model_dict[model_name]

def Edit_map(model_name):
    mixture_model_dict = {
        'SASFTOracle': 'SelfAskSFT',
        'MASFTOracle': 'MaskSFT',
        'BlockOracle': 'BlockSFT',
        'SASFTSparseRAG': 'SelfAskSFT',
        'MASFTSparseRAG': 'MaskSFT',
        'BlockSparseRAG': 'BlockSFT',
        'SASFTDenseRAG': 'SelfAskSFT',
        'MASFTDenseRAG': 'MaskSFT',
        'BlockDenseRAG': 'BlockSFT',
        'SASFTTreeRAG': 'SelfAskSFT',
        'MASFTTreeRAG': 'MaskSFT',
        'BlockTreeRAG': 'BlockSFT',
        'SASFTGraphRAG': 'SelfAskSFT',
        'MASFTGraphRAG': 'MaskSFT',
        'BlockGraphRAG': 'BlockSFT'
    }

    return model_name if model_name not in mixture_model_dict else mixture_model_dict[model_name]

def save_json_file(path, obj):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(obj, json_file, ensure_ascii=False, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_jsonl_add(path, data_dict):
    with open(path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_line + '\n')

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

class LLM():
    def __init__(self, llm_config):
        self.llm_config = llm_config

        if llm_config['name'] not in LocalModelList:
            self.client = OpenAI(api_key=self.llm_config['api_key'], base_url=self.llm_config['api_base'])

    def parse_response(self, response):
        return {'result': response.choices[0].message.content}

    def request(self, route_path, data):
        post_path = os.path.join(self.llm_config['api_base'], route_path)
        response = requests.post(post_path, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            return False

    def run(self, message_list):
        if self.llm_config['name'] not in LocalModelList:
            response = self.client.chat.completions.create(
                model=self.llm_config['name'],
                messages=message_list,
                temperature=0.9
            )
            response = self.parse_response(response)
        else:
            res = self.request('inference/', data={
            'messages': message_list,
            'kwargs': {}
            })
            if not res:
                response = {'result': False}
            else:
                response = {'result': res['response']}
            # print(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        max_retry = 5
        while not response['result']:
            max_retry -= 1
            response = self.run([{"role": "user", "content": query}])
            print('LLM Inference Retry.')
            if max_retry == 0:
                return 'None'
        return response['result']

def llm_inference(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, max_length=4096, padding='max_length', truncation=True)[0]
    return response

def llm_fast_inference(model, tokenizer, query):
    return llm_inference(model, tokenizer, [{"role": "user", "content": query}])

def map_tokenizer_sft_batch(batch_sample, tokenizer):
    prompt_message_list = [[
            {"role": "user", "content": batch_sample['input'][idx]},
        ] for idx in range(len(batch_sample['input']))]
    
    prompt_text_list = tokenizer.apply_chat_template(
        prompt_message_list,
        tokenize=False,
        add_generation_prompt=True,
    )

    response_text_list = [
        batch_sample['output'][idx] + tokenizer.eos_token
    for idx in range(len(batch_sample['input']))]

    tokenized_prompts = tokenizer(prompt_text_list, return_tensors="pt", max_length=4096, padding='max_length', truncation=True, add_special_tokens=False)
    tokenized_responses = tokenizer(response_text_list, return_tensors="pt", max_length=512, padding='max_length', truncation=True, add_special_tokens=False)

    input_ids = torch.concat([tokenized_prompts['input_ids'], tokenized_responses['input_ids']], dim=1)
    attention_mask = torch.concat([tokenized_prompts['attention_mask'], tokenized_responses['attention_mask']], dim=1)
    labels = torch.tensor([
        [-100] * len(tokenized_prompts['input_ids'][idx]) + tokenized_responses['input_ids'][idx].tolist()
    for idx in range(len(batch_sample['input']))], dtype=int)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }