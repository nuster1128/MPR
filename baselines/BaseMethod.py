from utils import LLM
from prompts import *
import os, torch
from utils import load_json, llm_fast_inference
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseMethod():
    def __init__(self, config):
        self.config = config
    
    def store(self, messages):
        raise NotImplementedError
    
    def response(self, call_type, param_dict, ref, knowledge_key):
        raise NotImplementedError
    
    def inference(self, call_type, param_dict):
        raise NotImplementedError

class RAGMethod(BaseMethod):
    def __init__(self, config):
        super().__init__(config)

        self.llm = LLM(config['llm_config'])

    def __make_index__(self, mid_list, text_list, messages):
        raise NotImplementedError

    def store(self, messages):
        mid_list, text_list = [], []
        for k, v in messages.items():
            mid_list.append(k)
            text_list.append(v)
        
        self.__make_index__(mid_list, text_list, messages)

    def __retrieve_message__(self, question, ref):
        raise NotImplementedError

    def response(self, call_type, param_dict, ref, knowledge_key):
        param_dict['references'] = self.__retrieve_message__(knowledge_key, ref)
        return self.inference(call_type, param_dict)

    def inference(self, call_type, param_dict):
        prompt = eval(f'{call_type}_RAG_Prompt_Template').format(**param_dict)
        print('[Prompt]::',prompt)
        response = self.llm.fast_run(prompt)
        print('[Response]::',response)
        return response

class EditMethod(BaseMethod):
    def __init__(self, config):
        super().__init__(config)

        self.model_path = config['edit_config']['base_model_path']
        self.model_load_path = self.config['edit_config']['model_load_path']

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def __edit_process__(self, messages):
        raise NotImplementedError

    def store(self, messages):
        if not self.model_load_path:
            self.__edit_process__(messages)
            print('[Edit Finishes]')
        else:
            self.model = PeftModel.from_pretrained(self.model, model_id=self.model_load_path)
            print('[Load Peft from Saved Files.]')
    
    def response(self, call_type, param_dict, ref, knowledge_key):
        return self.inference(call_type, param_dict)
    
    def inference(self, call_type, param_dict):
        prompt = eval(f'{call_type}_Ignoramus_Prompt_Template').format(**param_dict)
        print('[Prompt]::',prompt)
        response = llm_fast_inference(self.model, self.tokenizer, prompt)
        print('[Response]::',response)
        return response

class MixtureMethod(RAGMethod, EditMethod):
    def __init__(self, config):
        BaseMethod.__init__(self, config)
        self.model_path = config['edit_config']['base_model_path']
        self.model_load_path = self.config['edit_config']['model_load_path']

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def store(self, messages):
        RAGMethod.store(self, messages)
        EditMethod.store(self, messages)
    
    def response(self, call_type, param_dict, ref, knowledge_key):
        return RAGMethod.response(self, call_type, param_dict, ref, knowledge_key)
    
    def inference(self, call_type, param_dict):
        prompt = eval(f'{call_type}_RAG_Prompt_Template').format(**param_dict)
        print('[Prompt]::',prompt)
        response = llm_fast_inference(self.model, self.tokenizer, prompt)
        print('[Response]::',response)
        return response
