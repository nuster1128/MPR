from BaseMethod import BaseMethod
from utils import LLM
from prompts import *

class Ignoramus(BaseMethod):
    def __init__(self, config):
        super().__init__(config)

        self.llm = LLM(config['llm_config'])

    def store(self, messages):
        pass
    
    def response(self, call_type, param_dict, ref, knowledge_key):
        return self.inference(call_type, param_dict)
    
    def inference(self, call_type, param_dict):
        prompt = eval(f'{call_type}_Ignoramus_Prompt_Template').format(**param_dict)
        print('[Prompt]::',prompt)
        response = self.llm.fast_run(prompt)
        print('[Response]::',response)
        return response
