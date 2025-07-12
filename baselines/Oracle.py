from BaseMethod import RAGMethod, MixtureMethod
from SelfAskSFT import SelfAskSFT
from MaskSFT import MaskSFT
from BlockSFT import BlockSFT
import torch
from sentence_transformers import SentenceTransformer

class Oracle(RAGMethod):
    def __init__(self, config):
        super().__init__(config)

    def __make_index__(self, mid_list, text_list, messages):
        self.messages = messages
    
    def __retrieve_message__(self, question, ref):
        references = '\n'.join([self.messages[mid] for mid in ref])
        return references

class SASFTOracle(MixtureMethod, Oracle, SelfAskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

class MASFTOracle(MixtureMethod, Oracle, MaskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

class BlockOracle(MixtureMethod, Oracle, BlockSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(config['edit_config']['encoder_path']).to(self.device)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']
        self.block_num = self.config['edit_config']['block_num']
        self.lora_load_index = self.config['edit_config']['lora_load_index']
    
    def store(self, messages):
        RAGMethod.store(self, messages)
        BlockSFT.store(self, messages)
    
    def response(self, call_type, param_dict, ref, knowledge_key):
        param_dict['references'] = self.__retrieve_message__(knowledge_key, ref)
        return self.inference(call_type, param_dict)
    def inference(self, call_type, param_dict):
        return BlockSFT.inference(self, call_type, param_dict)