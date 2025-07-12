from BaseMethod import RAGMethod, MixtureMethod
from SelfAskSFT import SelfAskSFT
from MaskSFT import MaskSFT
from BlockSFT import BlockSFT
from sentence_transformers import SentenceTransformer
import torch
import bm25s

class SparseRAG(RAGMethod):
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['retrieval_config']['topk']

    def __make_index__(self, mid_list, text_list, messages):
        self.messages = messages
        self.mid_store = mid_list
        corpus_tokens = bm25s.tokenize(text_list, stopwords="en")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def __retrieve_message__(self, question, ref):
        query_tokens = bm25s.tokenize(question, stopwords="en")
        results, scores = self.retriever.retrieve(query_tokens, k=self.topk)
        message_ids = [self.mid_store[ind] for ind in results[0]]
        references = '\n'.join([self.messages[mid] for mid in message_ids])
        return references
    
class SASFTSparseRAG(MixtureMethod, SparseRAG, SelfAskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

class MASFTSparseRAG(MixtureMethod, SparseRAG, MaskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

class BlockSparseRAG(MixtureMethod, SparseRAG, BlockSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
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