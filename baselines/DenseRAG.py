from BaseMethod import RAGMethod, MixtureMethod
from SelfAskSFT import SelfAskSFT
from MaskSFT import MaskSFT
from BlockSFT import BlockSFT
import torch
from sentence_transformers import SentenceTransformer

class DenseRAG(RAGMethod):
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.mid_store = None
        self.tensor_store = None

    def encode_single(self, text):
        embeddings = self.encode_batch([text])
        return embeddings

    def encode_batch(self, text_list):
        embeddings = self.encoder.encode(text_list, normalize_embeddings=True)
        return torch.from_numpy(embeddings).to(self.device)

    def __make_index__(self, mid_list, text_list, messages):
        self.messages = messages
        embeddings = self.encode_batch(text_list)
        self.mid_store = mid_list
        self.tensor_store = embeddings

    def __retrieve_message__(self, question, ref):
        query_embedding = self.encode_single(question)
        scores = torch.matmul(self.tensor_store, query_embedding.squeeze())
        scores, indices = torch.sort(scores, descending=True)
        scores, indices = scores[:self.topk], indices[:self.topk]
        message_ids = [self.mid_store[ind] for ind in indices]
        references = '\n'.join([self.messages[mid] for mid in message_ids])
        return references
    
class SASFTDenseRAG(MixtureMethod, DenseRAG, SelfAskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.mid_store = None
        self.tensor_store = None

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

class MASFTDenseRAG(MixtureMethod, DenseRAG, MaskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.mid_store = None
        self.tensor_store = None

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

class BlockDenseRAG(MixtureMethod, DenseRAG, BlockSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.mid_store = None
        self.tensor_store = None

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