import torch, time, pickle
from utils import LLM
from sentence_transformers import SentenceTransformer
import numpy as np
from BaseMethod import RAGMethod, MixtureMethod
from SelfAskSFT import SelfAskSFT
from MaskSFT import MaskSFT
from BlockSFT import BlockSFT

class TreeStorage():
    def __init__(self, base_threshold, max_depth, merge_llm_config):
        self.base_threshold = base_threshold
        self.decrease_rate = - np.log(self.base_threshold)
        self.max_depth = max_depth

        self.tree = {}
        self.root_id = None
        self.llm = LLM(merge_llm_config)

        self.mid_store = None
        self.tensor_store = None
    
    def get_threshold(self, current_depth):
        return self.base_threshold * np.exp(self.decrease_rate * current_depth / self.max_depth)

    def add_new_node(self, embedding, text, parent_id, mid):
        node_id = '#ND%06d' % len(self.tree)
        node_element = {
            'node_id': node_id,
            'embedding': embedding,
            'text': text,
            'parent': parent_id,
            'children': [],
            'depth': 0 if parent_id is None else self.tree[parent_id]['depth'] + 1,
            'mid': [mid]
        }
        self.tree[node_id] = node_element
        depth = node_element['depth']
        print(f'[Add]:: {parent_id} --> {node_id} (Depth {depth})')

        if parent_id is not None:
            self.tree[parent_id]['children'].append(node_id)
        else:
            self.root_id = node_id

    def update_node_content(self, node_id):
        print(f'[Update]:: {node_id}')
        current_node = self.tree[node_id]
        child_information = '\n'.join([self.tree[child_node_id]['text'] for child_node_id in current_node['children']] + [current_node['text']])
        prompt = f"""Please help me merge the following information into a complete paragraph.
Information:
{child_information}

You should only output the paragraph in one line (no code block), without any other descriptions."""
        print('[Prompt (Update Node)]::',prompt)
        response = self.llm.fast_run(prompt)
        print('[Response (Update Node)]::',response)
        self.tree[node_id]['text'] = response
        self.tree[node_id]['mid'] = [self.tree[node_id]['mid'][0]]
        for child_node_id in current_node['children']:
           self.tree[node_id]['mid'] += self.tree[child_node_id]['mid']

    def tranverse(self, current_node_id, embedding, text, mid):
        # Special Exit: Empty Tree.
        if current_node_id is None:
            self.add_new_node(embedding, text, None, mid)
            return

        current_node = self.tree[current_node_id]
        # Normal Exit 1: Reach Leaf Node
        if len(current_node['children']) == 0:
            self.add_new_node(embedding, text, current_node_id, mid)
            self.update_node_content(current_node_id)
            return
        
        children_embedding = torch.stack([self.tree[child_node_id]['embedding'] for child_node_id in current_node['children']])
        scores = torch.matmul(children_embedding, embedding)
        scores, indices = torch.sort(scores, descending=True)
        max_score, max_index = scores[0], indices[0]
        tranverse_threshold = self.get_threshold(self.tree[current_node_id]['depth'])
        depth = self.tree[current_node_id]['depth']
        print(f'[Tranverse]:: Threshold: {tranverse_threshold} | Max Score: {max_score}. | Depth: {depth}')
        if max_score <= tranverse_threshold:
            # Normal Exit 2: Less Similarity Than Any Children
            self.add_new_node(embedding, text, current_node_id, mid)
            self.update_node_content(current_node_id)
            return
        else:
            # Tranverse Deeper
            self.tranverse(current_node['children'][max_index], embedding, text, mid)
            self.update_node_content(current_node_id)

    def update_embedding_index(self, encoder):
        node_id_list = []
        embedding_list = []
        for node_id, node in self.tree.items():
            node_id_list.append(node_id)
            ct_embedding = encoder.encode([node['text']], normalize_embeddings=True).squeeze()
            # self.tree[node_id]['embedding'] = torch.from_numpy(ct_embedding).to(self.device)
            self.tree[node_id]['embedding'] = None
            embedding_list.append(torch.from_numpy(ct_embedding))
        self.nid_store = node_id_list
        self.tensor_store = torch.stack(embedding_list)

    def retrieval(self, query_embedding, topk):
        scores = torch.matmul(self.tensor_store, query_embedding.squeeze())
        scores, indices = torch.sort(scores, descending=True)
        scores, indices = scores[:topk], indices[:topk]
        return [self.tree[self.nid_store[ind]]['text'] for ind in indices]

class TreeRAG(RAGMethod):
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.storage = TreeStorage(config['retrieval_config']['base_threshold'], config['retrieval_config']['max_depth'], config['retrieval_config']['merge_llm_config'])

    def encode_single(self, text):
        embeddings = self.encode_batch([text])
        return embeddings

    def encode_batch(self, text_list):
        embeddings = self.encoder.encode(text_list, normalize_embeddings=True)
        # return torch.from_numpy(embeddings).to(self.device)
        return torch.from_numpy(embeddings).cpu()

    def add_single_document(self, embedding, text, mid):
        self.storage.tranverse(self.storage.root_id, embedding, text, mid)

    def save_storage(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.storage, f)
    
    def load_storage(self, path):
        with open(path, 'rb') as f:
            self.storage = pickle.load(f)
            self.storage.tensor_store.to(self.device)

    def __make_index__(self, mid_list, text_list, messages):
        self.messages = messages
        if self.config['retrieval_config']['cache_mode']:
            self.load_storage(self.config['retrieval_config']['cache_path'])
            return
        
        text_embeddings = self.encode_batch(text_list)
        for index in range(len(mid_list)):
            self.add_single_document(text_embeddings[index], text_list[index], mid_list[index])

        self.storage.update_embedding_index(self.encoder)
        
        if self.config['retrieval_config']['cache_path']:
            self.save_storage(self.config['retrieval_config']['cache_path'])

    def __retrieve_message__(self, question, ref):
        query_embedding = self.encode_single(question)
        references = '\n'.join(self.storage.retrieval(query_embedding, self.config['retrieval_config']['topk']))
        return references


class SASFTTreeRAG(MixtureMethod, TreeRAG, SelfAskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.storage = TreeStorage(config['retrieval_config']['base_threshold'], config['retrieval_config']['max_depth'], config['retrieval_config']['merge_llm_config'])

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

class MASFTTreeRAG(MixtureMethod, TreeRAG, MaskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.storage = TreeStorage(config['retrieval_config']['base_threshold'], config['retrieval_config']['max_depth'], config['retrieval_config']['merge_llm_config'])

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

class BlockTreeRAG(MixtureMethod, TreeRAG, BlockSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.topk = config['retrieval_config']['topk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)
        self.storage = TreeStorage(config['retrieval_config']['base_threshold'], config['retrieval_config']['max_depth'], config['retrieval_config']['merge_llm_config'])

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