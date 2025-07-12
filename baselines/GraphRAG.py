from BaseMethod import RAGMethod, MixtureMethod
from SelfAskSFT import SelfAskSFT
from MaskSFT import MaskSFT
from BlockSFT import BlockSFT
import torch, time, pickle, re
from utils import LLM
from sentence_transformers import SentenceTransformer
import numpy as np

class GraphStorage():
    def __init__(self, extract_llm_config, encoder_path, node_topk, relation_topk):
        self.llm = LLM(extract_llm_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_topk =node_topk
        self.relation_topk = relation_topk

        self.encoder = SentenceTransformer(encoder_path).to(self.device)
        self.graph = {}
        self.entity_list = None
        self.entity_embedding = None
        self.relation_embedding = None

    def encode_single(self, text):
        embeddings = self.encode_batch([text])
        return embeddings

    def encode_batch(self, text_list):
        embeddings = self.encoder.encode(text_list, normalize_embeddings=True)
        return torch.from_numpy(embeddings).to(self.device)

    def parse_triples(self, triples_string):
        pattern = r'\(([^,]+), ([^,]+), ([^\)]+)\)'
        tuples = re.findall(pattern, triples_string)
        parsed_triples = [(a.strip(), b.strip(), c.strip()) for a,b,c in tuples]

        return parsed_triples[0]

    def add_node(self, entity):
        if entity not in self.graph:
            self.graph[entity] = {
                'out_edge': [],
                'in_edge': [],
            }
    
    def add_edge(self, from_entity, relation, to_entity, mid):
        self.graph[from_entity]['out_edge'].append((relation, to_entity, mid))
        self.graph[to_entity]['in_edge'].append((relation, from_entity, mid))

    def add_triples(self, triple_list, mid):
        for (sub, rel, obj) in triple_list:
            if sub != 'None':
                self.add_node(sub)
            if obj != 'None':
                self.add_node(obj)
            if sub != 'None' and rel != 'None' and obj != 'None':
                self.add_edge(sub, rel, obj, mid)

    def add_new_message(self, text, mid):
        prompt = f"""Please help me extract the (subject, relation, object) triplets of entities and relationships from the following text.
Text:
{text}

Requirements:
1. Each triplet is on a separate line, in the format of (subject, relation, object), such as (Alice, friend_of, Bob).
2. The extracted entities and relationships should be concise.
3. You should only output the lines of triplets, without any other descriptions.
4. If there are no entities and relations in the text, please output only None without any other descriptions."""
        print('[Prompt (Extract Triples)]::',prompt)
        triple_list = []
        for attempt in range(5):
            try:
                response = self.llm.fast_run(prompt)
                response_lines = response.split('\n')
                print('[Response (Extract Triples)]::',response_lines)
                for line in response_lines:
                    triple = self.parse_triples(line)
                    triple_list.append(triple)
                break
            except Exception as e:
                print(e)
                if attempt == 4:
                    return
        self.add_triples(triple_list, mid)
    
    def update_embedding_index(self):
        entity_list = []
        entity_embedding = []
        relation_embedding = {}
        for k, v in self.graph.items():
            entity_list.append(k)
            relation_embedding[k] = {
                'out_rel': self.encode_batch([tp[0] for tp in v['out_edge']]),
                'in_rel': self.encode_batch([tp[0] for tp in v['in_edge']]),
            }
        entity_embedding = self.encode_batch(entity_list)
        self.entity_list = entity_list
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    def retrieval(self, query):
        text_list = []

        query_embedding = self.encode_single(query)
        scores = torch.matmul(self.entity_embedding, query_embedding.squeeze())
        scores, indices = torch.sort(scores, descending=True)
        scores, indices = scores[:min(self.node_topk, len(scores))], indices[:min(self.node_topk, len(scores))]
        for index in indices:
            current_entity = self.entity_list[index]
            if len(self.graph[current_entity]['in_edge']) != 0:
                in_scores = torch.matmul(self.relation_embedding[current_entity]['in_rel'], query_embedding.squeeze())
                in_scores, in_indices = torch.sort(in_scores, descending=True)
                in_scores, in_indices = in_scores[:min(len(in_scores), self.relation_topk)], in_indices[:min(len(in_scores), self.relation_topk)]
                for in_id in in_indices:
                    tp = self.graph[current_entity]['in_edge'][in_id]
                    # text_list += ['Relation (%s, %s, %s)' % (tp[1], tp[0], current_entity)]
                    text_list += [tp[2]]

            if len(self.graph[current_entity]['out_edge']) != 0:
                out_scores = torch.matmul(self.relation_embedding[current_entity]['out_rel'], query_embedding.squeeze())
                out_scores, out_indices = torch.sort(out_scores, descending=True)
                out_scores, out_indices = out_scores[:min(len(out_scores), self.relation_topk)], out_indices[:min(len(out_scores), self.relation_topk)]
                for out_id in out_indices:
                    tp = self.graph[current_entity]['out_edge'][out_id]
                    # text_list += ['Relation (%s, %s, %s)' % (current_entity, tp[0], tp[1])]
                    text_list += [tp[2]]
        
        return text_list

class GraphRAG(RAGMethod):
    def __init__(self, config):
        super().__init__(config)

        self.storage = GraphStorage(
            self.config['retrieval_config']['extract_llm_config'],
            config['retrieval_config']['encoder_path'],
            config['retrieval_config']['node_topk'],
            config['retrieval_config']['relation_topk'],
            )

    def __make_index__(self, mid_list, text_list, messages):
        self.messages = messages
        if self.config['retrieval_config']['cache_mode']:
            with open(self.config['retrieval_config']['cache_path'], 'rb') as f:
                self.storage = pickle.load(f)
            return
        
        for index in range(len(mid_list)):
            self.storage.add_new_message(text_list[index], mid_list[index])
        self.storage.update_embedding_index()

        if self.config['retrieval_config']['cache_path']:
            with open(self.config['retrieval_config']['cache_path'], 'wb') as f:
                pickle.dump(self.storage, f)

    def __retrieve_message__(self, question, ref):
        message_ids = self.storage.retrieval(question)
        references = '\n'.join([self.messages[mid] for mid in message_ids])
        return references

class SASFTGraphRAG(MixtureMethod, GraphRAG, SelfAskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.storage = GraphStorage(
            self.config['retrieval_config']['extract_llm_config'],
            config['retrieval_config']['encoder_path'],
            config['retrieval_config']['node_topk'],
            config['retrieval_config']['relation_topk'],
            )
        
        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

class MASFTGraphRAG(MixtureMethod, GraphRAG, MaskSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.storage = GraphStorage(
            self.config['retrieval_config']['extract_llm_config'],
            config['retrieval_config']['encoder_path'],
            config['retrieval_config']['node_topk'],
            config['retrieval_config']['relation_topk'],
            )
        
        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

class BlockGraphRAG(MixtureMethod, GraphRAG, BlockSFT):
    def __init__(self, config):
        MixtureMethod.__init__(self, config)
        self.storage = GraphStorage(
            self.config['retrieval_config']['extract_llm_config'],
            config['retrieval_config']['encoder_path'],
            config['retrieval_config']['node_topk'],
            config['retrieval_config']['relation_topk'],
            )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(config['retrieval_config']['encoder_path']).to(self.device)

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