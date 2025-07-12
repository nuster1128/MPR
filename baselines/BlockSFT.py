from BaseMethod import EditMethod
from utils import llm_fast_inference, save_json_file, map_tokenizer_sft_batch, load_json
from prompts import *
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import os, torch, pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from peft import PeftModel
import threading

class BlockSFT(EditMethod):
    def __init__(self, config):
        super().__init__(config)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']
        self.block_num = self.config['edit_config']['block_num']
        self.lora_load_index = self.config['edit_config']['lora_load_index']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(config['edit_config']['encoder_path']).to(self.device)

    def store(self, messages):
        if not self.model_load_path:
            self.__edit_process__(messages)
            print('[Edit Finishes]')
        else:
            # Load Cluster Models and LoRA Adapters
            with open(f'{self.model_save_path}/cluster.pkl', 'rb') as f:
                self.cluster = pickle.load(f)

            for block_id in range(self.block_num):
                checkpoint_path = f'{self.model_save_path}/lora_{block_id}'
                all_tags = sorted([int(filename.split('-')[1]) for filename in os.listdir(checkpoint_path)])

                checkpoint_path = '%s/%s' % (checkpoint_path, f'checkpoint-{all_tags[self.lora_load_index]}')
                if os.path.exists(checkpoint_path):
                    if block_id == 0:
                        self.edit_model = PeftModel.from_pretrained(self.model, checkpoint_path, adapter_name=f'{block_id}')
                    else:
                        self.edit_model.load_adapter(checkpoint_path, adapter_name=f'{block_id}')
                    print(f'[Load LoRA Adapter from {checkpoint_path}]')
                else:
                    raise "No Lora Checkpoint Found"

    def __generate_trainset__(self, messages):
        ask_trainset = []
        for k, v in messages.items():
            part_list = []
            max_try = 0
            while len(part_list) < self.ask_proportion and max_try < 10 * self.ask_proportion:
                max_try += 1
                prompt = SelfAskSFT_Ask_Prompt_Template.format(**{'message': v})
                response = llm_fast_inference(self.model, self.tokenizer, prompt)
                response = response.split('\n')
                if len(response) != 2:
                    print(response)
                    continue
                part_list.append({
                    'message': v,
                    'question': response[0],
                    'answer': response[1]
                })

            ask_trainset += part_list
            print(k, len(part_list), part_list)
            # print(part_list)
        
        save_json_file(self.trainset_path, ask_trainset)
        return ask_trainset
    
    def __encode_single__(self, text):
        embeddings = self.encode_batch([text])
        return embeddings

    def __encode_batch__(self, text_list):
        embeddings = self.encoder.encode(text_list, normalize_embeddings=True)
        return embeddings

    def __sft_single_block__(self, block_id, block, cuda_id=None):
        if cuda_id:
            device = torch.device(f'cuda:{cuda_id}')
        else:
            device = self.device
        checkpoint_path = f'{self.model_save_path}/lora_{block_id}'
        # Prepare_dataset
        sft_data = Dataset.from_list([{
            'input': d['question'],
            'output': d['answer']
            }for d in block
        ])
        sft_data = sft_data.map(map_tokenizer_sft_batch, fn_kwargs={'tokenizer': self.tokenizer}, batched=True)
        # SFT Process
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        self.edit_model = get_peft_model(self.model, lora_config).to(device)

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        train_args = TrainingArguments(
            output_dir=checkpoint_path,
            num_train_epochs=10, # 1
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16, # 16
            learning_rate=1e-5, # 1e-5
            logging_steps=100, 
            save_strategy='epoch'
        )

        # [Attention]
        # Our target is storing ALL the information into our model, so we let eval_dataset = train_dataset.
        trainer = Trainer(
            model=self.edit_model,
            args=train_args,
            train_dataset=sft_data,
            eval_dataset=sft_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        )

        trainer.train()

    def __parallel_sft__(self, gpu_id, task_list):
        for block_id, block in task_list:
            print(f'[On GPU({gpu_id}) Training Block {block_id}] Start::', len(block))
            self.__sft_single_block__(block_id, block, gpu_id)
            print(f'[End:: On GPU({gpu_id}) Training Block {block_id}]', len(block))


    def __edit_process__(self, messages):
        # Generate Dataset
        if os.path.exists(self.trainset_path):
            trainset = load_json(self.trainset_path)
        else:
            trainset = self.__generate_trainset__(messages)
        
        # Divide Whole Messages into Blocks
        text_list = [t['message'] for t in trainset]
        ## Embed Messags into Semantic Vectors and Get Clusters by Scikit-learn
        semantic_blocks = {i:[] for i in range(self.block_num)}
        text_embeddings = self.__encode_batch__(text_list)
        self.cluster = KMeans(n_clusters = self.block_num, random_state=42, n_init=10)
        cluster_labels = self.cluster.fit_predict(text_embeddings)
        
        for index, label in enumerate(cluster_labels):
            semantic_blocks[label].append(trainset[index])

        with open(f'{self.model_save_path}/cluster.pkl', 'wb') as f:
            pickle.dump(self.cluster, f)

        ## Training All LoRA Adapters
        parallel_gpu_list = self.config['edit_config']['distributional_training']

        # Non-Parallel Training
        if not parallel_gpu_list:
            for block_id, block in semantic_blocks.items():
                self.__sft_single_block__(block_id, block)
            return 

        # Parallel Training
        meta_task_list = [[] for _ in range(len(parallel_gpu_list))]
        for block_id, block in semantic_blocks.items():
            meta_task_list[block_id % len(parallel_gpu_list)].append((block_id, block))
        
        schedular = []
        for idx, gpu_id in enumerate(parallel_gpu_list):
            thread = threading.Thread(target=self.__parallel_sft__, args=(gpu_id, meta_task_list[idx]))
            schedular.append(thread)
        
        for thread in schedular:
            thread.start()

        for thread in schedular:
            thread.join()
        print('[Edit Finishes]')
    
    def inference(self, call_type, param_dict):
        # Retrive LoRA Adaptors with Cluster Models
        if 'references' not in param_dict:
            print('Do Not Set Adaptor')
            prompt = eval(f'{call_type}_RAG_Prompt_Template').format(**param_dict)
            print('[Prompt]::',prompt)
            response = llm_fast_inference(self.model, self.tokenizer, prompt)
            print('[Response]::',response)
            return response
        references = param_dict['references'].split('\n')
        ref_embeddings = self.__encode_batch__(references)
        ref_cluster_labels = self.cluster.predict(ref_embeddings)
        ## Vote the label with the most references
        label_votes = {}
        for label in ref_cluster_labels:
            if label in label_votes:
                label_votes[label] += 1
            else:
                label_votes[label] = 1
        sorted_labels = sorted(label_votes.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_labels[0][0]

        ## Load loara_adapter
        self.edit_model.set_adapter(f'{top_label}')
        print('Set Adapter to', f'{top_label}')
        
        # Inference to Get Response
        prompt = eval(f'{call_type}_RAG_Prompt_Template').format(**param_dict)
        print('[Prompt]::',prompt)
        response = llm_fast_inference(self.model, self.tokenizer, prompt)
        print('[Response]::',response)
        return response

