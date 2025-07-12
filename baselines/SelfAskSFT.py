from BaseMethod import EditMethod
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import torch, os
from utils import load_json, llm_fast_inference, save_json_file, map_tokenizer_sft_batch
from prompts import *
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class SelfAskSFT(EditMethod):
    def __init__(self, config):
        super().__init__(config)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.ask_proportion = self.config['edit_config']['ask_proportion']

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
        
        save_json_file(self.trainset_path, ask_trainset)
        return ask_trainset
    
    def __edit_process__(self, messages):
        # Generate Dataset
        if os.path.exists(self.trainset_path):
            trainset = load_json(self.trainset_path)
        else:
            trainset = self.__generate_trainset__(messages)
        # Prepare_dataset
        sft_data = Dataset.from_list([{
                'input': d['question'],
                'output': d['answer']
                }for d in trainset
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

        self.model = get_peft_model(self.model, lora_config)

        train_args = TrainingArguments(
            output_dir=self.model_save_path,
            num_train_epochs=10,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            logging_steps=1500
        )

        # [Attention]
        # Our target is storing ALL the information into our model, so we let eval_dataset = train_dataset.
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=sft_data,
            eval_dataset=sft_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        )

        trainer.train()