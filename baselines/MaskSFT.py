from BaseMethod import EditMethod
from utils import llm_fast_inference, save_json_file, map_tokenizer_sft_batch, load_json
from prompts import *
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import os

class MaskSFT(EditMethod):
    def __init__(self, config):
        super().__init__(config)

        self.trainset_path = self.config['edit_config']['trainset_cache_path']
        self.model_save_path = self.config['edit_config']['model_save_path']
        self.mask_proportion = self.config['edit_config']['mask_proportion']

    def __generate_trainset__(self, messages):
        mask_trainset = []
        for k, v in messages.items():
            prompt = MaskSFT_Mask_Prompt_Template.format(**{'message': v})
            response = llm_fast_inference(self.model, self.tokenizer, prompt)

            mask_list = response.split('\n')
            part_list = []
            for mask in mask_list:
                original_message = v
                masked_message = original_message.replace(mask, '[MASK]')
                if '[MASK]' in masked_message:
                    part_list.append({
                        'masked_message': masked_message,
                        'mask': mask,
                        'input': MaskSFT_QA_Convert_Prompt_Template.format(message = masked_message),
                        'output': mask
                    })
            mask_trainset += part_list[:self.mask_proportion]
            print(k, len(part_list), len(mask_list), mask_list)
            # print(part_list)
        
        save_json_file(self.trainset_path, mask_trainset)
        return mask_trainset

    def __edit_process__(self, messages):
        # Generate Dataset
        if os.path.exists(self.trainset_path):
            trainset = load_json(self.trainset_path)
        else:
            trainset = self.__generate_trainset__(messages)
        # Prepare_dataset
        sft_data = Dataset.from_list(trainset)
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

