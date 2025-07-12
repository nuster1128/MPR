import numpy as np

# ---- Global Settings ----

Qwen_2_5_7B_model_path = '[MODEL_PATH]'
Qwen_2_5_3B_model_path = '[MODEL_PATH]'
E5_base_v2_model_path = '[MODEL_PATH]'
Root_dir = '[PROJECT_ROOT_PATH]'

gpt_4o_config = {
    'name': 'gpt-4o',
    'api_key': '[API_KEY]',
    'api_base': '[API_BASE]'
}

qwen_2_5_3B_config = {
    'name': 'Qwen2.5-3B',
    'api_base': 'http://127.0.0.1:3923'
}

qwen_2_5_7B_config = {
    'name': 'Qwen2.5-7B',
    'api_base': 'http://127.0.0.1:3933'
}

retrieval_topk = 20

# ---- Experiment Settings ----
environment_config = {
    'data_title': '[DATA_TITLE]',
    'data_path_dir': '[DATA_DIR_PATH]'
}

# ---- Reasoning Settings ----
Naive_default = {
    'reason_method': 'NaiveReason'
}

Decomposition_default = {
    'reason_method': 'DecompositionReason',
    'decompose_num': 5
}

Sequential_default = {
    'reason_method': 'SequentialReason',
    'max_step': 5
}

Structured_default = {
    'reason_method': 'StructuredReason',
    'max_step': 5,
    'branch_size': 2
}

# ---- Baseline Settings ----
Oracle_default = {
    'knowledge_method': 'Oracle',
    'llm_config': qwen_2_5_7B_config,
}

Ignoramus_default = {
    'knowledge_method': 'Ignoramus',
    'llm_config': qwen_2_5_7B_config,
}

SparseRAG_default = {
    'knowledge_method': 'SparseRAG',
    'llm_config': qwen_2_5_7B_config,
    'retrieval_config': {
        'topk': retrieval_topk
    }
}

DenseRAG_default = {
    'knowledge_method': 'DenseRAG',
    'llm_config': qwen_2_5_7B_config,
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path
    }
}

TreeRAG_default = {
    'knowledge_method': 'TreeRAG',
    'llm_config': qwen_2_5_7B_config,
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'max_depth': 8,
        'base_threshold': 0.7,
        'merge_llm_config': qwen_2_5_7B_config
    }
}

GraphRAG_default = {
    'knowledge_method': 'GraphRAG',
    'llm_config': qwen_2_5_7B_config,
    'retrieval_config': {
        'node_topk': int(np.ceil(np.sqrt(retrieval_topk))),
        'relation_topk': int(np.floor(retrieval_topk / np.floor(np.sqrt(retrieval_topk)))),
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'extract_llm_config': qwen_2_5_7B_config
    }
}

MaskSFT_default = {
    'knowledge_method': 'MaskSFT',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    }
}

SelfAskSFT_default = {
    'knowledge_method': 'SelfAskSFT',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    }
}

SASFTOracle_default = {
    'knowledge_method': 'SASFTOracle',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    }
}

MASFTOracle_default = {
    'knowledge_method': 'MASFTOracle',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    }
}

SASFTSparseRAG_default = {
    'knowledge_method': 'SASFTSparseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk
    }
}

MASFTSparseRAG_default = {
    'knowledge_method': 'MASFTSparseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk
    }
}

SASFTDenseRAG_default = {
    'knowledge_method': 'SASFTDenseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path
    }
}

MASFTDenseRAG_default = {
    'knowledge_method': 'MASFTDenseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path
    }
}

SASFTTreeRAG_default = {
    'knowledge_method': 'SASFTTreeRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'max_depth': 8,
        'base_threshold': 0.7,
        'merge_llm_config': qwen_2_5_7B_config
    }
}

MASFTTreeRAG_default = {
    'knowledge_method': 'MASFTTreeRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    },
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'max_depth': 8,
        'base_threshold': 0.7,
        'merge_llm_config': qwen_2_5_7B_config
    }
}

SASFTGraphRAG_default = {
    'knowledge_method': 'SASFTGraphRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'ask_proportion': 2
    },
    'retrieval_config': {
        'node_topk': int(np.ceil(np.sqrt(retrieval_topk))),
        'relation_topk': int(np.floor(retrieval_topk / np.floor(np.sqrt(retrieval_topk)))),
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'extract_llm_config': qwen_2_5_7B_config
    }
}

MASFTGraphRAG_default = {
    'knowledge_method': 'MASFTGraphRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'mask_proportion': 2
    },
    'retrieval_config': {
        'node_topk': int(np.ceil(np.sqrt(retrieval_topk))),
        'relation_topk': int(np.floor(retrieval_topk / np.floor(np.sqrt(retrieval_topk)))),
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'extract_llm_config': qwen_2_5_7B_config
    }
}

# ---- New Methods ----
Block_num = 50

BlockSFT_default = {
    'knowledge_method': 'BlockSFT',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num
    }
}

BlockOracle_default = {
    'knowledge_method': 'BlockOracle',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num,
        'distributional_training': None
    }
}

BlockDenseRAG_default = {
    'knowledge_method': 'BlockDenseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num,
        'distributional_training': None
    },
        'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path
    }
}

BlockSparseRAG_default = {
    'knowledge_method': 'BlockSparseRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num,
        'distributional_training': None
    },
    'retrieval_config': {
        'topk': retrieval_topk
    }
}

BlockTreeRAG_default = {
    'knowledge_method': 'BlockTreeRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num,
        'distributional_training': None
    },
    'retrieval_config': {
        'topk': retrieval_topk,
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'max_depth': 8,
        'base_threshold': 0.7,
        'merge_llm_config': qwen_2_5_7B_config
    }
}

BlockGraphRAG_default = {
    'knowledge_method': 'BlockGraphRAG',
    'edit_config': {
        'base_model_path': Qwen_2_5_7B_model_path,
        'trainset_cache_path': '[TRAINSET_CACHE_PATH]',
        'model_save_path': '[CHECKPOINT_DIR_PATH]',
        'model_load_path': '[CHECKPOINT_PATH]',
        'lora_load_index': 0,
        'encoder_path': E5_base_v2_model_path,
        'ask_proportion': 2,
        'block_num': Block_num,
        'distributional_training': None
    },
    'retrieval_config': {
        'node_topk': int(np.ceil(np.sqrt(retrieval_topk))),
        'relation_topk': int(np.floor(retrieval_topk / np.floor(np.sqrt(retrieval_topk)))),
        'encoder_path': E5_base_v2_model_path,
        'cache_mode': True, # False for re-generating, and True for loading.
        'cache_path': '[CACHE_PATH]',
        'extract_llm_config': qwen_2_5_7B_config
    }
}

# ---- Register ----
method_config = {
    # RAG Methods (6)
    'Oracle': Oracle_default,
    'Ignoramus': Ignoramus_default,
    'DenseRAG': DenseRAG_default,
    'SparseRAG': SparseRAG_default,
    'TreeRAG': TreeRAG_default,
    'GraphRAG': GraphRAG_default,
    # Edit Methods (2)
    'MaskSFT': MaskSFT_default,
    'SelfAskSFT': SelfAskSFT_default,
    # Mixture Methods (10)
    'SASFTOracle': SASFTOracle_default,
    'MASFTOracle': MASFTOracle_default,
    'SASFTSparseRAG': SASFTSparseRAG_default,
    'MASFTSparseRAG': MASFTSparseRAG_default,
    'SASFTDenseRAG': SASFTDenseRAG_default,
    'MASFTDenseRAG': MASFTDenseRAG_default,
    'SASFTTreeRAG': SASFTTreeRAG_default,
    'MASFTTreeRAG': MASFTTreeRAG_default,
    'SASFTGraphRAG': SASFTGraphRAG_default,
    'MASFTGraphRAG': MASFTGraphRAG_default,
    # New Methods (5)
    'BlockOracle': BlockOracle_default,
    'BlockDenseRAG': BlockDenseRAG_default,
    'BlockSparseRAG': BlockSparseRAG_default,
    'BlockTreeRAG': BlockTreeRAG_default,
    'BlockGraphRAG': BlockGraphRAG_default,
}

reason_config = {
    'Naive': Naive_default,
    'Decomposition': Decomposition_default,
    'Sequential': Sequential_default,
    'Structured': Structured_default
}


