import argparse, importlib, sys, os
from config import method_config, environment_config, reason_config, Root_dir
from utils import load_json, save_jsonl_add, RAG_map
sys.path.append(f'{Root_dir}/baselines')
sys.path.append(f'{Root_dir}/reasonings')

Oracle_list = ['Oracle', 'SASFTOracle', 'MASFTOracle', 'BlockOracle']

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('reason', type=str, help='Reasoning Structure')
    parser.add_argument('method', type=str, help='Knowledge Baseline')
    args, extend_args = parser.parse_known_args()
    reason = args.reason
    method = args.method
    extend_keys, extend_values = [k[2:] for k in extend_args[0::2]] , extend_args[1::2]
    extend_config = dict(zip(extend_keys, extend_values))

    config = environment_config | method_config[method] | reason_config[reason]
    for k,v in extend_config.items():
        config[k] = v
    return config

def run(config):
    reason_name = config['reason_method']
    method_name = config['knowledge_method']
    data_path_dir = config['data_path_dir']
    data_title = config['data_title']

    input_path = f'{data_path_dir}/{data_title}.json'
    aux = '' if 'aux' not in config else config['aux']
    if not os.path.exists(f'results/{reason_name}'):
        os.mkdir(f'results/{reason_name}')
    output_path = f'results/{reason_name}/{data_title}-{method_name}{aux}.jsonl'

    data = load_json(input_path)
    QAs = data['qa']

    knowledge_cls = getattr(importlib.import_module(f'baselines.{RAG_map(method_name)}'), method_name)
    knowledge_model = knowledge_cls(config)

    reason_cls = getattr(importlib.import_module(f'reasonings.{reason_name}'), reason_name)
    reason_model = reason_cls(config, knowledge_model)
    
    knowledge_model.store(data['message'])

    for hop_num, hop_dict in QAs.items():
        for rc_id, qa_info in hop_dict.items():
            if method_name in Oracle_list:
                response = reason_model.solve_qa(qa_info['question'], qa_info['message'])
            else:
                response = reason_model.solve_qa(qa_info['question'])
            response_dict = {
                'k_hop': hop_num,
                'rc_id': rc_id,
                'response': response,
                'ground_truth': qa_info['answer'],
                'fold_tag': qa_info['fold_tag']
            }
            save_jsonl_add(output_path, response_dict)

if __name__ == '__main__':
    config = get_config()
    run(config)