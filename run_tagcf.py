import argparse
from recbole.config import Config
from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="LogicRec", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="amazon_movie", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=2, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )


    parameter_dict = {
    'gpu_id': '0,1',
    'field_separator': ',',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'load_col': {'inter': ['user_id',  'item_id', 'time_ms'],
                 'item': ['item_id', 'user_tag_id_list','item_tag_id_list']},
    'TIME_FIELD': 'time_ms',
    'ITEM_LIST_LENGTH_FIELD': 'item_length',
    'ITEM_ID_LIST_FIELD': 'item_id_list',
    'MAX_ITEM_LIST_LENGTH': 50,
    # model config
    'hidden_size': 128,
    'learner': 'adamw',
    'weight_decay': 0.001,
    'full_tag_infer': True,
    'branch_factor': 50,
    'tag_decay': 0.01,
    'inner_size': 256,
    'lamda': 0.95,
    'tag_col_name': 'user_tag_id_list',
    'hidden_act': "gelu",
    'mode': 'user_tag',
    'learning_rate': 0.001,
    'n_layers': 2,
    'n_heads': 2,
    'initializer_range': 0.02,
    'layer_norm_eps': 1e-12,
    'hidden_dropout_prob': 0.2,
    'attn_dropout_prob': 0.2,
    'feat_hidden_dropout_prob': 0.3,
    'feat_attn_dropout_prob': 0.3,
    'loss_type': 'BCE',  
    # Training and evaluation config
    'epochs': 300,
    'train_batch_size': 4096,
    'eval_batch_size': 4096,
    'MODEL_INPUT_TYPE': 'InputType.PAIRWISE',
    'seed': 2024,
    'valid_metric': 'itemcoverage@20',
    'inter_matrix_type': '01',
    'topk': [10, 20],
    'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
    'stopping_step': 50,
    'metrics': ['Recall', 'NDCG', 'MRR', 'GAUC','ItemCoverage', 'GiniIndex'],
    'eval_args': {
        'split': {'LS': 'valid_and_test'},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full',

    },
    
    
}
    run(
        args.model,
        args.dataset,
        config_dict=parameter_dict,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )
