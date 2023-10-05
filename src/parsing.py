import argparse
import sys

def add_shared_args(parser):
    parser.add_argument('--name', default='webqsp', type=str)
    parser.add_argument('--data_folder', default='data/webqsp/', type=str)
    parser.add_argument('--max_train', default=200000, type=int)

    # embeddings
    # parser.add_argument('--word2id', default='vocab.txt', type=str)
    parser.add_argument('--word2id', default='vocab_new.txt', type=str)
    parser.add_argument('--relation2id', default='relations.txt', type=str)
    parser.add_argument('--entity2id', default='entities.txt', type=str)
    parser.add_argument('--char2id', default='chars.txt', type=str)
    parser.add_argument('--entity_emb_file', default=None, type=str)
    parser.add_argument('--relation_emb_file', default=None, type=str)
    parser.add_argument('--relation_word_emb', default=False, type=bool)
    parser.add_argument('--word_emb_file', default='word_emb.npy', type=str)
    parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)
    parser.add_argument('--kge_frozen', default=0, type=int)
    parser.add_argument('--lm', default='lstm', type=str, choices=['lstm', 'bert', 'roberta', 'sbert', 't5','sbert2'])
    parser.add_argument('--lm_frozen', default=1, type=int)

    # dimensions, layers, dropout
    parser.add_argument('--entity_dim', default=100, type=int)
    parser.add_argument('--kg_dim', default=100, type=int)
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--lm_dropout', default=0.3, type=float)
    parser.add_argument('--linear_dropout', default=0.2, type=float)

    # optimization
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--fact_scale', default=3, type=int)
    parser.add_argument('--eval_every', default=2, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--gradient_clip', default=1.0, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--decay_rate', default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr_schedule', action='store_true')
    parser.add_argument('--label_smooth', default=0.1, type=float)
    parser.add_argument('--fact_drop', default=0, type=float)
    parser.add_argument('--encode_type', action='store_true')

    # model options
    parser.add_argument('--is_eval', action='store_true')
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--load_experiment', default=None, type=str)
    parser.add_argument('--load_ckpt_file', default=None, type=str)
    parser.add_argument('--eps', default=0.95, type=float) # threshold for f1
    parser.add_argument('--test_batch_size', default=20, type=int)
    parser.add_argument('--q_type', default='seq', type=str)



def add_parse_args(parser):
    
    subparsers = parser.add_subparsers(help='Neural Tree Search')

    parser = subparsers.add_parser("NuTrea")
    create_parser_nutrea(parser)


def create_parser_nutrea(parser):

    parser.add_argument('--model_name', default='NuTrea', type=str)
    parser.add_argument('--num_iter', default=2, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_expansion_ins', default=3, type=int)
    parser.add_argument('--num_backup_ins', default=2, type=int)
    parser.add_argument('--backup_depth', default=2, type=int)
    parser.add_argument('--loss_type', default='kl', type=str)
    parser.add_argument('--use_self_loop', default=True, type=bool)
    parser.add_argument('--normalized_gnn', default=False, type=bool)
    parser.add_argument('--data_eff', action='store_true')
    parser.add_argument('--context_coef', default=1.0, type=float)
    parser.add_argument('--optimizer', type=str, default='radam')
    parser.add_argument('--rf_ief', action='store_true')
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--log_base', default=None, type=float)
    parser.add_argument('--pos_emb', action='store_true')
    parser.add_argument('--debug', action='store_true')


    add_shared_args(parser)
