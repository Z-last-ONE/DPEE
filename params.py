import random
import argparse
import torch
import os
import numpy as np


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(str):
    return True if str.lower() == 'true' else False


def parse_args(task='FewFC'):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    if task=='FewFC':
        parser.add_argument("--data_path", type=str, default='datasets/FewFC', help="Path of the dataset.")
        parser.add_argument("--test_path", type=str, default='datasets/FewFC/data/test.json', help="Path of the testset.")
    if task=='FNDEE':
        parser.add_argument("--data_path", type=str, default='datasets/FNDEE', help="Path of the dataset.")
        parser.add_argument("--test_path", type=str, default='datasets/FNDEE/data/test.json',help="Path of the testset.")

    parser.add_argument("--do_train", default=True, type=str2bool)
    parser.add_argument("--do_eval", default=False, type=str2bool)
    parser.add_argument("--do_test", default=True, type=str2bool)

    parser.add_argument("--output_result_path", type=str, default='models_save/results.txt.json')
    parser.add_argument("--output_model_path", default="./models_save/", type=str,
                        help="Path of the output model.")
    parser.add_argument("--best_model_path", default="./models_save/epoch:13-2024-08-22-13-35-1724333721/model.bin", type=str,
                        help="Path of the output model.")

    parser.add_argument("--model_name_or_path", default="/home/mengfanshen/bert-base-chinese/", type=str,
                        help="Path of the output model.")
    parser.add_argument("--cache_dir", default="./plm", type=str,
                        help="Where do you want to store the pre-trained models downloaded")
    parser.add_argument("--do_lower_case", action="store_true", help="")
    parser.add_argument("--seq_length", default=512, type=int, help="Sequence length.")

    parser.add_argument("--device_id", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=2333, help="Random seed.")
    parser.add_argument("--lr_bert", type=float, default=2e-5, help="Learning rate for BERT.")
    parser.add_argument("--lr_task", type=float, default=1e-4, help="Learning rate for task layers.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch_size.")
    if task=='FewFC':
        parser.add_argument("--epochs_num", type=int, default=20, help="Number of epochs.")
    if task=='FNDEE':
        parser.add_argument("--epochs_num", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=50, help="Specific steps to print prompt.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay value")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout on BERT")
    parser.add_argument("--decoder_dropout", type=float, default=0.3, help="Dropout on decoders")

    # Model options.
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--w3", type=float, default=1.0)
    parser.add_argument("--w4", type=float, default=1.0)
    parser.add_argument("--pow_0", type=int, default=1)
    parser.add_argument("--pow_1", type=int, default=1)
    parser.add_argument("--pow_2", type=int, default=1)
    parser.add_argument("--pow_3", type=int, default=1)

    parser.add_argument("--rp_size", type=int, default=64)
    parser.add_argument("--decoder_num_head", type=int, default=1)

    parser.add_argument("--threshold_0", type=float, default=0.5)
    parser.add_argument("--threshold_1", type=float, default=0.5)
    parser.add_argument("--threshold_2", type=float, default=0.5)
    parser.add_argument("--threshold_3", type=float, default=0.5)
    parser.add_argument("--threshold_4", type=float, default=0.5)

    parser.add_argument("--step", type=str, choices=["dev", "test"])

    args = parser.parse_args()

    seed_everything(args.seed)
    return args
