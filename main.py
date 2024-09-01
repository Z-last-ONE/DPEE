# -*- encoding:utf -*-
import datetime
import json

from tqdm import tqdm

from torch.utils.data import DataLoader
from params import parse_args
from models.model import DPEE
from sklearn.metrics import *
import transformers
from framework import Framework
from utils.data_loader import get_dict, collate_fn_dev, collate_fn_train, collate_fn_test, Data
import torch
import os
from utils.metric import gen_idx_event_dict
from utils.tsas_eval import match_event
from utils.utils_io_data import read_jsonl, write_jsonl
import logging
from transformers import BertConfig, BertTokenizer, BertModel
# from models.modeling_bert import BertModel

log = logging.getLogger()
log_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%s")
logging.basicConfig(filename='./logs/log-' + log_name + '.txt', level=logging.ERROR)


def main():
    if not os.path.exists('plm'):
        os.makedirs('plm')
    if not os.path.exists('models_save'):
        os.makedirs('models_save')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    config = parse_args()
    log.error("configs: {}".format(config))
    print("configs: {}".format(config))
    config.type_id, config.id_type, config.args_id, config.id_args, config.ty_args, config.ty_args_id, config.args2id = get_dict(
        config.data_path)

    config.args_num = len(config.args2id.keys())
    config.type_num = len(config.type_id.keys())
    device = torch.device("cuda:" + str(config.device_id))
    config.device = device
    config.model_type = 'bert'

    config_plm = BertConfig.from_pretrained(config.model_name_or_path)
    config.hidden_size = config_plm.hidden_size
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    model_weight = BertModel.from_pretrained(config.model_name_or_path,
                                             from_tf=bool('.ckpt' in config.model_name_or_path))

    model = DPEE(config, model_weight, pos_emb_size=config.rp_size)
    framework = Framework(config, model)

    if config.do_train:
        train_set = Data(task='train', fn=config.data_path + '/cascading_sampled/train.json', tokenizer=tokenizer,
                         seq_len=config.seq_length, args2id=config.args2id, type_id=config.type_id)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_train)
        dev_set = Data(task='eval_with_oracle', fn=config.data_path + '/cascading_sampled/dev.json',
                       tokenizer=tokenizer, seq_len=config.seq_length, args2id=config.args2id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_dev)
        test_set = Data(task='eval_without_oracle', fn=config.test_path, tokenizer=tokenizer, seq_len=config.seq_length,
                        args2id=config.args2id, type_id=config.type_id)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn_test)

        framework.train(train_loader, dev_loader, test_loader)

    if config.do_eval:
        framework.load_model(config.best_model_path)
        log.error("Dev set evaluation with oracle.")
        dev_set = Data(task='eval_with_oracle', fn=config.data_path + '/cascading_sampled/dev.json',
                       tokenizer=tokenizer, seq_len=config.seq_length, args2id=config.args2id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_dev)
        c_ps, c_rs, c_fs, t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = framework.evaluate_with_oracle(config, model, dev_loader,
                                                                                              config.device,
                                                                                              config.ty_args_id,
                                                                                              config.id_type)
        f1_mean_all = (c_fs + t_fs + a_fs) / 3
        log.error('Evaluate on all types:')
        log.error("Type P: {:.3f}, Type R: {:.3f}, Type F: {:.3f}".format(c_ps, c_rs, c_fs))
        log.error("Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(t_ps, t_rs, t_fs))
        log.error("Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(a_ps, a_rs, a_fs))
        log.error("F1 Mean All: {:.3f}".format(f1_mean_all))
        log.error(
            "-----------------------------------------------------------------------------------------------------")
        log.error("Test set evaluation with oracle.")
        test_set = Data(task='eval_with_oracle', fn=config.data_path + '/cascading_sampled/test.json',
                        tokenizer=tokenizer, seq_len=config.seq_length, args2id=config.args2id, type_id=config.type_id)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn_dev)
        c_ps, c_rs, c_fs, t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = framework.evaluate_with_oracle(config, model,
                                                                                              test_loader,
                                                                                              config.device,
                                                                                              config.ty_args_id,
                                                                                              config.id_type)
        f1_mean_all = (c_fs + t_fs + a_fs) / 3
        log.error('Evaluate on all types:')
        log.error("Type P: {:.3f}, Type R: {:.3f}, Type F: {:.3f}".format(c_ps, c_rs, c_fs))
        log.error("Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(t_ps, t_rs, t_fs))
        log.error("Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(a_ps, a_rs, a_fs))
        log.error("F1 Mean All: {:.3f}".format(f1_mean_all))

    def load_data(file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        records = []
        for line in lines:
            record = json.loads(line)
            records.append(record)
        return records

    if config.do_test:
        # for i in range(200):
        #     for j in range(200):
        #         print('i:', i)
        #         print('j:', j)
        if config.batch_size != 1:
            log.error('For simplicity, reset batch_size=1 to extract each sentence')
            config.batch_size = 1
        framework.load_model(config.best_model_path)
        # Evaluation on test set given oracle predictions.
        log.error("Test set evaluation.")
        print("Test set evaluation.")
        test_set = Data(task='eval_without_oracle', fn=config.test_path, tokenizer=tokenizer,
                        seq_len=config.seq_length,
                        args2id=config.args2id, type_id=config.type_id)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn_test)
        log.error("The number of testing instances:{}".format(len(test_set)))
        print("The number of testing instances:{}".format(len(test_set)))
        ti_p, ti_r, ti_f1, tc_p, tc_r, tc_f1, ai_p, ai_r, ai_f1, ac_p, ac_r, ac_f1, event_result_list = framework.evaluate_without_oracle(
            config, model, test_loader, config.device, config.seq_length, config.id_type, config.id_args,
            config.ty_args_id)
        log.error('TI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ti_p * 100, ti_r * 100, ti_f1 * 100))
        log.error('TC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(tc_p * 100, tc_r * 100, tc_f1 * 100))
        log.error('AI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ai_p * 100, ai_r * 100, ai_f1 * 100))
        log.error('AC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ac_p * 100, ac_r * 100, ac_f1 * 100))
        print('TI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ti_p * 100, ti_r * 100, ti_f1 * 100))
        print('TC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(tc_p * 100, tc_r * 100, tc_f1 * 100))
        print('AI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ai_p * 100, ai_r * 100, ai_f1 * 100))
        print('AC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ac_p * 100, ac_r * 100, ac_f1 * 100))
        if 'FewFC' in config.data_path:
            gold_list = load_data('./datasets/FewFC/data/test.json')
        else:
            gold_list = load_data('./datasets/FNDEE/data/test.json')
        match_event(gold_list, event_result_list)


if __name__ == '__main__':
    main()
