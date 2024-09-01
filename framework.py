import datetime
import logging
import os
import time
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils import predict_with_oracle
from utils.predict_without_oracle import extract_all_items_without_oracle
from utils.tsas_eval import predict_events, extract_all_items_without_oracle_new
from utils.utils_io_model import load_model, save_model
import torch
import numpy as np
from sklearn.metrics import *
from utils.predict_with_oracle import predict_one
from tqdm import tqdm
from utils.metric import score, gen_idx_event_dict, cal_scores, cal_scores_ti_tc_ai_ac, calculate_f1
from utils.utils_io_data import read_jsonl, write_jsonl

log = logging.getLogger()


class Framework(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model.to(config.device)

    def load_model(self, model_path):
        self.model = load_model(self.model, model_path)

    def set_learning_setting(self, config, train_loader, dev_loader, model, test_loader):
        instances_num = len(train_loader.dataset)
        train_steps = int(instances_num * config.epochs_num / config.batch_size) + 1

        log.error("Batch size: {}".format(config.batch_size))
        log.error("The number of training instances: {}".format(instances_num))
        log.error("The number of evaluating instances: {}".format(len(dev_loader.dataset)))
        log.error("The number of test instances: {}".format(len(test_loader.dataset)))
        print("Batch size: {}".format(config.batch_size))
        print("The number of training instances: {}".format(instances_num))
        print("The number of evaluating instances: {}".format(len(dev_loader.dataset)))
        print("The number of test instances: {}".format(len(test_loader.dataset)))

        bert_params = list(map(id, model.bert.parameters()))

        other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        optimizer_grouped_parameters = [{'params': model.bert.parameters()},
                                        {'params': other_params, 'lr': config.lr_task}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr_bert, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps * config.warmup,
                                                    num_training_steps=train_steps)

        # if torch.cuda.device_count() > 1:
        #     log.error("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        #     self.model = torch.nn.DataParallel(model)

        return scheduler, optimizer

    def train(self, train_loader, dev_loader, test_loader):
        scheduler, optimizer = self.set_learning_setting(self.config, train_loader, dev_loader, self.model, test_loader)
        # going to train
        total_loss = 0.0
        ed_loss = 0.0
        te_loss = 0.0
        ae_loss = 0.0
        t2a_loss = 0.0
        ro_loss = 0.0
        best_f1 = 0.0
        best_epoch = 0
        for epoch in range(1, self.config.epochs_num + 1):
            log.error('Training...')
            print('Training...')
            self.model.train()
            for i, (
                    idx, d_t, t_v, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e,
                    role_labels, tri2arg_labels,
                    word_mask1d, word_mask2d, triu_mask2d,prompt) in enumerate(train_loader):
                self.model.zero_grad()
                d_t = torch.LongTensor(d_t).to(self.config.device)
                t_v = torch.FloatTensor(np.asarray(t_v)).to(self.config.device)
                token = torch.LongTensor(token).to(self.config.device)
                seg = torch.LongTensor(seg).to(self.config.device)
                mask = torch.LongTensor(mask).to(self.config.device)
                t_l = torch.BoolTensor(trigger_labels).to(self.config.device)
                a_l = torch.BoolTensor(argument_labels).to(self.config.device)
                t_s = torch.FloatTensor(np.asarray(t_s)).to(self.config.device)
                t_e = torch.FloatTensor(np.asarray(t_e)).to(self.config.device)
                a_s = torch.FloatTensor(np.asarray(a_s)).to(self.config.device)
                a_e = torch.FloatTensor(np.asarray(a_e)).to(self.config.device)
                role_labels = torch.FloatTensor(np.asarray(role_labels)).to(self.config.device)
                tri2arg_labels = torch.FloatTensor(np.asarray(tri2arg_labels)).to(self.config.device)
                word_mask1d = torch.BoolTensor(word_mask1d).to(self.config.device)
                word_mask2d = torch.BoolTensor(word_mask2d).to(self.config.device)
                triu_mask2d = torch.BoolTensor(triu_mask2d).to(self.config.device)
                prompt = torch.stack(prompt).to(self.config.device)

                loss, type_loss, trigger_loss, args_loss, trigger2ara_loss, role_loss = self.model(token, seg, mask,
                                                                                                   d_t, t_v, t_l, t_s,
                                                                                                   t_e, a_l,
                                                                                                   a_s, a_e,
                                                                                                   role_labels,
                                                                                                   tri2arg_labels,
                                                                                                   word_mask1d,
                                                                                                   word_mask2d,
                                                                                                   triu_mask2d,prompt)
                # if torch.cuda.device_count() > 1:
                #     loss = torch.mean(loss)
                #     type_loss = torch.mean(type_loss)
                #     trigger_loss = torch.mean(trigger_loss)
                #     args_loss = torch.mean(args_loss)
                #     role_loss = torch.mean(role_loss)

                total_loss += loss.item()
                ed_loss += type_loss.item()
                te_loss += trigger_loss.item()
                ae_loss += args_loss.item()
                t2a_loss += trigger2ara_loss.item()
                ro_loss += role_loss.item()

                if (i + 1) % self.config.report_steps == 0:
                    log.error(
                        "Epoch id: {}, Training steps: {}, ED loss:{:.6f},TE loss:{:.6f}, AE loss:{:.6f}, T2A loss:{:.6f}, Ro loss:{:.6f}, Avg loss: {:.6f}".format(
                            epoch, i + 1, ed_loss / self.config.report_steps, te_loss / self.config.report_steps,
                                   ae_loss / self.config.report_steps, t2a_loss / self.config.report_steps,
                                   ro_loss / self.config.report_steps,
                                   total_loss / self.config.report_steps))
                    print(
                        "Epoch id: {}, Training steps: {}, ED loss:{:.6f},TE loss:{:.6f}, AE loss:{:.6f}, T2A loss:{:.6f}, Ro loss:{:.6f}, Avg loss: {:.6f}".format(
                            epoch, i + 1, ed_loss / self.config.report_steps, te_loss / self.config.report_steps,
                                   ae_loss / self.config.report_steps, t2a_loss / self.config.report_steps,
                                   ro_loss / self.config.report_steps,
                                   total_loss / self.config.report_steps))
                    total_loss = 0.0
                    ed_loss = 0.0
                    te_loss = 0.0
                    ae_loss = 0.0
                    t2a_loss = 0.0
                    ro_loss = 0.0

                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss = 0.0
            ed_loss = 0.0
            te_loss = 0.0
            ae_loss = 0.0
            t2a_loss = 0.0
            ro_loss = 0.0

            log.error('Evaluating...')
            print('Evaluating...')
            c_ps, c_rs, c_fs, t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = self.evaluate_with_oracle(self.config, self.model,
                                                                                             dev_loader,
                                                                                             self.config.device,
                                                                                             self.config.ty_args_id,
                                                                                             self.config.id_type)
            f1_mean_all = (c_fs + t_fs + a_fs) / 3
            log.error('Evaluate on all types:')
            print('Evaluate on all types:')
            log.error("Epoch id: {}, Type P: {:.4f}, Type R: {:.4f}, Type F: {:.4f}".format(epoch, c_ps, c_rs, c_fs))
            log.error(
                "Epoch id: {}, Trigger P: {:.4f}, Trigger R: {:.4f}, Trigger F: {:.4f}".format(epoch, t_ps, t_rs, t_fs))
            log.error("Epoch id: {}, Args P: {:.4f}, Args R: {:.4f}, Args F: {:.4f}".format(epoch, a_ps, a_rs, a_fs))
            log.error("Epoch id: {}, F1 Mean All: {:.4f}".format(epoch, f1_mean_all))
            print("Epoch id: {}, Type P: {:.4f}, Type R: {:.4f}, Type F: {:.4f}".format(epoch, c_ps, c_rs, c_fs))
            print(
                "Epoch id: {}, Trigger P: {:.4f}, Trigger R: {:.4f}, Trigger F: {:.4f}".format(epoch, t_ps, t_rs, t_fs))
            print("Epoch id: {}, Args P: {:.4f}, Args R: {:.4f}, Args F: {:.4f}".format(epoch, a_ps, a_rs, a_fs))
            print("Epoch id: {}, F1 Mean All: {:.4f}".format(epoch, f1_mean_all))

            if f1_mean_all > best_f1:
                best_f1 = f1_mean_all
                best_epoch = epoch
                path = os.path.join(self.config.output_model_path, 'epoch:' + str(epoch) +
                                    '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%s"))
                if not os.path.exists(path):
                    os.makedirs(path)
                save_model(self.model, os.path.join(path, 'model.bin'))
                self.config.best_model_path = os.path.join(path, 'model.bin')
            log.error("The Best F1 Is: {:.4f}, When Epoch Is: {}, Path Is: {}\n".format(best_f1, best_epoch,
                                                                                        self.config.best_model_path))
            print("The Best F1 Is: {:.4f}, When Epoch Is: {}, Path Is: {}\n".format(best_f1, best_epoch,
                                                                                    self.config.best_model_path))

            log.error("Eval on test set without oracle")
            ti_p, ti_r, ti_f1, tc_p, tc_r, tc_f1, ai_p, ai_r, ai_f1, ac_p, ac_r, ac_f1, event_result_list = self.evaluate_without_oracle(
                self.config, self.model, test_loader, self.config.device, self.config.seq_length, self.config.id_type,
                self.config.id_args, self.config.ty_args_id)
            log.error('TI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ti_p * 100, ti_r * 100, ti_f1 * 100))
            log.error('TC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(tc_p * 100, tc_r * 100, tc_f1 * 100))
            log.error('AI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ai_p * 100, ai_r * 100, ai_f1 * 100))
            log.error('AC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ac_p * 100, ac_r * 100, ac_f1 * 100))

            print("Eval on test set without oracle")
            ti_p, ti_r, ti_f1, tc_p, tc_r, tc_f1, ai_p, ai_r, ai_f1, ac_p, ac_r, ac_f1, event_result_list = self.evaluate_without_oracle(
                self.config, self.model, test_loader, self.config.device, self.config.seq_length, self.config.id_type,
                self.config.id_args, self.config.ty_args_id)
            print('TI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ti_p * 100, ti_r * 100, ti_f1 * 100))
            print('TC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(tc_p * 100, tc_r * 100, tc_f1 * 100))
            print('AI: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ai_p * 100, ai_r * 100, ai_f1 * 100))
            print('AC: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(ac_p * 100, ac_r * 100, ac_f1 * 100))

    def evaluate_with_oracle(self, config, model, dev_data_loader, device, ty_args_id, id2type):
        # if hasattr(model, "module"):
        #     model = model.module
        model.eval()
        type_pred_dict = {}
        type_truth_dict = {}
        num_ti_r = 0
        num_ti_p = 0
        num_ti_c = 0
        num_ac_r = 0
        num_ac_p = 0
        num_ac_c = 0
        args_pred_tuples_dict = {}
        args_truth_tuples_dict = {}
        for i, (
                idx, typ_oracle, typ_truth, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e,
                role_labels, tri2arg_labels, word_mask2d, triu_mask2d, tuples_ti, tuples_ac, t_t, a_t,prompt) in enumerate(
            dev_data_loader):
            typ_oracle = torch.LongTensor(typ_oracle).to(device)
            typ_truth = torch.FloatTensor(np.asarray(typ_truth)).to(device)
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            word_mask2d = torch.BoolTensor(word_mask2d).to(device)
            triu_mask2d = torch.BoolTensor(triu_mask2d).to(device)
            t_l = torch.BoolTensor(trigger_labels).to(device)
            a_l = torch.BoolTensor(argument_labels).to(device)
            t_s = torch.FloatTensor(np.asarray(t_s)).to(device)
            t_e = torch.FloatTensor(np.asarray(t_e)).to(device)
            a_s = torch.FloatTensor(np.asarray(a_s)).to(device)
            a_e = torch.FloatTensor(np.asarray(a_e)).to(device)
            role_labels = torch.BoolTensor(role_labels).to(device)
            tri2arg_labels = torch.BoolTensor(tri2arg_labels).to(device)
            prompt = torch.stack(prompt).to(self.config.device)

            type_pred, ti, args_pred_tuples, args_truth_tuples = predict_one(model, config, typ_oracle, token, seg,
                                                                             mask, ty_args_id, word_mask2d,
                                                                             triu_mask2d, a_t, config,prompt)
            type_truth = typ_truth.view(self.config.type_num).cpu().numpy().astype(int)

            idx = idx[0]

            ti_r, ti_p, ti_c = predict_with_oracle.decode('ti', tuples_ti, self.config.ty_args_id, ti=ti)
            # ac_r, ac_p, ac_c = predict_with_oracle.decode('ac', tuples_ac, self.config.ty_args_id, ai=ai, ac=ac,
            #                                               type_oracle=typ_oracle)

            # collect type predictions
            if idx not in type_pred_dict:
                type_pred_dict[idx] = type_pred
            if idx not in type_truth_dict:
                type_truth_dict[idx] = type_truth

            # collect argument predictions
            if idx not in args_pred_tuples_dict:
                args_pred_tuples_dict[idx] = []
            args_pred_tuples_dict[idx].extend(args_pred_tuples)
            if idx not in args_truth_tuples_dict:
                args_truth_tuples_dict[idx] = []
            args_truth_tuples_dict[idx].extend(args_truth_tuples)

            num_ti_r += ti_r
            num_ti_p += ti_p
            num_ti_c += ti_c
            num_ac_r += 0
            num_ac_p += 0
            num_ac_c += 0

        type_pred_s, type_truth_s = [], []
        for idx in type_truth_dict.keys():
            type_pred_s.append(type_pred_dict[idx])
            type_truth_s.append(type_truth_dict[idx])
        c_ps = precision_score(type_truth_s, type_pred_s, average='macro')
        c_rs = recall_score(type_truth_s, type_pred_s, average='macro')
        c_fs = f1_score(type_truth_s, type_pred_s, average='macro')
        ti_f1, ti_r, ti_p = calculate_f1(num_ti_r, num_ti_p, num_ti_c)
        ac_f1, ac_r, ac_p = calculate_f1(num_ac_r, num_ac_p, num_ac_c)
        a_p, a_r, a_f = score(args_pred_tuples_dict, args_truth_tuples_dict)
        return c_ps, c_rs, c_fs, ti_p, ti_r, ti_f1, a_p, a_r, a_f



    def evaluate_without_oracle(self, config, model, data_loader, device, seq_len, id_type, id_args, ty_args_id
                                ):
        # if torch.cuda.device_count() > 1:
        #     model = model.module
        model.eval()
        event_result_list = []

        ti_r, ti_p, ti_c = 0, 0, 0
        tc_r, tc_p, tc_c = 0, 0, 0
        ai_r, ai_p, ai_c = 0, 0, 0
        ac_r, ac_p, ac_c = 0, 0, 0
        for i, (idx, content, token, seg, mask, word_mask1d, word_mask2d, triu_mask2d, ti_tuple, tc_tuple, ai_tuple,
                ac_tuple,prompt) in enumerate(data_loader):
            idx = idx[0]
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            word_mask2d = torch.BoolTensor(word_mask2d).to(device)
            triu_mask2d = torch.BoolTensor(triu_mask2d).to(device)
            prompt = torch.stack(prompt).to(self.config.device)

            result = extract_all_items_without_oracle(model, device, idx, content, token, seg, mask, seq_len,
                                                      id_type, id_args, ty_args_id, word_mask2d, ti_tuple, tc_tuple,
                                                      ai_tuple, ac_tuple, config, triu_mask2d,prompt)
            event = extract_all_items_without_oracle_new(model, device, idx, content, token, seg, mask, seq_len,
                                                         id_type, id_args, ty_args_id, word_mask2d, ti_tuple, tc_tuple,
                                                         ai_tuple, ac_tuple, config, triu_mask2d,prompt)
            for e in event:
                event_result_list.append(e)
            _ti_r, _ti_p, _ti_c = result['ti']
            _tc_r, _tc_p, _tc_c = result['tc']
            _ai_r, _ai_p, _ai_c = result['ai']
            _ac_r, _ac_p, _ac_c = result['ac']
            ti_r += _ti_r
            ti_p += _ti_p
            ti_c += _ti_c

            tc_r += _tc_r
            tc_p += _tc_p
            tc_c += _tc_c

            ai_r += _ai_r
            ai_p += _ai_p
            ai_c += _ai_c

            ac_r += _ac_r
            ac_p += _ac_p
            ac_c += _ac_c
        ti_f1, ti_r, ti_p = calculate_f1(ti_r, ti_p, ti_c)
        tc_f1, tc_r, tc_p = calculate_f1(tc_r, tc_p, tc_c)
        ai_f1, ai_r, ai_p = calculate_f1(ai_r, ai_p, ai_c)
        ac_f1, ac_r, ac_p = calculate_f1(ac_r, ac_p, ac_c)
        return ti_p, ti_r, ti_f1, tc_p, tc_r, tc_f1, ai_p, ai_r, ai_f1, ac_p, ac_r, ac_f1, event_result_list
