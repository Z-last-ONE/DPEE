import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils_padding import pad_2d, pad_3d

prompt_url='./prompt.npy'
def get_dict(fn):
    with open(fn + '/cascading_sampled/ty_args.json', 'r', encoding='utf-8') as f:
        ty_args = json.load(f)
    if not os.path.exists(fn + '/cascading_sampled/shared_args_list.json'):
        args_list = set()
        for ty in ty_args:
            for arg in ty_args[ty]:
                args_list.add(arg)
        args_list = list(args_list)
        with open(fn + '/cascading_sampled/shared_args_list.json', 'w', encoding='utf-8') as f:
            json.dump(args_list, f, ensure_ascii=False)
    else:
        with open(fn + '/cascading_sampled/shared_args_list.json', 'r', encoding='utf-8') as f:
            args_list = json.load(f)

    args2id = {}
    for i in range(len(args_list)):
        s = args_list[i]
        args2id[s] = i
    id_type = {i: item for i, item in enumerate(ty_args)}
    type_id = {item: i for i, item in enumerate(ty_args)}

    id_args = {i: item for i, item in enumerate(args_list)}
    args_id = {item: i for i, item in enumerate(args_list)}
    ty_args_id = {}
    for ty in ty_args:
        args = ty_args[ty]
        tmp = [args_id[a] for a in args]
        ty_args_id[type_id[ty]] = tmp
    return type_id, id_type, args_id, id_args, ty_args, ty_args_id, args2id


def read_labeled_data(fn):
    loaded_prompt_tensor_dict = np.load(prompt_url, allow_pickle=True).item()
    ''' Read Train Data / Dev Data '''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    data_type = []
    data_occur = []
    data_triggers = []
    data_args = []
    data_events = []
    prompt = []
    for line in lines:

        line_dict = json.loads(line.strip())
        data_ids.append(line_dict.get('id', 0))
        prompt.append(loaded_prompt_tensor_dict[line_dict.get('id', 0)])
        data_occur.append(line_dict['occur'])
        data_type.append(line_dict['type'])
        data_content.append(line_dict['content'])
        data_triggers.append(line_dict['triggers'])
        data_events.append(line_dict['events'])
        data_arg = {}
        for event in line_dict['events']:
            args = event['args']
            for k, v in args.items():
                if k not in data_arg.keys():
                    data_arg[k] = v.copy()
                else:
                    temp = data_arg[k]
                    for o in v:
                        if o not in temp: temp.append(o)
                    data_arg[k] = temp
        data_args.append(data_arg)
    return data_ids, data_occur, data_type, data_content, data_triggers, data_args, data_events, prompt


def read_unlabeled_data(fn):
    ''' Read Test Data'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict['id'])
        data_content.append(line_dict['content'])
    return data_ids, data_content


class Data(Dataset):
    def __init__(self, task, fn, tokenizer=None, seq_len=None, args2id=None, type_id=None):
        assert task in ['train', 'eval_with_oracle', 'eval_without_oracle']
        self.task = task
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.args2id = args2id
        self.args_num = len(args2id.keys())
        self.type_id = type_id
        self.type_num = len(type_id.keys())
        self.loaded_prompt_tensor_dict = np.load(prompt_url, allow_pickle=True).item()

        if self.task == 'eval_without_oracle':
            data_ids, data_content = read_unlabeled_data(fn)
            self.data_ids = data_ids
            self.data_content = data_content
            data_ids, token_ids, segs_ids, masks_ids, word_mask1d, word_mask2d, triu_mask2d, ti_tuple, tc_tuple, ai_tuple, ac_tuple, prompt = self.read_labeled_data_test(
                fn)

            self.token = token_ids
            self.seg = segs_ids
            self.mask = masks_ids
            self.word_mask1d = word_mask1d
            self.word_mask2d = word_mask2d
            self.triu_mask2d = triu_mask2d
            self.ti_tuple = list(ti_tuple)
            self.tc_tuple = list(tc_tuple)
            self.ai_tuple = list(ai_tuple)
            self.ac_tuple = list(ac_tuple)
            self.prompt = prompt


        else:
            data_ids, data_occur, data_type, data_content, data_triggers, data_args, data_events, _ = read_labeled_data(
                fn)
            self.data_ids = data_ids
            self.data_occur = data_occur
            self.data_triggers = data_triggers
            self.data_content = data_content
            self.data_events = data_events
            self.data_args = data_args
            triggers_truth_s, args_truth_s = self.results_for_eval()
            self.triggers_truth = triggers_truth_s
            self.args_truth = args_truth_s

            tokens_ids, segs_ids, masks_ids, data_type_id_s, type_vec_s, trigger_labels, t_s, t_e, argument_labels, a_s, a_e, role_labels, tri2arg_labels, \
            word_mask1d, word_mask2d, triu_mask2d, tuples_ti, tuples_ai, tuples_ac, prompt = self.data_to_id_train(
                data_content, data_type, data_occur, data_triggers, data_args, data_events)
            self.data_type_id_s = data_type_id_s
            self.type_vec_s = type_vec_s
            self.token = tokens_ids
            self.seg = segs_ids
            self.mask = masks_ids
            self.trigger_labels = trigger_labels
            self.argument_labels = argument_labels
            self.t_s = t_s
            self.t_e = t_e
            self.a_s = a_s
            self.a_e = a_e
            self.role_labels = role_labels
            self.tri2arg_labels = tri2arg_labels
            self.word_mask1d = word_mask1d
            self.word_mask2d = word_mask2d
            self.triu_mask2d = triu_mask2d
            self.tuples_ti = tuples_ti
            self.tuples_ai = tuples_ai
            self.tuples_ac = tuples_ac
            self.prompt = prompt

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        if self.task == 'train':
            return self.data_ids[index], \
                   self.data_type_id_s[index], \
                   self.type_vec_s[index], \
                   torch.LongTensor(self.token[index]), \
                   torch.LongTensor(self.seg[index]), \
                   torch.LongTensor(self.mask[index]), \
                   torch.BoolTensor(self.trigger_labels[index]), \
                   torch.BoolTensor(self.t_s[index]), \
                   torch.BoolTensor(self.t_e[index]), \
                   torch.BoolTensor(self.argument_labels[index]), \
                   torch.BoolTensor(self.a_s[index]), \
                   torch.BoolTensor(self.a_e[index]), \
                   torch.BoolTensor(self.role_labels[index]), \
                   torch.BoolTensor(self.tri2arg_labels[index]), \
                   torch.BoolTensor(self.word_mask1d[index]), \
                   torch.BoolTensor(self.word_mask2d[index]), \
                   torch.BoolTensor(self.triu_mask2d[index]), \
                   self.prompt[index]

        elif self.task == 'eval_with_oracle':
            return self.data_ids[index], \
                   self.data_type_id_s[index], \
                   self.type_vec_s[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   torch.BoolTensor(self.trigger_labels[index]), \
                   torch.BoolTensor(self.t_s[index]), \
                   torch.BoolTensor(self.t_e[index]), \
                   torch.BoolTensor(self.argument_labels[index]), \
                   torch.BoolTensor(self.a_s[index]), \
                   torch.BoolTensor(self.a_e[index]), \
                   torch.BoolTensor(self.role_labels[index]), \
                   torch.BoolTensor(self.tri2arg_labels[index]), \
                   torch.BoolTensor(self.word_mask1d[index]), \
                   torch.BoolTensor(self.word_mask2d[index]), \
                   torch.BoolTensor(self.triu_mask2d[index]), \
                   self.tuples_ti[index], \
                   self.tuples_ac[index], \
                   self.triggers_truth[index], \
                   self.args_truth[index], \
                   self.prompt[index]
        elif self.task == 'eval_without_oracle':
            return self.data_ids[index], \
                   self.data_content[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   torch.BoolTensor(self.word_mask1d[index]), \
                   torch.BoolTensor(self.word_mask2d[index]), \
                   torch.BoolTensor(self.triu_mask2d[index]), \
                   self.ti_tuple[index], \
                   self.tc_tuple[index], \
                   self.ai_tuple[index], \
                   self.ac_tuple[index], \
                   self.prompt[index]
        else:
            raise Exception('task not define !')

    def data_to_id_train(self, data_contents, data_type, data_occur, data_triggers, data_args, data_events):
        tokens_ids, segs_ids, masks_ids = [], [], []
        data_type_id_s, type_vec_s = [], []
        trigger_labels, t_s_list, t_e_list, argument_labels, a_s_list, a_e_list, role_labels, tri2arg_labels = [], [], [], [], [], [], [], []
        word_mask1d, word_mask2d, triu_mask2d = [], [], []
        tuples_ti, tuples_ai, tuples_ac = [], [], []
        prompt = []
        for i, id in enumerate(self.data_ids):
            prompt.append(self.loaded_prompt_tensor_dict[id])
            data_content = data_contents[i]
            data_content = [token.lower() for token in data_content]
            data_content = list(data_content)
            inputs = self.tokenizer.encode_plus(data_content, add_special_tokens=True, max_length=self.seq_len,
                                                truncation=True)
            tokens, segs, masks = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']

            data_type_id = self.type_id[data_type[i]]
            type_vec = np.array([0] * self.type_num)
            for occ in data_occur[i]:
                idx = self.type_id[occ]
                type_vec[idx] = 1
            length = len(tokens)
            trigger_label, t_s, t_e, ti = self.get_trigger_labels(data_triggers[i], length)
            argument_label, a_s, a_e, ai, ac = self.get_argument_labels(data_args[i], length)
            role_label = self.get_role_labels(data_triggers[i], data_events[i], length)
            _tri2arg_label = self.get_tri2arg_labels(data_triggers[i], data_events[i], length)
            tuples_ti.append(ti)
            tuples_ai.append(ai)
            tuples_ac.append(ac)
            _word_mask1d = np.array([1] * length)
            _word_mask2d = np.triu(np.ones((length, length), dtype=bool))
            _triu_mask2d = np.ones((length, length), dtype=bool)

            tokens_ids.append(tokens)
            segs_ids.append(segs)
            masks_ids.append(masks)
            data_type_id_s.append(data_type_id)
            type_vec_s.append(type_vec)
            trigger_labels.append(trigger_label)
            t_s_list.append(t_s)
            t_e_list.append(t_e)
            argument_labels.append(argument_label)
            a_s_list.append(a_s)
            a_e_list.append(a_e)
            role_labels.append(role_label)
            tri2arg_labels.append(_tri2arg_label)
            word_mask1d.append(_word_mask1d)
            word_mask2d.append(_word_mask2d)
            triu_mask2d.append(_triu_mask2d)

        return tokens_ids, segs_ids, masks_ids, data_type_id_s, type_vec_s, trigger_labels, t_s_list, t_e_list, argument_labels, a_s_list, a_e_list, role_labels, \
               tri2arg_labels, word_mask1d, word_mask2d, triu_mask2d, tuples_ti, tuples_ai, tuples_ac, prompt

    def get_trigger_labels(self, triggers_one, length):
        tuples_ti = set()
        _tri_label = np.zeros((length, length), dtype=bool)
        _t_s = np.zeros((length), dtype=bool)
        _t_e = np.zeros((length), dtype=bool)
        for trigger in triggers_one:
            t_s = trigger[0]
            t_e = trigger[1]
            _tri_label[t_s + 1, t_e + 1 - 1] = 1
            _t_s[t_s + 1] = 1
            _t_e[t_e] = 1
            tuples_ti.add((t_s + 1, t_e))
        return _tri_label, _t_s, _t_e, tuples_ti

    def get_argument_labels(self, argument_one, length):
        tuples_ai, tuples_ac = set(), set()
        _arg_label = np.zeros((length, length), dtype=bool)
        _a_s = np.zeros((self.args_num, length), dtype=bool)
        _a_e = np.zeros((self.args_num, length), dtype=bool)

        for args_k, args_v in argument_one.items():
            for arg_v in args_v:
                a_s = arg_v[0]
                a_e = arg_v[1]
                _arg_label[a_s + 1, a_e + 1 - 1] = 1
                _a_s[self.args2id[args_k], a_s + 1] = 1
                _a_e[self.args2id[args_k], a_e] = 1
                tuples_ai.add((a_s + 1, a_e))
                tuples_ac.add((a_s + 1, a_e, self.args2id[args_k]))
        return _arg_label, _a_s, _a_e, tuples_ai, tuples_ac

    def get_role_labels(self, data_triggers, data_events, length):
        _role_label = np.zeros((length, length), dtype=bool)
        for i, (trigger) in enumerate(data_triggers):
            for data_event in data_events:
                if data_event['trigger_idx'] == i:
                    args = data_event['args']
                    for _, v in args.items():
                        for arg in v:
                            a_s = arg[0]
                            a_e = arg[1]
                            # '''
                            for _, v2 in args.items():
                                for arg2 in v2:
                                    if arg[0] != arg2[0] and arg[1] != arg2[1]:
                                        a_s2 = arg2[0]
                                        a_e2 = arg2[1]
                                        _role_label[a_s + 1:a_e + 1, a_s2 + 1:a_e2 + 1] = 1
                                        _role_label[a_s2 + 1:a_e2 + 1, a_s + 1:a_e + 1] = 1
                            # '''
        return _role_label

    def get_tri2arg_labels(self, data_triggers, data_events, length):
        _tri2arg_label = np.zeros((length, length), dtype=bool)
        for i, (trigger) in enumerate(data_triggers):
            t_s = trigger[0]
            t_e = trigger[1]
            for data_event in data_events:
                if data_event['trigger_idx'] == i:
                    args = data_event['args']
                    for _, v in args.items():
                        for arg in v:
                            a_s = arg[0]
                            a_e = arg[1]
                            _tri2arg_label[t_s + 1:t_e + 1, a_s + 1:a_e + 1] = 1
                            '''
                            for _, v2 in args.items():
                                for arg2 in v2:
                                    if arg[0] != arg2[0] and arg[1] != arg2[1]:
                                        a_s2 = arg2[0]
                                        a_e2 = arg2[1]
                                        _tri2arg_label[a_s + 1:a_e + 1, a_s2 + 1:a_e2 + 1] = 1
                                        _tri2arg_label[a_s2 + 1:a_e2 + 1, a_s + 1:a_e + 1] = 1
                            # '''
        return _tri2arg_label

    # self.token = tokens_ids
    # self.seg = segs_ids
    # self.mask = masks_ids
    # self.word_mask1d = word_mask1d
    # self.word_mask2d = word_mask2d
    # self.triu_mask2d = triu_mask2d
    # self.ti_tuple = list(ti_tuple)
    # self.tc_tuple = list(tc_tuple)
    # self.ai_tuple = list(ai_tuple)
    # self.ac_tuple = list(ac_tuple)

    def read_labeled_data_test(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data_ids = []
        token_ids = []
        segs_ids = []
        masks_ids = []
        prompt = []
        word_mask1d, word_mask2d, triu_mask2d = [], [], []
        ti_tuple, tc_tuple, ai_tuple, ac_tuple = [], [], [], []

        for line in lines:
            line_dict = json.loads(line.strip())
            data_ids.append(line_dict.get('id', 0))
            prompt.append(self.loaded_prompt_tensor_dict[line_dict.get('id', 0)])
            data_content = line_dict.get('content')
            events = line_dict.get('events')
            data_content = [token.lower() for token in data_content]
            data_content = list(data_content)
            inputs = self.tokenizer.encode_plus(data_content, add_special_tokens=True, max_length=self.seq_len,
                                                truncation=True)
            tokens, segs, masks = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']

            length = len(tokens)
            _word_mask1d = np.array([1] * length)
            _word_mask2d = np.triu(np.ones((length, length), dtype=bool))
            _triu_mask2d = np.ones((length, length), dtype=bool)
            _ti_tuple, _tc_tuple, _ai_tuple, _ac_tuple = set(), set(), set(), set()
            for event in events:
                event_type = event['type']
                trigger = event['trigger']
                t_span = trigger['span']
                args = event['args']
                _ti_tuple.add((t_span[0] + 1, t_span[1]))
                _tc_tuple.add((t_span[0] + 1, t_span[1], self.type_id[event_type]))
                for k, v in args.items():
                    for arg in v:
                        a_span = arg['span']
                        _ai_tuple.add((a_span[0] + 1, a_span[1], self.type_id[event_type]))
                        _ac_tuple.add((a_span[0] + 1, a_span[1], self.type_id[event_type], self.args2id[k]))
            word_mask1d.append(_word_mask1d)
            word_mask2d.append(_word_mask2d)
            triu_mask2d.append(_triu_mask2d)
            token_ids.append(tokens)
            segs_ids.append(segs)
            masks_ids.append(masks)
            ti_tuple.append(_ti_tuple)
            tc_tuple.append(_tc_tuple)
            ai_tuple.append(_ai_tuple)
            ac_tuple.append(_ac_tuple)
        return data_ids, token_ids, segs_ids, masks_ids, word_mask1d, word_mask2d, triu_mask2d, ti_tuple, tc_tuple, ai_tuple, ac_tuple, prompt

    def data_to_id_test(self, data_contents, data_type, data_occur, data_triggers, data_args):
        tokens_ids, segs_ids, masks_ids = [], [], []
        word_mask1d, word_mask2d, triu_mask2d = [], [], []
        _ti_tuple, _tc_tuple, _ai_tuple, _ac_tuple = [], [], [], []
        for i in range(len(self.data_ids)):
            data_content = data_contents[i]
            data_content = [token.lower() for token in data_content]
            data_content = list(data_content)
            inputs = self.tokenizer.encode_plus(data_content, add_special_tokens=True, max_length=self.seq_len,
                                                truncation=True)
            tokens, segs, masks = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']
            d_t = self.type_id[data_type[i]]
            d_tri = data_triggers[i]
            d_a = data_args[i]
            ti_tuple, tc_tuple, ai_tuple, ac_tuple = set(), set(), set(), set()
            for trigger in d_tri:
                ti_tuple.add((trigger[0] + 1, trigger[1]))
                tc_tuple.add((trigger[0] + 1, trigger[1], d_t))
            for k, v in d_a.items():
                for a_span in v:
                    k_idx = self.args2id[k]
                    ai_tuple.add((a_span[0] + 1, a_span[1]))
                    ac_tuple.add((a_span[0] + 1, a_span[1], k_idx))
            _ti_tuple.append(ti_tuple)
            _tc_tuple.append(tc_tuple)
            _ai_tuple.append(ai_tuple)
            _ac_tuple.append(ac_tuple)
            length = len(tokens)
            _word_mask1d = np.array([1] * length)
            _word_mask2d = np.triu(np.ones((length, length), dtype=bool))
            _triu_mask2d = np.ones((length, length), dtype=bool)
            word_mask1d.append(_word_mask1d)
            word_mask2d.append(_word_mask2d)
            triu_mask2d.append(_triu_mask2d)
            tokens_ids.append(tokens)
            segs_ids.append(segs)
            masks_ids.append(masks)

        return tokens_ids, segs_ids, masks_ids, word_mask1d, word_mask2d, triu_mask2d, _ti_tuple, _tc_tuple, _ai_tuple, _ac_tuple

    def results_for_eval(self):
        '''
        read structured ground truth, for evaluating model performance
        '''
        triggers_truth_s = []
        args_truth_s = []
        for i in range(len(self.data_ids)):
            triggers = self.data_triggers[i]
            args = self.data_args[i]
            # plus 1 for additional <CLS> token
            triggers_truth = [(span[0] + 1, span[1] + 1 - 1) for span in triggers]
            args_truth = {i: [] for i in range(self.args_num)}
            for args_name in args:
                s_r_i = self.args2id[args_name]
                for span in args[args_name]:
                    # plus 1 for additional <CLS> token
                    args_truth[s_r_i].append((span[0] + 1, span[1] + 1 - 1))
            triggers_truth_s.append(triggers_truth)
            args_truth_s.append(args_truth)
        return triggers_truth_s, args_truth_s


from torch.nn.utils.rnn import pad_sequence


def collate_fn_train(data):
    idx, d_t, t_v, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e, role_labels, tri2arg_labels, word_mask1d, word_mask2d, triu_mask2d, prompt = zip(
        *data)

    batch_size = len(token)
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    '''padding tokens'''
    token = pad_sequence(token, True)
    seg = pad_sequence(seg, True)
    mask = pad_sequence(mask, True)

    '''padding mask'''
    word_mask1d = pad_sequence(word_mask1d, True)
    word_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    word_mask2d = pad_2d(word_mask2d, word_mat)
    triu_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    triu_mask2d = pad_2d(triu_mask2d, triu_mat)

    '''padding labels'''
    tri_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    trigger_labels = pad_2d(trigger_labels, tri_mat)
    arg_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    argument_labels = pad_2d(argument_labels, arg_mat)
    t2a_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    tri2arg_labels = pad_2d(tri2arg_labels, t2a_mat)
    role_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    role_labels = pad_2d(role_labels, role_mat)
    t_s = pad_sequence(t_s, True)
    t_e = pad_sequence(t_e, True)

    as_mat = torch.zeros((batch_size, a_s[0].size(0), max_tokens), dtype=torch.bool)
    a_s = pad_2d(a_s, as_mat)
    ae_mat = torch.zeros((batch_size, a_e[0].size(0), max_tokens), dtype=torch.bool)
    a_e = pad_2d(a_e, ae_mat)
    return idx, d_t, t_v, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e, role_labels, tri2arg_labels, word_mask1d, word_mask2d, triu_mask2d, prompt


def collate_fn_dev(data):
    idx, d_t, t_v, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e, role_labels, tri2arg_labels, word_mask1d, word_mask2d, \
    triu_mask2d, tuples_ti, tuples_ac, t_t, a_t, prompt = zip(*data)
    batch_size = len(token)
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    '''padding mask'''
    word_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    word_mask2d = pad_2d(word_mask2d, word_mat)
    triu_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    triu_mask2d = pad_2d(triu_mask2d, triu_mat)

    '''padding labels'''
    tri_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    trigger_labels = pad_2d(trigger_labels, tri_mat)
    arg_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    argument_labels = pad_2d(argument_labels, arg_mat)
    t2a_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    tri2arg_labels = pad_2d(tri2arg_labels, t2a_mat)
    role_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    role_labels = pad_2d(role_labels, role_mat)
    t_s = pad_sequence(t_s, True)
    t_e = pad_sequence(t_e, True)

    as_mat = torch.zeros((batch_size, a_s[0].size(0), max_tokens), dtype=torch.bool)
    a_s = pad_2d(a_s, as_mat)
    ae_mat = torch.zeros((batch_size, a_e[0].size(0), max_tokens), dtype=torch.bool)
    a_e = pad_2d(a_e, ae_mat)
    return idx, d_t, t_v, token, seg, mask, trigger_labels, t_s, t_e, argument_labels, a_s, a_e, role_labels, tri2arg_labels, word_mask2d, triu_mask2d, \
           tuples_ti[0], tuples_ac[0], t_t, a_t, prompt


def collate_fn_test(data):
    idx, dc, token, seg, mask, word_mask1d, word_mask2d, triu_mask2d, ti_tuple, tc_tuple, ai_tuple, ac_tuple, prompt = zip(
        *data)
    batch_size = len(token)
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    '''padding mask'''
    word_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    word_mask2d = pad_2d(word_mask2d, word_mat)
    triu_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    triu_mask2d = pad_2d(triu_mask2d, triu_mat)

    return idx, dc, token, seg, mask, word_mask1d, word_mask2d, triu_mask2d, ti_tuple[0], tc_tuple[0], ai_tuple[0], \
           ac_tuple[0], prompt
