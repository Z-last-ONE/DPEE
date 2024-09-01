import torch
import numpy as np

ARG_LEN_DICT = {
    'collateral': 14,
    'proportion': 37,
    'obj-org': 34,
    'number': 18,
    'date': 27,
    'sub-org': 35,
    'target-company': 59,
    'sub': 38,
    'obj': 36,
    'share-org': 19,
    'money': 28,
    'title': 8,
    'sub-per': 15,
    'obj-per': 18,
    'share-per': 20,
    'institution': 22,
    'way': 8,
    'amount': 19
}

ARG_LEN_DICT_FNDEE = {
    "Subject": 20,
    "Equipment": 20,
    "Date": 20,
    "Location": 20,
    "Area": 20,
    "Content": 20,
    "Militaryforce": 20,
    "Object": 20,
    "Materials": 20,
    "Result": 20,
    "Quantity": 20
}


def extract_all_items_without_oracle(model, device, idx, content: str, token, seg, mask, seq_len, id_type: dict,
                                     id_args: dict, ty_args_id: dict, word_mask2d, ti_tuple, tc_tuple, ai_tuple,
                                     ac_tuple, config, triu_mask2d,prompt):
    ti_pred = set()
    tc_pred = set()
    ai_pred = set()
    ac_pred = set()
    assert token.size(0) == 1
    content = content[0]
    result = {'id': idx, 'content': content}
    text_emb = model.plm(token, seg, mask,prompt)

    args_id = {id_args[k]: k for k in id_args}
    if 'FewFC' in config.data_path:
        args_len_dict = {args_id[k]: ARG_LEN_DICT[k] for k in ARG_LEN_DICT}
    else:
        args_len_dict = {args_id[k]: ARG_LEN_DICT_FNDEE[k] for k in ARG_LEN_DICT_FNDEE}
    p_type, type_emb = model.predict_type(text_emb, mask)
    type_pred = np.array(p_type > config.threshold_0, dtype=bool)
    type_pred = [i for i, t in enumerate(type_pred) if t]
    events_pred = []
    for type_pred_one in type_pred:
        type_rep = type_emb[type_pred_one, :]
        type_rep = type_rep.unsqueeze(0)
        a_s, a_e, p_s, p_e, t2a_logits, role_logits = model.predict_logits(text_emb, type_rep, mask, word_mask2d,
                                                                           triu_mask2d)
        t2a_logits = t2a_logits.data.cpu().numpy()
        role_logits = role_logits.data.cpu().numpy()
        t2a_set = set()
        role_set = set()
        _, x, y = np.where(t2a_logits > 0)
        for k in y:
            t2a_set.add(k)
        _, x2, y2 = np.where(role_logits > 0)
        for k in y2:
            role_set.add(k)
        f_set = t2a_set.union(role_set)

        trigger_s = np.where(a_s[0] > config.threshold_1)[0]
        trigger_e = np.where(a_e[0] > config.threshold_2)[0]
        for i in trigger_s:
            es = trigger_e[trigger_e >= i]
            if len(es) > 0:
                e = es[0]
                if e - i + 1 <= 4:
                    ti_pred.add((i, e))
                    tc_pred.add((i, e, type_pred_one))

        p_s = np.transpose(p_s)
        p_e = np.transpose(p_e)
        args_candidates = ty_args_id[type_pred_one]
        for i in args_candidates:
            args_s = np.where(p_s[i] > config.threshold_3)[0]
            args_e = np.where(p_e[i] > config.threshold_4)[0]
            for j in args_s:
                es = args_e[args_e >= j]
                if len(es) > 0:
                    e = es[0]
                    if e - j + 1 <= args_len_dict[i]:
                        s = int(j)
                        e = int(e)
                        # if is_integers_between_in_set(s, e, f_set):
                        ai_pred.add((s, e, type_pred_one))
                        ac_pred.add((s, e, type_pred_one, i))

    result['ti'] = [len(ti_tuple), len(ti_pred), len(ti_tuple & ti_pred)]
    result['tc'] = [len(tc_tuple), len(tc_pred), len(tc_tuple & tc_pred)]
    result['ai'] = [len(ai_tuple), len(ai_pred), len(ai_tuple & ai_pred)]
    result['ac'] = [len(ac_tuple), len(ac_pred), len(ac_tuple & ac_pred)]
    return result


def is_integers_between_in_set(start, end, my_set):
    # 使用range函数创建两个数字之间的整数范围
    integers_between = set(range(start, end + 1))

    # 判断两个集合的交集是否等于integers_between
    return integers_between.issubset(my_set)
