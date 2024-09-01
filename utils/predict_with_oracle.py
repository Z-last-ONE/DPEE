import torch
import numpy as np


def extract_specific_item_with_oracle(model, d_t, token, seg, mask, args_num, ty_args_id, word_mask2d, triu_mask2d,
                                      config,prompt):
    assert token.size(0) == 1
    data_type = d_t.item()
    text_emb = model.plm(token, seg, mask,prompt)

    '''预测事件类型'''
    p_type, type_emb = model.predict_type(text_emb, mask)
    type_pred = np.array(p_type > config.threshold_0, dtype=int)
    type_rep = type_emb[d_t, :]

    '''获取三矩阵'''
    # tri_logits, arg_logits, role_logits = model.predict_logits(text_emb, type_rep, mask, word_mask2d, triu_mask2d)
    a_s, a_e, p_s, p_e, t2a_logits, role_logits = model.predict_logits(text_emb, type_rep, mask, word_mask2d,
                                                                       triu_mask2d)

    '''三矩阵解码'''
    # tri_b_index, tri_x_index, tri_y_index = ((tri_logits > 0).long() + word_mask2d.long()).eq(2).nonzero(as_tuple=True)
    # arg_b_index, arg_x_index, arg_y_index = ((arg_logits > 0).long() + word_mask2d.long()).eq(2).nonzero(as_tuple=True)
    role_b_index, role_x_index, role_r_index = (t2a_logits > 0).nonzero(as_tuple=True)

    '''填充触发词列表'''
    # trigger_spans = []
    # triggers = torch.cat([x.unsqueeze(-1) for x in [tri_b_index, tri_x_index, tri_y_index]], dim=-1).cpu().numpy()
    # for _, x, y in triggers:
    #     trigger_spans.append((x, y))
    trigger_s = np.where(a_s[0] > config.threshold_1)[0]
    trigger_e = np.where(a_e[0] > config.threshold_2)[0]
    trigger_spans = []
    for i in trigger_s:
        es = trigger_e[trigger_e >= i]
        if len(es) > 0:
            e = es[0]
            trigger_spans.append((i, e))

    '''填充论元列表'''
    args_spans = {i: [] for i in range(args_num)}
    for i in ty_args_id[data_type]:
        args_s = np.where(p_s.swapaxes(0, 1)[i] > config.threshold_3)[0]
        args_e = np.where(p_e.swapaxes(0, 1)[i] > config.threshold_4)[0]
        for j in args_s:
            es = args_e[args_e >= j]
            if len(es) > 0:
                e = es[0]
                args_spans[i].append((j, e))

    '''填充role列表'''
    ai = args_spans
    ac = torch.cat([x.unsqueeze(-1) for x in [role_b_index, role_x_index, role_r_index]], dim=-1).cpu().numpy()
    return type_pred, trigger_spans, ai, ac


def predict_one(model, args, typ_oracle, token, seg, mask, ty_args_id, word_mask2d,
                triu_mask2d, a_t, config,prompt):
    type_pred, ti, ai, ac = extract_specific_item_with_oracle(model, typ_oracle, token, seg, mask,
                                                              args.args_num, ty_args_id, word_mask2d,
                                                              triu_mask2d, config,prompt)
    args_truth = a_t[0]
    args_pred_tuples = []  # (type, tri_sta, tri_end, arg_sta, arg_end, arg_role), 6-tuple
    args_truth_tuples = []
    args_candidates = ty_args_id[typ_oracle.item()]  # type constrain
    for i in args_candidates:
        typ = typ_oracle
        arg_role = i
        for args_pred_one in ai[i]:
            arg_sta = args_pred_one[0]
            arg_end = args_pred_one[1]
            args_pred_tuples.append((typ, arg_sta, arg_end, arg_role))

        for args_truth_one in args_truth[i]:
            arg_sta = args_truth_one[0]
            arg_end = args_truth_one[1]
            args_truth_tuples.append((typ, arg_sta, arg_end, arg_role))

    return type_pred, ti, args_pred_tuples, args_truth_tuples


def decode(task, labels, tri_args, ti=None, ai=None, ac=None, type_oracle=None):
    if task == 'ti':
        pred = ti
        if pred is None:
            pred_set = set()
        else:
            pred_set = set(pred)
        return len(labels), len(pred_set), len(pred_set & labels)

    if task == "ac":
        arg_dict = {}
        for b, x, y in ai:
            for c in range(x, y + 1):
                if (b, c) in arg_dict:
                    arg_dict[(b, c)].append((b, x, y))
                else:
                    arg_dict[(b, c)] = [(b, x, y)]
        new_pred_set = set()
        pred_set = set([tuple(x.tolist()) for x in ac])
        for b, x, r in pred_set:
            if (b, x) in arg_dict:
                for (b_a, x_a, y_a) in arg_dict[(b, x)]:
                    new_pred_set.add((x_a, y_a, r))
        pred_set = set([x for x in new_pred_set if (x[-1]) in tri_args[type_oracle.item()]])
        return len(labels), len(pred_set), len(pred_set & labels)
