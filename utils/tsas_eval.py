import networkx as nx
from scipy.optimize import linear_sum_assignment
import numpy as np


def calculate_f1(r, p, c):
    if r == 0 or p == 0 or c == 0:
        return 0, 0, 0
    r = c / r
    p = c / p
    f1 = (2 * r * p) / (r + p)
    return f1, r, p


def find_cliques(graph):
    G = nx.Graph(graph)
    cliques = list(nx.find_cliques(G))
    return cliques


def predict_events(model, device, idx, content: str, token, seg, mask, seq_len, id_type: dict,
                   id_args: dict, ty_args_id: dict, word_mask2d, ti_tuple, tc_tuple, ai_tuple,
                   ac_tuple, config, triu_mask2d,prompt):
    event_list = []
    assert token.size(0) == 1
    content = content[0]
    result = {'id': idx, 'content': content}
    text_emb = model.plm(token, seg, mask,prompt)

    args_id = {id_args[k]: k for k in id_args}
    # args_len_dict = {args_id[k]: ARG_LEN_DICT[k] for k in ARG_LEN_DICT}

    p_type, type_emb = model.predict_type(text_emb, mask)
    type_pred = np.array(p_type > config.threshold_0, dtype=bool)
    type_pred = [i for i, t in enumerate(type_pred) if t]
    events_pred = []
    event_3tuple = set()
    event_tuple = set()
    role_logits = None
    for type_pred_one in type_pred:
        type_rep = type_emb[type_pred_one, :]
        type_rep = type_rep.unsqueeze(0)
        a_s, a_e, p_s, p_e, t2a_logits, role_logits = model.predict_logits(text_emb, type_rep, mask, word_mask2d,
                                                                           triu_mask2d)
        t2a_logits = t2a_logits.data.cpu().numpy()
        role_logits = role_logits.data.cpu().numpy()

        trigger_s = np.where(a_s[0] > config.threshold_1)[0]
        trigger_e = np.where(a_e[0] > config.threshold_2)[0]
        trigger_list = []
        for i in trigger_s:
            es = trigger_e[trigger_e >= i]
            if len(es) > 0:
                e = es[0]
                if e - i + 1 <= 4:
                    tri_s = i
                    tri_e = e
                    trigger_list.append((tri_s, tri_e, type_pred_one))
        argument_list = []
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
                    if e - j + 1 <= 100:
                        s = int(j)
                        e = int(e)
                        arg_s = s
                        arg_e = e
                        role = i
                        argument_list.append((arg_s, arg_e, type_pred_one, role))
        for trigger in trigger_list:
            t_s = trigger[0]
            t_e = trigger[1]
            for argument in argument_list:
                argument_s = argument[0]
                argument_e = argument[1]
                argument_type = argument[2]
                argument_role = argument[3]
                tri2arg = t2a_logits[0][t_s:t_e + 1, argument_s:argument_e + 1]
                prob = tri2arg.mean()
                if prob > -10:
                    event_3tuple.add((t_s, t_e, argument_type))
                    event_tuple.add((t_s, t_e, argument_type, argument_s, argument_e, argument_role))
    events = []
    for tri_s, tri_e, e_t in event_3tuple:
        event = {}
        tri_s = int(tri_s) - 1
        tri_e = int(tri_e)
        e_t = int(e_t)
        event['id'] = idx
        event['event_type'] = id_type[e_t].lower()
        event['trigger'] = [tri_s, tri_e]
        args = {}
        flag_need_spilt = False
        for _tri_s, _tri_e, _e_t, _argument_s, _argument_e, _argument_role in event_tuple:
            _arg_s = int(_argument_s) - 1
            _arg_e = int(_argument_e)
            _role = id_args[_argument_role].lower()
            if idx == idx and tri_s == _tri_s - 1 and tri_e == _tri_e and e_t == _e_t:
                if _role not in args.keys():
                    args[_role] = [[_arg_s, _arg_e]]
                else:
                    tmp = args[_role]
                    tmp.append([_arg_s, _arg_e])
                    args[_role] = tmp
                    flag_need_spilt = True
        event['args'] = args

        if flag_need_spilt:
            example_graph = []
            argument_list = []
            for role, spans in args.items():
                for span in spans:
                    argument_list.append((span[0], span[1], role))

            for i, argument_tuple_source in enumerate(argument_list):
                for j, argument_tuple_target in enumerate(argument_list):
                    if i != j:
                        area_matrix = role_logits[0][argument_tuple_source[0] + 1:argument_tuple_source[1] + 1,
                                      argument_tuple_target[0] + 1:argument_tuple_target[1] + 1]
                        area_matrix_value_mean = area_matrix.mean()
                        # x, y = np.where(area_matrix > -5)
                        # is_relation = True if len(x) != 0 else False
                        is_relation = False
                        if is_relation:
                            example_graph.append((i, j))
                    else:
                        example_graph.append((i, j))

            cliques = find_cliques(example_graph)

            if len(cliques) < 1:
                events.append(event)
            else:
                for i, clique_nodes in enumerate(cliques, start=1):
                    new_event = event.copy()
                    args = {}
                    for index in clique_nodes:
                        arg = argument_list[index]
                        if arg[2] not in args.keys():
                            args[arg[2]] = [[arg[0], arg[1]]]
                        else:
                            tmp = args[arg[2]]
                            tmp.append([arg[0], arg[1]])
                            args[arg[2]] = tmp
                    new_event['args'] = args
                    events.append(new_event)
        else:
            events.append(event)
    events_pred.append(events)

    return events_pred


def extract_all_items_without_oracle_new(model, device, idx, content: str, token, seg, mask, seq_len, id_type: dict,
                                         id_args: dict, ty_args_id: dict, word_mask2d, ti_tuple, tc_tuple, ai_tuple,
                                         ac_tuple, config, triu_mask2d,prompt):
    events_pred = predict_events(model, device, idx, content, token, seg, mask, seq_len, id_type, id_args,
                                 ty_args_id,
                                 word_mask2d, ti_tuple, tc_tuple, ai_tuple, ac_tuple, config, triu_mask2d,prompt)

    return events_pred


def match_event(gold_list, predict_list):
    ti_r = 0
    ti_p = 0
    ti_c = 0
    tc_r = 0
    tc_p = 0
    tc_c = 0
    ai_r = 0
    ai_p = 0
    ai_c = 0
    ac_r = 0
    ac_p = 0
    ac_c = 0
    tc_f1_total = 0.0
    ac_f1_total = 0.0
    for gold_one in gold_list:
        sentence_id = gold_one['id']
        for i, predict_one in enumerate(predict_list):
            predict_id = predict_one[0]['id'] if len(predict_one) != 0 else None
            if predict_id == None:
                continue
            if predict_id == sentence_id:
                predict_event_list = predict_one
                gold_event_list = gold_one['events']
                tc_f1, ac_f1 = calculate_confused_score(gold_event_list, predict_event_list)
                tc_f1_total += tc_f1
                ac_f1_total += ac_f1
    #             ti_r += tir
    #             ti_p += tip
    #             ti_c += tic
    #             tc_r += tcr
    #             tc_p += tcp
    #             tc_c += tcc
    #             ai_r += air
    #             ai_p += aip
    #             ai_c += aic
    #             ac_r += acr
    #             ac_p += acp
    #             ac_c += acc
    # ti_f1, r_ti, p_ti = calculate_f1(ti_r, ti_p, ti_c)
    # tc_f1, r_tc, p_tc = calculate_f1(tc_r, tc_p, tc_c)
    # ai_f1, r_ai, p_ai = calculate_f1(ai_r, ai_p, ai_c)
    # ac_f1, r_ac, p_ac = calculate_f1(ac_r, ac_p, ac_c)
    print('\n\n tc:{:.3f}, ac:{:.3f}\n\n'.format(tc_f1_total / len(gold_list), ac_f1_total / len(gold_list)))
    return


def calculate_confused_score(gold_event_list, predict_event_list):
    sample = np.zeros((len(gold_event_list), len(predict_event_list)))
    for i, gold_event in enumerate(gold_event_list):
        for j, predict_event in enumerate(predict_event_list):
            gold_type = gold_event['type'].lower()
            gold_trigger = gold_event['trigger']['span']
            predict_type = predict_event['event_type'].lower()
            predict_trigger = predict_event['trigger']
            if gold_type == predict_type and gold_trigger == predict_trigger:
                gold_arg_ti_list = set()
                predict_ti_arg_list = set()

                gold_arg_list = set()
                predict_arg_list = set()

                gold_args = gold_event['args']
                predict_args = predict_event['args']
                for key, value in gold_args.items():
                    key = key.lower()
                    for v in value:
                        span = v['span']
                        gold_arg_ti_list.add((span[0], span[1]))
                        gold_arg_list.add((span[0], span[1], key))
                for key, value in predict_args.items():
                    key = key.lower()
                    for v in value:
                        predict_ti_arg_list.add((v[0], v[1]))
                        predict_arg_list.add((v[0], v[1], key))

                f1, _, _ = calculate_f1(len(gold_arg_list), len(predict_arg_list),
                                        len(gold_arg_list & predict_arg_list))
                f1_ti, _, _ = calculate_f1(len(gold_arg_ti_list), len(predict_ti_arg_list),
                                           len(gold_arg_ti_list & predict_ti_arg_list))
                sample[i, j] = f1

            else:
                sample[i, j] = 0

    sample_cost = 1 / sample
    sample_cost[np.isinf(sample_cost)] = 99999
    row_ind, col_ind = linear_sum_assignment(sample_cost)
    match_list = []
    for i in range(len(col_ind)):
        if sample_cost[row_ind[i], col_ind[i]] < 99998:
            match_list.append([row_ind[i], col_ind[i], sample[row_ind[i], col_ind[i]]])

    gold_ti_r = 0
    gold_ti_p = 0
    gold_ti_c = 0
    gold_tc_r = 0
    gold_tc_p = 0
    gold_tc_c = 0
    gold_ai_r = 0
    gold_ai_p = 0
    gold_ai_c = 0
    gold_ac_r = 0
    gold_ac_p = 0
    gold_ac_c = 0
    trigger_set = {}
    for gold_event in gold_event_list:
        gold_tc_r += 1
        gold_trigger = gold_event['trigger']
        gold_trigger_span = gold_event['trigger']['span']
        gold_trigger_type = gold_event['type']
        _tuple = (gold_trigger_span[0], gold_trigger_span[1], gold_trigger_type.lower())
        if _tuple not in trigger_set.keys():
            trigger_set[_tuple] = 1
        else:
            tmp = trigger_set[_tuple]
            tmp += 1
            trigger_set[_tuple] = tmp

    for pred_event in predict_event_list:
        gold_tc_p += 1
        event_type = pred_event['event_type']
        trigger = pred_event['trigger']
        _tuple = (trigger[0], trigger[1], event_type)
        if _tuple in trigger_set.keys():
            num = trigger_set[_tuple]
            if num > 0:
                gold_tc_c += 1
                num = num - 1
                trigger_set[_tuple] = num

    total_arg_f1 = 0.0
    for match in match_list:
        gold_event = gold_event_list[match[0]]
        pred_event = predict_event_list[match[1]]
        arg_f1 = match[2]
        total_arg_f1 += arg_f1
        gold_type = gold_event['type'].lower()
        gold_trigger = gold_event['trigger']['span']

        pred_type = pred_event['event_type'].lower()
        pred_trigger = pred_event['trigger']

    tc_f1, r_tc, p_tc = calculate_f1(len(gold_event_list), len(match_list), len(match_list))
    ac_f1 = total_arg_f1 / (len(gold_event_list) + 1e-10)
    return tc_f1, ac_f1
