from models.layers import *
from models.networks import QKAttention, Gate, CNNAttention
import numpy as np
from utils.global_pointer_utils import _pointer, multilabel_categorical_crossentropy

import torch.nn.functional as F

class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.config = config
        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

        pretrained_weights = np.load('/home/mengfanshen/EE/CCEE/utils/schema.npy')
        pretrained_weights = torch.tensor(pretrained_weights, dtype=torch.float32)
        # self.type_emb = nn.Embedding.from_pretrained(pretrained_weights)
        self.type_emb = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)

        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c]
        p_type = torch.sigmoid(pred)
        return p_type, type_emb


class TriggerRec(nn.Module):
    def __init__(self, config, hidden_size):
        super(TriggerRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(hidden_size, 1, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config

        self.gate = Gate(config, hidden_size)

    def forward(self, type_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''

        h_cln = self.ConditionIntegrator(text_emb, type_emb)
        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        # sa_pro = self.SA(type_emb, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        inp = self.layer_norm(h_sa + h_cln)
        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, config, hidden_size, num_labels):
        super(ArgsRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.config = config

        self.gate = Gate(config, hidden_size)

    def forward(self, text_emb, mask, type_emb):
        h_sa = self.SA(text_emb, text_emb, text_emb, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + text_emb)
        inp = gelu(self.hidden(h_sa))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))
        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))  # [b, l]
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)
        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class Tri2argCls(nn.Module):
    def __init__(self, config, hidden_size):
        super(Tri2argCls, self).__init__()
        self.eve_hid_size = 64
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config
        self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        # self.QKA = QKAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.role_linear_s = nn.Linear(hidden_size, self.eve_hid_size * 2)

        self.gate = Gate(config, hidden_size)

    def forward(self, text_emb, type_emb, mask, triu_mask2d):
        B = text_emb.shape[0]
        L = text_emb.size(1)
        h_sa = self.SA(text_emb, text_emb, text_emb, mask)
        inp = self.layer_norm(h_sa + text_emb)
        role_reps = gelu(self.role_linear_s(inp).view(B, L, -1, self.eve_hid_size * 2))
        role_reps = self.dropout(role_reps)

        role_s_qw, role_s_kw = torch.chunk(role_reps, 2, dim=-1)
        role_logits = _pointer(role_s_qw, role_s_kw, triu_mask2d, inp.device).squeeze(1)
        return role_logits


class RoleCls(nn.Module):
    def __init__(self, config, hidden_size):
        super(RoleCls, self).__init__()
        self.eve_hid_size = 64
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config
        self.role_linear_s = nn.Linear(hidden_size, self.eve_hid_size * 2)

    def forward(self, text_emb, triu_mask2d):
        B = text_emb.shape[0]
        L = text_emb.size(1)

        role_reps = gelu(self.role_linear_s(text_emb).view(B, L, -1, self.eve_hid_size * 2))
        role_reps = self.dropout(role_reps)

        role_s_qw, role_s_kw = torch.chunk(role_reps, 2, dim=-1)
        role_logits = _pointer(role_s_qw, role_s_kw, triu_mask2d, text_emb.device).squeeze(1)
        return role_logits


class DPEE(nn.Module):
    def __init__(self, config, model_weight, pos_emb_size):
        super(DPEE, self).__init__()
        self.bert = model_weight
        self.tri_hid_size = 256
        self.arg_hid_size = 256
        self.eve_hid_size = 256
        self.config = config
        self.args_num = config.args_num
        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        # self.trigger_cls = TriggerCls(config, self.tri_hid_size)
        # self.arg_cls = ArgCls(config, self.arg_hid_size)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.args_num)
        self.tri2arg_cls = Tri2argCls(config, config.hidden_size)
        self.role_cls = RoleCls(config, config.hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')
        self.gates = nn.ModuleList([nn.Linear(768, 32 * 768) for i in range(12)])

    def forward(self, tokens, segment, mask, type_id, type_vec, t_l, t_s, t_e, a_l, a_s, a_e, role_labels,
                tri2arg_labels,
                word_mask1d, word_mask2d, triu_mask2d, prompt):
        prompt_attention_mask, prompt_guids = self.get_text_prompt(True, prompt, word_mask1d)
        outputs = self.bert(
            tokens,
            attention_mask=prompt_attention_mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            past_key_values=prompt_guids
        )

        output_emb = outputs[0]
        B, L, H = output_emb.size()
        '''识别事件类型'''
        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)
        type_loss = torch.sum(type_loss)
        type_rep = type_emb[type_id, :]
        '''三个矩阵进行学习'''
        '''# ---'''
        # tri_logits = self.trigger_cls(output_emb, type_rep, mask, word_mask2d, B, L)
        # trigger_loss = multilabel_categorical_crossentropy(tri_logits[word_mask2d], t_l[word_mask2d])
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask)
        p_s = p_s.pow(self.config.pow_1)
        p_e = p_e.pow(self.config.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, t_s)
        trigger_loss_e = self.loss_1(p_e, t_e)
        mask_t = mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))
        trigger_loss = trigger_loss_s + trigger_loss_e

        '''# ---'''
        # arg_logits = self.arg_cls(output_emb, type_rep, mask, word_mask2d, B, L)
        # args_loss = multilabel_categorical_crossentropy(arg_logits[word_mask2d], a_l[word_mask2d])
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, mask, type_rep)
        p_s = p_s.pow(self.config.pow_2)
        p_e = p_e.pow(self.config.pow_2)
        as_loss = self.loss_2(p_s, a_s.transpose(1, 2))
        ae_loss = self.loss_2(p_e, a_e.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(as_loss).float()  # [b, t, l]
        args_loss_s = torch.sum(as_loss.mul(mask_a))
        args_loss_e = torch.sum(ae_loss.mul(mask_a))
        args_loss = args_loss_s + args_loss_e

        '''# ---'''
        tri2arg_logits = self.tri2arg_cls(text_rep_type, type_rep, mask, triu_mask2d)
        tri2arg_loss = multilabel_categorical_crossentropy(tri2arg_logits, tri2arg_labels)

        role_logits = self.role_cls(text_rep_type, triu_mask2d)
        role_loss = multilabel_categorical_crossentropy(role_logits, role_labels)

        # role_logits = self.role_linear(h_cln)
        # role_logits = torch.sigmoid(role_logits)
        # role_loss = self.loss_1(role_logits, r_l)
        # role_loss = torch.sum(role_loss)

        type_loss = self.config.w1 * type_loss
        trigger_loss = self.config.w2 * trigger_loss
        args_loss = self.config.w3 * args_loss
        tri2arg_loss = self.config.w4 * tri2arg_loss
        role_loss = self.config.w4 * role_loss
        loss = type_loss + trigger_loss + args_loss + tri2arg_loss + role_loss
        return loss, type_loss, trigger_loss, args_loss, tri2arg_loss, role_loss

    def plm(self, tokens, segment, mask,prompt):
        assert tokens.size(0) == 1
        prompt_attention_mask, prompt_guids = self.get_text_prompt(True, prompt, mask)
        outputs = self.bert(
            tokens,
            attention_mask=prompt_attention_mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            past_key_values=prompt_guids
        )
        output_emb = outputs[0]
        return output_emb

    def get_text_prompt(self, use_prompt, promptText, att_mask):
        if use_prompt == True:
            promptText = promptText.to(self.config.device)
            bsz = promptText.size(0)
            # 得到第一个文本的提示
            prompt_guids = promptText  # [bsz, 384]
            # 将prompt_guid的四层提示在维度1上面进行拼接，1*4*3840----需要对得到的提示进行四次线性变化吗？
            # prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
            # 1*4*(4*2*768)
            # split_prompt_guids = self.linear_layer(prompt_guids)   # [bsz, 32*768]
            # result保存着12个transformer层对应的key、value
            result = []
            for idx in range(12):  # 12
                # [8,32*768]
                key_val = F.softmax(F.leaky_relu(self.gates[idx](prompt_guids)), dim=-1)
                key_val = key_val.view(bsz, -1, 1536)
                key_val = key_val.split(768, dim=-1)
                key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,
                                                                                                  64).contiguous()  # bsz, 12, 4, 64
                temp_dict = (key, value)
                result.append(temp_dict)
            prompt_guids = result
            # key,value都是1*12*16*64,
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = att_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.config.device)
            # 1*96,将两个attention进行拼接。
            prompt_attention_mask = torch.cat((prompt_guids_mask, att_mask), dim=1)
        else:
            prompt_attention_mask = att_mask
            prompt_guids = None
        return prompt_attention_mask, prompt_guids

    def predict_type(self, output_emb, mask):
        assert output_emb.size(0) == 1
        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.view(self.config.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_logits(self, output_emb, type_rep, mask, word_mask2d, triu_mask2d):
        B, L, H = output_emb.size()

        # tri_logits = self.trigger_cls(output_emb, type_rep, mask, word_mask2d, B, L)
        a_s, a_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask)
        mask_t = mask.float()  # [1, t]
        a_s = a_s.squeeze(-1)
        a_e = a_e.squeeze(-1)
        a_s = a_s.mul(mask_t)
        a_e = a_e.mul(mask_t)
        a_s = a_s.data.cpu().numpy()  # [b, t]
        a_e = a_e.data.cpu().numpy()
        ''''''
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, mask, type_rep)
        mask_a = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask_a)
        p_e = p_e.mul(mask_a)
        p_s = p_s.view(L, self.args_num).data.cpu().numpy()
        p_e = p_e.view(L, self.args_num).data.cpu().numpy()
        ''''''
        t2a_logits = self.tri2arg_cls(text_rep_type, type_rep, mask, triu_mask2d)
        role_logits = self.role_cls(text_rep_type, triu_mask2d)
        return a_s, a_e, p_s, p_e, t2a_logits, role_logits

# class TriggerCls(nn.Module):
#     def __init__(self, config, tri_hid_size):
#         super(TriggerCls, self).__init__()
#         self.dropout = nn.Dropout(config.decoder_dropout)
#         self.config = config
#         self.tri_hid_size = tri_hid_size
#         self.tri_linear = nn.Linear(config.hidden_size, tri_hid_size * 2)
#         self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
#         self.SA = MultiHeadedAttention(config.hidden_size, heads_num=config.decoder_num_head,
#                                        dropout=config.decoder_dropout)
#         self.layer_norm = nn.LayerNorm(config.hidden_size)
#
#     def forward(self, text_emb, type_rep, mask, word_mask2d, B, L):
#         h_cln = self.ConditionIntegrator(text_emb, type_rep)
#         h_cln = self.dropout(h_cln)
#         h_sa = self.SA(h_cln, h_cln, h_cln, mask)
#         h_sa = self.dropout(h_sa)
#         h_sa = self.layer_norm(h_sa + h_cln)
#
#         tri_reps = self.tri_linear(h_sa).view(B, L, -1, self.tri_hid_size * 2)
#         tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)
#         tri_logits = _pointer(tri_qw, tri_kw, word_mask2d, tri_reps.device).squeeze(1)
#
#         return tri_logits
# class ArgCls(nn.Module):
#     def __init__(self, config, arg_hid_size):
#         super(ArgCls, self).__init__()
#         self.dropout = nn.Dropout(config.decoder_dropout)
#         self.config = config
#         self.arg_hid_size = arg_hid_size
#         self.arg_linear = nn.Linear(config.hidden_size, arg_hid_size * 2)
#         self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
#         self.SA = MultiHeadedAttention(config.hidden_size, heads_num=config.decoder_num_head,
#                                        dropout=config.decoder_dropout)
#         self.layer_norm = nn.LayerNorm(config.hidden_size)
#
#     def forward(self, text_emb, type_rep, mask, word_mask2d, B, L):
#         h_cln = self.ConditionIntegrator(text_emb, type_rep)
#         h_cln = self.dropout(h_cln)
#         h_sa = self.SA(h_cln, h_cln, h_cln, mask)
#         h_sa = self.dropout(h_sa)
#         h_sa = self.layer_norm(h_sa + h_cln)
#
#         arg_reps = self.arg_linear(h_sa).view(B, L, -1, self.arg_hid_size * 2)
#         arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)
#         arg_logits = _pointer(arg_qw, arg_kw, word_mask2d, arg_reps.device).squeeze(1)
#
#         return arg_logits
