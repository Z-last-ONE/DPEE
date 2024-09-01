import torch


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_true = y_true.float().detach()
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def _sinusoidal_position_embedding(batch_size, seq_len, output_dim, device):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


def _pointer(qw, kw, word_mask2d, device):
    B, L, K, H = qw.size()
    pos_emb = _sinusoidal_position_embedding(B, L, H, device)
    # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
    cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
    qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
    qw2 = qw2.reshape(qw.shape)
    qw = qw * cos_pos + qw2 * sin_pos
    kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
    kw2 = kw2.reshape(kw.shape)
    kw = kw * cos_pos + kw2 * sin_pos

    logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

    grid_mask2d = word_mask2d.unsqueeze(1).expand(B, K, L, L).float()
    logits = logits * grid_mask2d - (1 - grid_mask2d) * 1e12
    return logits
