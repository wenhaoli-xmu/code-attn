import torch


def compute_attn_supervise_loss(draft_attn, true_attn, max_top, max_oth, maskout):
    loss = torch.tensor(0, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 计算出true attn的sort index
    mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
    indices = torch.argsort(true_attn, dim=-1, descending=True)

    # 切分出来top 0.01 的indices，和other 0.98的indices
    top_cnt = int(indices.shape[-1] * (1 - maskout))
    top_indices = indices[..., :top_cnt]
    oth_indices = indices[..., top_cnt:]

    if max_top is not None:
        top_rnd_indices = torch.randperm(top_cnt, dtype=torch.int64, device=indices.device)[:max_top]
        top_indices = top_indices[..., top_rnd_indices]
    if max_oth is not None:
        oth_rnd_indices = torch.randperm(indices.shape[-1] - top_cnt, dtype=torch.int64, device=indices.device)[:max_oth]
        oth_indices = oth_indices[..., oth_rnd_indices]

    top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
    oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

    top_draft_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
    oth_draft_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

    residual = top_draft_attn - oth_draft_attn
    residual_mask = (top_mask | oth_mask).expand_as(residual).flatten(-3)

    logits = residual.flatten(-3)[~residual_mask.bool()]
    labels = torch.ones_like(logits, dtype=torch.float32)
    loss += criterion(logits, labels.type(torch.float32)).cpu()

    # 算一下排序误差
    diff = torch.count_nonzero(logits < 0) / logits.numel()

    return diff, loss
