"""
intro
-----
* v4版本使用ranknet的pairwise ranking loss来学习排序
"""

from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, adjust_lr

import argparse
from tqdm import tqdm
import torch
import os
from functools import partial

import deepspeed
import time, gc
from corpus import get_processor, RandomSampleCorpus
from torch.utils.data import ConcatDataset


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


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = RandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def get_optimizer_and_lr_adjuster(model, max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster



