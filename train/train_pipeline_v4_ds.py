"""
intro
-----
* v4版本使用ranknet的pairwise ranking loss来学习排序
"""

from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, Saver, Evaluator, get_optimizer_and_lr_adjuster

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import wandb
from corpus import LazyRandomSampleCorpus, get_processor

import deepspeed


def compute_attn_supervise_loss(attentions, max_top, max_oth):
    loss = torch.tensor(0, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss()

    for draft_attn, true_attn in zip(attentions[0], attentions[1]):
        cond1 = draft_attn is not None
        cond2 = true_attn is not None

        if cond1 and cond2:
            
            # 计算出true attn的sort index
            mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
            true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
            indices = torch.argsort(true_attn, dim=-1, descending=True)

            # 切分出来top 0.01 的indices，和other 0.98的indices
            top_cnt = int(indices.shape[-1] * 0.02)
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
            print(f"loss: {loss.item():<.3f}. diff: {diff.item():<.3f}")

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--max_top', type=int, default=None)
    parser.add_argument('--max_oth', type=int, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": args.local_rank}

    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    saver = Saver(model, **env_conf['train'])
    model.train()

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.ft_params())

    model.freeze_model()
    model.unfreeze_model()

    # load datasets
    sum_partition = 0
    num_iters = env_conf["train"]["train_iters"]
    corpus = []
    for info in env_conf["train"]["corpus"]:
        sum_partition += info["partition"]
        num_instance = int(info["partition"] * num_iters)

        proc = get_processor(info["conf"], tokenizer)
        corp = LazyRandomSampleCorpus(info["data"], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    dataset = ConcatDataset(corpus)

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for step, batch in tqdm(enumerate(loader), disable=True):
        outputs = model_engine(**batch)
        loss = compute_attn_supervise_loss(outputs.attentions, args.max_top, args.max_oth) / accum_grad
        model_engine.backward(loss) 

        model_engine.step()
        model_engine.model.reset()
        saver.step()

        torch.cuda.empty_cache()
