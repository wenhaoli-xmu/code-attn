from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, Saver, Evaluator, get_optimizer_and_lr_adjuster

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import wandb
from corpus import LazyRandomSampleCorpus, get_processor
import random


def get_color(value):
    from pygments.console import colorize
    if value > 75:
        return colorize("green", f"{value: 2d}")
    elif value > 50:
        return colorize("yellow", f"{value: 2d}")
    else:
        return colorize("red", f"{value: 2d}")


def compute_attn_supervise_loss(model, attentions, classify=False):
    loss = torch.tensor(0, dtype=torch.float32)
    num_layers = len(attentions[0])

    for draft_attn, true_attn in zip(attentions[0], attentions[1]):
        cond1 = draft_attn is not None
        cond2 = true_attn is not None

        if cond1 and cond2:

            if not classify:
                draft_attn = torch.clamp(draft_attn, min=torch.finfo(draft_attn.dtype).min).flatten(0,2)
                true_attn = torch.softmax(true_attn.flatten(0,2), dim=-1, dtype=torch.float32).type(true_attn.dtype)
                loss += torch.nn.functional.cross_entropy(draft_attn, true_attn, reduce='none').cpu() / num_layers
            else:
                draft_attn = torch.clamp(draft_attn, min=torch.finfo(draft_attn.dtype).min).flatten(0,2)
                true_attn = true_attn.flatten(0,2)

                """
                这一步是将true_attn变成只有前百分之多少是1, 其余都是0的形状
                """
                label = torch.zeros_like(true_attn)
                remain = true_attn.shape[-1] - int(true_attn.shape[-1] * model.conf['draft_kwargs']['mask_out'])
                _, indices,  = torch.topk(true_attn, k=remain)
                label = torch.scatter(label, dim=-1, index=indices, value=1)

                """
                这一步是将mask施加到label上, 让原本为-inf的变成0
                """
                mask1 = true_attn == torch.finfo(draft_attn.dtype).min
                mask2 = true_attn == -torch.inf
                label = torch.masked_fill(label, mask1 | mask2, value=0)

                """
                最后进行loss的累加
                """
                loss += torch.nn.functional.binary_cross_entropy_with_logits(draft_attn.type(torch.float32), label).cpu() / num_layers

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    parser.add_argument('--classify', action='store_true', default=False)
    args = parser.parse_args()


    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf['train'])
    saver = Saver(model, **env_conf['train'])
    evaluator = Evaluator(model, tokenizer, **env_conf['train'])
    model.train()


    def callback():
        if not model.model.decoder.is_benchmark_mode():
            return

        ratios = model.model.decoder.get_ratios(reset=True)
        num_heads = 32

        print("      ", end='')
        for head_id in range(num_heads):
            print(f'#{head_id:<2}', end=' ')
        print(f"avg", end=None)

        mean_ratios = [[] for _ in range(num_heads + 1)]

        for idx, layer_ratio in enumerate(ratios):
            if layer_ratio is not None:
                print(f"{idx: >3}", end=': ')
                for hid, head_ratio in enumerate(layer_ratio):
                    value = int(head_ratio * 100)
                    print(get_color(value), end=' ')
                    mean_ratios[hid].append(value)
                value = int(sum(layer_ratio) / len(layer_ratio) * 100)
                print(get_color(value))
                mean_ratios[-1].append(value)
        
        print(f"     ", end='')
        for head_ratio in mean_ratios:
            head_ratio = sum(head_ratio) // len(head_ratio)
            print(get_color(head_ratio), end=' ')


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

    wandb.init(project='hybird')

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for step, batch in tqdm(enumerate(loader), disable=True):
        lr_adjuster(step=step)

        batch.update({
            "attn_supervise": True,
            "attn_supervise_reduce": 128
            })
        outputs = model(**batch)
        loss = compute_attn_supervise_loss(model, outputs.attentions, args.classify) / accum_grad
        loss.backward()

        wandb.log({
            "Train": {
                "Samples": {
                    "train_loss": loss.item()
                }
            }
        })

        # update the parameters
        if (step + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            optim.step()
            optim.zero_grad()

        # reset the model and clear cuda cache
        model.reset()
        saver.step()
        evaluator.step()

        torch.cuda.empty_cache()

    wandb.finish()
