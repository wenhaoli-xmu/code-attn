import torch
from code_attn.misc import get_model_and_tokenizer, get_env_conf, adjust_lr
import gc, time
import os

import deepspeed
from corpus import LazyRandomSampleCorpus, get_processor

from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import torch.distributed as dist

from functools import partial
import tqdm
import argparse

from concurrent.futures import ThreadPoolExecutor
import concurrent


def compute_attn_supervise_loss(
        draft_attn, 
        true_attn, 
        query_index, 
        max_top, 
        max_oth, 
        maskout, 
        beta: float = 1.0, 
        margin: float = 0.0):

    loss = torch.tensor(0, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss()

    # prepare & apply mask
    num_kv = true_attn.shape[-1]
    mask = torch.triu(torch.ones((num_kv, num_kv), dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    if query_index is not None:
        mask = mask[..., query_index, :]
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
    labels = torch.ones_like(logits)
    loss += criterion(logits * beta - margin, labels).cpu()

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
        corp = LazyRandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def clear_cache(local_rank, max_trial=10):
    torch.cuda.set_device(local_rank)
    while max_trial > 0:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(0.1)
        max_trial -= 1


def collate_fn(batch, pad_token_id, max_tokens):
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = [x.get('input_ids') for x in batch]
    input_len = [len(x) for x in input_ids]

    # padding
    input_ids = [x + [pad_token_id] * (max_tokens - len(x)) for x in input_ids]

    # to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)

    return {"input_ids": input_ids, "input_len": input_len}


def reset_buffer_dir():
    if dist.get_rank() == 0:
        for file in os.listdir("buffer"):
            os.remove(os.path.join("buffer", file))
    dist.barrier()


def train(args):
    deepspeed.init_distributed()

    # 计算一些变量 & 例行检查
    env_conf = get_env_conf(args.env_conf)
    num_gpus = dist.get_world_size()
    if not os.path.exists("buffer"):
        os.mkdir("buffer")
    else:
        reset_buffer_dir()

    assert env_conf['train']['train_iters'] % args.instance_per_cycle == 0
    assert args.num_layers % num_gpus == 0
    assert args.instance_per_cycle % args.prepare_batch_size_per_gpu == 0
    assert (args.instance_per_cycle // args.prepare_batch_size_per_gpu) % num_gpus == 0
    assert env_conf['model']['model_dtype'] in ('bf16', 'fp16')

    num_inn_cycle = env_conf['train']['train_iters'] // args.instance_per_cycle
    num_out_cycle = args.num_layers // num_gpus

    # 开始训练pipeline
    for out_cycle_idx in range(num_out_cycle):
        torch.manual_seed(42)

        for gpu_id in range(num_gpus):
            if args.local_rank == gpu_id:
                # 加载模型 & tokenizer
                layer_idx = out_cycle_idx * num_gpus + args.local_rank
                layer_indices = [out_cycle_idx * num_gpus + i for i in range(num_gpus)]
                env_conf["model"]["device_map"] = {"": args.local_rank}
                tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

                # 将模型只保存某个layer
                model.train()
                model.freeze_model()
                model.unfreeze_model()
                layer = model.dump_as_attn_modules()[layer_idx]
                params = model.layer_ft_params(layer_idx)
                del model
                clear_cache(args.local_rank)
                print(f"RANK-{args.local_rank} training started !")
            dist.barrier()

        # 构造数据集
        corpus = build_dataset(env_conf, tokenizer)
        partial_collate_fn = partial(
            collate_fn, 
            pad_token_id=tokenizer.pad_token_id, 
            max_tokens=args.max_tokens)
        sampler = DistributedSampler(
            corpus, 
            num_replicas=num_gpus, 
            rank=args.local_rank, 
            shuffle=True)
        loader = DataLoader(
            corpus, 
            batch_size=args.prepare_batch_size_per_gpu, 
            sampler=sampler,
            collate_fn=partial_collate_fn)
        data_iter = iter(loader)
        sampler.set_epoch(0)

        # 构造优化器 & 学习率调节器
        optim, lr_adjust = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)
        
        # 一些参数
        accum_grad = env_conf["train"]["accum_grad"]
        clip_grad = env_conf["train"]["clip_grad"]
        step = 0

        history_loss = []
        history_diff = []

        compute_loss = partial(
            compute_attn_supervise_loss,
            max_top=args.max_top, 
            max_oth=args.max_oth,
            maskout=args.maskout,
            beta=args.beta,
            margin=args.margin)

        for inn_cycle_idx in range(num_inn_cycle):

            # 加载数据准备模型
            for gpu_id in range(num_gpus):
                if args.local_rank == gpu_id:
                    _, model = get_model_and_tokenizer(**env_conf['model'])
                    model.eval()
                dist.barrier()

            increment = num_gpus * args.prepare_batch_size_per_gpu
            executor = ThreadPoolExecutor(max_workers=args.max_prepare_workers)
            futures = []

            for idx in tqdm.tqdm(range(0, args.instance_per_cycle, increment)):

                inputs = next(data_iter)
                length = inputs.get("input_len")
                inputs.update({"return_inputs": True})

                #前向传播 & 获取每层的输入数据
                outputs = model(**inputs)
                inputs = [outputs.hidden_states[i].cuda(args.local_rank) for i in layer_indices]
                inputs = torch.stack(inputs, dim=0)

                # 进程之间通信-1, 交换padded input hidden states
                inputs_gather = [torch.empty_like(inputs) for _ in range(num_gpus)]
                dist.all_gather(inputs_gather, inputs)
                inputs_gather = [inputs[args.local_rank] for inputs in inputs_gather]
                inputs_gather = torch.cat(inputs_gather, dim=0).cpu()

                # 进程之间通信-2, 交换input hidden states的尺寸
                length = torch.tensor(length, dtype=torch.int64, device=args.local_rank)
                length_gather = [torch.empty_like(length) for _ in range(num_gpus)]
                dist.all_gather(length_gather, length)
                length_gather = torch.cat(length_gather)

                # 保存数据
                buffer = (inputs_gather, length_gather)
                buffer_file = f"buffer/inputs_buffer_rank_{args.local_rank}_{idx:05d}.pt"

                future = executor.submit(torch.save, buffer, buffer_file)
                futures.append(future)
                if len(futures) >= args.max_prepare_workers:
                    concurrent.futures.wait(futures)
                    futures = []

            # 先准备好数据的进程等待未准备完成的进程
            del model, inputs, outputs
            clear_cache(args.local_rank)
            dist.barrier()

            # 将buffer文件夹下的所有文件进行排序
            buffer_files = os.listdir("buffer")
            buffer_files = sorted(filter(
                lambda x: x.startswith(f"inputs_buffer_rank_{args.local_rank}"), 
                buffer_files))

            # 先读取第1个数据
            inputs_gather, length_gather = torch.load(os.path.join("buffer", buffer_files[0]))
            buffer_files = [*buffer_files[1:], buffer_files[0]]

            # 遍历所有的buffer files
            for buffer_file in buffer_files:

                # 下一个数据的prefetch
                future = executor.submit(torch.load, os.path.join("buffer", buffer_file))

                for hidden_states, length in zip(inputs_gather, length_gather):
                    hidden_states = hidden_states[:length, ...].unsqueeze(0)
                    lr_adjust(step=step)

                    # forward & backward
                    if args.max_que is not None:
                        random_query_index = torch.randperm(
                            hidden_states.shape[-2], 
                            dtype=torch.int64, 
                            device=hidden_states.device)[:args.max_que]
                    else:
                        random_query_index = None

                    _, _, draft_attn, true_attn = layer(
                        hidden_states=hidden_states.to(args.local_rank), 
                        early_exit=True,
                        query_index=random_query_index)


                    if args.backward_per_head:
                        grad = torch.zeros_like(draft_attn)

                        # per head calculation to save GPU memory
                        for head_idx, (draft_attn_head, true_attn_head) in enumerate(zip(
                            torch.chunk(draft_attn, chunks=draft_attn.shape[1], dim=1),
                            torch.chunk(true_attn, chunks=draft_attn.shape[1], dim=1),
                        )):
                            draft_attn_head = draft_attn_head.detach()
                            true_attn_head = true_attn_head.detach()
                            draft_attn_head.requires_grad_(True)

                            diff, loss = compute_loss(draft_attn_head, true_attn_head, random_query_index)
                            loss.backward()

                            grad[:, head_idx, ...] = draft_attn_head.grad.data[:]

                            history_loss.append(loss.item())
                            history_diff.append(diff.item())

                        grad /= accum_grad
                        draft_attn.backward(gradient=grad)
                    else:
                        # direct calculation
                        diff, loss = compute_loss(draft_attn, true_attn, random_query_index)
                        history_loss.append(loss.item())
                        history_diff.append(diff.item())
                        loss /= accum_grad
                        loss.backward()

                    # update the parameters
                    if (step + 1) % accum_grad == 0:
                        if clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad)
                        optim.step()
                        optim.zero_grad()

                    step += 1

                # 已经prefetch好的数据
                inputs_gather, length_gather = future.result()

            print(f"layer: {layer_idx}\tstep: {step}\tloss: {sum(history_loss) / len(history_loss):<.3f}\tdiff: {sum(history_diff) / len(history_diff):<.3f}", flush=True)
            history_loss = []
            history_diff = []
            
            clear_cache(args.local_rank)
            dist.barrier()
            reset_buffer_dir()

        # 创建目录
        save_path = args.env_conf.split('/')[-1]
        if dist.get_rank() == 0:
            if not os.path.exists(f"train_results/{save_path}"):
                os.mkdir(f"train_results/{save_path}")
        dist.barrier()

        # 保存训练好的参数
        torch.save(params, f"train_results/{save_path}/{layer_idx}.pth")
        print(f"RANK-{args.local_rank} training done !")
        dist.barrier()

        # 清空缓存
        del layer, optim
        clear_cache(args.local_rank)
        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 和模型结构有关的参数，需要根据模型的不同而相应地调整
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--maskout", type=float, default=0.98)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=-10.0)

    # 和模型无关的参数
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    # 和资源开销有关的参数
    parser.add_argument("--instance_per_cycle", type=int, default=1000)
    parser.add_argument("--prepare_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--max_prepare_workers", type=int, default=4)
    parser.add_argument("--max_top", type=int, default=None)
    parser.add_argument("--max_oth", type=int, default=None)
    parser.add_argument("--max_que", type=int, default=None)
    parser.add_argument("--backward_per_head", action='store_true')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    train(args)