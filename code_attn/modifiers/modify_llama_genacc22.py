import torch
import types
from .modify_llama import check_and_apply_qk_rope
from transformers.models.llama.modeling_llama import repeat_kv, CausalLMOutputWithPast, CrossEntropyLoss
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
from lsh_kernel import lsh_attn_dx_u8


@contextmanager
def do_nothing():
    yield


@dataclass
class Genacc22Output(CausalLMOutputWithPast):
    past_key_hash_codes: Optional[Tuple[torch.Tensor]] = None


# ================================================================================================================================
# NOTE: 注释掉，这一部分无需使用
# @torch.no_grad()
# def log_diffs(true_attn, draft_attn, layer_idx):
#     mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
#     true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
#     indices = torch.argsort(true_attn, dim=-1, descending=True)

#     top_cnt = int(indices.shape[-1] * 0.02)
#     top_indices = indices[..., :top_cnt]
#     oth_indices = indices[..., top_cnt:]

#     top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
#     oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

#     top_draft_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
#     oth_draft_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

#     total_diff = 0
#     total_compare = 0

#     for top_seg, oth_seg, top_mask_seg, oth_mask_seg in tqdm.tqdm(
#         zip(
#         segment(top_draft_attn, dim=1, n=1),
#         segment(oth_draft_attn, dim=1, n=1),
#         segment(top_mask, dim=1, n=1),
#         segment(oth_mask, dim=1, n=1)),
#         desc=f'layer-{layer_idx}'
#     ):

#         residual = (top_seg - oth_seg)
#         residual_mask = (top_mask_seg | oth_mask_seg).expand_as(residual).flatten(-3)
#         logits = residual.flatten(-3)[~residual_mask.bool()]

#         diff = torch.count_nonzero(logits < 0)
#         compare = logits.numel()

#         total_diff += diff
#         total_compare += compare

#     # 算一下排序误差
#     diff = total_diff / total_compare
#     print(f"diff: {diff.item():<.3f}")
# ================================================================================================================================
    

def get_hash_code(x_after_rope, rot_mat1, rot_mat2, relu, return_np=True):

    assert x_after_rope.ndim == 4, f"Expect input tensor of 4 dimensionality, got {x_after_rope.shape}"
    assert x_after_rope.shape[-1] % 8 == 0, f"Expect last dimension of `x_after_rope` divisible by 8, got {x_after_rope.shape}"

    def pack_bits(bits_tensor):
        meta = {"dtype": bits_tensor.dtype, "device": bits_tensor.device}

        bit_chunks = bits_tensor.unflatten(-1, (-1, 8))
        bit_mask = torch.tensor([1 << i for i in range(8)], **meta)[None, None, None, None, :]
        packed_bytes = (bit_chunks * bit_mask).sum(dim=-1).to(**meta)

        return packed_bytes

    x_after_rope = relu(x_after_rope @ rot_mat1) @ rot_mat2 > 0
    x_after_rope = x_after_rope.type(torch.uint8)
    return pack_bits(x_after_rope)


def lsh_attn(q_hash, k_hash, return_np=True):
    meta = {"dtype": k_hash.dtype, "device": k_hash.device}
    sim = torch.bitwise_not(torch.bitwise_xor(q_hash, k_hash))
    bit_count_table = torch.tensor([bin(i).count('1') for i in range(256)], **meta)
    count_of_ones = bit_count_table[sim]
    return count_of_ones.sum(dim=-1)


def random_rotation_matrix(dim, dtype, device):
    """
    随机生成一个 n 维旋转矩阵
    :param dim: 维度大小 (n)
    :return: n x n 随机旋转矩阵
    """
    # 使用QR分解生成随机正交矩阵
    random_matrix = torch.randn((dim, dim), dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    
    # 调整使其行列式为1
    if torch.det(q) < 0:
        q[:, 0] *= -1

    return q.type(dtype).to(device)


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    k_hash_cache: List[torch.Tensor] = None,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache, k_hash_cache = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache,
        k_hash_cache=k_hash_cache)
    
    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return Genacc22Output(
        loss=loss, 
        logits=logits, 
        past_key_values=kv_cache,
        past_key_hash_codes=k_hash_cache)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    k_hash_cache: List[torch.Tensor] = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    if k_hash_cache is None:
        k_hash_cache = [None] * len(self.layers)

    for layer_idx, (decoder_layer, kv_cache_layer, k_hash_layer) in enumerate(zip(self.layers, kv_cache, k_hash_cache)):
        hidden_states, kv_cache_layer, k_hash_layer = decoder_layer(
            hidden_states, 
            kv_cache_layer,
            k_hash_layer)

        kv_cache[layer_idx] = kv_cache_layer
        k_hash_cache[layer_idx] = k_hash_layer

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, k_hash_cache


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    k_hash_cache: torch.Tensor = None,
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, kv_cache, k_hash_cache = self.self_attn(
        hidden_states, 
        kv_cache,
        k_hash_cache)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache, k_hash_cache


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    k_hash_cache: torch.Tensor = None,
):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    # some condition things to identify current phase
    prefill_cond1 = hidden_states.shape[-2] > 1
    prefill_cond2 = kv_cache is None
    is_prefill = prefill_cond1 and prefill_cond2

    cond1 = self.draft_kwargs['enable'] is True
    cond2 = not self.is_fix_layer
    do_sparse_attn = cond1 and cond2

    # qkv projection
    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    seq_len = hidden_states.shape[-2] + (kv_cache[0].shape[-2] if kv_cache is not None else 0)

    # rope embedding
    cos, sin = self.rotary_emb(vals, seq_len=seq_len)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos=cos, sin=sin, pos=seq_len)

    # qk hashing
    q_hash, k_hash = None, None
    if do_sparse_attn:
        if is_prefill:
            k_hash = get_hash_code(keys, self.rot_mat1, self.rot_mat2, self.relu1)
        else:
            assert k_hash_cache is not None, f"`k_hash_cache` is required in the decoding phase."
            q_hash = get_hash_code(ques, self.rot_mat1, self.rot_mat2, self.relu1)
            k_hash = get_hash_code(keys, self.rot_mat1, self.rot_mat2, self.relu1)

        if k_hash_cache is not None:
            k_hash = torch.cat([k_hash_cache, k_hash], dim=-2)

        # key hash related things
        k_hash_cache = k_hash
        k_hash = repeat_kv(k_hash, num_kv_group)

    # kv cache related things
    if kv_cache is not None:
        key_cache, val_cache = kv_cache
        keys = torch.cat([key_cache, keys], dim=-2)
        vals = torch.cat([val_cache, vals], dim=-2)
    kv_cache = (keys.data, vals.data)

    # for MQA models
    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    if do_sparse_attn and not is_prefill:

        assert ques.shape[-2] == 1, f"The number of queries in the decoding phase should always be 1 rather than {ques.shape[-2]}"

        # low precision attention based on NXOR similarity
        low_precision_attn = lsh_attn_dx_u8(q_hash, k_hash)

        # calculate the number of kv pairs to maintain
        num_kv_pair = low_precision_attn.shape[-1]
        num_remain = num_kv_pair - int(num_kv_pair * self.draft_kwargs['mask_out'])
        num_remain = max(min(num_kv_pair, self.draft_kwargs['min_remain']), num_remain)

        remain_indices = low_precision_attn.topk(k=num_remain, dim=-1).indices.unsqueeze(-1).expand(-1,-1,-1,keys.shape[-1])
        keys_subset = torch.gather(keys, dim=-2, index=remain_indices)
        vals_subset = torch.gather(vals, dim=-2, index=remain_indices)

        # =========================================================================================================
        # NOTE: test
        # diff, loss = compute_attn_supervise_loss(draft_score, true_score, max_top=None, max_oth=1024, maskout=0.98)
        # print(f"layer-{self.layer_idx}: {diff}, {loss}")
        # =========================================================================================================


        # ==================================================================================
        # NOTE: test
        # if self.draft_kwargs['bench_mark']:

        #     # 2.5 run benchmark to evaluate the performance of draft strategy
        #     true_score = get_attn_score(query=ques, key=keys, cos=cos, sin=sin)
        #     true_indices = aggregate_topk(true_score, num_remain)
        #     self.ratios = []

        #     for draft_head, true_head in zip(draft_indices[0], true_indices[0]):
        #         ratios = []

        #         for qid, (draft_query, true_query) in enumerate(zip(draft_head, true_head)):
        #             draft_set = set(draft_query[:qid + 1].tolist())
        #             true_set = set(true_query[:qid + 1].tolist())

        #             intersect = draft_set.intersection(true_set)
        #             union = draft_set.union(true_set)
        #             ratio = len(intersect) / len(union)
        #             ratios.append(ratio)
                
        #         self.ratios.append(sum(ratios) / len(ratios))
        # ==================================================================================

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=ques,
            key=keys_subset,
            value=vals_subset)

        attn_output = attn_output.transpose(1,2).flatten(2)
        return self.o_proj(attn_output), kv_cache, k_hash_cache

    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=ques,
            key=keys,
            value=vals,
            is_causal=is_prefill)
        
        attn_output = attn_output.transpose(1,2).flatten(2)
        return self.o_proj(attn_output), kv_cache, k_hash_cache


class Decoder(torch.nn.Module):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        self.decoder = get_peft_model(self.decoder, peft_config)


    @property
    def layers(self):
        if self.enable_lora:
            return self.decoder.base_model.model.model.layers
        else:
            return self.decoder.model.layers


    @property
    def model(self):
        if self.enable_lora:
            return self.decoder.base_model.model
        else:
            return self.decoder


    def reset(self):
        ...


    def __init__(
            self, 
            decoder, 
            enable_lora: bool = False,
            lora_kwargs: dict = None,
            fix_layers: list = [],
            draft_kwargs: dict = {"use_draft": False}):

        super().__init__()
        self.decoder = decoder
        self.enable_lora = False
        self.fix_layers = fix_layers
        self.draft_kwargs = draft_kwargs

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer_idx, layer in enumerate(self.layers):

            info = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype}
            
            layer.self_attn.is_fix_layer = layer_idx in fix_layers

            # modify the forward function
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            def get_rot_mat():
                rot_mats = []
                for _ in range(32):
                    rot_mats.append(random_rotation_matrix(dim=128, **info))
                return torch.stack(rot_mats, dim=0).unsqueeze(0)

            if not layer.self_attn.is_fix_layer:
                layer.self_attn.rot_mat1 = torch.nn.Parameter(get_rot_mat(), requires_grad=True)
                layer.self_attn.relu1 = torch.nn.SiLU()
                layer.self_attn.rot_mat2 = torch.nn.Parameter(get_rot_mat(), requires_grad=True)


    def is_benchmark_mode(self):
        return self.draft_kwargs['bench_mark']

    
    def enable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = True

    
    def disable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = False


    def get_ratios(self, reset=False):
        ratios = []
        for idx, layer in enumerate(self.layers):
            if idx in self.fix_layers:
                ratios.append(None)
            else:
                ratios.append(layer.self_attn.ratios)
                del layer.self_attn.ratios
        return ratios
    

    def layer_ft_params(self, layer):
        layer = self.layers[layer]
        if layer.self_attn.is_fix_layer:
            return []
        return [layer.self_attn.rot_mat1, layer.self_attn.rot_mat2]


    def ft_params(self):
        params = []

        for layer in self.layers:
            if not layer.self_attn.is_fix_layer:
                params += [
                    layer.self_attn.rot_mat1,
                    layer.self_attn.rot_mat2,
                ]

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            kv_cache=None,
            k_hash_cache=None):

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            kv_cache=kv_cache,
            k_hash_cache=k_hash_cache)

        return outputs


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder
        ):
        super().__init__()
        self.decoder = decoder

    def ft_params(self):
        params = self.decoder.ft_params()
        return params


    def reset(self):
        self.decoder.reset()


    def forward(
            self,
            input_ids,
            kv_cache=None,
            labels=None,
            local_rank=None,
            k_hash_cache=None,
            **kwargs
        ):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]

        # sign
        label_exist = labels is not None
        rank_exist = local_rank is not None

        # maybe extend the dim of input_ids
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        # put inputs-ids to the same device
        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        outputs = self.decoder(
            input_ids, 
            labels=labels,
            kv_cache=kv_cache,
            k_hash_cache=k_hash_cache)

        return outputs


class LlamaGenAcc22(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        draft_kwargs = self.conf['draft_kwargs']
        fix_layers = [] if "fix_layers" not in self.conf else self.conf["fix_layers"]
        
        decoder = Decoder(
            model, 
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            fix_layers=fix_layers,
            draft_kwargs=draft_kwargs)

        decoder = Model(decoder)

        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        return self.model.ft_params()


    def reset(self):
        self.model.reset()


    def is_benchmark_mode(self):
        return self.model.decoder.is_benchmark_mode()

    
    def enable_benchmark_mode(self):
        return self.model.decoder.enable_benchmark_mode()

    
    def disable_benchmark_mode(self):
        return self.model.decoder.disable_benchmark_mode()


    @torch.no_grad()
    def generate(self, input_ids, kv_cache=None, max_new_tokens=128, eos_token_id=[2], prof=None):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        output = self.model(input_ids=input_ids)
        logits, kv_cache, k_hash_cache = output.logits, output.past_key_values, output.past_key_hash_codes
        new_tok = logits.argmax(dim=-1)
        new_ids = [new_tok]

        # generation
        new_tok = input_ids[:, -1:]
        new_ids = []

        with prof if prof is not None else do_nothing():
            while len(new_ids) < max_new_tokens:
                output = self.model(input_ids=new_tok, kv_cache=kv_cache, k_hash_cache=k_hash_cache)
                logits, kv_cache, k_hash_cache = output.logits, output.past_key_values, output.past_key_hash_codes

                new_tok = logits.argmax(dim=-1)
                if new_tok.ravel().item() in eos_token_id: break
                new_ids.append(new_tok.ravel().item())

        self.model.reset()
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)
