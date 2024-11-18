import torch
import types
from .modify_llama import do_sdpa_attn, generate_mask, get_attn_score, check_and_apply_qk_rope, rotate_half
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple, Optional
from profiler import WallTime

import math

def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    attn_supervise: bool = False,
    attn_supervise_layers: Optional[list] = None,
    attn_supervise_reduce: Optional[int] = None,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache, draft_attn, true_attn = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache,
        attn_supervise=attn_supervise,
        attn_supervise_layers=attn_supervise_layers,
        attn_supervise_reduce=attn_supervise_reduce)
    
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

    return CausalLMOutputWithPast(
        loss=loss, 
        logits=logits, 
        past_key_values=kv_cache,
        attentions=(draft_attn, true_attn))


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    attn_supervise: bool = False,
    attn_supervise_layers: Optional[list] = None,
    attn_supervise_reduce: Optional[list] = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    draft_attns = []
    true_attns = []

    for layer_idx, (decoder_layer, kv_cache_layer) in enumerate(zip(self.layers, kv_cache)):
        if torch.is_grad_enabled():
            layer_output = checkpoint(
                decoder_layer,
                hidden_states,
                kv_cache_layer,
                attn_supervise,
                attn_supervise_layers,
                attn_supervise_reduce,
                use_reentrant=False)
        else:
            layer_output = decoder_layer(
                hidden_states, 
                kv_cache_layer,
                attn_supervise,
                attn_supervise_layers,
                attn_supervise_reduce)

        hidden_states, kv_cache_layer, draft_attn, true_attn = layer_output
        draft_attns.append(draft_attn)
        true_attns.append(true_attn)

        kv_cache[layer_idx] = kv_cache_layer

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, draft_attns, true_attns

def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    attn_supervise: bool = False,
    attn_supervise_layers: Optional[list] = None,
    attn_supervise_reduce: Optional[int] = None,
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache, draft_attn, true_attn = self.self_attn(
        hidden_states, 
        kv_cache, 
        attn_supervise,
        attn_supervise_layers,
        attn_supervise_reduce)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache, draft_attn, true_attn


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    attn_supervise: bool = False,
    attn_supervise_layers: Optional[list] = None,
    attn_supervise_reduce: Optional[int] = None,
):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    prefill_cond1 = hidden_states.shape[-2] > 1
    prefill_cond2 = kv_cache is None
    is_prefill = prefill_cond1 and prefill_cond2

    # ======================================
    # NOTE: 注释掉
    # is_fix_layer = self.q_proj2 is None
    # ======================================

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    # ==============================================================================================
    # NOTE: 注释掉
    # if not is_fix_layer:
    #     ques2 = self.q_proj2(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    #     keys2 = self.k_proj2(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)  
    #     keys2 = repeat_kv(keys2, num_kv_group)
    # ==============================================================================================

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    if kv_cache is not None:
        key_cache, val_cache = kv_cache
        keys = torch.cat([key_cache, keys], dim=-2)
        vals = torch.cat([val_cache, vals], dim=-2)
        
        # =================================================
        # NOTE: 注释掉
        # if not is_fix_layer:
        #     keys2 = torch.cat([key_cache, keys2], dim=-2)
        # =================================================

    kv_cache = (keys.data, vals.data)
    ret_attn = (None, None)

    cos, sin = self.rotary_emb(vals, seq_len=4096)
    cond1 = self.draft_kwargs['enable'] is True

    # ========================
    # NOTE: 修改
    cond2 = not self.is_fix_layer
    # cond2 = cond1
    # ========================

    if cond1 and cond2:

        # 0. apply randomly reduce
        is_reduce = isinstance(attn_supervise_reduce, int)
        random_idx = torch.randperm(ques.shape[-2])[:attn_supervise_reduce] if is_reduce is not None else None   

        def get_attn_score_using_lowrank_proj(hidden_states):
            """
            projections
            -----------
            self.mixq : (4096, 32)
            self.mixk : (4096, 32)
            self.q_pos : 4096 * [n_heads, 32, 128]
            self.k_pos : 4096 * [n_heads, 32, 128]
            """

            low_rank_q = hidden_states @ self.mixq
            low_rank_k = hidden_states @ self.mixk

            """
            low_rank_q : (m, 32)
            low_rank_k : (n, 32)
            """

            low_rank_q = low_rank_q[..., None, :].repeat(1, 1, num_heads, 1)
            low_rank_k = low_rank_k[..., None, :].repeat(1, 1, num_heads, 1)

            """
            low_rank_q : (1, m, n_heads, 32)
            low_rank_k : (1, n, n_heads, 32)
            """

            position_q = torch.stack(self.q_pos[:hidden_states.shape[-2]], dim=0)
            position_k = torch.stack(self.k_pos[:hidden_states.shape[-2]], dim=0)
            
            """
            position_q: (m, n_heads, 32, 128)
            position_k: (n, n_heads, 32, 128)
            """

            low_rank_q = torch.einsum("bmhi,mhij->bmhj", low_rank_q, position_q)
            low_rank_k = torch.einsum("bmhi,mhij->bmhj", low_rank_k, position_k)

            low_rank_q = low_rank_q.transpose(1,2)
            low_rank_k = low_rank_k.transpose(1,2)

            return low_rank_q @ low_rank_k.transpose(-1,-2)


        true_score = get_attn_score(query=ques, key=keys, cos=cos, sin=sin)
        draft_score = get_attn_score_using_lowrank_proj(hidden_states)

        # pre-filling stage should do causal attention
        if is_prefill:
            mask = generate_mask(*draft_score.shape[-2:], dtype=draft_score.dtype, device=draft_score.device)
            draft_score += mask
            if true_score is not None:
                true_score += mask

        # 2. compute the topk indices
        def aggregate_topk(x, k):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            _, x_topk = x.topk(k=k, dim=-1)
            return x_topk

        num_kv_pair = draft_score.shape[-1]
        num_remain = num_kv_pair - int(num_kv_pair * self.draft_kwargs['mask_out'])
        draft_indices = aggregate_topk(draft_score, num_remain)

        cond_a = attn_supervise_layers is None 
        cond_b = not cond_a and self.layer_idx in attn_supervise_layers

        if attn_supervise and (cond_a or cond_b):
            assert self.draft_kwargs['bench_mark'] is False

            # construct the train mask
            true_indices = aggregate_topk(true_score, num_remain)
            attn_mask = generate_mask(ques.shape[-2], keys.shape[-2], dtype=draft_score.dtype, device=draft_score.device)
            
            if is_reduce:
                attn_mask = attn_mask[..., random_idx, :]

            draft_score += attn_mask
            true_score += attn_mask
            ret_attn = (draft_score, true_score)

        if self.draft_kwargs['bench_mark']:

            # 2.5 run benchmark to evaluate the performance of draft strategy
            true_indices = aggregate_topk(true_score, num_remain)
            self.ratios = []

            for draft_head, true_head in zip(draft_indices[0], true_indices[0]):
                ratios = []

                for qid, (draft_query, true_query) in enumerate(zip(draft_head, true_head)):
                    draft_set = set(draft_query[:qid + 1].tolist())
                    true_set = set(true_query[:qid + 1].tolist())

                    intersect = draft_set.intersection(true_set)
                    union = draft_set.union(true_set)
                    ratio = len(intersect) / len(union)
                    ratios.append(ratio)
                
                self.ratios.append(sum(ratios) / len(ratios))


        # 3. discard the unimportant token while keep the important 
        if attn_supervise:
            mask = None
        else:
            mask = torch.full(
                (1, num_heads, ques.shape[-2], num_kv_pair), 
                fill_value=torch.finfo(draft_score.dtype).min, 
                dtype=draft_score.dtype, 
                device=draft_score.device)
            mask = mask.scatter_(dim=-1, index=draft_indices, value=0)

        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            mask=mask,
            out_proj=self.o_proj)

    else:
        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)

    return attn_output, kv_cache, *ret_attn


class Decoder(torch.nn.Module):

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
        for layer in self.layers:
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache


    def __init__(
            self, 
            decoder, 
            enable_lora: bool = False,
            lora_kwargs: dict = None,
            rank: int = 512,
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

        def get_reduce_mat(x, target):
            from sklearn.decomposition import PCA
            assert x.ndim == 2
            info = {
                "dtype": x.dtype,
                "device": x.device}

            x = x.float().cpu().numpy()
            pca = PCA(n_components=target)
            pca.fit(x)
            return torch.tensor(pca.components_, **info).T

        for layer_idx, layer in enumerate(self.layers):

            info = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype
            }

            # modify the forward function
            layer.self_attn.is_fix_layer = layer_idx in self.fix_layers
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            num_heads = layer.self_attn.config.num_attention_heads
            num_kv_heads = layer.self_attn.config.num_key_value_heads
            assert num_heads == num_kv_heads

            # pre process the query projection matrix
            Wq = layer.self_attn.q_proj.weight.data.T
            Dq = get_reduce_mat(Wq, rank)
            Uq = Dq.T.chunk(num_heads, dim=-1)
            Wq = Wq.unflatten(-1, (num_heads, -1)).transpose(0,1)
            Dq = Dq.unflatten(0, (num_heads, -1))
            layer.self_attn.mixq = torch.einsum("mij,mjk->ik", Wq, Dq)

            # pre process the key projection matrix
            Wk = layer.self_attn.k_proj.weight.data.T
            Dk = get_reduce_mat(Wk, rank)
            Uk = Dk.T.chunk(num_heads, dim=-1)
            Wk = Wk.unflatten(-1, (num_heads, -1)).transpose(0,1)
            Dk = Dk.unflatten(0, (num_heads, -1))
            layer.self_attn.mixk = torch.einsum("mij,mjk->ik", Wk, Dk)

            base_vecs = torch.diag_embed(torch.ones(128, **info))
            cos, sin = layer.self_attn.rotary_emb(base_vecs, seq_len=4096)

            def apply_rotary_pos_emb(mat, cos, sin, position_ids):
                cos = cos[position_ids]
                sin = sin[position_ids]
                mat_embed = (mat * cos) + (rotate_half(mat) * sin)

                return mat_embed

            import os
            if not os.path.exists(f"/data/lwh/cache/q_pos{layer_idx}.pth"):
                layer.self_attn.q_pos = [[] for _ in range(4096)]
                layer.self_attn.k_pos = [[] for _ in range(4096)]

                # prepare RoPE matrix
                for hid in range(num_heads):

                    Uq_head, Uk_head = Uq[hid], Uk[hid]

                    for pos in range(4096):
                        r11 = torch.diag_embed(cos[pos,:64])
                        r12 = torch.diag_embed(sin[pos,:64])
                        r21 = torch.diag_embed(-sin[pos,64:])
                        r22 = torch.diag_embed(cos[pos,64:])
                        rotary_mat = torch.concat([torch.concat([r11, r12], dim=-1), torch.concat([r21, r22], dim=-1)], dim=0)
                    
                        layer.self_attn.q_pos[pos].append(Uq_head @ rotary_mat)
                        layer.self_attn.k_pos[pos].append(Uk_head @ rotary_mat)

                for pos in range(4096):
                    layer.self_attn.q_pos[pos] = torch.stack(layer.self_attn.q_pos[pos], dim=0)
                    layer.self_attn.k_pos[pos] = torch.stack(layer.self_attn.k_pos[pos], dim=0)

                torch.save(layer.self_attn.q_pos, f"/data/lwh/cache/q_pos{layer_idx}.pth")
                torch.save(layer.self_attn.k_pos, f"/data/lwh/cache/k_pos{layer_idx}.pth")

            else:
                layer.self_attn.q_pos = torch.load(f"/data/lwh/cache/q_pos{layer_idx}.pth")
                layer.self_attn.k_pos = torch.load(f"/data/lwh/cache/k_pos{layer_idx}.pth")

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


    def ft_params(self):
        params = []

        for idx, layer in enumerate(self.layers):
            params += [
                layer.self_attn.q_down_proj.data,
                layer.self_attn.q_up_proj.data,
                layer.self_attn.k_down_proj.data,
                layer.self_attn.k_up_proj.data]

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            attn_supervise=False,
            attn_supervise_layers=None,
            attn_supervise_reduce=None):

        # assertions
        if attn_supervise_layers is not None:
            for layer_idx in attn_supervise_layers:
                assert layer_idx not in self.fix_layers

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            attn_supervise=attn_supervise,
            attn_supervise_layers=attn_supervise_layers,
            attn_supervise_reduce=attn_supervise_reduce)

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
            labels=None,
            local_rank=None,
            attn_supervise=False,
            attn_supervise_layers=None,
            attn_supervise_reduce=None,
            **kwargs
        ):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]

        label_exist = labels is not None
        rank_exist = local_rank is not None

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        outputs = self.decoder(
            input_ids, 
            labels=labels, 
            attn_supervise=attn_supervise,
            attn_supervise_layers=attn_supervise_layers,
            attn_supervise_reduce=attn_supervise_reduce)

        return outputs


class LlamaGenAcc10(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        rank = self.conf["rank"]

        draft_kwargs = self.conf['draft_kwargs']
        fix_layers = [] if "fix_layers" not in self.conf else self.conf["fix_layers"]
        
        decoder = Decoder(
            model, 
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            rank=rank,
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
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2]):

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        prefill_ids = input_ids[:, :-1]
        self.model(input_ids=prefill_ids)

        # generation
        new_tok = input_ids[:, -1:]
        new_ids = []
        while len(new_ids) < max_new_tokens:
            logits = self.model(input_ids=new_tok).logits
            new_tok = logits.argmax(dim=-1)
            if new_tok.ravel().item() in eos_token_id: break
            new_ids.append(new_tok.ravel().item())

        self.model.reset()
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)
