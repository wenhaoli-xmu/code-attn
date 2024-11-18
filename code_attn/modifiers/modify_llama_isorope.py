from ..modifier import Modifier
from .modify_llama import check_and_apply_rope, generate_mask, check_and_apply_qk_rope, do_sdpa_attn
from peft import LoraConfig, get_peft_model, TaskType


import torch
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv

from sklearn.decomposition import PCA
import math


def get_reduce_mat(x, target):
    assert x.ndim == 2
    info = {
        "dtype": x.dtype,
        "device": x.device}

    x = x.float().cpu().numpy()
    pca = PCA(n_components=target)
    pca.fit(x)
    return torch.tensor(pca.components_, **info).T

def reconstruct(x, rank):
    pca = PCA(n_components=rank)
    x_np = x.cpu().float().numpy()
    pca.fit(x_np)
    y_np = pca.components_
    z_np = x_np @ y_np.T @ y_np
    z = torch.tensor(z_np, dtype=x.dtype, device=x.device)
    return z


def merge_wq_wk(q_proj, k_proj, q_down, k_down, n_heads=32):
    head_wise_wq = q_proj.weight.data.T.chunk(n_heads, dim=-1)
    head_wise_wk = k_proj.weight.data.T.chunk(n_heads, dim=-1)
    head_wise_wq = torch.stack(head_wise_wq, dim=0)
    head_wise_wk = torch.stack(head_wise_wk, dim=0)

    head_wise_wq = torch.einsum("ij,hjk->hik", q_down.T, head_wise_wq)
    head_wise_wk = torch.einsum("ij,hjk->hik", k_down.T, head_wise_wk)
    merged = torch.einsum("hij,hjk->hik", head_wise_wq, head_wise_wk.transpose(-1,-2))
    
    return merged




def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_caches: torch.Tensor = None,
    **kwargs
):
    rets = self.model(
        input_ids=input_ids,
        kv_caches=kv_caches)

    hidden_states = rets
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

    return CausalLMOutputWithPast(loss=loss, logits=logits)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_caches: torch.Tensor = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_caches is None:
        kv_caches = [[None] * len(self.layers)] * 2

    for decoder_layer, key_cache, val_cache in zip(self.layers, *kv_caches):
        hidden_states = decoder_layer(
            hidden_states,
            (key_cache, val_cache))

    hidden_states = self.norm(hidden_states)
    return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor = None,
):
    device = self.self_attn.v_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
):
    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    head_dim = embed_dim // num_heads

    cond1 = hasattr(self.config, "max_sequence_length")
    cond2 = hasattr(self.config, "max_position_embeddings")
    cond3 = hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None
    max_pos_embed = int(max(
        self.config.max_sequence_length if cond1 else 0, 
        self.config.max_position_embeddings if cond2 else 0,
        self.config.max_position_embeddings * self.config.rope_scaling["factor"] if cond2 and cond3 else 0))

    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = self.config.num_attention_heads // num_kv_heads

    if self.is_fix_layer:
        ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
        keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    elif self.is_merged:
        ques = hidden_states @ self.q_down
        keys = hidden_states @ self.k_down
    else:
        q_proj = self.q_proj.weight.data @ self.q_down @ self.q_down.T
        k_proj = self.k_proj.weight.data @ self.k_down @ self.k_down.T
        ques = (hidden_states @ q_proj.T).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
        keys = (hidden_states @ k_proj.T).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    cos, sin = self.rotary_emb(vals, seq_len=max_pos_embed)

    # =====================================================
    # TODO: 补全这里的kv cache以及GQA机制

    # if kv_cache is not None and kv_cache[0] is not None:
    #     k_cache, v_cache = kv_cache
    #     keys = torch.cat([k_cache, keys], dim=-2)
    #     vals = torch.cat([v_cache, vals], dim=-2)

    # keys = repeat_kv(keys, num_kv_group)
    # vals = repeat_kv(vals, num_kv_group)
    # =====================================================

    
    def do_causal_attn(query, key, value, cos, sin, out_proj = None):

        info = {
            "device": query.device,
            "dtype": query.dtype}
        
        # # 1. 计算RoPE偏移量
        iso_que = torch.ones((1, self.config.num_attention_heads, query.shape[-2], head_dim), **info)
        iso_key = torch.ones((1, self.config.num_attention_heads, key.shape[-2], head_dim), **info)
        iso_que, iso_key = check_and_apply_qk_rope(iso_que, iso_key, cos, sin)
        iso_attn = iso_que @ iso_key.transpose(-1,-2) / math.sqrt(head_dim)

        if self.is_merged is False:
            # 2. 将RoPE的偏移量融合进入attention maks中
            attn_mask = generate_mask(ques.shape[-2], key.shape[-2], query.dtype, query.device)
            attn_mask = iso_attn * self.gamma + attn_mask
            attn_mask = torch.clamp(attn_mask, min=torch.finfo(attn_mask.dtype).min)

            # 3. 使用sdpa计算attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                is_causal=False)

        else:
            # 2. 计算attention matrix
            attn_score = torch.einsum("ni,hij,jm->hnm", query.squeeze(0), self.merged, key.squeeze(0).transpose(-1,-2))
            attn_score = attn_score.unsqueeze(0) / math.sqrt(head_dim)

            # 3. 加上偏置，通过softmax
            attn_score = attn_score + iso_attn * self.gamma
            attn_mask = generate_mask(ques.shape[-2], key.shape[-2], query.dtype, query.device)
            attn_score = torch.softmax(attn_score + attn_mask, dim=-1, dtype=torch.float32).type(query.dtype)

            attn_output = attn_score @ value

        attn_output = attn_output.transpose(1,2).flatten(2)
        if out_proj is not None:
            attn_output = out_proj(attn_output)

        return attn_output

    if not self.is_fix_layer:
        attn_output = do_causal_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos, sin=sin,
            out_proj=self.o_proj)
    else:
        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)
        
    return attn_output


class LlamaIsoRoPE(Modifier):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn)\.(q|k)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        self.decoder = get_peft_model(self.model, peft_config)


    def __init__(self, model, save_ckp, load_ckp, config):
        super().__init__(model, save_ckp, load_ckp)
        # self._init_lora(lora_rank=128, lora_alpha=256, lora_dropout=0)

        fix_layers = [0,1]

        import types
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer_idx, layer in enumerate(self.model.model.layers):

            kwargs = {
                "dtype": layer.self_attn.q_proj.weight.data.dtype,
                "device": layer.self_attn.q_proj.weight.data.device
            }

            layer.self_attn.is_fix_layer = layer_idx in fix_layers
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            if not layer.self_attn.is_fix_layer:
                """
                不在[0,1]中才可以进行rope解耦合 + 降维的操作
                """
                layer.self_attn.gamma = torch.full((1, 32, 1, 1), 0.4, **kwargs)
                layer.self_attn.is_merged = False

                layer.self_attn.q_down = get_reduce_mat(layer.self_attn.q_proj.weight.data, 64)
                layer.self_attn.k_down = get_reduce_mat(layer.self_attn.k_proj.weight.data, 64)

        # self.reconstruct_params()

    
    def reconstruct_params(self):
        for layer in self.model.model.layers:
            if not layer.self_attn.is_fix_layer:
                layer.self_attn.is_merged = True
                layer.self_attn.merged = merge_wq_wk(
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.q_down,
                    layer.self_attn.k_down,
                    self.model.config.num_attention_heads)
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj


    def ft_params(self):
        params = []

        for layer in self.model.model.layers:
            if not layer.self_attn.is_fix_layer:
                params += [
                    layer.self_attn.gamma,
                    layer.self_attn.q_down,
                    layer.self_attn.k_down]
            
        return params

    
    def reset(self):
        pass


    def forward(self, input_ids, pos=None, labels=None, **kwargs):
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        elif isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)[None, :]

        if labels is not None:
            if isinstance(labels, torch.Tensor) and labels.ndim == 3:
                labels = labels.flatten(0, 1)
            elif isinstance(labels, list):
                labels = torch.tensor(labels)[None, :]

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        return self.model(input_ids=input_ids, labels=labels, pos=pos)
