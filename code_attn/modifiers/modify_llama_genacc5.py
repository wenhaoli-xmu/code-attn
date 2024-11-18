import torch
import types
from .modify_llama import do_sdpa_attn, do_causal_attn
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple, Optional
import numpy as np
from multiprocessing import Pool


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_attn_score: bool = False,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache, attn_scores = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache,
        return_attn_score=return_attn_score)

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
        attentions=attn_scores)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_attn_score: bool = False
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    attn_scores = []

    for layer_idx, (decoder_layer, kv_cache_layer) in enumerate(zip(self.layers, kv_cache)):
        if torch.is_grad_enabled():
            hidden_states, kv_cache_layer, attn_score = checkpoint(
                decoder_layer,
                hidden_states,
                kv_cache_layer,
                return_attn_score,
                use_reentrant=False)
        else:
            hidden_states, kv_cache_layer, attn_score = decoder_layer(
                hidden_states, 
                kv_cache_layer,
                return_attn_score)
            
        kv_cache[layer_idx] = kv_cache_layer
        attn_scores.append(attn_score)

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, attn_scores

def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    return_attn_score: bool = False,
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache, attn_score = self.self_attn(
        hidden_states, 
        kv_cache,
        return_attn_score)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache, attn_score


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    return_attn_score: bool = False
):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    if kv_cache is not None:
        key_cache, val_cache = kv_cache
        keys = torch.cat([key_cache, keys], dim=-2)
        vals = torch.cat([val_cache, vals], dim=-2)

    kv_cache = (keys.data, vals.data)
    cos, sin = self.rotary_emb(vals, seq_len=4096)

    if return_attn_score:
        attn_output, attn_score = do_causal_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj,
            return_attn_score=return_attn_score)
    else:
        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)
        attn_score = None

    return attn_output, kv_cache, attn_score.cpu()


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

        for idx, layer in enumerate(self.layers):

            # modify the forward function
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            kwargs = {
                "dtype": layer.self_attn.q_proj.weight.dtype,
                "device": layer.self_attn.q_proj.weight.device}

            if idx not in fix_layers:
                layer.self_attn.query_down_proj = torch.nn.Parameter(torch.empty((4096, rank), **kwargs), requires_grad=True)
                layer.self_attn.key_down_proj = torch.nn.Parameter(torch.empty((4096, rank), **kwargs), requires_grad=True)
                torch.nn.init.xavier_uniform_(layer.self_attn.query_down_proj)
                torch.nn.init.xavier_uniform_(layer.self_attn.key_down_proj)
            else:
                layer.self_attn.query_down_proj = None
                layer.self_attn.key_down_proj = None

        self.enable_lora = enable_lora
        if self.enable_lora is True:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = []

        for idx, layer in enumerate(self.layers):

            if idx not in self.fix_layers:
                params += [
                    layer.self_attn.query_down_proj,
                    layer.self_attn.key_down_proj,]

            if self.enable_lora:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.k_proj.lora_A.default.weight,
                    layer.self_attn.k_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight,
                    layer.self_attn.o_proj.lora_A.default.weight,
                    layer.self_attn.o_proj.lora_B.default.weight,
                    layer.mlp.gate_proj.lora_A.default.weight,
                    layer.mlp.gate_proj.lora_B.default.weight,
                    layer.mlp.up_proj.lora_A.default.weight,
                    layer.mlp.up_proj.lora_B.default.weight,
                    layer.mlp.down_proj.lora_A.default.weight,
                    layer.mlp.down_proj.lora_B.default.weight,]

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            return_attn_score=False):

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            return_attn_score=return_attn_score)

        return outputs


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder
        ):
        super().__init__()
        self.decoder = decoder
        self.return_attn_score = False

    def ft_params(self):
        params = self.decoder.ft_params()
        return params


    def reset(self):
        self.decoder.reset()

    
    def always_return_attn_score(self):
        self.return_attn_score = True


    def forward(
            self,
            input_ids,
            labels=None,
            local_rank=None,
            return_attn_score=False,
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
            return_attn_score=return_attn_score or self.return_attn_score)

        return outputs


class LlamaGenAcc5(Modifier):
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
