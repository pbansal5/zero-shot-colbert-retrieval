import json
from typing import Dict

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

import transformers

from ralm.rerankers.base_reranker import BaseReranker 

FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi")

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt', 'gpt' and 'falcon' are supported"

class ColLLMReranker(BaseReranker):
    def __init__(
        self,
        tokenizer,
        model,
        device,
        max_length: int,
        min_layer: int,
        max_layer: int,
        attention: bool = False,
        similarity: str = "max"
    ) -> None:
        super(ColLLMReranker, self).__init__(tokenizer=tokenizer, model=model, device=device)
        self.max_length = max_length
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.attention = attention
        self.similarity = similarity
        if isinstance(model, nn.DataParallel):
            self.model_attr = self.model.module
        else:
            self.model_attr = self.model
        if self.attention:
            n_heads, n_key_heads = self._get_num_heads()
            self.n_heads = n_heads
            self.n_key_heads = n_key_heads

    def _apply_similarity(self, token_similarity: torch.Tensor) -> torch.Tensor:
        if self.similarity == "max":
            return torch.max(token_similarity, dim=-1).values
        elif self.similarity == "avg":
            return torch.mean(token_similarity, dim=-1)
        elif self.similarity == "topk-avg":
            return torch.mean(torch.topk(token_similarity, 2).values, dim=-1)
        else:
            raise ValueError(f"Unknown Similarity Metric {self.similarity}")

    def _maxsim(self, query_embed: torch.Tensor, docs_embed: torch.Tensor):
        import pdb; pdb.set_trace()
        query_embed_normalized = query_embed / torch.norm(query_embed, dim=-1, keepdim=True).clamp(min=1e-5)
        docs_embed_normalized = docs_embed / torch.norm(docs_embed, dim=-1, keepdim=True).clamp(min=1e-5)

        tokenwise_similarity = einsum('qtd,red->qrte', query_embed_normalized, docs_embed_normalized).squeeze(dim=0)

        token_score = self._apply_similarity(tokenwise_similarity)
        score = token_score.sum(dim=-1) 
        return score
    
    def _get_query_key_projections(self, query_embed: torch.Tensor, doc_embed: torch.Tensor, projections):
        # proj_arr = [name for name in sequential if str(i) == name.split("_")[-1]]
        # curr_layer = [all_sublayers[i][name] for name in proj_arr]
        model_name = self.model_attr.config.model_type
        if model_name == "gpt2":
            
            for key, _ in projections.items():
                if "ln" in key:
                    query_embed = projections[key](query_embed)
                    doc_embed = projections[key](doc_embed)

            k = False
            q = False
            for key, _ in projections.items():
                if "query" in key:
                    q = True
                    q_key = key
                if "key" in key:
                    k = True
                    k_key = key
            
            if q and k:
                q_attn, c_attn = projections[q_key], projections[k_key]
                query_proj = q_attn(query_embed)
                docs_proj, _= c_attn(doc_embed).split(self.model_attr.config.hidden_size, dim=2)
            else:
                c_attn = projections[k_key]
                query_proj, _, _ = c_attn(query_embed).split(self.model_attr.config.hidden_size, dim=2)
                _, docs_proj, _ = c_attn(doc_embed).split(self.model_attr.config.hidden_size, dim=2)

        # if model_name == "gpt2":
        #     for key, value in projections.items():
        #         if "query" not in projections.values():
        #             c_attn = projections["key"]
        #             query_proj, _, _ = c_attn(query_embed).split(self.model_attr.config.hidden_size, dim=2)
        #             _, docs_proj, _ = c_attn(doc_embed).split(self.model_attr.config.hidden_size, dim=2)
        #         else:
        #             q_attn, c_attn = projections["query"], projections["key"]
        #             query_proj = q_attn(query_embed)
        #             docs_proj, _= c_attn(doc_embed).split(self.model_attr.config.hidden_size, dim=2)
        elif model_name in [*LLAMA_LIKE, *FALCON_TYPES, "opt", "bert"]:
            for key, _ in projections.items():
                if "query" in key:
                    proj_q = projections[key]
                if "key" in key:
                    proj_k = projections[key]
            query_projection, key_projection = proj_q, proj_k
            query_proj = query_projection(query_embed)
            docs_proj = key_projection(doc_embed)
        else:
            raise ValueError(MODEL_ERROR_MSG.format(model_name))
        return query_proj, docs_proj
    
    def _get_num_heads(self):
        if self.model_attr.config.model_type == "gpt2":
            n_heads = self.model_attr.config.n_head
        else:
            n_heads = self.model_attr.config.n_heads
        try:
            n_key_heads = self.model_attr.config.num_key_value_heads
        except AttributeError:
            n_key_heads = n_heads
        return n_heads, n_key_heads
    
    def _maxsim_attention(
        self,
        query_embed: torch.Tensor,
        doc_embed: torch.Tensor,
        projections,
        attention_mask: torch.Tensor
        ) -> torch.Tensor :
        query_proj, docs_proj = self._get_query_key_projections(query_embed, doc_embed, projections)
        query_proj = query_proj * attention_mask[0:1,:,:] # If projection has a bias we need to zero out padded tokens
        docs_proj = docs_proj * attention_mask[1:,:,:]

        # Get number of attention_heads
        n_heads, n_key_heads = self.n_heads, self.n_key_heads

        query_states = rearrange(query_proj, "b s (h d) -> b h s d", h=n_heads)
        key_states = rearrange(docs_proj, "b s (h d) -> b h s d", h=n_key_heads)

        norm_query_states = query_states / torch.norm(query_states, dim=-1, keepdim=True).clamp(min=1e-5)
        norm_key_states = key_states / torch.norm(key_states, dim=-1, keepdim=True).clamp(min=1e-5)

        token_similarity = einsum("bhqe,dhke->bdhqk", norm_query_states, norm_key_states).squeeze(dim=0)

        token_score = self._apply_similarity(token_similarity)
        score = token_score.sum(dim=(1, 2))
        return score
    
    def _get_layers(self):
        if self.model_attr.config.model_type in LLAMA_LIKE:
            return self.model_attr.model.layers
        elif self.model_attr.config.model_type.lower() in FALCON_TYPES:
            return self.model_attr.transformer.h
        elif self.model_attr.config.model_type == "opt":
            return self.model_attr.model.decoder.layers
        elif self.model_attr.config.model_type == "gpt2":
            return self.model_attr.transformer.h
        elif self.model_attr.config.model_type == "bert":
            return self.model_attr.encoder.layer
        else:
            raise ValueError(MODEL_ERROR_MSG.format(self.model_attr.config.model_type))

    def _find_sublayers(self, module: nn.Module, i, layers=(torch.nn.Linear, torch.nn.Conv1d, transformers.pytorch_utils.Conv1D)):
        res = {}
        if self.model_attr.config.model_type == "gpt2":
            for name, layer in module.named_modules():
                if isinstance(layer, layers) and ("c_attn" in name or "q_attn" in name or "ln_1" in name):
                    if "c_attn" in name:
                        res[name+"_key_"+str(i)] = layer
                    if "q_attn" in name:
                        res[name+"_query_"+str(i)] = layer
                    if "ln_1" in name:
                        res[name+"_ln_"+str(i)] = layer

        else:
            for name, layer in module.named_modules():
                if isinstance(layer, layers) and ("k" in name or "q" in name):
                    if "k" in name:
                        res[name+"_key_"+str(i)] = layer
                    elif "q" in name:
                        res[name+"_query_"+str(i)] = layer
                    # Need to add layernorm
        return res

    def rerank(self, query_info: dict, k=1) -> None:
        query = [q["query"] for q in query_info]
        documents = [q["retrieved_docs"][:min(len(q["retrieved_docs"]), k)] for q in query_info]
        text = [[q] + [d["title"] + "\n" + d["text"] if "title" in d else d["text"] for d in docs] for q,docs in zip(query, documents)]

        # Tokenize the input
        tokenized_text = [self.tokenizer(
            t,
            truncation=True,
            max_length=self.max_length,
            padding="max_length", 
            return_tensors='pt'
        ) for t in text]

        with torch.no_grad():
            model_hidden_states = [self.model( tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
                output_hidden_states=True
            )["hidden_states"] for tokens in tokenized_text]

            model_hidden_states = [
                [hidden_state * tokenized_text[i]["attention_mask"][:, :, None].to(self.device) for hidden_state in query_doc_hidden_state]
                for i, query_doc_hidden_state in enumerate(model_hidden_states)
            ]

            
            # Rerank on hidden states and sublayer weights
            if self.attention:
                layers = self._get_layers()
                # list of dicts conatining the key = layer_name and value = corresponding layer
                all_sublayers = [self._find_sublayers(layers[i], i) for i in range(len(layers))]
                scores = []
                for m, qd_states in enumerate(model_hidden_states):
                    i = 0
                    score_vec = []
                    for layer_state in qd_states[self.min_layer:self.max_layer]:
                        mask = tokenized_text[m]["attention_mask"][:,:,None].to(self.device)
                        score_vec.append(self._maxsim_attention(layer_state[0:1], layer_state[1:], all_sublayers[i], mask))
                        i += 1
                    scores.append(torch.stack(score_vec))
            
            # Rerank on hidden states
            else:
                scores = [torch.stack([self._maxsim(layer_state[0:1], layer_state[1:]) for layer_state in qd_states[self.min_layer:self.max_layer+1]]) for qd_states in model_hidden_states]

            # Set Rank
            for score, query in zip(scores,query_info):
                for i, doc in enumerate(query["retrieved_docs"]):
                    doc["score"] = score[:,i].tolist()
