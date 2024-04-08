import json

import torch
from torch import einsum

import transformers

from ralm.rerankers.base_reranker import BaseReranker 

FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi")

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"

class ColbertReranker(BaseReranker):
    def __init__(self, tokenizer, model, device, max_length, min_layer, max_layer) -> None:
        super(ColbertReranker, self).__init__(tokenizer=tokenizer, model=model, device=device)
        self.max_length = max_length
        self.min_layer = min_layer
        self.max_layer = max_layer

    def _maxsim(self, query_embed, docs_embed):
        query_embed_normalized = (
            query_embed / torch.norm(query_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )
        docs_embed_normalized = (
            docs_embed / torch.norm(docs_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )

        tokenwise_similarity = einsum('qtd,red->qrte', query_embed_normalized, docs_embed_normalized).squeeze(dim=0)

        max_over_doctokens_similarity = torch.max(
            tokenwise_similarity, dim=2
        ).values  # max_over_doctokens_similarity has shape (# of docs) x (# of query tokens)
        score = max_over_doctokens_similarity.sum(
            dim=1
        )  # sum_over_querytokens_similarity has shape (# of docs)
        return score
    
    def new_maxsim(self, query_embed, docs_embed, query, key):
        query_embed = torch.einsum('ijk,kl->ijl', query_embed, query.weight.data)
        if query.bias.data is not None:
            query_embed += query.bias.data
        docs_embed = torch.einsum('ijk,kl->ijl', docs_embed, key.weight.data)
        if key.bias.data is not None:
            docs_embed += key.bias.data

        query_embed_normalized = (
            query_embed / torch.norm(query_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )
        docs_embed_normalized = (
            docs_embed / torch.norm(docs_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )

        tokenwise_similarity = einsum('qtd,red->qrte', query_embed_normalized, docs_embed_normalized).squeeze(dim=0)

        max_over_doctokens_similarity = torch.max(
            tokenwise_similarity, dim=2
        ).values  # max_over_doctokens_similarity has shape (# of docs) x (# of query tokens)
        score = max_over_doctokens_similarity.sum(
            dim=1
        )  # sum_over_querytokens_similarity has shape (# of docs)
        return score
    
    def new_maxsim_gpt(self, embed, c_attn):
        Q,K,V = c_attn(torch.ones_like(embed)).split(self.model.config.hidden_size, dim=2)
        query_embed = torch.einsum('ijk,mjk->imj', embed[0:1], Q)
        docs_embed = torch.einsum('ijk,mjk->imj', embed[1:], K)
        
        query_embed_normalized = (
            query_embed / torch.norm(query_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )
        docs_embed_normalized = (
            docs_embed / torch.norm(docs_embed, dim=-1).clamp(min=1e-5)[:, :, None]
        )

        tokenwise_similarity = einsum('qtd,red->qrte', query_embed_normalized, docs_embed_normalized).squeeze(dim=0)

        max_over_doctokens_similarity = torch.max(
            tokenwise_similarity, dim=2
        ).values  # max_over_doctokens_similarity has shape (# of docs) x (# of query tokens)
        score = max_over_doctokens_similarity.sum(
            dim=1
        )  # sum_over_querytokens_similarity has shape (# of docs)
        return score

    def get_layers(self):
        if self.model.config.model_type in LLAMA_LIKE:
            return self.model.model.layers
        elif self.model.config.model_type.lower() in FALCON_TYPES:
            return self.model.transformer.h
        elif self.model.config.model_type == "opt":
            return self.model.model.decoder.layers
        elif self.model.config.model_type == "gpt2":
            return self.model.h
        elif self.model.config.model_type == "bert":
            return self.model.encoder.layer
        else:
            raise ValueError(MODEL_ERROR_MSG.format(self.model.config.model_type))

    def find_sublayers(self, module, layers=(torch.nn.Linear, torch.nn.Conv1d, transformers.pytorch_utils.Conv1D)):
        res = {}
        if self.model.config.model_type == "gpt2":
            for name, layer in module.named_modules():
                if isinstance(layer, layers) and ("c_attn" in name):
                    res[name] = layer

        for name, layer in module.named_modules():
            if isinstance(layer, layers) and ("k" in name or "q" in name):
                res[name] = layer
        return res

    def rerank(self, query_info, k=1) -> None:
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
            model_hidden_states = [self.model(
                tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
                output_hidden_states=True
            )["hidden_states"] for tokens in tokenized_text]

            model_hidden_states = [
                [hidden_state * tokenized_text[i]["attention_mask"][:, :, None].to(self.device) for hidden_state in query_doc_hidden_state]
                for i, query_doc_hidden_state in enumerate(model_hidden_states)
            ]

            layers = self.get_layers()
            all_sublayers = [self.find_sublayers(layer) for layer in layers]
            sequential = [list(sublayer.keys()) for sublayer in all_sublayers]
            
            # Rerank on hidden states and sublayer weights
            scores = []
            for qd_states in model_hidden_states:
                i = 0
                score_vec = []
                for layer_state in qd_states[self.min_layer:self.max_layer-1]:
                    curr_layer = all_sublayers[i]
                    if self.model.config.model_type == "gpt2":
                        score_vec.append(self.new_maxsim_gpt(layer_state, curr_layer[sequential[i][0]]))
                    else:
                        score_vec.append(self.new_maxsim(layer_state[0:1], layer_state[1:], curr_layer[sequential[i][0]], curr_layer[sequential[i][1]]))
                    i += 1
                scores.append(torch.stack(score_vec))
            
            # Rerank on hidden states
            # scores = [torch.stack([self._maxsim(layer_state[0:1], layer_state[1:]) for layer_state in qd_states[self.min_layer:self.max_layer]]) for qd_states in model_hidden_states]

            # Set Rank
            for score, query in zip(scores,query_info):
                for i, doc in enumerate(query["retrieved_docs"]):
                    doc["score"] = score[:,i].tolist()
