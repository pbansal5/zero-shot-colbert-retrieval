import json
from typing import List

import torch
from torch import einsum

from ralm.rerankers.base_reranker import BaseReranker 

class ContrieverReranker(BaseReranker):
    def __init__(self, tokenizer, model, device, max_length) -> None:
        super(ContrieverReranker, self).__init__(tokenizer=tokenizer, model=model, device=device)
        self.max_length = max_length


    def _mean_pooling(token_embeddings: List[torch.Tensor], mask: List[torch.Tensor]) -> List[int]:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        document_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return document_embeddings 


    def rerank(self, query_info: dict, k:int=1) -> None:
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

        
        # Rerank the Documents
        with torch.no_grad():
            token_embeddings = [self.model(**t) for t in tokenized_text]
            text_embedding = torch.stack([
                self._mean_pooling(t[0], tokenized_text[i]["attention_mask"]) for i, t in enumerate(token_embeddings)
            ])
            scores = einsum('bqe,bde->bqd', text_embedding[:,0:1,:], text_embedding[:,1:,:])

            for score, query in zip(scores, query_info):
                for i, doc in enumerate(query["retrieved_docs"]):
                    doc["score"] = score[:,i].tolist()

