import json

import torch

from ralm.rerankers.base_reranker import BaseReranker 

class BertReranker(BaseReranker):
    def __init__(self, tokenizer, model, device) -> None:
        pass

    def rerank(self, query_info, k=1) -> None:
        pass
