class BaseReranker:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def rerank(self, query_info, k):
        raise NotImplementedError
