import torch
import torch.nn as nn

class PrefixEmbeddingParameters(nn.Module):
    def __init__(self, num_tokens, embedding_dimension, split_tower_tokens):
        super().__init__()
        self.prefix1 = nn.Parameter(torch.rand(num_tokens, embedding_dimension))
        self.prefix2 = None
        if split_tower_tokens:
            self.prefix2 = nn.Parameter(torch.rand(num_tokens, embedding_dimension))

    def get_query_prefix(self):
        return self.prefix1

    def get_document_prefix(self):
        if self.prefix2 is not None:
            return self.prefix2
        return self.prefix1

class PrefixTokenParameters(nn.Module):
    def __init__(self, num_tokens, split_tower_embeddings, init_string = None):
        if init_string is None:
            self.prefix1 = "!" * num_tokens
            self.prefix2 = None
            if split_tower_embeddings:
                self.prefix2 = "!" * num_tokens
        else:
            if split_tower_embeddings:
                self.prefix1 = init_string[0]
                self.prefix2 = init_string[1]
            else:
                self.prefix1 = init_string
                self.prefix2 = None

    def  get_query_prefix(self):
        return self.prefix1

    def get_document_prefix(self):
        return self.prefix2
