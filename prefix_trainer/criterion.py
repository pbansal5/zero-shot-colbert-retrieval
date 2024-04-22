import torch
import torch.nn.functional as F


class FinegrainLoss:
    def __call__(self, query_embeddings, document_embeddings, labels):
        norm_query_embeddings = F.normalize(query_embeddings, dim=-1)
        norm_document_embeddings = F.normalize(document_embeddings, dim=-1)

        similarity = torch.einsum('bqse,bdce->bqdsc', norm_query_embeddings, norm_document_embeddings)
        score = similarity.amax(dim=-1).sum(dim=-1).squeeze(dim=1) # "b d" shape
        cross_entropy_score = (labels * (-F.log_softmax(score, dim=-1))).sum(dim=-1)
        return cross_entropy_score.mean(dim=0)
