import torch


class FinegrainLoss:
    def __call__(self, query_embeddings, document_embeddings):
        norm_query_embeddings = query_embeddings / torch.norm(query_embeddings, dim=-1, keepdim=True)
        norm_document_embeddings = document_embeddings / torch.norm(document_embeddings, dim=-1, keepdim=True)

        similarity = torch.einsum('bqse,bdce->bqdsc', norm_query_embeddings, norm_document_embeddings)
        score = similarity.amax(dim=-1).sum(dim=-1)
        softmax_score = torch.softmax(score, dim=-1)[:,:,0].squeeze(dim=(-1,-2)) # First Doc should be our golden doc
        return softmax_score.sum(dim=0)
