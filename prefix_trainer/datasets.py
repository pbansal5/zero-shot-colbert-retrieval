import torch
from torch.utils.data import Dataset

class PerplexityDataset(Dataset):
    def __init__(self, retrieval_info, best_doc):
        self.retrieval_info = retrieval_info
        self.best_doc = best_doc

    def __len__(self):
        return len(self.retrieval_info)

    def __getitem__(self, index):
        query = self.retrieval_info[index]["query"]
        documents = self.retrieval_info[index]["retrieved_docs"]
        documents = [d["title"] + "\n" + d["text"] if "title" in d else d["text"] for d in documents]
        return query, documents, self.best_doc[index]

