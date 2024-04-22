import logging
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import repeat, pack, unpack, rearrange 

from .config import TrainingConfig
from .prefix import PrefixTokenParameters, PrefixEmbeddingParameters
from .criterion import FinegrainLoss

class DefaultTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, num_of_tokens, max_length, split_tower_tokens = False) -> None:
        self.model = model
        if isinstance(model, nn.DataParallel):
            self.model_attr = self.model.module
        else:
            self.model_attr = self.model
        if "gpt2" in self.model_attr.config.model_type:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_of_tokens = num_of_tokens
        self.max_length = max_length
        self.split_tower_tokens = split_tower_tokens
        self.prefix = self.get_prefix()

    def get_prefix(self):
        raise NotImplementedError 

    def get_optimizer(self, training_config: TrainingConfig) -> None:
        if self.prefix is None:
            raise ValueError
        if training_config.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.prefix.parameters(), **training_config.optimizer_params)
            return
        elif training_config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.prefix.parameters(), **training_config.optimizer_params)
            return
        raise ValueError

    def get_scheduler(self, training_config: TrainingConfig):
        if training_config.learning_rate_schedule_params["total_steps"] == -1:
            training_config.learning_rate_schedule_params["total_steps"] = (len(self.train_dataset) // training_config.batch_size) * training_config.epochs
        if training_config.learning_rate_schedule == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, **training_config.learning_rate_schedule_params)
            return 
        raise ValueError

    def get_loss(self, training_config: TrainingConfig):
        if training_config.loss == "finegrain":
            self.criterion = FinegrainLoss()

    def train(self, training_config: TrainingConfig, device):
        # Create DataLoader
        train_dataloader = DataLoader(self.train_dataset, batch_size=training_config.batch_size, shuffle=True)
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = DataLoader(self.eval_dataset, batch_size=training_config.batch_size, shuffle=True)

        # Setup additionals
        self.get_loss(training_config)
        self.get_optimizer(training_config)
        self.get_scheduler(training_config)

        for epoch in range(training_config.epochs):
            logging.info(f"Begging Epoch {epoch}")
            loss_all = []
            for query, documents, best_doc in train_dataloader:
                self.optimizer.zero_grad()
                query, documents, labels = self.process_tokens(query, documents, best_doc, device)
                query_embeddings = self.get_model_embeddings(query, training_config, device)
                document_embeddings = self.get_model_embeddings(documents, training_config, device)

                # Apply Attention masks
                query_embeddings = query_embeddings * query["attention_mask"]
                document_embeddings = document_embeddings * documents["attention_mask"]

                # Unpack prefix embeddings and reshape for criterion
                _, final_query_embeddings = unpack(query_embeddings, [[self.num_of_tokens][query_embeddings.shape[1] - self.num_of_tokens]], "b * d")
                _, final_document_embeddings = unpack(document_embeddings, [[self.num_of_tokens][document_embeddings.shape[1] - self.num_of_tokens]], "b * d")
                final_document_embeddings = rearrange(final_document_embeddings, "(d b) s l -> b d s l", d=training_config.batch_size)
                final_query_embeddings = repeat(final_query_embeddings, "b s l -> b q s l", q=1)

                # Loss should be pased in the shape of batch (documents/queries) sequence_length embedding_dimension
                loss = self.criterion(final_query_embeddings, final_document_embeddings, labels)
                loss_all.append(loss.item())
                self.update_tokens(loss)
            self.scheduler.step()

            if eval_dataloader is not None:
                with torch.no_grad():
                    loss_all = []
                    for query, documents, best_doc in eval_dataloader:
                        query, documents = self.process_tokens(query, documents, best_doc, device)

                        query_embeddings = self.get_model_embeddings(query, training_config, device)
                        document_embeddings = self.get_model_embeddings(documents, training_config, device)

                        loss = self.criterion(query_embeddings, document_embeddings)
                        loss_all.append(loss.item())
                    logging.info(f"Eval Loss is {sum(loss_all)/len(loss_all)}")

        logging.info("Completed Training Run")
        if isinstance(self.prefix, PrefixEmbeddingParameters):
            return self.prefix.get_query_prefix(), self.prefix.get_document_prefix()
        raise NotImplementedError
    
    def get_model_embeddings(self, value, training_config, device):
        raise NotImplementedError

    def process_tokens(self, query, documents, best_doc, device):
        raise NotImplementedError

    def update_tokens(self, loss):
        raise NotImplementedError

class TokenTrainer(DefaultTrainer):

    def get_prefix(self):
        pass

    def process_tokens(self, query, documents, best_doc, device):
        pass

class EmbeddingTrainer(DefaultTrainer):

    def get_prefix(self):
        embedding_dimension = self._get_embedding_dimension()
        return PrefixEmbeddingParameters(num_tokens=self.num_of_tokens, split_tower_tokens=self.split_tower_tokens, embedding_dimension=embedding_dimension)

    def _get_embedding_dimension(self):
        if self.model_attr.config.model_type == "gpt2":
            return self.model_attr.config.n_embd
        raise ValueError

    def get_model_embeddings(self, value, training_config, device):
        return self.model(
            inputs_embeds=value["embeddings"].to(device),
            attention_mask=value["attention_mask"].to(device),
            output_hidden_states=True
        )["hidden_states"][training_config.layer]

    def _project_tokens(self, tokens):
        if self.model_attr.config.model_type == "gpt2":
            return self.model_attr.transformer.wte(tokens)
        raise ValueError

    def process_tokens(self, query, documents, best_doc, device):
        # Make Labels
        labels = torch.zeros((len(documents), len(documents[0])), requires_grad=False)
        for i, best in enumerate(best_doc):
            labels[best, i] = 1
        labels = rearrange(labels, "d b -> b d")

        # Flatten Documents
        documents = list(chain.from_iterable(documents))

        # Tokenize Everything
        tokenized_query = self.tokenizer(
                query,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors='pt'
        )
        tokenized_documents = self.tokenizer(
                documents,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors='pt'
        )

        # Embed Everything
        embedding_query = self._project_tokens(tokenized_query["input_ids"].to(device))
        embedding_documents = self._project_tokens(tokenized_documents["input_ids"].to(device))
        # Shape will be [B,L,D] and [B * N, L, D] where N is the number of documents per query

        # Append prefix
        query_prefix = self.prefix.get_query_prefix().to(device)
        document_prefix = self.prefix.get_document_prefix().to(device)

        query_prefix = repeat(query_prefix, "l d -> b l d", b=embedding_query.shape[0])
        document_prefix = repeat(document_prefix, "l d -> b l d", b=embedding_documents.shape[0])

        final_query, _ = pack([query_prefix, embedding_query], "b * d")
        final_documents, _ = pack([document_prefix, embedding_documents], "b * d")

        tokenized_query["embeddings"] = final_query
        tokenized_documents["embeddings"] = final_documents

        #Expand attention mask
        query_prefix_attention = torch.ones((final_query.shape[0], query_prefix.shape[1]))
        document_prefix_attention = torch.ones((final_documents.shape[0], document_prefix.shape[1]))

        final_query_attention, _ = pack([query_prefix_attention, tokenized_query["attention_mask"]], "b *")
        final_document_attention, _ = pack([document_prefix_attention, tokenized_documents["attention_mask"]], "b *")

        tokenized_documents["attention_mask"] = final_document_attention
        tokenized_query["attention_mask"] = final_query_attention

        return tokenized_query, tokenized_documents, labels


    def update_tokens(self, loss) -> None:
        loss.backward()
        self.optimizer.step()
