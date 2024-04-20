import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import repeat

from config import TrainingConfig
from prefix import PrefixTokenParameters, PrefixEmbeddingParameters
from criterion import FinegrainLoss

class DefaultTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, num_of_tokens, max_length, split_tower_tokens = False) -> None:
        self.model = model
        if isinstance(model, nn.DataParallel):
            self.model_attr = self.model.module
        else:
            self.model_attr = self.model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_of_tokens = num_of_tokens
        self.max_length = max_length
        self.split_tower_tokens = split_tower_tokens
        self.prefix = None

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
        if training_config.learning_rate_schedule == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(**training_config.learning_rate_schedule_params)
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
        self.get_scheduler(training_config)
        self.get_optimizer(training_config)

        for epoch in range(training_config.epochs):
            logging.info(f"Begging Epoch {epoch}")
            loss_all = []
            for query, documents, best_doc in train_dataloader:
                self.optimizer.zero_grad()
                query, documents = self.process_tokens(query, documents, best_doc)
                query_embeddings = self.get_model_embeddings(query, training_config, device)
                document_embeddings = self.get_model_embeddings(documents, training_config, device)

                # Apply Attention masks
                print("FIX ME")

                loss = self.criterion(query_embeddings, document_embeddings)
                loss_all.append(loss.item())
                self.update_tokens(loss)
            self.scheduler.step()

            if eval_dataloader is not None:
                with torch.no_grad():
                    loss_all = []
                    for query, documents, best_doc in eval_dataloader:
                        query, documents = self.process_tokens(query, documents, best_doc)

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

    def process_tokens(self, query, documents, best_doc):
        raise NotImplementedError

    def update_tokens(self, loss):
        raise NotImplementedError

class TokenTrainer(DefaultTrainer):
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, num_of_tokens, max_length, split_tower_tokens = False) -> None:
        self.model = model
        if isinstance(model, nn.DataParallel):
            self.model_attr = self.model.module
        else:
            self.model_attr = self.model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_of_tokens = num_of_tokens
        self.max_length = max_length
        self.split_tower_tokens = split_tower_tokens
        # Need to add custom init functionality
        self.prefix = PrefixTokenParameters(num_tokens=num_of_tokens, split_tower_embeddings=split_tower_tokens)

    def process_tokens(self, query, documents, best_doc):
        pass

class EmbeddingTrainer(DefaultTrainer):
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, num_of_tokens, max_length, split_tower_tokens = False) -> None:
        self.model = model
        if isinstance(model, nn.DataParallel):
            self.model_attr = self.model.module
        else:
            self.model_attr = self.model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_of_tokens = num_of_tokens
        self.max_length = max_length
        self.split_tower_tokens = split_tower_tokens
        # Need to add custom init functionality
        embedding_dimension = self._get_embedding_dimension()
        self.prefix = PrefixEmbeddingParameters(num_tokens=num_of_tokens, split_tower_tokens=split_tower_tokens, embedding_dimension=embedding_dimension)

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
            return self.model_attr.wte(tokens)
        raise ValueError

    def process_tokens(self, query, documents, best_doc):
        # Make the best document first
        for docs, best in zip(documents, best_doc):
            docs[0], docs[best] = docs[best], docs[0]

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
        embedding_query = self._project_tokens(tokenized_query)
        embedding_documents = self._project_tokens(tokenized_documents)

        # Append prefix
        query_prefix = self.prefix.get_query_prefix()
        document_prefix = self.prefix.get_document_prefix()

        query_prefix = repeat(query_prefix, "l d -> b q l d", b=embedding_query.shape[0], q=1)
        document_prefix = repeat(document_prefix, "l d -> b q l d", b=embedding_documents.shape[0], q=embedding_documents.shape[1])

        final_query = torch.cat((query_prefix, embedding_query), dim=-2)
        final_documents = torch.cat((document_prefix, embedding_documents), dim=-2)

        tokenized_query["embeddings"] = final_query
        tokenized_documents["embeddings"] = final_documents

        return tokenized_query, tokenized_documents


    def update_tokens(self, loss) -> None:
        loss.backward()
        self.optimizer.step()
