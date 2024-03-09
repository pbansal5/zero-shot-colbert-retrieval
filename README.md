## TODO 
- [ ] Run BM25 script on Wikitext-103
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 16 
- [ ] Add perplexity benchmarking code mimicking https://github.com/AI21Labs/in-context-ralm/tree/main
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 100


## TODO Refactoring
- [ ] Add text and title embeddings in zeroshot_rerank.py (This will make it easier to plug and play on benchmark)
- [ ] Need to test project setup.py
- [ ] Need to refactor model and tokenizer (Allows us to keep it consistent across files)
- [ ] Attempt running and debugging
- [ ] create logging, so that parameters can be saved to file when running the benchmark (create file_utils.py or just create object in benchmarking)


## Additional Notes

Do we need to run BM25 on a large retrieval (i.e. top 1000?) so that we thin down the corpus, but don't attribute the benefit to retrieval of bm25

Alternatively, we can just use bm25 retrieval with colbert reranking?

This is something we need to talk about, otherwise it would be hard to attribute where we are getting the benefits from.
