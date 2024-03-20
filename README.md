## TODO 
- [ ] Run BM25 script on Wikitext-103
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 16 
- [ ] Add perplexity benchmarking code mimicking https://github.com/AI21Labs/in-context-ralm/tree/main
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 100


## TODO Refactoring
- [x] Add text and title embeddings in zeroshot_rerank.py (This will make it easier to plug and play on benchmark)
- [x] Need to test project setup.py
- [ ] Need to refactor model and tokenizer (Allows us to keep it consistent across files) # Not as important
- [ ] Attempt running and debugging
- [x] create logging, so that parameters can be saved to file when running the benchmark (create file_utils.py or just create object in benchmarking)

## Setup

In order to setup run the following commands. Clone the repo and then run the following after installing python=3.8:

```bash
pip install -e .
```

Then ensure that JDK=11 is installed. We can do that with conda, through the following command.


```bash
conda install openjdk=11
```

