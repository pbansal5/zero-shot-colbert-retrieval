## Setup

In order to setup run the following commands. Clone the repo and then run the following after installing python=3.8:

```bash
pip install -e .
```

Then ensure that JDK=11 is installed. We can do that with conda, through the following command.


```bash
conda install openjdk=11
```

## TODO 
- [x] Run BM25 script on Wikitext-103
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 16 
- [x] Add perplexity benchmarking code mimicking https://github.com/AI21Labs/in-context-ralm/tree/main
- [ ] Run Re-rankers (BERT-base and GPT2-small) on top 100


