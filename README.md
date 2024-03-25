## Setup

In order to setup run the following commands. Clone the repo and then run the following after installing python=3.8:

```bash
pip install -e .
```

Then ensure that JDK=11 is installed. We can do that with conda, through the following command.


```bash
conda install openjdk=11
```

## DeBugging TODO
- [ ] Split up reranking so that each layer gets it's own json file. (Current creates 64GB of overhead ~ 180 GB >= of RAM)

## TODO 
- [x] Run BM25 script on Wikitext-103
- [x] Run Re-rankers (BERT-base and GPT2-small) on top 16 
- [x] Add perplexity benchmarking code mimicking https://github.com/AI21Labs/in-context-ralm/tree/main
- [x] Run Re-rankers (BERT-base and GPT2-small) on top 100

## Running Retrieval


## Running ReRanking


## Benchmarking


