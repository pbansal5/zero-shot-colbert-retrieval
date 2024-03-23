python3 -m cbralm.zeroshot_rerank \
    --data-dir '' \
    --project-name 'logs/wikitext_rql_32_rs_4_topK_100' \
    --run-name 'bert_reranked_wikitext_rql_32_rs_4_topK_16' \
    --retrieval-corpus 'wikipedia-dpr-100w' \
    --rerank-model 'bert-base-uncased' \
    --topK 16 \
    --batch-size 8 \
