python3 -m cbralm.zeroshot_rerank \
    --data_dir '' \
    --bm25_file 'jsons/wikitext_rql_32_rs_4_topK_100.json' \
    --retrieval_corpus 'wikipedia-dpr-100w' \
    --reranked_file 'jsons/bert_reranked_wikitext_rql_32_rs_4_topK_16.json' \
    --rerank_model 'bert-base-uncased' \
    --topK 16 \
