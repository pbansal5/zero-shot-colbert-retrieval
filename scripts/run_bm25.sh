python3 src/bm25_retrieval.py \
    --data_dir '' \
    --output_file 'jsons/wikitext_rql_32_rs_4_topK_100.json' \
    --retrieval_corpus 'wikipedia-dpr-100w' \
    --query_corpus 'wikitext' \
    --tokenizer 'gpt2' \
    --retrieval_query_length 32 \
    --retrieval_stride 4 \
    --topK 100 \
    --forbidden_titles 'jsons/wikitext_forbidden_titles.txt' \
