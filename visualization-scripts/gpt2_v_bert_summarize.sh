python summarize_results.py \
    --benchmark-dir ./benchmark \
    --sub-group bm25_retrieval_gpt2_colbert_rerank_layer bm25_retrieval_bert_based_uncased_colbert_rerank_gpt2_layer \
    --sub-group-rename 'Bert-Base' 'GPT-2 S' \
    --layer-lower 0 0\
    --layer-upper 11 11\
    --baseline baseline \
    --random bm25_random \
    --first bm25 \
    --oracle bm25_oracle \
    --retriever bm25 \
    --reranker gpt2 \
    --strategy zs-llms \
    --plot-type plot \
    --plot-style ticks \
    --plot-targets first \
    --plot-targets-rename "Top BM25"\
    --plot-save gpt2_v_bert


#  --plot-name 'GPT2 AvgSim' \
