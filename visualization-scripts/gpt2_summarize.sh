python summarize_results.py \
    --benchmark-dir ./benchmark \
    --sub-group bm25_retrieval_gpt2_colbert_rerank_layer bm25_retrieval_gpt2_medium_colbert_rerank_gpt2_layer bm25_retrieval_gpt2_large_colbert_rerank_gpt2_layer bm25_retrieval_gpt2_xlarge_colbert_rerank_gpt2_layer\
    --sub-group-rename 'GPT-2 S' 'GPT-2 M' 'GPT-2 L' 'GPT-2 XL'\
    --layer-lower 0 0 0 0\
    --layer-upper 11 23 35 47\
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
    --plot-save gpt2_comparison


#  --plot-name 'GPT2 AvgSim' \
