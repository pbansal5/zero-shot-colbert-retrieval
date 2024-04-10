python summarize_results.py \
    --benchmark-dir ./benchmark \
    --sub-group bm25_retrieval_gpt2_avgsim_rerank_gpt2_layer bm25_retrieval_gpt2_colbert_rerank_layer\
    --sub-group-rename AvgSim 'MaxSim(GPT-2 S)'\
    --layer-lower 0 0 \
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
    --indicate-best \
    --plot-save gpt2_small_avgsim


#  --plot-name 'GPT2 AvgSim' \
