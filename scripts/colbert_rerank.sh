python rerank_retrieval_data.py \
    --reranking_type colbert \
    --model_name gpt2 \
    --batch_size 1 \
    --output_file logs/bm25_retrieval_colbert_rerank \
    --retrieved_file logs/bm25_retrieval \
    --max_length 256 \
    --num_docs_to_rank 16 
