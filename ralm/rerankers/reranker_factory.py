def add_reranker_args(parser, reranker_type):
    if reranker_type == "zs-llms":
        parser.add_argument("--max_length", type=int, default=256)
        parser.add_argument("--min_layer", type=int, default=0)
        parser.add_argument("--max_layer", type=int, default=-1)
        parser.add_argument("--attention", action="store_true", default=False)
        parser.add_argument("--similarity", type=str, default="max")
    elif reranker_type == "contriever":
        parser.add_argument("--max_length", type=int, default=256)
    elif reranker_type == "finegrain":
        raise NotImplementedError
    elif reranker_type == "coarse":
        raise NotImplementedError
    else:
        raise ValueError


def get_reranker(reranker_type, args, tokenizer, model, device):
    if reranker_type == "zs-llms":
        from ralm.rerankers.zsllm_reranker import ColLLMReranker 
        if "gpt2" in args.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        if "llama" in args.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        return ColLLMReranker(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            attention=args.attention,
            similarity=args.similarity
        )
    elif reranker_type == "colbert-attention":
        raise NotImplementedError
    elif reranker_type == "coarse":
        raise NotImplementedError
    elif reranker_type == 'contriever':
        from ralm.rerankers.contriever_reranker import ContrieverReranker
        return ContrieverReranker(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length
        )
    elif reranker_type == 'spider':
        raise NotImplementedError
    elif reranker_type == 'dpr':
        raise NotImplementedError
    raise ValueError

