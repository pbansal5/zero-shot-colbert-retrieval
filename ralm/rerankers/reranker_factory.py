def add_reranker_args(parser, reranker_type):
    if reranker_type == "colbert":
        parser.add_argument("--max_length", type=int, default=256)
        parser.add_argument("--min_layer", type=int, default=0)
        parser.add_argument("--max_layer", type=int, default=-1)
    else:
        raise ValueError


def get_reranker(reranker_type, args, tokenizer, model, device):
    if reranker_type == "colbert":
        from ralm.rerankers.colbert_reranker import ColbertReranker
        if args.model_name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
        return ColbertReranker(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            min_layer=args.min_layer,
            max_layer=args.max_layer
        )
    raise ValueError
