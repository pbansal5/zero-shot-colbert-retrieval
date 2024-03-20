import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


def print_args(args, output_dir=None, output_file=None, retrieval_info=None):
    assert output_dir is None or output_file is None

    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None or output_file is not None:
        output_file = output_file or os.path.join(output_dir, "args.txt")
        with open(output_file, "w") as f:
            f.write(f" **************** Benchmark Params **************** ")
            for key, val in sorted(vars(args).items()):
                keystr = "{}".format(key) + (" " * (30 - len(key)))
                f.write(f"{keystr}   {val}\n")
            if retrieval_info is not None:
                bm25_logging_info = retrieval_info["bm25_logging_info"]
                reranking_args = retrieval_info["reranking_args"]
                f.write(f"\n\n\n")
                f.write(f" **************** BM25 Logging INFO **************** ")
                for key, val in sorted(bm25_logging_info.items()):
                    keystr = "{}".format(key) + (" " * (30 - len(key)))
                    f.write(f"{keystr}   {val}\n")

                f.write(f"\n\n\n")
                f.write(f" **************** Reranking Logging INFO **************** ")
                for key, val in sorted(reranking_args.items()):
                    keystr = "{}".format(key) + (" " * (30 - len(key)))
                    f.write(f"{keystr}   {val}\n")
