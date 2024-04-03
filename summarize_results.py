import os
import json
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable

PLOT_CHOICES = [
    "plot",
    "bar",
]

TARGET_CHOICES = ["baseline", "random", "first", "oracle"]

TARGET_COLORS = {
    "baseline": "red",
}


def main(args):
    runs = [
        name
        for name in os.listdir(args.benchmark_dir)
        if os.path.isdir(os.path.join(args.benchmark_dir, name))
    ]
    assert (
        args.sub_group is None or args.run_names is None
    ), "Can't have sub group and run names both set"
    assert (
        args.sub_group is not None or args.run_names is not None
    ), "Need to specify a sub group or run names"

    if isinstance(args.sub_group, str):
        args.sub_group = [args.sub_group]

    group_runs = []
    if args.sub_group is not None:
        group_runs = [
            [name for name in runs if group in name] for group in args.sub_group
        ]

    if args.run_names is not None:
        group_runs = [[name for name in runs if name in args.run_names]]

    if args.layer_lower is not None or args.layer_upper is not None:
        assert (
            args.layer_lower is not None and args.layer_upper is not None
        ), "Need both lower layer and upper layer set"

        group_runs = [
            [
                name
                for name in group
                if int(name.strip().split("_")[-1])
                in range(args.layer_lower, args.layer_upper + 1)
            ]
            for group in group_runs
        ]
        group_runs = [
            sorted(group, key=lambda x: int(x.strip().split("_")[-1]))
            for group in group_runs
        ]

    results = [
        [
            json.load(open(os.path.join(args.benchmark_dir, name, "eval.json")))
            for name in group
        ]
        for group in group_runs
    ]

    targets = {}
    target_names = [args.baseline, args.random, args.first, args.oracle]
    for target in target_names:
        if target is not None:
            targets[target] = json.load(
                open(os.path.join(args.benchmark_dir, target, "eval.json"))
            )["eval_perplexity"]

    # Sub groups assume similar hyper-parameters of benchmarking
    if args.sub_group is not None:
        for name, runs in zip(args.sub_group, group_runs):
            print(f"********** Configuration {name} **********")
            with open(os.path.join(args.benchmark_dir, runs[0], "args.txt"), "r") as f:
                for line in f:
                    print(line)
            print(f"********** Configuration {name} **********")

    vals = [[run["eval_perplexity"] for run in r] for r in results]

    # Summarize results into a table
    if args.sub_group is not None:
        for group in vals:
            table = PrettyTable(
                ["Retriever", "Reranker", "Strategy", "Layer", "Perplexity"]
            )

            # Add Target Stats
            if args.baseline is not None:
                table.add_row(["~", "~", "~", "~", targets[args.baseline]])
            if args.random is not None:
                table.add_row(
                    [args.retriever, "~", "random", "~", targets[args.random]]
                )
            if args.first is not None:
                table.add_row([args.retriever, "~", "first", "~", targets[args.first]])
            if args.oracle is not None:
                table.add_row(
                    [args.retriever, "~", "oracle", "~", targets[args.oracle]]
                )

            # Add Group Stats
            for i, perp in enumerate(group):
                table.add_row(
                    [args.retriever, args.reranker, args.strategy, i + 1, perp]
                )
            print(table)

    if args.run_names is not None:
        raise NotImplementedError

    # Plot results
    if args.plot_type == "plot":
        df = {"x": [x + 1 for x in range(args.layer_lower, args.layer_upper + 1)]}
        for name, vals in zip(args.sub_group, vals):
            df[name] = vals
        df = pd.DataFrame(df)
        for key in df.keys():
            if key != "x":
                sns.lineplot(data=df, x="x", y=df[key])

        for target in args.plot_targets:
            plt.axhline(y=targets[str(getattr(args, target))], linestyle="--")

        plt.xlabel("Layer")
        plt.ylabel("Perplexity")

    if args.plot_type == "bar":
        assert len(vals) == 1, "Bar does not support multiple groups"

    if args.plot_type is not None:
        assert args.plot_name is not None, "Plot name is not specified"

        plt.title(args.plot_name)
        if args.plot_style is not None:
            sns.set_style(args.plot_style)
        if args.plot_save is not None:
            plt.savefig(os.path.join(args.benchmark_dir, f"{args.plot_save}.png"))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Runs
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark")
    parser.add_argument(
        "--sub-group", type=str, nargs="+", default=None
    )  # choose between this and run-names
    parser.add_argument("--run-names", type=str, nargs=-1, default=None)
    parser.add_argument("--layer-lower", type=int, default=None)
    parser.add_argument("--layer-upper", type=int, default=None)

    # Targets
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--random", type=str, default=None)
    parser.add_argument("--first", type=str, default=None)
    parser.add_argument("--oracle", type=str, default=None)

    # Table
    parser.add_argument("--retriever", type=str, default=None)
    parser.add_argument("--reranker", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)

    # Plots
    parser.add_argument("--plot-style", type=str, default=None)
    parser.add_argument("--plot-type", type=str, choices=PLOT_CHOICES, default=None)
    parser.add_argument(
        "--plot-targets", type=str, nargs="+", choices=TARGET_CHOICES, default=None
    )
    parser.add_argument("--plot-name", type=str, default=None)
    parser.add_argument("--plot-save", type=str, default=None)
    args = parser.parse_args()

    main(args)
