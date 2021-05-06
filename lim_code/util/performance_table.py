import pandas as pd
import pathlib
from lim_code.experiment import time_format
from datetime import datetime
from lim_code.lim_logger import logger


def main(results_dir="results"):
    """
    Produces data for tables
    tab:performance-short-simulations,
    tab:performance-long-simulations
    in the paper.
    """

    honest_df, malicious_df = combine_csvs(results_dir)
    for df in [honest_df, malicious_df]:
        df["classifier"] = df.classifier.replace('no-lim', "SAFEW")
        df["classifier"] = df.classifier.replace('no-privacy', "Centralized SAFEW")
        df["classifier"] = df.classifier.replace('lim', "LiM")
        df["classifier"] = df.classifier.replace('baseline', "Baseline")

        metrics = ["fp", "f1"]
        for metric in metrics:
            df[metric] = pd.to_numeric(df[metric])
        decimals = {metric: 3 for metric in metrics}
        for place in df.place.unique():
            logger.info(f"Place: {place}")
            df_place = df[df.place == place]
            groups = [
                df_place.baseline,
                df_place.no_features,
                df_place.classifier
            ]
            mean = df_place[metrics].groupby(groups).mean()
            mean = mean.round(decimals=decimals)

            caption = f"Performance in {place}"
            if "poisoned" in df.classifier.unique():
                caption = ' '.join([caption, "with 50\% adversarial clients"])
            table = mean.unstack().to_latex(caption=f"{caption}.")

            logger.info(table)


def combine_csvs(results_dir):
    """
    Combines the results CSVs.

    It parses time, baseline classifier, and number of features.
    """

    names = [
        "classifier",
        "federation_round",
        "place",
        "id",
        "tn",
        "fp",
        "fn",
        "tp",
        "f1",
        "accuracy",
        "poisoned",
        "weights",
        "membership_tp",
        "membership_fp",
    ]
    honest_df = pd.DataFrame(columns=names)
    malicious_df = pd.DataFrame(columns=names)

    for f in pathlib.Path(results_dir).glob("**/results.csv"):
        df = pd.read_csv(f, names=names, header=0)
        time, configuration = [p.stem for p in list(f.parents)[0:2]]
        baseline = " ".join(configuration.split("_base_")[0].split("_")[1:])
        no_features = configuration.split("_")[-3]

        df["time"] = datetime.strptime(
            time,
            time_format
        )
        df["baseline"] = baseline
        df["no_features"] = no_features

        if "poisoned" in df.classifier.unique():
            malicious_df = pd.concat([malicious_df, df])
        else:
            honest_df = pd.concat([honest_df, df])

    return honest_df, malicious_df


if __name__ == '__main__':
    main()
