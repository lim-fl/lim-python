from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from lim_code.lim_logger import logger


def simple_report(df):
    errors = []
    Line = namedtuple("Line",
                      ["error",
                       "place",
                       "classifier",
                       "average",
                       "std",
                       "total"])
    df["precision"] = df.tp/(df.tp+df.fp)
    df["recall"] = df.tp/(df.tp+df.fn)

    for place in ["client", "cloud"]:
        client = df[df.place == place]
        for classifier in df.classifier.unique():
            classifier_df = client[client.classifier == classifier]
            line = Line(error="False positives",
                        place=place,
                        classifier=classifier,
                        # weights=classifier_df.weights,
                        average=classifier_df.groupby(by="federation_round").fp.mean().mean(),
                        std=classifier_df.groupby(by="federation_round").fp.std().mean(),
                        total=classifier_df.fp.mean(),
            )
            errors.append(line)

            with_malware = classifier_df[classifier_df.tp + classifier_df.fn > 0]
            if with_malware.shape[0] > 0:
                # line = Line(error="Poisoned",
                #             place=place,
                #             classifier=classifier,
                #             average=classifier_df[["federation_round", "poisoned"]].astype(int).groupby(by="federation_round").poisoned.sum().mean(),
                #             std=classifier_df.groupby(by="federation_round").poisoned.sum().std(),
                #             total=classifier_df.poisoned.astype(int).sum().mean(),)
                # errors.append(line)

                line = Line(error="Recall",
                            place=place,
                            classifier=classifier,
                            # weights=classifier_df.weights,
                            average=with_malware.groupby(by="federation_round").recall.mean().mean(),
                            std=with_malware.groupby(by="federation_round").recall.std().mean(),
                            total=(with_malware.tp + with_malware.fp).mean(),)
                errors.append(line)

                line = Line(error="Overall performance (F1 score)",
                            place=place,
                            classifier=classifier,
                            # weights=classifier_df.weights,
                            average=with_malware.groupby(by="federation_round").f1.mean().mean(),
                            std=with_malware.groupby(by="federation_round").f1.std().mean(),
                            total=(with_malware.tp + with_malware.fn + with_malware.fp + with_malware.tn).mean(),)
                errors.append(line)
    report = pd.DataFrame(data=errors)
    for error in report.error.unique():
        logger.info(f"{error}\n{report[report.error == error]}")

    return report.to_string()


def plot_place(df, place, metrics=["recall", "precision", "fp", "f1"]):
    df["precision"] = df.tp/(df.tp+df.fp)
    df["recall"] = df.tp/(df.tp+df.fn)

    df = df[df.place == place]
    by = df.groupby(by=["federation_round", "classifier"]).mean().unstack()

    styles = ['bs-', 'ro-', 'y^-', 'k+-', 'v:-']
    style = styles[:df.classifier.unique().shape[0]]

    # metric = "fp"
    # if metric in by:
    #     by[metric].plot(title=metric, style=style)
    #     f = Path(f"{place}_{metric}.png")
    #     plt.savefig(str(f))
    #     plt.close()

    with_malware = df[df.tp + df.fn > 0]
    if with_malware.shape[0] > 0:
        for metric in metrics:
            by = with_malware.groupby(by=["federation_round", "classifier"]).mean().unstack()
            if metric in by:
                results_metric = by[metric]
                if 'poisoned' in results_metric.columns:
                    results_metric = results_metric.reindex(columns= ['baseline', 'no-lim', 'lim', 'poisoned'])
                else:
                    results_metric = results_metric.reindex(columns= ['baseline', 'no-lim', 'lim'])
                results_metric = results_metric.rename(columns={"no-lim": "SAFEW", "lim": "LiM"})

                ylim = None if metric is "fp" else [-0.1, 1.1]
                results_metric.plot(
                    title=metric,
                    ylim=ylim,
                    style=style,
                )

                f = Path(f"{place}_{metric}.png")
                plt.savefig(str(f))
                plt.close()


def plot_cloud(df):
    plot_place(df, "cloud")


def plot_clients(df):
    plot_place(df, "client")
