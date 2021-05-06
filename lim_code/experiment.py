from lim_code.model_train.lim import LiM
from lim_code.model_test import results
from lim_code.model_test.evaluate_lim import plot_cloud, plot_clients
from lim_code.model_test.evaluate_lim import simple_report
from lim_code.lim_logger import logger

from lim_code.dataset_generate.malware_data import LiMData

from sklearn import ensemble, feature_selection
import gc

from datetime import datetime
import pathlib
import shutil

BASELINE_MODEL = ensemble.RandomForestClassifier(n_estimators=200)
BASE_MODELS = [
    ensemble.RandomForestClassifier(n_estimators=200),
    ensemble.RandomForestClassifier(n_estimators=200),
    ensemble.RandomForestClassifier(n_estimators=200),
    ensemble.RandomForestClassifier(n_estimators=200),
    ensemble.RandomForestClassifier(n_estimators=200),
]

time_format = "%a%d_%b_%Y_%H_%M"


def move_results(name):
    files = [
        "client_f1.png",
        "client_precision.png",
        "client_recall.png",
        "client_fp.png",
        "cloud_f1.png",
        "cloud_precision.png",
        "cloud_recall.png",
        "cloud_fp.png",
        "results.csv",
        "report.txt",
    ]

    now_str = datetime.now().strftime(time_format)
    results_dir = pathlib.Path("results") / name / now_str
    results_dir.mkdir(parents=True)
    for f in files:
        path = pathlib.Path(f)
        if path.exists():
            shutil.move(str(path), str(results_dir))


def experiment(
        name,
        baseline_model=BASELINE_MODEL,
        base_models=BASE_MODELS,
        top_k=50,
        top_k_features=20,
        n_rounds=50,
        n_clients=500,
        p_install=0.6,
        p_malware=0.1,
        unlabeled_data_proportion=0.8,
        client_unlabeled_proportion=0.8,
        k_best_features=100,
        adversarial_proportion=0.5,
        n_max_apps_per_round=5,
):

    logger.info(f"Name of the experiment: {name}")
    logger.info(f"Baseline model: {baseline_model}")
    logger.info(f"Base models: {base_models}")
    logger.info(f"K best features: {k_best_features}")
    logger.info(f"Number of popular apps: {top_k}")
    logger.info(f"Number of rounds: {n_rounds}")
    logger.info(f"Number of clients: {n_clients}")
    logger.info(f"Install an app with probability {p_install}")
    logger.info(f"Install a malware app with probability {p_malware}")
    logger.info(f"Top k features: {top_k_features}")
    logger.info(f"Proportion of testing data: {unlabeled_data_proportion}")
    logger.info(f"Proportion of testing data for clients: {client_unlabeled_proportion}")

    feature_selector = feature_selection.SelectKBest(
        feature_selection.chi2,
        k=k_best_features)
    lim_data = LiMData(
        unlabeled_data_proportion=unlabeled_data_proportion,
        client_unlabeled_proportion=client_unlabeled_proportion,
        top_k=top_k,
        top_k_features=top_k_features,
        feature_selector=feature_selector,
        random_state=42).get()
    lim = LiM(
        data=lim_data,
        baseline_model=baseline_model,
        base_models=base_models,
        n_rounds=n_rounds,
        n_clients=n_clients,
        p_install=p_install,
        p_malware=p_malware,
        adversarial_proportion=adversarial_proportion,
        n_max_apps_per_round=n_max_apps_per_round,
    )
    df = lim.run_federation()
    df.to_csv("results.csv")

    with pathlib.Path("report.txt").open("w") as f:
        f.write(simple_report(df))
    logger.info(f"{simple_report(df)}")

    plot_cloud(df)
    plot_clients(df)

    move_results(name)
    results.clear()
    gc.collect()
