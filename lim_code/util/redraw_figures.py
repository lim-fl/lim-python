from lim_code.model_test.evaluate_lim import plot_place

import pandas as pd

from pathlib import Path

configurations = {
    "knn3": {
        "dir": "baseline_KNN_n3_base_LR_c1_SVM_c1_RF_n200_RF_n100_RF_n50_200_best_features",
        "poisoned": "Thu28_May_2020_17_20",
        "not_poisoned": "Thu28_May_2020_00_08"},
    "lr": {
        "dir": "baseline_LR_c1_base_KNN_n3_SVM_c1_RF_n200_RF_n100_RF_n50_200_best_features",
        "poisoned": "Wed24_Jun_2020_15_09",
        "not_poisoned": "Thu28_May_2020_02_45"},
    "svm": {
        "dir": "baseline_SVM_c1_base_KNN_n3_LR_c1_RF_n200_RF_n100_RF_n50_200_best_features",
        "poisoned": "Fri29_May_2020_05_03",
        "not_poisoned": "Thu28_May_2020_05_11"
    }
}

results = Path("results")
for classifier in configurations:
    configuration = configurations[classifier]
    metrics = ["fp", "f1"]
    place = "client"
    for integrity in ["poisoned", "not_poisoned"]:
        f = results / configuration["dir"] / configuration[integrity] / "results.csv"
        df = pd.read_csv(f)
        plot_place(df, place=place, metrics=metrics)
        for metric in metrics:
            png = Path(f"{place}_{metric}.png")
            png.rename(f"{place}_{metric}_200_features_baseline_{classifier}_{integrity}.png")
        
