from lim_code.experiment import experiment
from sklearn import ensemble, neighbors, linear_model, svm


def main(adversarial_proportion=0):
    models = {
        "KNN_n3": neighbors.KNeighborsClassifier(n_neighbors=3),
        "LR_c1": linear_model.LogisticRegression(C=1),
        "SVM_c1": svm.LinearSVC(C=1),
        "RF_n200": ensemble.RandomForestClassifier(n_estimators=200),
        "RF_n100": ensemble.RandomForestClassifier(n_estimators=100),
        "RF_n50": ensemble.RandomForestClassifier(n_estimators=50),
    }
    p_install = 0.6
    p_malware = 0.1
    n_clients = 200
    top_k = 50
    n_max_apps_per_round = 5
    k_best_features = [100, 200, 500]
    n_rounds = 50

    for model in models:
        baseline = models[model]
        base_names = [b for b in models if b is not model]
        base = [models[b] for b in base_names]

        for k in k_best_features:
            name = f"baseline_{model}_base_" + '_'.join(base_names) + f"_{k}_best_features"
            experiment(
                name=name,
                baseline_model=baseline,
                base_models=base,
                k_best_features=k,
                top_k=top_k,
                n_rounds=n_rounds,
                n_clients=n_clients,
                p_install=p_install,
                p_malware=p_malware,
                adversarial_proportion=adversarial_proportion,
                n_max_apps_per_round=n_max_apps_per_round,
            )


if __name__ == "__main__":
    main()
