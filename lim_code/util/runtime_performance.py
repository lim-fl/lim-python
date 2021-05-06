from lim_code.experiment import experiment
from sklearn import ensemble, neighbors, linear_model, svm

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
n_clients = 2
top_k = 50
adversarial_proportion = 0.5
k_best_features = 200
n_rounds = 1

def main():
    baseline_knn()
    baseline_lr()
    baseline_svm()
    baseline_rf200()
    baseline_rf100()
    baseline_rf50()


def baseline_knn():
    runtime_experiment("KNN_n3")


def baseline_lr():
    runtime_experiment("LR_c1")


def baseline_svm():
    runtime_experiment("SVM_c1")


def baseline_rf200():
    runtime_experiment("RF_n200")


def baseline_rf100():
    runtime_experiment("RF_n100")


def baseline_rf50():
    runtime_experiment("RF_n50")


def runtime_experiment(model):
    baseline = models[model]
    base_names = [b for b in models if b is not model]
    base = [models[b] for b in base_names]

    name = f"baseline_{model}_base_" + '_'.join(base_names) + f"_{k_best_features}_best_features"
    experiment(
        name=name,
        baseline_model=baseline,
        base_models=base,
        k_best_features=k_best_features,
        top_k=top_k,
        n_rounds=n_rounds,
        n_clients=n_clients,
        p_install=p_install,
        p_malware=p_malware,
        adversarial_proportion=adversarial_proportion)


if __name__ == "__main__":
    main()
