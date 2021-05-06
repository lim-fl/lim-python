from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn import ensemble
from collections import namedtuple

from lim_code.dataset_generate.malware_data import LiMData
from lim_code.model_train.safew import safew

from lim_code.lim_logger import logger

models = {
    "RF N=50-1": ensemble.RandomForestClassifier(n_estimators=50),
    "RF N=50-2": ensemble.RandomForestClassifier(n_estimators=50),
    "RF N=50-3": ensemble.RandomForestClassifier(n_estimators=50),
    "RF N=50-4": ensemble.RandomForestClassifier(n_estimators=50),
    "RF N=50-5": ensemble.RandomForestClassifier(n_estimators=50),
}

lim_data = LiMData(
    unlabeled_data_proportion=0.8,
    client_unlabeled_proportion=0.8,
    random_state=42).get()

def train_all(models):
    X = lim_data.X_train
    y_true = lim_data.y_train
    for classifier in models:
        model = models[classifier]
        model.fit(X, y_true)
    return models

def test_all(models):
    X = lim_data.cloud_X_test
    y_true = lim_data.cloud_y_test

    for classifier in models:
        model = models[classifier]
        y_pred = model.predict(X)
        logger.info(scores(classifier, y_true, y_pred))

        baseline = classifier
        y_pred, weights = safew(
            baseline_model=models[baseline],
            base_models=[models[classifier] for classifier in models if classifier is not baseline],
            X=X,
        )

        logger.info(
            scores(
                f"SAFEW baseline {baseline}",
                y_true,
                y_pred
            ))
    

def scores(classifier, y_true, y_pred):
    Scores = namedtuple("Scores",
                        ["classifier",
                         "tn", "fp", "fn", "tp",
                         "f1", "accuracy"])
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    scores = Scores(
        classifier=classifier,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        f1=f1,
        accuracy=accuracy,
    )
    return scores

def add_model(name, model):
    if name not in models:
        models[name] = model
    model.fit(
        lim_data.X_train,
        lim_data.y_train
    )
    test_all(models)
