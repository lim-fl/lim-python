from pathlib import Path
import itertools
from multiprocessing import Pool
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import configparser

from lim_code.lim_logger import logger


from lim_code.dataset_generate.feature_extraction import lim_features

config = configparser.ConfigParser()
config.read("lim_code/generate_dataset.conf")


def main():
    f_feature_lists = Path("feature_lists.pickle")
    f_app_names = Path("app_names.pickle")
    if f_feature_lists.exists() and f_app_names.exists():
        logger.info("Found pickle files for feature_lists and app_names")
        with f_feature_lists.open(mode="rb") as f:
            feature_lists = pickle.load(f)
        with f_app_names.open(mode="rb") as f:
            app_names = pickle.load(f)
    else:
        app_names, feature_lists = get_features(config.get("dataset", "path"))
        with f_feature_lists.open(mode="wb") as f:
            pickle.dump(feature_lists, f, pickle.HIGHEST_PROTOCOL)
        with f_app_names.open(mode="wb") as f:
            pickle.dump(app_names, f, pickle.HIGHEST_PROTOCOL)

    features, labels, feature_names = vectorize_features(
        feature_lists, label=config.get("dataset", "label"))

    sparse.save_npz("features.npz", features)
    sparse.save_npz("labels.npz", labels)
    with Path("feature_names.pickle").open(mode="wb") as f:
        pickle.dump(feature_names, f, pickle.HIGHEST_PROTOCOL)


def get_features(dataset_path):
    dataset = Path(dataset_path)
    files = dataset.glob("**/*.apk")
    with Pool() as p:
        results = p.map(lim_features, files)

    names = []
    features = []
    for r in results:
        if r[0] and r[1]:
            names.append(r[0])
            features.append(r[1])

    return names, features


def vectorize_features(as_lists, label):
    vectorizer = TfidfVectorizer(
        use_idf=False,
        norm=None,
        binary=True,
    )  # Get only 1s and 0s
    as_documents = [features_document(row) for row in as_lists if isinstance(row, list)]
    corpus = make_corpus(as_documents)
    vectorized = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    if label in feature_names:
        label_index = feature_names.index(label)
        sparse_columns = vectorized.tocsc()
        labels = sparse_columns.getcol(label_index)
        feature_names_without_label = itertools.dropwhile(
            lambda f: f is label, feature_names
        )
        features = sparse_columns[
            :, [i for i in range(len(feature_names)) if i != label_index]
        ]
        return features, labels, feature_names
    else:
        raise ValueError(f"label {label} is not in the list of features")


def features_document(feature_list):
    return " ".join(feature_list).replace(".", "_")


def make_corpus(documents):
    for doc in documents:
        yield doc


if __name__ == "__main__":
    main()
