import pandas as pd
from sklearn import feature_selection
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict
import itertools
import pickle
from androguard.core.bytecodes.apk import APK
from lim_code.lim_logger import logger

from lim_code.dataset_generate.malware_data import (
    LiMData,
    load_feature_names,
    FEATURE_NAMES_PICKLE
)

CATEGORY_FEATURES_PICKLE = "category_features.pickle"


def categories_table(selected_features, n_features=[100, 200, 500]):
    category_features = get_features_in_categories()
    counts_dict = {}

    for n in n_features:
        row_counts = {category: 0 for category in category_features}
        for feature in selected_features[:n]:
            for category in category_features:
                if feature in category_features[category]:
                    row_counts[category] += 1
            occurrences = sum([feature in category_features[c] for c in category_features])
            categories = [c for c in category_features if feature in category_features[c]]
            if occurrences > 1:
                logger.info(f"Feature {feature} is in {occurrences} categories: {categories}")
        counts_dict[n] = row_counts

    df = pd.DataFrame(data=counts_dict)
    return df


def load_backup(path):
    with path.open(mode="rb") as f:
        return pickle.load(f)


def map_features_to_categories(files):
    with Pool() as p:
        features_per_file = p.map(lim_features_categories, files)

    features_per_file = [p for p in features_per_file if p is not None]
    save_backup(
        Path("map_features_categories_per_file.pickle"),
        features_per_file)

    return category_features(features_per_file)


def category_features(features_per_file):
    """features are a dictionary category: features, for multiple categories (see lim_features_categories)."""
    category_features = defaultdict(set)
    # import pdb
    # pdb.set_trace()
    for features in features_per_file:
        for category in features:
            category_features[category].update(features[category])

    return category_features


def save_backup(backup_f, variable):
    with backup_f.open(mode="wb") as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)


def get_features_in_categories():
    backup_f = Path(CATEGORY_FEATURES_PICKLE)
    if backup_f.exists():
        return load_backup(backup_f)
    else:
        dataset = Path("dataset/raw")
        files = dataset.glob("**/*.apk")
        category_features = map_features_to_categories(files)
        save_backup(backup_f, category_features)

        return category_features


def lim_features_categories(apk_filepath):
    try:
        apk = APK(apk_filepath)
        info = {
            'declared permissions': sorted(apk.get_permissions()),
            'activities': apk.get_activities(),
            'services': apk.get_services(),
            'intent filters': apk.get_intent_filters('receiver', ''),
            'content providers': apk.get_providers(),
            'broadcast receivers': apk.get_receivers(),
            'hardware components': apk.get_features()}

        for category in info:
            info[category] = [
                feature.replace(".", "_").lower()
                for feature in info[category]
            ]

        return info
    except:
        # We just do not process the APK
        pass


def main():
    feature_names = load_feature_names(FEATURE_NAMES_PICKLE)
    for k in [100, 200, 500]:
        lim_data = LiMData(
            feature_selector=feature_selection.SelectKBest(
                feature_selection.chi2,
                k=k)).get()

        feature_selector = lim_data.fit_selector()
        selection_mask = feature_selector.get_support()
        selected_features = list(itertools.compress(feature_names, selection_mask))

        with Path(f"top_{k}_features.txt").open("w") as f:
            print(*selected_features, sep="\n", file=f)

    table = categories_table(selected_features)
    with Path("categories_table.tex").open(mode="w") as f:
        print(
            table.to_latex(caption="Number of features per category"),
            file=f)


if __name__ == "__main__":
    main()
