from multiprocessing import Pool

import numpy as np
from scipy import sparse
from collections import defaultdict

from lim_code.lim_logger import logger
from lim_code.model_train.safew import safew
from lim_code.model_train.malware_classifier import MalwareSAFEWClient
from lim_code.model_test import results, evaluate_lim
from lim_code.model_train.adversarial import poison_weights, is_poisoned


class LiM(object):
    """
    The LiM classifier is a federated binary SAFEW for malware
    """

    def __init__(self,
                 data,
                 baseline_model,
                 base_models,
                 p_malware=0.1,
                 p_install=0.2,
                 unlabeled_data_proportion=0.99,
                 client_proportion=0.5,
                 n_clients=20,
                 adversarial_proportion=0.5,
                 n_max_apps_per_round=5,
                 n_rounds=5):
        """
        p_malware: probability of a client installing a malware app
        n_clients: number of clients in the federation
        """
        super(LiM, self).__init__()

        self.data = data
        self.cloud_weights = None
        self.n_clients = n_clients
        self.adversarial_proportion = adversarial_proportion
        self.clients = []
        self.n_max_apps_per_round = n_max_apps_per_round
        self.p_malware = p_malware
        self.p_install = p_install
        self.n_rounds = n_rounds
        self.federation_round = 0

        self.baseline_model = baseline_model
        self.base_models = base_models

    def run_federation(self):
        logger.info(f"Round {self.federation_round}")
        self.create_cloud()
        if self.adversarial_proportion > 0:
            # Simulate a blank slate adversarial client
            client = self.create_client_worker([0, True])
            self.data.select_malicious(
                baseline_model=self.baseline_model,
                base_models=self.base_models,
                weights=client.weights)

        # Let adversarial clients find an malicious app in later rounds
        self.create_clients()

        while self.federation_round < self.n_rounds:
            self.federation_round += 1
            logger.info(f"Round {self.federation_round}")
            self.federate_cloud()
            self.federate_clients()
            df_tmp = results.to_df()
            df_tmp.to_csv("results_tmp.csv")
            evaluate_lim.plot_cloud(df_tmp)
            evaluate_lim.plot_clients(df_tmp)

        return results.to_df()

    def create_cloud(self):
        data = self.data
        X_train = data.X_train
        y_train = data.y_train
        self.baseline_model.fit(X_train, y_train)
        for model in self.base_models:
            model.fit(X_train, y_train)

        X = data.cloud_X_test
        y_true = data.cloud_y_test

        pred, w = safew(
            baseline_model=self.baseline_model,
            base_models=self.base_models,
            X=X,)
        baseline_pred = self.baseline_model.predict(X)
        predictions = {
            "baseline": baseline_pred,
            "no-lim": pred,
            "lim": pred,
            "no-privacy": pred,
        }
        weights = {
            "no-lim": w,
            "lim": w,
            "no-privacy": w,
        }
        labels = {
            "baseline": y_true,
            "no-lim": y_true,
            "lim": y_true,
            "no-privacy": y_true,
        }
        results.add(
            place="cloud",
            federation_round=self.federation_round,
            id_=-1,
            poisoned=False,
            labels=labels,
            predictions=predictions,
            weights=weights,
        )
        self.cloud_weights = weights["lim"]

    def cloud_better_than_baseline(self):
        df = results.to_df()
        df = df[
            (df.federation_round == 0) &
            (df.place == "cloud")
        ]

        baseline_accuracy = df[df.classifier == "baseline"].accuracy.to_numpy()[0]
        lim_accuracy = df[df.classifier == "lim"].accuracy.to_numpy()[0]

        return lim_accuracy > baseline_accuracy

    def create_clients(self):
        n_adversarial = int(self.adversarial_proportion * self.n_clients)
        n_honest = self.n_clients - n_adversarial
        args = [
            [id_, malicious]
            for id_, malicious in zip(
                    list(range(self.n_clients)),
                    [True]*n_adversarial
                    + [False]*n_honest
                    )
            ]

        with Pool() as p:
            clients = p.map(self.create_client_worker, args)

        for client in clients:
            X = client.X_installed
            y_true = client.y_installed
            pred, w = safew(
                baseline_model=self.baseline_model,
                base_models=self.base_models,
                X=X,)
            client.weights = w

        compromised_weights = [c.weights for c in clients if c.malicious]
        for client in clients:
            baseline_pred = self.baseline_model.predict(X)
            predictions = {
                "baseline": baseline_pred,
                "no-lim": pred,
                "lim": pred,
            }
            weights = {
                "no-lim": w,
                "lim": w,
            }
            labels = {
                "baseline": y_true,
                "no-lim": y_true,
                "lim": y_true,
            }
            if client.malicious:
                if self.data.X_malicious is None:
                    self.data.select_malicious(
                        baseline_model=self.baseline_model,
                        base_models=self.base_models,
                        weights=client.weights)
                    found_app = self.data.X_malicious is not None
                    # Install new selected malicious app in all malicious clients
                    if found_app:
                        for c in clients:
                            if c.malicious: c.install_malicious_app(self.data.X_malicious)
                        
                if self.data.X_malicious is not None:
                    poisoned_weights = poison_weights(
                        baseline_model=self.baseline_model,
                        base_models=self.base_models,
                        lim_weights=weights["lim"],
                        cloud_weights=self.cloud_weights,
                        compromised_weights=compromised_weights,
                        X_malicious=self.data.X_malicious,
                    )
                    poisoned_pred, poisoned_weights = safew(
                        baseline_model=self.baseline_model,
                        base_models=self.base_models,
                        X=X,
                        weights=poisoned_weights,
                    )
                    labels["poisoned"] = y_true
                    weights["poisoned"] = poisoned_weights
                    predictions["poisoned"] = poisoned_pred
                    client.weights = poisoned_weights

                    poisoned = is_poisoned(
                        baseline_model=self.baseline_model,
                        base_models=self.base_models,
                        poisoned_weights=client.weights,
                        X_malicious=self.data.X_malicious,
                    )

        # No adversarial clients or no adversarial sample
        if n_adversarial == 0 or self.data.X_malicious is None:
            poisoned = False
        results.add(
            place="client",
            federation_round=self.federation_round,
            id_=client.id_,
            labels=labels,
            poisoned=poisoned,
            predictions=predictions,
            weights=weights)
        self.clients = clients

    def create_client_worker(self, args):
        id_, malicious = args
        return MalwareSAFEWClient(
            self.data,
            self.cloud_weights,
            malicious,
            id_,
        )

    def federate_cloud(self):
        client_weights = np.mean(
            [client.weights for client in self.clients],
            axis=0)

        data = self.data
        X = data.cloud_X_test
        y_true = data.cloud_y_test
        N = data.cloud_N_test

        no_lim_pred, no_lim_weights = safew(
            baseline_model=self.baseline_model,
            base_models=self.base_models,
            X=X,
        )

        new_weights = np.median(
            [client_weights, no_lim_weights],
            axis=0)

        lim_pred, lim_weights = safew(
            baseline_model=self.baseline_model,
            base_models=self.base_models,
            X=X,
            weights=new_weights,
        )
        if self.federation_round > 1:
            self.previous_lim_weights = self.last_lim_weights
            membership_tp, membership_fp = self.infer_membership(X=X, N=N)
        else:
            membership_tp = -1
            membership_fp = -1
        self.last_lim_weights = lim_weights

        X_global, y_global = self.no_privacy_dataset(
            X_cloud=X,
            y_cloud=y_true,
            N_cloud=N,
            clients=self.clients,
        )

        no_privacy_pred, no_privacy_weights = safew(
            baseline_model=self.baseline_model,
            base_models=self.base_models,
            X=X_global,
        )
        predictions = {
            "baseline": self.baseline_model.predict(X),
            "no-lim": no_lim_pred,
            "lim": lim_pred,
            "no-privacy": no_privacy_pred,
        }
        weights = {
            "no-lim": no_lim_weights,
            "lim": lim_weights,
            "no-privacy": no_privacy_weights,
        }
        labels = {
            "baseline": y_true,
            "no-lim": y_true,
            "lim": y_true,
            "no-privacy": y_global,
        }
        results.add(
            place="cloud",
            federation_round=self.federation_round,
            id_=-1,
            labels=labels,
            poisoned=False,
            predictions=predictions,
            weights=weights,
            membership_tp=membership_tp,
            membership_fp=membership_fp,
        )
        self.cloud_weights = lim_weights

    def infer_membership(self, X, N):
        membership_tp = 0
        membership_fp = 0
        for membership_x, membership_N in zip(X, N):
            if self.federation_round == 1:
                membership_x = sparse.vstack((self.data.X_preinstalled, membership_x))
            _, membership_weights = safew(
                baseline_model=self.baseline_model,
                base_models=self.base_models,
                X=membership_x,
            )
            for client in self.clients:
                # Undo the average operation with last cloud weights
                client_nolim_weights = (2*client.weights) - self.last_lim_weights
                if self.federation_round > 1:
                    client_previous_nolim_weights = (2*client.previous_weights) - self.previous_lim_weights
                    same_weights = np.all(membership_weights == client_nolim_weights - client_previous_nolim_weights)
                else:
                    same_weights = np.all(membership_weights == client_nolim_weights)
                if same_weights:
                    if membership_N in client.N_installed:
                        membership_tp += 1
                    else:
                        membership_fp += 1

            return membership_tp, membership_fp

    def no_privacy_dataset(self, X_cloud, y_cloud, N_cloud, clients):
        client_apps = None
        client_labels = None
        client_N = None
        for client in clients:
            if client_apps is None:
                client_apps = client.X_installed
                client_labels = client.y_installed
                client_N = client.N_installed
            else:
                client_apps = sparse.vstack((client_apps, client.X_installed))
                client_labels = np.append(client_labels, client.y_installed)
                client_N = client_N + client.N_installed
        X_global = sparse.vstack((X_cloud, client_apps))
        y_global = np.append(y_cloud, client_labels)
        N_global = N_cloud + client_N

        d = defaultdict(list)
        for i, item in enumerate(N_global):
            d[item].append(i)
        deduplicated_index = [d[item][0] for item in d]
        X_global = X_global.tocsr()[deduplicated_index]
        y_global = y_global[deduplicated_index]
        # N_global = [name for name, i in enumerate(N_global) if i in deduplicated_index]

        return X_global, y_global
    
    def federate_clients(self):
        compromised_weights = [c.weights for c in self.clients if c.malicious]
        args = [[self.n_max_apps_per_round,
                 self.p_install,
                 self.p_malware,
                 client,
                 self.baseline_model,
                 self.base_models,
                 self.cloud_weights,
                 self.federation_round,
                 compromised_weights,
                 self.data.X_malicious]
                for client in self.clients]

        with Pool() as p:
            client_r = p.map(update_client_worker, args)

        for worker in client_r:
            for i, client in enumerate(self.clients):
                if worker["client"].id_ == client.id_:
                    self.clients[i] = worker["client"]
            weights = worker["weights"]
            if worker["client"].malicious and self.data.X_malicious is not None and "poisoned" in weights:
                poisoned_weights = weights["poisoned"]
            else:
                poisoned_weights = weights["lim"]
            if self.data.X_malicious is None:
                self.data.select_malicious(
                    baseline_model=self.baseline_model,
                    base_models=self.base_models,
                    weights=weights["lim"])
                found_app = self.data.X_malicious is not None
                # Install new selected malicious app in all malicious clients
                if found_app:
                    for c in self.clients:
                        if c.malicious: c.install_malicious_app(self.data.X_malicious)
            if len(compromised_weights) > 0 and self.data.X_malicious is not None:
                poisoned = is_poisoned(
                    baseline_model=self.baseline_model,
                    base_models=self.base_models,
                    poisoned_weights=poisoned_weights,
                    X_malicious=self.data.X_malicious,
                )
            else:
                poisoned = False
            results.add(place="client",
                        federation_round=self.federation_round,
                        id_=worker["id"],
                        labels=worker["labels"],
                        predictions=worker["predictions"],
                        poisoned=poisoned,
                        weights=worker["weights"])


def update_client_worker(args):
    n_max_apps, p_install, p_malware, client, baseline_model, base_models, cloud_weights, federation_round, compromised_weights, X_malicious = args
    rng = np.random.default_rng()
    n_to_install_apps = rng.binomial(
        n=n_max_apps,
        p=p_install)
    for i in range(n_to_install_apps):
        client.install_app(p_malware=p_malware)
    X = client.X_installed
    y_true = client.y_installed

    no_lim_pred, no_lim_weights = safew(
        baseline_model=baseline_model,
        base_models=base_models,
        X=X,
    )

    new_weights = np.mean(
        [cloud_weights, no_lim_weights],
        axis=0)
    lim_pred, lim_weights = safew(
        baseline_model=baseline_model,
        base_models=base_models,
        X=X,
        weights=new_weights,
    )
    client.previous_weights = client.weights
    client.weights = lim_weights

    predictions = {
        "baseline": baseline_model.predict(X),
        "no-lim": no_lim_pred,
        "lim": lim_pred,
    }
    weights = {
        "no-lim": no_lim_weights,
        "lim": lim_weights,
    }
    labels = {
        "baseline": y_true,
        "no-lim": y_true,
        "lim": y_true,
    }
    if client.malicious and X_malicious is not None:
        poisoned_weights = poison_weights(
            baseline_model=baseline_model,
            base_models=base_models,
            lim_weights=lim_weights,
            cloud_weights=cloud_weights,
            compromised_weights=compromised_weights,
            X_malicious=X_malicious,
        )
        poisoned_pred, poisoned_weights = safew(
            baseline_model=baseline_model,
            base_models=base_models,
            X=X,
            weights=poisoned_weights,
        )
        labels["poisoned"] = y_true
        weights["poisoned"] = poisoned_weights
        predictions["poisoned"] = poisoned_pred
        client.weights = poisoned_weights

    return {
        "client": client,
        "id": client.id_,
        "labels": labels,
        "predictions": predictions,
        "weights": weights,
    }
