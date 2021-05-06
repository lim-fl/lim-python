import numpy as np
import cvxpy as cp

from lim_code.model_train.safew import safew


def poison_weights(
        baseline_model,
        base_models,
        lim_weights,
        cloud_weights,
        compromised_weights,
        X_malicious):
    """
    Computes a set of adversarial weights.

    Adversarial weights maximize the chances of LiM misclassifying X_malicious.

    It tries to augment the weights of the base models that are wrong.
    """

    compromised_weights = np.array(compromised_weights)

    wrong_models = np.asarray([
        model.predict(X_malicious)[0] == -1
        for model in base_models
    ])
    if wrong_models.sum() > 0:
        n = lim_weights.shape[0]
        weights = cp.Variable(n, nonneg=True)

        n_wrong_models = wrong_models.sum()
        unitary_vector = np.ones((1, n_wrong_models))
        threshold = 0.5 + 1e-3  # Weights are rounded to 3 decimals

        local_constraint = unitary_vector @ weights[wrong_models] >= threshold

        mean_client_weights = np.mean(
            compromised_weights,
            axis=0)
        # Iterative mean http://www.heikohoffmann.de/htmlthesis/node134.html
        # Add new weights to the mean, to make sure it does not fades away 
        c_t = mean_client_weights
        t = mean_client_weights.shape[0]
        guessed_client_weights = c_t + (weights - c_t)/(t+1)
        clients_constraint = unitary_vector @ guessed_client_weights[wrong_models] >= threshold

        median_cloud = (cloud_weights+guessed_client_weights)/2  # Median of n=2 is just dividing their sum by two
        cloud_constraint = unitary_vector @ median_cloud[wrong_models] >= threshold

        objective = cp.Minimize(cp.norm(lim_weights - weights))
        constraints = [
            np.ones((1, n)) @ weights == 1,
            local_constraint,
            clients_constraint,
            cloud_constraint,
        ]

        return best_poisoned_weights(
            objective=objective,
            constraints=constraints,
            variable=weights,
            lim_weights=lim_weights)
    else:
        return lim_weights


def best_poisoned_weights(objective, constraints, variable, lim_weights):
    if len(constraints) > 1:
        try:
            weights = solve_problem(
                objective,
                constraints,
                variable)
            return np.round_(weights, decimals=3)
        except UnsolvedException:
            relaxed_constraints = constraints[:-1]
            return best_poisoned_weights(
                objective,
                relaxed_constraints,
                variable,
                lim_weights)
    else:
        return lim_weights


class UnsolvedException(cp.SolverError):
    pass


def solve_problem(objective, constraints, result):
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(verbose=False)
    except cp.SolverError as e:
        raise UnsolvedException from e

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return result.value
    else:
        raise UnsolvedException


def is_poisoned(
        baseline_model,
        base_models,
        X_malicious,
        poisoned_weights,
):

    poisoned_pred, _ = safew(
        baseline_model=baseline_model,
        base_models=base_models,
        X=X_malicious,
        weights=poisoned_weights)

    return poisoned_pred[0] == -1
