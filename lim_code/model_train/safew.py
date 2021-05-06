import cvxpy as cp
import numpy as np


def safew(baseline_model, base_models, X, weights=None):
    # Compute baseline and base predictions
    baseline_pred, candidate_pred = safew_base_predictions(
        baseline_model=baseline_model,
        base_models=base_models,
        X=X
    )

    # Compute weights
    if weights is None:
        weights = safew_weights(baseline_pred, candidate_pred)

    # Compute SAFEW prediction
    safe_prediction = np.zeros((X.shape[0], 1))
    for i, weight in enumerate(weights):
        safe_prediction[:, 0] = safe_prediction[:,0] + weight * candidate_pred[:, i]

    safe_prediction[np.where(safe_prediction < 0)] = -1
    safe_prediction[np.where(safe_prediction >= 0)] = 1

    return safe_prediction, weights


def safew_base_predictions(baseline_model, base_models, X):
    baseline_pred = baseline_model.predict(X)
    candidate_pred = np.array(
        [model.predict(X) for model in base_models], dtype=np.double
    ).T
    candidate_pred[candidate_pred < 0] = -1
    candidate_pred[candidate_pred >= 0] = 1

    return baseline_pred, candidate_pred


def safew_weights_guo(baseline_pred, candidate_pred):
    candidate_num = candidate_pred.shape[1]

    H = np.dot(baseline_pred.T, candidate_pred)
    x = cp.Variable(candidate_num)
    lb = np.ones((1, candidate_num))
    objective = cp.Minimize(H@x)
    constraints = [x>=0, lb@x==1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    x_value = x.value
    weights = np.round_(x_value, decimals=3)

    return weights

def safew_weights(baseline_pred, candidate_pred):
    candidate_num = candidate_pred.shape[1]
    m = candidate_num
    n_samples = candidate_pred.shape[0]
    n = n_samples
    C = candidate_pred.T @ candidate_pred
    delta = 0.5
    epsi = cp.Variable(n)
    weights = cp.Variable(m)
    objective = cp.Minimize(
        np.ones((1, n)) @ epsi + cp.norm(candidate_pred @ weights, p=1))
    constraints = [
        weights >= 0,
        np.ones((1, m)) @ weights == 1,
        cp.multiply(baseline_pred, (candidate_pred @ weights)) >= np.ones(n) - epsi,
        C @ weights >= np.ones(m) * delta,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    weights = np.round_(weights.value, decimals=3)

    return weights
