import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.integrate import quad, dblquad

def pMSE_tree(synth, original):
    """
    Calculate the propensity score MSE using a decision tree classifier.
    args:
        synth(np.array): synthetic data
        original(np.array): original data
    """
    synth = synth.reshape(-1, 1)
    original = original.reshape(-1, 1)
    n = synth.shape[0]
    m = original.shape[0]
    p_true = n / (n + m)
    p_true = np.ones(n + m) * p_true
    X = np.concatenate([synth, original])
    y = np.concatenate([np.ones(n), np.zeros(m)])

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Fit the classifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    p_hat = clf.predict_proba(X)[:, 1]
    return mean_squared_error(p_true, p_hat)

def pMSE_int(syn, ori, syn_fn, ori_fn, window, dim=2, mu_syn=None):
    """
    Calculate the propensity score MSE using the intensity.
    args:
        syn(np.array): synthetic data
        ori(np.array): original data
        syn_fn(function): intensity function for the synthetic data
        ori_fn(function): intensity function for the original data
        window(tuple): window of the data
        mu_syn(float): integral of the synthetic intensity function over the window
    """
    n, m = syn.shape[0], ori.shape[0]
    points = np.concatenate([syn, ori])

    if dim == 2:
        def syn_fn_(x, y):
            return syn_fn(np.array([x, y]).reshape(1, -1))

        int_syn = syn_fn(points)
        int_ori = ori_fn(points[:, 0], points[:, 1])
        if mu_syn is None:
            mu_syn = dblquad(syn_fn_, *window)[0]
        mu_ori = dblquad(ori_fn, *window)[0]
    elif dim == 1:
        int_syn = syn_fn(points)
        int_ori = ori_fn(points)
        if mu_syn is None:
            mu_syn = np.trapz(syn_fn(np.linspace(*window, 1000)), np.linspace(*window, 1000))
        mu_ori = np.trapz(ori_fn(np.linspace(*window, 1000)), np.linspace(*window, 1000))
    else:
        raise ValueError("Dimension must be 1 or 2.")

    int_syn = int_syn / mu_syn
    int_ori = int_ori / mu_ori

    int_syn = int_syn.reshape(-1, 1)
    int_ori = int_ori.reshape(-1, 1)

    p_hat = int_syn / (int_syn + int_ori)
    p_true = np.ones(n + m) * n / (n + m)

    return mean_squared_error(p_true, p_hat)